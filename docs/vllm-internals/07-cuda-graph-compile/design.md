# 模块 07 · CUDA Graph / torch.compile 编译优化 —— 设计文档

> 范围：vLLM V1 如何用 **torch.compile（Dynamo + Inductor）做图捕获与算子融合**，又如何用 **piecewise CUDA graph** 把"除 attention 之外的静态计算"录成 CUDA graph、在运行期 replay，从而消除小 batch decode 时的 Python/CPU 启动开销与 kernel launch 开销。
> 架构：**V1 为主**。V1 的核心创新是把"整图 CUDA graph"换成"**torch.compile 切图 + piecewise CUDA graph**"，关键处对比 V0 的整图 cudagraph。
> 这是一个**横切模块**：它服务于所有模型的 forward，被 [模块 00](../00-request-lifecycle/design.md)（请求生命周期）的 `execute_model` 直接调用，与 [模块 02](../02-paged-attention-kvcache/design.md)（attention backend）在"attention 为何留在 graph 外"处强耦合。

---

## 1. 这个模块解决什么问题

LLM 推理 decode 阶段的一个反直觉事实：**当 batch 很小（比如 1~8 条序列各生成 1 个 token）时，GPU 算得飞快，瓶颈反而在 CPU**。

一次 forward 要在 GPU 上启动几百到上千个 kernel（每层 attention、norm、MLP、各种 elementwise）。每个 kernel 都要：

1. **Python 解释器**走一遍 `nn.Module.__call__` → `forward` → 各层逻辑（含大量属性查找、dict 构造、dtype 检查）。
2. **框架开销**：PyTorch 的 dispatcher、autograd（即便 inference 也有开销）、shape 推断。
3. **kernel launch**：每次 `cudaLaunchKernel` 是一次 CPU→GPU 的提交，约几微秒。

小 batch 时，单个 kernel 的实际 GPU 计算可能只要几微秒，**和它的 launch 开销同量级**。于是出现典型的 **CPU-bound / launch-bound** 现象：GPU 利用率上不去，时间都耗在"CPU 准备下一个 kernel"上。decode 又恰恰是 LLM 服务里占比最高的阶段（一个 prompt 生成几百 token，就是几百步小 batch decode）。

vLLM 要同时解决两层开销：

| 开销层 | 来源 | 解法 |
|---|---|---|
| **Python / 框架开销** | 解释器逐层走 forward、dispatcher、shape 推断 | **torch.compile**：把整个 forward 用 Dynamo 抓成一张 FX 图，Inductor 编译成融合 kernel，绕开逐层 Python |
| **kernel launch 开销** | 每个算子一次 `cudaLaunchKernel` | **CUDA graph**：把一连串 kernel **录制一次**，之后用一条 `replay()` 指令重放整段，CPU 几乎不参与 |

> 设计动机出处：
> - kernel launch 开销与 CUDA graph 的提出：NVIDIA, *"Getting Started with CUDA Graphs"*（developer.nvidia.com/blog，2019）——CUDA graph 把"定义一次、重放多次"的执行模型引入，消除重复 launch 的 CPU 提交开销。
> - torch.compile / TorchDynamo / Inductor：PyTorch 2.x 文档与 *"PyTorch 2.0: Our next generation release"*（pytorch.org，2023）——Dynamo 做 Python 字节码层面的图捕获，Inductor 做后端代码生成与算子融合。
> - vLLM 的 piecewise 方案：vLLM 官方博客 *"vLLM V1: A Major Upgrade to vLLM's Core Architecture"*（blog.vllm.ai，2025-01）明确把 "torch.compile integration + piecewise CUDA graphs" 列为 V1 的核心改造之一。

---

## 2. 设计目标与约束

**目标**

- **同时拿到两种加速**：既要 torch.compile 的算子融合 / Python 消除，又要 CUDA graph 的 launch 消除。
- **不牺牲 PagedAttention**：attention 的输入是变长序列、读写的是不定形的 paged KV cache，**不能**被强行塞进固定 shape 的 CUDA graph。
- **覆盖动态 batch size**：服务负载下 batch size 时刻在变，CUDA graph 却只能 replay 固定 shape ⇒ 需要"捕获若干代表性 size + 运行期 pad 到最近"。
- **编译/捕获只在启动期做一次**：运行期热路径上只能是"查表 + replay"，不能有编译。
- **可关闭**：调试、不支持的硬件、超长序列等场景要能一键退回 eager（`enforce_eager`）。

**约束（来自 CUDA graph 与 torch.compile 的硬性限制）**

- **CUDA graph 要求固定的 shape 与固定的输入显存地址**：replay 时 GPU 按录制时的指针读写，输入张量**必须住在同一块地址**（⇒ 持久 buffer 是前提，见 §3.4）。
- **graph 内不能有 data-dependent 的控制流 / 动态分配 / CPU 同步**：attention 里基于序列长度的分支、KV block 的动态寻址天然违反。
- **Dynamo 需要"full graph"才能稳定**：图里若有 graph break（无法捕获的 Python），会退化成多段、guard 检查变多，收益打折。
- **编译有时间与显存代价**：Inductor 编译几十秒起步；每个捕获的 size 都要占一份 CUDA graph 显存。

---

## 3. 核心设计思想

### 3.0 一句话总览

> **torch.compile 负责"把 forward 变成一张可融合、可切分的图"；piecewise CUDA graph 负责"把这张图里除 attention 外的静态段录下来、运行期重放"。两者叠加：融合省算力、graph 省 launch、pad 适配动态 batch。**

四个 CompilationLevel（`config.py:3251` `CompilationLevel`）：

| Level | 名称 | 含义 |
|---|---|---|
| 0 | `NO_COMPILATION` | 纯 eager，不 compile 不 capture |
| 1 | `DYNAMO_AS_IS` | 用 Dynamo 抓图，但编译交给用户给的 backend，整图 |
| 2 | `DYNAMO_ONCE` | Dynamo 只抓一次（用自定义 dispatcher 跳过 guard） |
| 3 | `PIECEWISE` | **V1 默认**：切图 + 分段编译 + piecewise CUDA graph |

V1 在 `VllmConfig.__post_init__`（`config.py:3794-3808`）里，只要 `VLLM_USE_V1` 且没 `enforce_eager`，就**强制把 level 设为 PIECEWISE**，并打开 `use_cudagraph`、`use_inductor`、设好 `splitting_ops`。

### 3.1 torch.compile：图捕获 + 融合（解决 Python / 框架开销）

vLLM 不是在 model runner 里手写 `torch.compile(model)`，而是用一个**类装饰器** `@support_torch_compile`（`decorators.py:38`）把模型类的 `forward` 包起来。被装饰的模型（如 `LlamaModel`）在 `__init__` 时混入 `TorchCompileWrapperWithCustomDispatcher`（`wrapper.py:19`），后者在构造时调用 `torch.compile(self.forward, backend=...)`（`wrapper.py:44`）。

Dynamo 做的事：在 **Python 字节码层面**拦截 `forward`，把里面的 tensor 运算抓成一张 **FX 图**（graph），遇到无法捕获的 Python 就 graph break。vLLM 通过 `mark_dynamic`（`decorators.py:188`）把"batch / token 维度"标成**动态符号 shape**，这样**同一张图能服务任意 token 数**——只编译一次"general shape"图，而不是每个 size 重编。

backend 拿到 FX 图后（level 3 是 `VllmBackend`，`backends.py:280`），交给 **Inductor** 做后端编译：算子融合（把 norm+量化、elementwise 链等合成单个 kernel）、生成 Triton/C++ kernel。融合本身就减少了 kernel 数量与中间张量。

> 注：V1 默认 `custom_ops=["none"]`（`config.py:3801`），即关掉 vLLM 自己的手写 CUDA custom op，**让 Inductor 来融合**——因为 piecewise cudagraph 与某些手写 kernel 配合有问题，注释见 `config.py:3796-3799`。

### 3.2 piecewise CUDA graph：把 attention 切出去（解决 launch 开销，且不破坏 paged attention）

这是 V1 相对 V0 最关键的设计。

**为什么 attention 不能进 CUDA graph。** CUDA graph 的铁律是"固定 shape + 固定地址 + 无 data-dependent 控制流"。而 PagedAttention 恰恰处处违反：

- **变长序列**：一个 batch 里每条请求的已算长度、context 长度都不同，attention kernel 内部按 `seq_lens` / `block_tables` 做循环与分支——这是 **data-dependent 控制流**。
- **paged KV cache 不定形**：KV 不是一块连续张量，而是按 block table 散落在 block pool 里。每步要读哪些 block、写到哪个 slot，由 `slot_mapping` / `block_table` 这些**运行期才知道**的元数据决定。把这些指针固化进 graph 是不安全的。
- prefill / decode / chunked 混在一个 batch（见 [模块 00](../00-request-lifecycle/design.md) §3.3），attention 的"形状"在不同步之间剧烈变化。

V0 的整图 cudagraph 因此被迫**只对纯 decode、且 batch 内规整**的情形开 graph，prefill 一律 eager——覆盖面窄、prefill 退化。

**piecewise 的做法：把图按 attention 算子"切开"。** vLLM 把 attention 实现成一个 **PyTorch custom op** `torch.ops.vllm.unified_attention(_with_output)`（`attention/layer.py:332,363`），并把它登记为 **splitting op**（`config.py:3573` `set_splitting_ops_for_v1` → `["vllm.unified_attention", "vllm.unified_attention_with_output"]`）。

`VllmBackend` 拿到 Dynamo 抓来的整图后，用 `split_graph`（`backends.py:161`）**按这些 splitting op 把整图切成若干段**：

```
整图 = [embedding+layer0前半] → attn0 → [layer0后半+layer1前半] → attn1 → ... → [最后+norm]
                ↑静态段              ↑切点          ↑静态段           ↑切点      ↑静态段
        ── 可 CUDA graph ──    不入graph     ── 可 CUDA graph ──    不入graph  ── 可 CUDA graph ──
```

- **静态段**（attention 之间的所有计算：QKV 投影、norm、MLP、residual、最后的 lm_head 之前）shape 固定、无 data-dependent 控制流 ⇒ **逐段录成 CUDA graph**。
- **attention 段**留在 CUDA graph **外**，以普通 eager kernel 执行 ⇒ 它可以自由读变长 KV、走动态分支。

这样一次 forward 变成：`graph0.replay() → attn0(eager) → graph1.replay() → attn1(eager) → ...`。**绝大多数 kernel（投影/norm/MLP）被 replay 一笔带过省掉 launch；attention 保持灵活**。这就是 "piecewise"（分段）的含义。

attention 段如何拿到运行期的 KV 与元数据？靠 **forward context**（见 §3.3）：custom op 在 eager 执行时从全局 `ForwardContext` 取 `attn_metadata` 和该层的 attention 模块（`attention/layer.py:338-342`），所以它不需要把这些当作 graph 的输入张量。

### 3.3 forward context：把"非张量的运行期信息"旁路传进去

CUDA graph / 编译图的输入只能是固定的几个张量（`input_ids`、`positions` 等）。但 attention 还需要 `attn_metadata`（seq_lens、block_table、slot_mapping……）这些**每步都变、又不能当 graph 输入**的东西。

vLLM 用一个**进程内全局上下文** `ForwardContext`（`forward_context.py:33`）解决：`execute_model` 在调模型前用 `set_forward_context(attn_metadata, vllm_config)`（`forward_context.py:56`）把本步的 `attn_metadata` 塞进全局；attention custom op 在图外执行时通过 `get_forward_context()`（`forward_context.py:48`）取出来。

`ForwardContext` 还持有 `no_compile_layers`（即 `compilation_config.static_forward_context`，`forward_context.py:96`）——一个 `layer_name → Attention 模块` 的 dict（每个 `Attention` 在 `__init__` 时把自己登记进去，`attention/layer.py:146-149`）。custom op 只接收一个字符串 `layer_name`，再从这个 dict 反查真正的层对象（`attention/layer.py:340`）。**为什么绕这一圈？** 因为 Dynamo 不能把"一个 nn.Module 对象"当作图的输入；传一个字符串名字、运行期再查表，才能让 attention op 成为一个干净、可被 Dynamo 识别为 splitting 点的 custom op。

### 3.4 持久 buffer：CUDA graph 能 replay 的前提

CUDA graph replay 时，GPU 按**录制时记下的输入显存地址**去读数据。所以每一步必须把真实输入**写进同一块固定地址的 buffer**，graph 才会读到新数据。

model runner 在 `__init__` 里预分配了持久 buffer（`gpu_model_runner.py:203-208`）：

```python
self.input_ids  = torch.zeros(self.max_num_tokens, dtype=torch.int32,  device=...)
self.positions  = torch.zeros(self.max_num_tokens, dtype=torch.int64,  device=...)
```

每步 `_prepare_inputs` 把这一步的 token 写进 `self.input_ids[:n]`，forward 时取 `self.input_ids[:num_input_tokens]`（`gpu_model_runner.py:1040`）这个**切片视图**——视图和原 buffer 共享地址，所以捕获时与 replay 时地址一致。`PiecewiseBackend` 在 replay 前还会（debug 模式）断言输入地址不变（`backends.py:700-708`）。

这也是为什么 [模块 00](../00-request-lifecycle/impl.md) §T8 强调的"持久 InputBatch + 增量 copy_ 进固定 buffer"既是性能优化、**又是 CUDA graph 的硬前提**。

### 3.5 捕获多个 batch size + 运行期 pad 到最近

CUDA graph 一个 size 一张图。服务负载的 batch size 千变万化，不可能每个都捕获。折中：

1. **启动期捕获一组代表性 size**：`[1, 2, 4] + range(8, 513, 8)`，再按 `max_num_batched_tokens` 截断（`config.py:3907-3913`），形成 `cudagraph_capture_sizes`。
2. **预计算 pad 映射**：`init_with_cudagraph_sizes`（`config.py:3559-3571`）构造 `bs_to_padded_graph_size` 数组——对每个可能的 batch size，预存"应该 pad 到哪个捕获 size"。
3. **运行期 O(1) 查表**：`pad_for_cudagraph(n)`（`config.py:3704`）直接 `bs_to_padded_graph_size[n]`，把实际 token 数**向上 pad 到最近的捕获 size**，多出来的位置是 buffer 里的垃圾值（不影响正确性，因为对应输出会被丢弃）。
4. **超过 `max_capture_size` 就退回 eager**：`execute_model` 里 `num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]` 才走 graph（`gpu_model_runner.py:1001-1009`）；更大的 batch（通常是大 prefill）走普通编译图 / eager。

> pad 是"向上取整到捕获档位"。例如捕获了 `{8,16,24}`，实际来了 13 个 token，就 pad 到 16，跑 16 那张图，最后只取前 13 个结果。多算几个 token 的算力换 launch 的消除，小 batch 下非常划算。

### 3.6 编译与捕获的顺序：先 compile，再 capture

启动期严格遵守 "**先把每段编译好（拿到可运行的融合 kernel），再在这些 kernel 之上录 CUDA graph**" 的顺序（细节见 impl.md 调用链）：

1. **load_model** 时，被装饰的模型类已准备好 `torch.compile` 包装，但 JIT 编译是惰性的——要等第一次 forward 才触发。
2. **profile_run / compile_or_warm_up_model**：用 `_dummy_run` 跑 general shape（动态符号 shape），**触发 Dynamo 抓图 + 切图 + Inductor 编译**每个静态段（`PiecewiseCompileInterpreter.run`，`backends.py:455`）。这一步只编译、**不**录 graph。
3. **capture_model**：在 `graph_capture` 上下文里，**从大到小**遍历 `cudagraph_batch_sizes`，每个 size 先跑几次 warmup（`cudagraph_num_of_warmups`），再 `_dummy_run` 一次触发 `PiecewiseBackend` 录制该 size 的 CUDA graph（`gpu_model_runner.py:1613-1618`）。

顺序不能反：CUDA graph 录的是"实际会执行的那串 kernel"。若没先编译好，录下来的就是未优化的 eager kernel，甚至录到编译过程本身。warmup 则是为了让 cuBLAS/Inductor 的惰性初始化、autotune 在录制**之前**完成，否则会被错误地录进 graph。

> **从大 size 到小 size 捕获**（`reversed`，`gpu_model_runner.py:1614`）是个 memory trick：大 size 的 CUDA graph 显存池可被小 size 复用（共享 `global_graph_pool`，`backends.py:313`），反过来则要为每个 size 单独留池子。

### 3.7 V0 整图 cudagraph vs V1 piecewise（核心对比）

| 维度 | V0：整图 CUDA graph | V1：piecewise CUDA graph + torch.compile |
|---|---|---|
| 捕获粒度 | 整个 forward（含 attention）一张图 | 按 attention 切成多段，**只录 attention 之间的静态段** |
| attention | 也被塞进 graph ⇒ 只能对规整 decode 开 | **留在 graph 外** eager 执行，自由读 paged KV / 变长序列 |
| prefill | 几乎无法 graph，退 eager | 静态段照样 replay；只有 attention 段 eager，**prefill 也受益** |
| 算子融合 | 无（纯 cudagraph，kernel 还是手写 custom op） | **torch.compile + Inductor 融合**，再叠加 cudagraph |
| 动态 shape | 每个 batch size 一张整图，捕获面有限 | 编译用符号 shape 一张图通用；cudagraph 多 size + pad |
| 覆盖率 | 窄（仅 decode，且需关掉很多特性） | 宽（prefill/decode/chunked 混批都能部分 replay） |

一句话：**V0 把 attention 也硬塞进 graph，导致只能服务很窄的场景；V1 把 attention 切出来，让"灵活的 attention" 和 "可 replay 的静态计算" 各得其所。**

### 3.8 enforce_eager：一键关闭整条编译路径

`enforce_eager=True` 时，`VllmConfig.__post_init__` 的 PIECEWISE 分支被跳过（`config.py:3795` 的 `not ... enforce_eager` 条件），level 保持 0；`use_cuda_graph` 也为 False（`gpu_model_runner.py:187-189`）。于是：

- `support_torch_compile` 装饰器里 `do_not_compile=True`（`decorators.py:155-159`），`__call__` 直接走原始 `forward`（`decorators.py:171`），不 compile。
- `capture_model` 直接 warning 并 return（`gpu_model_runner.py:1600-1605`）。
- `execute_model` 里 `use_cuda_graph=False`，`num_input_tokens = num_scheduled_tokens`，不 pad（`gpu_model_runner.py:1007-1009`）。

用途：编译有时间/显存成本、有时触发 Inductor bug、超长序列或不支持的硬件——这些场景一键退回纯 eager，牺牲速度换确定性与兼容性。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `CompilationLevel` | `config.py:3251` | 0=无 / 1=dynamo as-is / 2=dynamo once / **3=piecewise（V1 默认）** |
| `CompilationConfig` | `config.py:3259` | 编译总配置：level、`splitting_ops`、`use_cudagraph`、`use_inductor`、`compile_sizes`、`cudagraph_capture_sizes`、`cudagraph_num_of_warmups` |
| `cudagraph_capture_sizes` | `config.py:3354` | 要捕获 CUDA graph 的 batch size 列表（降序） |
| `bs_to_padded_graph_size` | `config.py:3400` | 预计算的 `batch_size → 应 pad 到的捕获 size` 数组，`pad_for_cudagraph` O(1) 查 |
| `splitting_ops` | `config.py:3345` | 切图用的 op 名（V1=`unified_attention(_with_output)`），切点即 attention |
| `static_forward_context` | `config.py:3410` | `layer_name → Attention 模块` 的 dict，attention custom op 据名反查层对象 |
| `ForwardContext` | `forward_context.py:33` | 每步 forward 的全局上下文：`attn_metadata`、`no_compile_layers`、`dp_metadata` |
| `VllmBackend` | `backends.py:280` | level-3 的 torch.compile backend：切图 + 驱动分段编译 |
| `SplitItem` | `backends.py:153` | 切图产物：一个子图段（含 `is_splitting_graph` 标记它是否就是 attention 段） |
| `PiecewiseBackend` | `backends.py:537` | 每个**静态段**的运行期后端：管该段的"按 size 编译 + 捕获 + replay" |
| `ConcreteSizeEntry` | `backends.py:520` | 某段在某个具体 size 下的状态：`cudagraph`、`output`、`input_addresses`、是否已编译/捕获 |
| `CompilerManager` | `backends.py:29` | 管 Inductor/Eager 编译与磁盘缓存（`vllm_compile_cache.py`） |
| `CompilationCounter` | `counter.py:8` | 计数器（图数、piecewise 段数、捕获的 cudagraph 数），主要供测试断言 |

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| piecewise 切图（attention 切出） | prefill/decode/chunked 都能部分 replay；attention 保持灵活 | 一次 forward 有 N 段 graph + N 个 eager attention，段间切换有少量开销；实现复杂 |
| torch.compile + Inductor 融合 | 减少 kernel 数、消除逐层 Python | 启动编译慢（几十秒~分钟级，含 autotune）；偶发 Inductor bug ⇒ 需 `enforce_eager` 兜底 |
| 捕获多个 size + pad 到最近 | 用有限张图覆盖动态 batch | 每个 size 一份 graph 显存（几十 size 累积可观）；pad 会多算 padding 那部分 token |
| 持久 buffer + 固定地址 | CUDA graph replay 的前提；省每步分配 | buffer 按 `max_num_tokens` 预留显存；forward 中禁止改写 nn.Module buffer（`wrapper.py:110-115` 直接报错） |
| 启动期一次性 compile+capture | 运行期热路径只剩查表+replay | 启动变慢（"Graph capturing finished in 5~20 secs"，`gpu_model_runner.py:1624`）；首次冷启需编译 |
| 磁盘编译缓存（`CompilerManager`） | 二次启动直接 load 编译产物 | 缓存 key 要正确反映所有影响图的因素（env+config+traced 代码+compiler hash，`backends.py:359-394`），否则可能 stale |
| `custom_ops=["none"]`（V1） | 让 Inductor 统一融合，配合 piecewise cudagraph | 放弃部分手写 CUDA kernel 的极致性能，换正确性与可融合性 |

---

## 6. 小结：V1 编译栈的净收益

- **两种加速叠加**：torch.compile 砍 Python/框架开销 + 融合算力，CUDA graph 砍 kernel launch，二者正交相乘。
- **不牺牲 PagedAttention**：attention 切到 graph 外，paged KV / 变长序列照常工作，是 V1 能"既要 cudagraph 又要 paged attention"的关键。
- **覆盖面远超 V0**：prefill、chunked prefill、混合批的静态段都能 replay，不再像 V0 那样只对窄场景开 graph。
- **运行期零编译**：所有 compile/capture 集中在启动期，热路径只剩 `pad → 查表 → replay`。

> 逐行调用链、`file:line` 对照、实现精妙之处与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 7. 一图速览：piecewise 编译/捕获/重放全景

```
启动期（一次性）                                       运行期（每步 decode）
─────────────────────────────────────────            ─────────────────────────────
load_model
  └ @support_torch_compile 包好 forward               execute_model
     （torch.compile 惰性，未真编译）                    │ _prepare_inputs → 写入持久 buffer
                                                         │   self.input_ids[:n].copy_(...)
compile_or_warm_up_model                                 │
  │ ① _dummy_run(general/symbolic shape)                 │ num_in = pad_for_cudagraph(n)
  │    └ Dynamo 抓图 → VllmBackend.__call__               │   （O(1) 查 bs_to_padded_graph_size）
  │       └ split_graph(按 splitting_ops=attention)      │
  │          整图 ─┬─ 静态段0 ─┐                          │ set_forward_context(attn_metadata)
  │                ├─ attn (切点,不编译) │ 各静态段          │   └ model(input_ids[:num_in], ...)
  │                ├─ 静态段1            │ 经 Inductor      │      静态段0.replay() ← 固定地址读 buffer
  │                ├─ attn              ├ 编译为融合 kernel │      attn0(eager) ← 读 forward_context
  │                └─ 静态段2 ─┘         │  (PiecewiseBackend)     里的 attn_metadata + paged KV
  │ ② capture_model                                       │      静态段1.replay()
  │    graph_capture():                                   │      attn1(eager)
  │      for size in reversed(capture_sizes):  ← 大→小     │      静态段2.replay()
  │        warmup ×k                                       │
  │        _dummy_run(size)                                │ hidden_states[:n] → logits → sample
  │          └ PiecewiseBackend 对每段录 CUDAGraph         │
  │             (固定地址, 共享 global_graph_pool)          └─ 超过 max_capture_size 的大 batch:
  └ 先 compile 再 capture，顺序不可反                          不 pad、不 replay，走 eager/编译图

enforce_eager=True ⇒ 上述全部跳过，纯 eager forward
```
