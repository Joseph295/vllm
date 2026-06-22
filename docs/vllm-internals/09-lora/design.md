# 模块 09 · LoRA 多适配器 —— 设计文档

> 范围：vLLM 如何让**一个基座模型（base model）同时服务多个 LoRA 适配器**，并且**在同一个 batch 内**让不同请求各用各的 LoRA。本模块聚焦"多租户 LoRA 批量推理"这一核心机制：低秩增量 W+BA 怎么省显存、多 LoRA 怎么在一个 batch 内并行计算（SGMV/BGMV / Punica kernel）、token→lora 的 mapping 怎么构建、CPU/GPU 双层 LRU 缓存与动态加载/驱逐、为什么调度要限制 `max_loras`。
> 架构：**V1 为主**。LoRA 是横切关注点（见 [README](../README.md) 依赖图），它叠加在调度（[模块 01](../01-scheduler-batching/design.md)）、模型执行（[模块 00](../00-request-lifecycle/impl.md) 阶段 F）、CUDA graph（[模块 07](../07-cuda-graph-compile/design.md)）之上。
> [模块 00](../00-request-lifecycle/design.md) / [模块 01](../01-scheduler-batching/design.md) 已经提到调度器的 `max_loras` 约束与 `add_lora`/`remove_lora`/`list_loras` 接口，本模块深入这些接口背后的批量推理机制。

---

## 1. 背景与定位

> 一句话先垫底：**LoRA = 给基座大模型挂一个小的低秩增量（W+BA），多个租户共享同一份底座、各挂各的适配器。** 本模块讲清"多 LoRA 怎么在同一个 batch 内并行算完"和"成百上千适配器怎么在 CPU/GPU 间动态换入换出"。前置的 continuous batching 概念见 [PRIMER §2.4](../PRIMER.md#24-易混概念对照区别与联系专治听过但分不清)。

### 1.1 这个模块在系统里的位置

LoRA 是一个**横切关注点**，叠加在多个核心模块之上：

- **调度层**（[模块 01](../01-scheduler-batching/design.md)）：调度器组 batch 时要保证"同一步里出现的不同 LoRA 数 ≤ `max_loras`"（GPU 槽位数）——这就是 `max_loras` 既是显存约束又是调度约束的原因。
- **本模块**：启动期把被适配的 Linear/Embedding/Logits 层换成 `*WithLoRA` 包装层、预分配 GPU stacked 权重；每 batch 构建 token→lora 的 mapping 并把热的适配器激活到 GPU 槽位；forward 时用 Punica kernel 把"基座大 GEMM"和"多 LoRA 增量"分开算。
- **执行层**（[模块 00](../00-request-lifecycle/impl.md) 阶段 F）+ **CUDA graph**（[模块 07](../07-cuda-graph-compile/design.md)）：LoRA 的 kernel 必须能被 graph capture，权重 buffer 必须固定地址。

一句话：**它是叠在基座 forward 之上的"多租户增量层"——基座只算一遍，每个被适配层再追加一个低秩增量。**

### 1.2 关键文件与类各干什么（先认人）

| 文件 / 类 | 主要作用（一句话） |
|---|---|
| `lora/request.py` 的 `LoRARequest` | 用户侧适配器标识：`lora_name` / `lora_int_id`（全局唯一）/ `lora_path` |
| `lora/lora.py` 的 `LoRALayerWeights` | 单层的 LoRA 权重 `lora_a`/`lora_b`/`scaling`；`optimize()` 把 scaling 折进 `lora_b` |
| `lora/models.py` 的 `LoRAModel` / `LoRAModelManager` | 一个完整适配器 + GPU 槽位表（`lora_index_to_id`）与双层 LRU 缓存、激活/驱逐/pin |
| `lora/layers.py` 的 `BaseLayerWithLoRA` 及子类 | 包装基座层的 LoRA 层，持有 `lora_a_stacked`/`lora_b_stacked` GPU 张量 |
| `lora/worker_manager.py` 的 `WorkerLoRAManager` | worker 侧入口：读盘加载适配器、`set_active_adapters`、转发到 ModelManager |
| `lora/punica_wrapper/punica_gpu.py` 的 `PunicaWrapperGPU` | 封装 `add_lora_linear` / `add_shrink` / `add_expand`，调 Triton kernel |
| `lora/ops/triton_ops/lora_shrink.py` / `lora_expand.py` | 分段并行的 Triton kernel：一次 launch 算完全 batch、全 LoRA 的增量 |
| `v1/worker/lora_model_runner_mixin.py` | V1 集成：`load_lora_model` / `set_active_loras` |

### 1.3 和相似模块 / 概念的区别与联系（专治"听过但分不清"）

- **LoRA(W+BA) vs 全量微调**：全量微调要为每个任务存一整份 `W`（7B 模型 ~14GB），100 个任务就是 100 份；**LoRA 冻结 `W`、只训一个低秩增量 `ΔW=B·A`**（`A:[in,r]`、`B:[r,out]`，秩 `r` 通常 8~64），一个适配器只有几十 MB。代价是表达能力受秩限制，但绝大多数下游任务够用——这换来了"一份基座服务无数租户"的部署红利。
- **多 LoRA 同 batch vs 按 LoRA 拆批**：vLLM 的吞吐来自把不同请求塞进同一个 GPU batch（continuous batching）。若请求 A 用 LoRA-1、B 用 LoRA-2、C 不用 LoRA，**朴素做法是按 LoRA 把 batch 拆开逐个跑**——会把一个大 GEMM 拆成许多瘦 GEMM、GPU 严重欠载。vLLM 坚持**基座大 GEMM 全 batch 只算一遍**，LoRA 增量用专门的分段 kernel **一次 launch 算完所有 LoRA**。
- **SGMV vs BGMV**：都是 Punica 提出的"一次 kernel 处理多 LoRA"的批量 GEMM。**SGMV（Segmented Gather Matrix-Vector）** 面向 prefill——连续 token 往往同属一个 LoRA，于是把 batch 按 LoRA **分段**、每段一个连续 token 区间、在 grid 的一维上并行各段；**BGMV（Batched ...）** 面向 decode——每请求每步只 1 个 token，逐 token gather 对应权重。vLLM V1 用统一的 Triton `lora_shrink`/`lora_expand`，prefill/decode 共用同一套 kernel。
- **base 权重 vs adapter**：**base（基座）权重** `W` 是冻结的、全租户共享、只加载一份、只算一遍；**adapter（适配器）** 是每租户各自的 `A`/`B`，按请求动态挂载，存活在"磁盘→CPU→GPU"三层缓存里。vLLM 刻意**不把 `ΔW` merge 进 `W`**——merge 会让每个请求的权重各不相同、砸掉基座共享的全部价值。
- **本模块 vs 量化（[模块 08](../08-quantization/design.md)）/ 编译（[模块 07](../07-cuda-graph-compile/design.md)）**：三者都是横切关注点。LoRA 叠在基座层之上、量化决定基座层权重怎么存、编译决定整段 forward 怎么 replay。LoRA 的 GPU 权重必须 stacked 静态预分配（固定地址），正是为了和 CUDA graph 兼容。

---

### 1.4 这个模块要解决的问题

LoRA（Low-Rank Adaptation）是当下最主流的参数高效微调（PEFT）方式：冻结基座权重 `W`，只训练一个低秩增量 `ΔW = B·A`（`A`、`B` 是两个瘦长矩阵，秩 `r` 通常 8~64）。一个 7B 模型的全量微调要存一整份 ~14GB 权重，而一个 LoRA 适配器往往只有几十 MB。

这带来一个极有价值的部署场景——**多租户 LoRA 服务**：

1. **共享基座、按需切换适配器**：100 个客户各自微调了一个 LoRA，但他们共享同一份基座权重。如果给每个 LoRA 都起一个独立的模型实例，显存会爆炸（100×14GB）；正确做法是**基座只加载一份**，100 个 LoRA 各自只占几十 MB，按请求动态挂载。
2. **同一 batch 内混合不同 LoRA**：vLLM 的核心吞吐来自 continuous batching（[模块 01](../01-scheduler-batching/design.md)）——把不同请求塞进同一个 GPU batch。但如果请求 A 用 LoRA-1、请求 B 用 LoRA-2、请求 C 不用 LoRA，这个混合 batch 该怎么算？朴素做法是"按 LoRA 把 batch 拆开、逐个 LoRA 跑 forward"，但这会把一个大 GEMM 拆成许多小 GEMM，GPU 利用率骤降。**vLLM 要做到：一个混合 batch 仍然只跑一遍 forward，LoRA 增量用专门的分段 kernel 一次性算完。**
3. **适配器数量远超显存容量**：注册的 LoRA 可能成百上千，但 GPU 上同时"激活"的槽位有限。需要一套**缓存 + 驱逐**机制，把热的 LoRA 放 GPU、冷的放 CPU 内存、更冷的留在磁盘，按访问动态搬运。

vLLM 的 LoRA 子系统，本质就是围绕"**基座只算一遍，LoRA 增量在同一个 batch 内分段并行**"和"**CPU/GPU 双层缓存动态调度适配器**"这两件事组织的。

---

## 2. 设计目标与约束

- **基座 forward 不受影响**：基座模型的权重、forward、CUDA graph 完全不动；LoRA 只在被适配的 Linear / Embedding / LogitsProcessor 层上**追加一个增量**。基座那一份大 GEMM 永远只跑一次。
- **同 batch 多 LoRA 并行**：一个 batch 内的 token 可能分属不同 LoRA（甚至无 LoRA），LoRA 增量必须用**一次 kernel 调用**算完，而不是按 LoRA 循环。这是 Punica 的 SGMV/BGMV kernel 要解决的问题。
- **动态加载 / 驱逐**：`add_lora` / `remove_lora` / `pin_lora` / `list_loras` 在运行时随时可调；GPU 槽位满了要按 LRU 驱逐；CPU 缓存满了也要驱逐到只留磁盘。
- **显存可控**：GPU 上为 LoRA 预留的显存是**静态**的——按 `max_loras` 个槽位、`max_lora_rank` 上限预分配 stacked 权重张量，运行时只往里 `copy_`，不重新分配。
- **与 CUDA graph 兼容**：V1 用 `torch.compile` / CUDA graph 捕获 forward。LoRA 的 kernel 必须能被 graph capture，且 batch 内 LoRA 数量变化、有无 LoRA 都不能破坏 graph 的固定 shape（用持久 buffer + CPU flag 早退，见 impl T 节）。
- **与调度协同**：调度器必须保证"同一步 batch 里出现的不同 LoRA 数 ≤ `max_loras`"（GPU 槽位数），否则激活会失败。这是 `max_loras` 既是显存约束、也是调度约束的原因。

**当前 V1 的限制**（来自 `config.py` 的 `LoRAConfig.verify_lora_support`）：V1 **不支持 long LoRA**（`long_lora_scaling_factors` 必须用 V0）；不支持 DoRA、`modules_to_save`（`peft_helper.py:_validate_features`）；多模态模型**只对语言模型部分加 LoRA**，vision tower / connector 被过滤掉。

---

## 3. 核心设计思想

### 3.1 LoRA = W + BA：为什么低秩分解能省显存，又能"加"上去

一个被 LoRA 适配的线性层，前向变成：

```
y = x · W            (基座，冻结，大矩阵 [in, out])
  + x · A · B · s    (LoRA 增量；A:[in, r]，B:[r, out]，s:标量 scaling)
```

关键点：

- **省显存**：`ΔW = A·B` 是一个 `[in,out]` 的满秩近似，但只用 `[in,r] + [r,out]` 两个低秩矩阵存储。`r << min(in,out)`（典型 r=16，in=out=4096），参数量从 `in×out`（1600 万）降到 `r×(in+out)`（13 万），**两个数量级**。这就是一个适配器只有几十 MB 的原因（Hu et al. 2021，LoRA 原论文）。
- **可加性 → 可并行**：因为增量是**加法**叠在基座输出上，基座那遍大 GEMM（所有 token、所有请求共享同一 `W`）只算一次；LoRA 增量是另一遍小计算，可以单独用 kernel 算完再加回去。vLLM 选择**不把 `ΔW` merge 进 `W`**（merge 会让每个请求的权重不同、破坏共享），而是**运行时分两步算 LoRA 增量**：先 `shrink`（`x·A`，把 hidden_dim 压到 rank），再 `expand`（`·B`，把 rank 还原到 out_dim）。中间张量只有 `[num_tokens, r]`，非常小。
- **scaling 折叠**：`s = lora_alpha / r`（或 rsLoRA 的 `alpha/√r`，`peft_helper.py:54-59`）。`optimize()`（`lora.py:40`）把 scaling **直接乘进 `lora_b`**，让 `scaling=1`，省掉前向时一次逐元素乘。

### 3.2 同一 batch 内多 LoRA 并行：SGMV / BGMV（Punica kernel）

这是整个模块最核心的机制。问题是：一个 batch 有 N 个 token，token-0..k 用 LoRA-1，token-k+1.. 用 LoRA-2，还有些 token 不用 LoRA。怎么一次算完所有 token 的 LoRA 增量？

**朴素做法**（按 LoRA 分组循环）：对每个 LoRA 取出它的 token 子集，做一次小 GEMM。N 个 LoRA 就是 N 次 kernel launch，每次都是"瘦"GEMM，GPU 严重欠载。

**Punica 的做法**（Chen et al. 2023, *Punica: Multi-Tenant LoRA Serving*，arXiv 2310.18547；SGMV 发表于 MLSys 2024）：用一个**分段 / 分组的 batched GEMM kernel**，把"每个 token 该用哪个 LoRA 的权重"作为**索引**喂进 kernel，让一次 kernel launch 内部并行处理所有 LoRA 段：

- **SGMV（Segmented Gather Matrix-Vector）**：prefill 场景，连续的 token 往往属于同一请求 / 同一 LoRA，于是把 batch 按 LoRA **分段**，每段是一个连续的 token 区间，对应一个 LoRA 权重，kernel 在 grid 的一个维度上并行处理各段（`lora_idx = tl.program_id(axis=2)`，`lora_shrink.py:40`）。
- **BGMV（Batched Gather Matrix-Vector）**：decode 场景，每个请求每步只有 1 个 token，batch 就是"每个 token 一个 LoRA 索引"，逐 token gather 对应权重做向量×矩阵。

vLLM 当前的 GPU 实现（`punica_gpu.py` + `triton_ops/`）用**统一的 Triton kernel `lora_shrink` / `lora_expand`**，prefill 和 decode 共用同一套 kernel（`lora_model_runner_mixin.py:65-68` 注释明示："On cuda platforms we use the same kernels for prefill and decode"）。kernel 内部按 `active_lora_ids` 这一维并行，每个 program 负责一个 LoRA 的若干 token block，靠 `token_indices_sorted_by_lora_ids`（按 LoRA 排序后的 token 下标）gather 出本 LoRA 的行。一次 launch 算完全 batch、全 LoRA。

整个 LoRA 增量被拆成两次这种 kernel：

```
shrink:  buffer[r]  = (x[hidden] @ A) * scale     # lora_shrink，hidden→rank
expand:  y[out]    += buffer[r]   @ B             # lora_expand，rank→out，原地加到基座输出上
```

（`punica_gpu.py:add_lora_linear` → `add_shrink` + `add_expand`。）

### 3.3 LoRA mapping：token→lora_index 怎么构建

kernel 需要知道"第 i 个 token 用哪个 LoRA"。这条信息分两段构建：

1. **请求级（per-request）**：每条请求带一个 `LoRARequest`（`request.py`），里面有全局唯一的 `lora_int_id`。`InputBatch` 维护一个 `request_lora_mapping` numpy 数组（`gpu_input_batch.py:192`），`request_lora_mapping[i] = 该请求的 lora_int_id`（无 LoRA 则为 0）。
2. **token 级（per-token）**：每步 forward 前，`make_lora_inputs`（`gpu_input_batch.py:615`）用 `np.repeat(req_lora_mapping, num_scheduled_tokens)` 把"每请求一个 LoRA id"**展开成"每 token 一个 LoRA id"**（`token_lora_mapping`），同时给出 `prompt_lora_mapping`（每请求一个，给 sampler / logits 用）。

注意这里有两层 id 映射：`lora_int_id`（全局唯一、用户可见）→ `lora_index`（0..max_loras-1 的 **GPU 槽位下标**）。后者由 `LoRAModelManager.lora_index_to_id`（`models.py:332`）维护——一个长 `max_loras` 的列表，`lora_index_to_id[slot] = 当前占用该槽位的 lora_int_id`。`convert_mapping`（`punica_wrapper/utils.py:44`）把 token 的 `lora_int_id` 通过 `lora_index_to_id.index()` 翻译成槽位下标，并把"无 LoRA"（id≤0）标成 `-1`，kernel 见到 `-1` 直接早退（`lora_shrink.py:43`）。

### 3.4 三层缓存：磁盘 → CPU（registered）→ GPU（active）

适配器的生命周期跨越三层存储，由两套 LRU 缓存管理（`models.py` 的 `LRUCacheLoRAModelManager`）：

```
磁盘 / HF Hub                         # 适配器权重的源头（adapter_model.safetensors）
   │  _load_adapter (worker_manager.py:85) —— 读盘、校验、转成 LoRAModel（device="cpu"）
   ▼
CPU 缓存 self._registered_adapters    # 容量 = max_cpu_loras，LRU 驱逐
   │  activate_adapter (models.py:381) —— copy_ 进 GPU stacked 张量的某个 slot
   ▼
GPU 激活 self._active_adapters        # 容量 = max_loras (= lora_slots)，LRU 驱逐
   │  set_adapter_mapping → punica_wrapper.update_metadata
   ▼
本步 batch 实际用到的 LoRA            # token_lora_mapping 指向 GPU slot
```

- **GPU 层**是真正参与 forward 的：`max_loras` 个槽位，每个槽位是 stacked 权重张量（`lora_a_stacked` / `lora_b_stacked`，`layers.py:325-342`）里的一个下标。激活就是把 CPU 上的 `LoRAModel` 权重 `copy_` 进对应 slot（`models.py:413` `module.set_lora(index, ...)`）。
- **CPU 层**是 `max_cpu_loras` 个反序列化好的 `LoRAModel`（驻留 CPU 内存，可 pin_memory）。GPU 驱逐一个 LoRA 时它仍在 CPU，下次激活无需重新读盘。
- **磁盘**是最终来源；CPU 缓存满了驱逐，下次要用再从磁盘 `_load_adapter`。

**pin_lora**（`models.py:761`）：把某个 LoRA 在 CPU 缓存和 GPU 缓存里都标记为"钉住"，使 LRU 永不驱逐它（典型用于高优先级 / 常驻适配器）。

**为什么 GPU 权重是"stacked"预分配的**：`lora_a_stacked` 的 shape 是 `[max_loras, 1, rank, in]`（`layers.py:325`），一开始就按最大容量、最大 rank 全零初始化。激活适配器只是往 `[index]` 这一片 `copy_`，**从不动态分配显存**——这样显存占用恒定、可预测，且与 CUDA graph 的固定地址要求兼容。

### 3.5 为什么调度要限制 max_loras

GPU 上只有 `max_loras` 个激活槽位。如果调度器把"用了 `max_loras+1` 种不同 LoRA 的请求"塞进同一个 batch，激活时会因"No free lora slots"失败（`models.py:391`），或在 `_apply_adapters` 处抛 RuntimeError（`worker_manager.py:221`）。

所以调度器（`scheduler.py:262-302`）在组 batch 时主动维护一个 `scheduled_loras` 集合：

- 先统计本步 RUNNING 请求已经用掉的不同 LoRA（`:265-267`），断言 ≤ `max_loras`。
- 再调度 WAITING 请求时，如果"加入这条请求会让 batch 内 LoRA 种类超过 `max_loras`、且它用的是一个新 LoRA"，就**跳过这条请求**、留到下一步（`:295-302`）。

这把 GPU 槽位约束**前移到调度层**，保证送到 worker 的 batch 永远不会超过激活容量。代价是：高 LoRA 多样性的负载下，部分请求要排队等槽位（公平性 vs 容量的权衡）。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `LoRARequest` | `lora/request.py` | 用户侧请求标识：`lora_name` / `lora_int_id`（全局唯一）/ `lora_path`。`__hash__`/`__eq__` 按 `lora_name` 比较，便于跨引擎识别 |
| `LoRAConfig` | `config.py:2573` | 全局配置：`max_lora_rank`、`max_loras`（GPU 槽位）、`max_cpu_loras`（CPU 缓存）、`lora_extra_vocab_size`、`bias_enabled`、`fully_sharded_loras` |
| `PEFTHelper` | `lora/peft_helper.py` | 解析 `adapter_config.json`；校验（rank ≤ max、不支持 DoRA / modules_to_save）；算 scaling factor |
| `LoRALayerWeights` | `lora/lora.py:13` | 单层的 LoRA 权重：`lora_a` / `lora_b` / `bias` / `scaling`。`optimize()` 把 scaling 折进 `lora_b` |
| `PackedLoRALayerWeights` | `lora/lora.py:121` | 打包层（qkv_proj、gate_up_proj）的 LoRA：多个子 LoRA 合成一个，每个子 slice 一份权重 |
| `LoRAModel` | `lora/models.py:61` | 一个完整适配器：`id` + `{module_name → LoRALayerWeights}`。`from_local_checkpoint` 读盘构建 |
| `BaseLayerWithLoRA` 及子类 | `lora/layers.py:82` | 包装基座层的 LoRA 层：`ColumnParallelLinearWithLoRA`、`MergedColumnParallel...`、`QKVParallel...`、`RowParallel...`、`VocabParallelEmbedding...`、`LogitsProcessor...`。持有 `lora_a_stacked` / `lora_b_stacked` GPU 张量 |
| `LoRAModelManager` / `LRUCacheLoRAModelManager` | `lora/models.py:304` / `:711` | 管理 GPU 槽位（`lora_index_to_id`）、激活 / 驱逐 / 双层 LRU 缓存 / pin |
| `WorkerLoRAManager` / `LRUCacheWorkerLoRAManager` | `lora/worker_manager.py` | worker 侧入口：加载适配器、`set_active_adapters`、转发到 ModelManager |
| `PunicaWrapperGPU` | `lora/punica_wrapper/punica_gpu.py` | 持有 mapping 元数据，封装 `add_lora_linear` / `add_shrink` / `add_expand`，调 Triton kernel |
| `LoRAKernelMeta` | `lora/ops/triton_ops/lora_kernel_metadata.py` | kernel 所需的元数据张量：`token_lora_mapping`、`token_indices_sorted_by_lora_ids`、`active_lora_ids`、`num_tokens_per_lora`、`lora_token_start_loc`、`no_lora_flag_cpu` |
| `LoRAMapping` | `lora/layers.py:78` | `index_mapping`（token→lora_int_id）+ `prompt_mapping`（请求→lora_int_id）+ `is_prefill` |

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 运行时算 `BA` 增量、**不 merge 进 W** | 基座权重全请求共享、只算一遍；适配器可即时切换 | 每个被适配层多两次小 kernel（shrink+expand）；中间 buffer 的 H2D / 计算开销 |
| Punica 分段 GEMM（SGMV/BGMV）一次算多 LoRA | 混合 batch 单次 kernel launch，GPU 不欠载 | kernel 复杂；需要 token 按 LoRA 排序、构造段元数据；小 batch 收益有限 |
| GPU 权重 stacked 预分配（按 max_loras × max_rank） | 显存恒定可预测；与 CUDA graph 固定地址兼容 | 即使实际 rank 小、LoRA 少，也按上限占显存（浪费） |
| CPU/GPU 双层 LRU 缓存 | 适配器数 >> GPU 槽位时仍可服务；CPU 命中免读盘 | 缓存 miss 要读盘 / H2D，延迟抖动；LRU 抖动场景退化 |
| 调度层强制 `max_loras` 约束 | 保证 batch 不超激活容量，激活永不失败 | 高 LoRA 多样性负载下请求排队，吞吐 / 公平性受影响 |
| `scaling` 折进 `lora_b`（optimize） | 前向省一次逐元素乘 | 权重被原地改写，`optimize()` 必须幂等（用 `scaling==1` 守卫） |
| `no_lora_flag` 用 CPU tensor 传给 torch op | 让 batch 全无 LoRA 时 kernel 内部早退、又不破坏 torch.compile 图 | 多一个 CPU↔判断；依赖 torch op 内部不被 trace 的实现细节（见 impl T 节注释） |

---

## 6. 设计动机出处

- **LoRA 低秩分解**：Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685。冻结 `W`、只训 `B·A`，`scaling=alpha/r`。vLLM 的 `LoRALayerWeights` / `W + BA·s` 前向直接对应该论文。
- **rsLoRA**（`scaling = alpha/√r`）：Kalajdzievski (2023), arXiv:2312.03732（`peft_helper.py:32` 注释）。
- **Punica / SGMV**：Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). *Punica: Multi-Tenant LoRA Serving*. arXiv:2310.18547（`punica_base.py:2-6` / `punica_gpu.py:2-6` 文件头直接引用）。SGMV（Segmented Gather Matrix-Vector）kernel 让一个 batch 内多个 LoRA 在单次 launch 中并行。后续发表于 MLSys 2024。
- **S-LoRA**：Sheng, Y., et al. (2023). *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*. arXiv:2311.03285。提出 Unified Paging + 大规模适配器的 CPU/GPU 分层缓存与统一内存管理，是 vLLM"成百上千适配器 + 双层 LRU 缓存 + 动态换入换出"思路的直接来源。
- **DoRA**（暂不支持，仅校验）：Liu et al. (2024), arXiv:2402.09353（`peft_helper.py:34`）。

---

## 7. 一图速览：LoRA 在前向中的位置

```
                              一个被 LoRA 适配的 Linear 层（如 q_proj）
   x [num_tokens, hidden]
        │
        ├──────────────► base_layer.quant_method.apply(x)  ── 基座大 GEMM，全 batch 共享 W，只算一遍
        │                          │
        │                          ▼  output [num_tokens, out]
        │                          │
        │   ┌──────────── punica_wrapper.add_lora_linear(output, x, A_stacked, B_stacked) ───────────┐
        │   │                                                                                         │
        └──►│  ① add_shrink:  buffer = lora_shrink(x, A_stacked, meta)   # [num_tokens, rank]，scale  │
            │      Triton kernel 按 active_lora_ids 这一维并行：                                       │
            │      每个 LoRA 段 gather 自己的 token（token_indices_sorted_by_lora_ids），做 x·A       │
            │                                                                                         │
            │  ② add_expand:  output += lora_expand(buffer, B_stacked, meta)  # rank→out，原地加      │
            │      lora_id == -1 的 token（无 LoRA）在 kernel 内早退，不被污染                         │
            └─────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼  y = x·W + (x·A·B)·s   一次 forward 完成全 batch、全 LoRA

   mapping 怎么来：
     InputBatch.request_lora_mapping[req] = lora_int_id
          │  np.repeat(·, num_scheduled_tokens)
          ▼
     token_lora_mapping[token] = lora_int_id
          │  convert_mapping: lora_int_id ──(lora_index_to_id.index)──► GPU slot 下标，无 LoRA→ -1
          ▼
     LoRAKernelMeta.prepare_tensors: sort by lora id → 段元数据(active_lora_ids/num_tokens_per_lora/start_loc)

   三层缓存：磁盘 ──load──► CPU(registered, max_cpu_loras, LRU) ──activate/copy_──► GPU(active, max_loras, LRU)
```

> 实现层面的逐行调用链、`file:line` 对照、精妙之处与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 8. 设计背后的考量与历史教训

### 8.1 设计背后的考量（深化"为什么"）

1. **为什么坚持"运行时算 BA、不 merge 进 W"——这是多租户的地基，不是一个性能选项**。理论上把 `ΔW=BA·s` merge 进 `W` 能让前向退化成单次 GEMM、零额外 kernel。但 merge 一旦发生，每个请求的 `W` 就各不相同，基座再也无法在 batch 内共享——这恰好砸掉了 §1 那个"100 个 LoRA 共享一份基座"的全部价值。所以 vLLM 在架构层面**否决了 merge 方案**，宁可每个被适配层多付 shrink+expand 两次小 kernel，换取"基座大 GEMM 全 batch 只算一遍"。这条约束反过来倒逼出 Punica 分段 kernel：既然不能 merge，就必须有办法在一个混合 batch 里一次算完所有 LoRA 的增量。

2. **kernel 的演进：从 SGMV/BGMV 双套到统一 Triton，再到 V1 全面收敛**。最早 vLLM 沿用 Punica 的两套 CUDA kernel——prefill 用 SGMV、decode 用 BGMV（`punica_base.py` 文件头引用 arXiv:2310.18547）。V1 重写时（#13096 *Add triton kernels for V1*）引入统一的 Triton `lora_shrink`/`lora_expand`，prefill 与 decode **共用同一套 kernel**（`lora_model_runner_mixin.py:65-68` 注释明示），最终在 #14685 *Retire SGMV and BGMV Kernels* 把老的手写 CUDA kernel 整体退役。这条脉络的动机是：维护两套 + 两种平台的手写 kernel 成本过高，而 Triton 让"按 active_lora_ids 维并行"的同一份逻辑覆盖所有场景，可读性与可移植性都更好。

3. **为什么 GPU 权重要 stacked 静态预分配——被 CUDA graph 的固定地址约束倒逼**。`lora_a_stacked` 一开始就按 `[max_loras, 1, max_rank, in]` 全零分配，激活只是往某个 slot `copy_`，从不动态分配显存。这并非单纯为了省事：V1 用 CUDA graph 捕获 forward（#14626 *Enable CUDAGraphs for V1*），而 graph replay 要求所有 buffer 地址恒定。若 LoRA 权重随激活/驱逐动态分配，地址就会漂移、graph 失效。于是"显存恒定可预测"这个收益其实是"必须兼容 CUDA graph"这个硬约束的副产品——代价是即便实际 rank 小、LoRA 少也按上限占显存。

4. **把 max_loras 约束前移到调度层，是"激活永不失败"的设计选择**。GPU 只有 `max_loras` 个槽位，理论上可以让 worker 在激活失败时回退/排队，但那会让错误处理散落在执行热路径上。vLLM 选择让调度器（`scheduler.py:262-302`）在组 batch 时就维护 `scheduled_loras` 集合、主动跳过会超容量的请求——把"容量约束"变成"调度不变式"，使送到 worker 的 batch **永远不会**触发"No free lora slots"。代价是高 LoRA 多样性负载下请求要排队（公平性 vs 容量），但换来了执行层的简单与确定性。

5. **"全 batch 无 LoRA 时早退"为何要靠 CPU flag 而非 Python 分支**。一个 batch 可能完全没有请求用 LoRA。最自然的写法是 Python 侧 `if no_lora: skip`，但这会让 forward 产生两条不同的代码路径、破坏 CUDA graph 的固定结构。#15152 *Skip LoRA kernels when not required* 的做法是引入 `no_lora_flag_cpu` 这个 CPU tensor，让 torch op **在 kernel 内部早退**而不改变图结构——这是"既要省掉无谓 kernel、又不能破坏 torch.compile 图"这对矛盾倒逼出的折中（见 impl T 节）。

### 8.2 重要 bug 修复（真实、精选）

下列 PR 号均经 `git log`/`git show` 在本仓库实际历史中核对。

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| **#16040** [Bugfix] LoRA: Fix the order in which the kernels process LoRAs | `lora_kernel_metadata.py:113` 构造 `active_lora_ids` 时用 `torch.unique(..., sorted=False)`，导致 kernel 处理 LoRA 的段顺序与元数据假设不一致 | 一行 `sorted=False → sorted=True` 的修复。教训：分段 kernel 的正确性高度依赖"段元数据与 token 排序顺序严格一致"这一隐式契约；`token_indices_sorted_by_lora_ids` 与 `active_lora_ids` 必须按同一个 key 排序，任何一处用 unsorted 都会错位 |
| **#11708** [Bugfix] Fix ColumnParallelLinearWithLoRA slice | TP 下切分 `lora_b` / bias 时误用 `self.output_dim` 作 `shard_size`，应为 `self.output_size`，导致每个 rank 切到错误的权重分片 | LoRA 是叠在已并行化的基座层上的"横切层"，必须复用基座层的分片语义；`output_dim`/`output_size` 一字之差就让 TP 分片错位。教训：LoRA 层的分片逻辑不能自创，要和基座 ColumnParallel 的语义严格对齐 |
| **#11727** [Bugfix] Validate lora adapters to avoid crashing server | 非法/不兼容的适配器（rank 超限、含不支持的字段）能一路加载到 worker 才抛错，直接拖垮整个 server | 催生了 `PEFTHelper._validate_features`（校验 DoRA/modules_to_save 等）这层前置校验。教训：动态加载用户提供的适配器是攻击面，校验必须前移到加载入口，不能让坏适配器进到执行热路径 |
| **#14508** [V1][Core] Fix memory issue with logits & sampling | logits/sampling 的显存占用未被正确计入，与 LoRA 的 `lora_model_runner_mixin` 路径相互作用导致显存核算偏差 | 该修复同时改了 `gpu_worker.py` / `gpu_model_runner.py` / `lora_model_runner_mixin.py`。教训：LoRA 作为横切关注点叠在显存 profiling 之上，任何对激活峰值的改动都要连带复核 LoRA 路径的显存账（这条 PR 此前还被 revert 过两次，见 #13775/#14504，说明显存核算的边界极易踩坑）|

无"挖不到 bug"的情况：本模块代码路径的 bug 修复历史非常丰富。
