# 模块 07 · CUDA Graph / torch.compile 编译优化 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的编译/捕获/重放调用链。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
> 阅读方式：分两条主干——**启动期**（compile → warmup → capture）与**运行期**（pad → replay），每步标注 `文件:行号 函数/类名`，摘录关键代码并解读。

---

## 1. 代码地图

```
编译框架 vllm/compilation/
  ├─ decorators.py        # @support_torch_compile：把模型 forward 包成可编译类
  ├─ wrapper.py           # TorchCompileWrapperWithCustomDispatcher：调 torch.compile + 自定义 dispatch
  ├─ backends.py          # 核心：VllmBackend(切图) / PiecewiseBackend(分段编译+capture+replay)
  │                       #       CompilerManager(编译+磁盘缓存) / split_graph / ConcreteSizeEntry
  ├─ compiler_interface.py# InductorAdaptor / EagerAdaptor：实际调 Inductor 或 eager
  ├─ pass_manager.py      # PostGradPassManager：post-grad Inductor passes（融合/noop 消除）
  ├─ fusion.py / noop_elimination.py  # 具体的融合 / no-op 消除 pass
  ├─ counter.py           # CompilationCounter：计数（图数/段数/捕获数），供测试断言
  └─ monitor.py           # start/end_monitoring_torch_compile：计时 + depyf debug dump

forward context  vllm/forward_context.py
  └─ ForwardContext / set_forward_context / get_forward_context

attention 切点   vllm/attention/layer.py
  └─ unified_attention(_with_output) custom op + static_forward_context 登记

配置             vllm/config.py
  └─ CompilationLevel / CompilationConfig / VllmConfig.{pad_for_cudagraph,_set_cudagraph_sizes,__post_init__}

V1 集成          vllm/v1/worker/
  ├─ gpu_model_runner.py  # 持久 buffer / execute_model(pad) / _dummy_run / capture_model / use_cuda_graph
  └─ gpu_worker.py        # compile_or_warm_up_model：warmup→capture 的编排
```

---

## 2. 启动期调用链：compile → warmup → capture

### 阶段 A：配置阶段 —— 决定开不开、捕获哪些 size

**A1.** `vllm/config.py:3794` `VllmConfig.__post_init__`（PIECEWISE 强制开启）
```python
if envs.VLLM_USE_V1 and self.model_config is not None and \
    not self.model_config.enforce_eager:
    self.compilation_config.custom_ops = ["none"]      # :3801 让 Inductor 融合
    self.compilation_config.use_cudagraph = True        # :3802
    self.compilation_config.use_inductor = True         # :3803
    self.compilation_config.cudagraph_num_of_warmups = 1# :3804
    self.compilation_config.level = CompilationLevel.PIECEWISE  # :3807
    self.compilation_config.set_splitting_ops_for_v1()  # :3808
```
> V1 + 非 enforce_eager ⇒ 一律 piecewise。`enforce_eager` 把这整块跳过，level 保持 0。

**A2.** `config.py:3573` `set_splitting_ops_for_v1`：切点 = attention custom op
```python
if not self.splitting_ops:
    self.splitting_ops = ["vllm.unified_attention",
                          "vllm.unified_attention_with_output"]
```

**A3.** `config.py:3810` → `config.py:3849` `_set_cudagraph_sizes`：算捕获 size 列表
- `:3907-3908` V1 默认候选：`[1, 2, 4] + [i for i in range(8, 513, 8)]`。
- `:3910-3913` 按 `max_num_batched_tokens` 截断。
- `:3915` 调 `init_with_cudagraph_sizes(batch_size_capture_list)`。

**A4.** `config.py:3524` `init_with_cudagraph_sizes`：降序 + 预计算 pad 映射
- `:3555-3557` `cudagraph_capture_sizes.sort(reverse=True)`；`max_capture_size` = 最大档。
- `:3559-3571` 构造 `bs_to_padded_graph_size`：对 `[0, max_capture_size]` 每个 bs 预存"应 pad 到的捕获档"。这是运行期 `pad_for_cudagraph` 的 O(1) 查找表。
- `:3539-3552` 处理 `compile_sizes`（额外要 Inductor 静态编译的 size，可写 `"cudagraph_capture_sizes"` 展开成所有捕获档）。

**A5.** `gpu_model_runner.py:187` 读出最终开关
```python
self.use_cuda_graph = (self.vllm_config.compilation_config.level
                       == CompilationLevel.PIECEWISE
                       and not self.model_config.enforce_eager)
self.cudagraph_batch_sizes = list(reversed(   # :194 转成升序，方便取 [-1] 是最大档
    self.vllm_config.compilation_config.cudagraph_capture_sizes))
```

**A6.** `gpu_model_runner.py:203-208` 预分配**持久 buffer**（CUDA graph replay 的地址前提）
```python
self.input_ids = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
```

---

### 阶段 B：load_model —— 把模型 forward 包成"可编译类"

**B1.** `gpu_model_runner.py:1278` `load_model` → `get_model(...)`：构造模型。被 `@support_torch_compile` 装饰的模型类（如各 `*Model`）此时混入编译能力。

**B2.** `decorators.py:38` `support_torch_compile`（类装饰器）
- `:98-122` `cls_decorator_helper`：从 `forward` 的类型注解**推断动态维度**——凡注解为 `torch.Tensor` / `IntermediateTensors` 的参数，其第 0 维标为动态（`:103-107`）。
- `:146` 把 `TorchCompileWrapperWithCustomDispatcher` 加进 `cls.__bases__`。
- `:150-165` 改写 `__init__`：若 level∈{NO_COMPILATION, DYNAMO_AS_IS} 或不支持 dynamo，则 `do_not_compile=True` 直接返回（`:155-160`）；否则调 wrapper 的 `__init__`。

**B3.** `wrapper.py:32` `TorchCompileWrapperWithCustomDispatcher.__init__`
```python
backend = vllm_config.compilation_config.init_backend(vllm_config)  # :42
compiled_callable = torch.compile(self.forward,
    fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE, backend=backend)  # :44
```
> **此处 `torch.compile` 是惰性的**：只是包了一层，真正编译要等第一次实际调用 `forward` 才触发。

**B4.** `config.py:3502` `init_backend`：level 3 返回 `VllmBackend(vllm_config)`（`:3519-3522`）。

---

### 阶段 C：第一次 forward —— Dynamo 抓图 + 切图 + 分段编译

由 §2 阶段 D 的 `compile_or_warm_up_model` → `_dummy_run` 触发。第一次调用被装饰模型的 `__call__`：

**C1.** `decorators.py:167` `__call__`（装饰器注入的分发逻辑）
- `:171` 若 `do_not_compile` 或正处于 `torch.compiler.is_compiling()`，直接走原始 `forward`。
- `:175-200` 首次编译：用 `torch._dynamo.mark_dynamic(arg, dims)`（`:188`）把 batch/token 维标成**符号动态 shape**——使一张图通用于任意 token 数。
- `:202` `start_monitoring_torch_compile`（计时 + 可选 depyf dump）。
- `:209-239` 走 `self.compiled_callable(...)`，触发 Dynamo → backend；同时 hook `InliningInstructionTranslator.inline_call`（`:230-237`）记录所有被 trace 的源码文件，用于编译缓存失效判定。

**C2.** `backends.py:350` `VllmBackend.__call__`（Dynamo 把抓好的 FX 图交给它）
- `:353-419` 计算编译缓存目录：env hash + vllm_config hash + 被 trace 文件的代码 hash + compiler hash（`:359-394`），任一变化则 cache 失效；初始化 `CompilerManager` 缓存。
- `:431` `assert not self._called`：**一个 backend 实例只能被调一次**。
- `:436` **切图**：`self.split_gm, self.piecewise_graphs = split_graph(graph, self.compilation_config.splitting_ops)`。

**C3.** `backends.py:161` `split_graph`：按 splitting op 切
```python
for node in graph.graph.nodes:
    if node.op == 'call_function' and str(node.target) in ops:  # :170 命中 attention op
        subgraph_id += 1                 # 切：attention 单独成段
        node_to_subgraph_id[node] = subgraph_id
        split_op_graphs.append(subgraph_id)
        subgraph_id += 1
    else:
        node_to_subgraph_id[node] = subgraph_id  # 静态计算归并进当前段
```
- `:182-186` 用 `split_module(..., keep_original_order=True)` 切；**保持原顺序**至关重要——否则有 mutation 的图语义会变（`:178-181` 注释）。
- 产出 `SplitItem` 列表，`is_splitting_graph` 标记该段是否就是 attention 段（`:200-201`）。

**C4.** `backends.py:448-457` 对**非 attention 段**（`not item.is_splitting_graph`）逐段编译
```python
submod_names_to_compile = [item.submod_name for item in self.piecewise_graphs
                           if not item.is_splitting_graph]
PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                            self.vllm_config, self.graph_pool, self).run(*example_inputs)
```

**C5.** `backends.py:247` `PiecewiseCompileInterpreter.call_module`：用 fake tensor 跑图、对目标段调编译
- `:260-268` 对每个要编译的段调 `compiler_manager.compile(submod, ..., runtime_shape=None)` ——**general（符号）shape** 编译，一张图通用。
- `:270-273` 把该段在 `split_gm` 里**替换成一个 `PiecewiseBackend` 实例**——运行期对该段的"按 size 编译 + capture + replay"都由它接管。
- `:275` `num_piecewise_capturable_graphs_seen += 1`。

**C6.** `backends.py:92` `CompilerManager.compile` → `compiler_interface.py` 的 `InductorAdaptor.compile`（`:163`）：真正调 Inductor 做 post-grad 融合（`PostGradPassManager`，`backends.py:336-348`）并生成 kernel；产物写入磁盘缓存（`backends.py:127-133`）。

> 这一阶段只**编译**（拿到融合 kernel），**不录 CUDA graph**。`PiecewiseBackend.__call__` 首次运行只跑 general shape 图（`backends.py:604-608`）。

---

### 阶段 D：compile_or_warm_up_model —— warmup 与 capture 的编排

**D0.** 调用入口：`v1/executor/abstract.py:65` `collective_rpc("compile_or_warm_up_model")`（在 `initialize_from_config` 里，KV cache 初始化之后）。

**D1.** `gpu_worker.py:202` `compile_or_warm_up_model`
```python
warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()  # :206
if not self.model_config.enforce_eager:
    warmup_sizes = [x for x in warmup_sizes               # :208-211 去掉会被 capture 的 size
                    if x not in ...cudagraph_capture_sizes]
for size in sorted(warmup_sizes, reverse=True):           # :212
    self.model_runner._dummy_run(size)                    # :214 触发对该 size 的 Inductor 静态编译
if not self.model_config.enforce_eager:
    self.model_runner.capture_model()                     # :216 ← 先 compile/warmup，再 capture
```
> 顺序铁律：**先把各 size 编译/warmup 好，最后才 `capture_model`**（注释 `:221`：capture 后才 warmup sampler，避免 `empty_cache` 清掉 graph 显存）。

**D2.** `gpu_model_runner.py:1390` `_dummy_run(num_tokens)`：用持久 buffer 跑一遍假 forward
- `:1416` `input_ids = self.input_ids[:num_tokens]`（取持久 buffer 切片，固定地址）。
- `:1437-1445` `with set_forward_context(None, vllm_config, num_tokens=...): model(...)`——`attn_metadata=None`，dummy 不需要真 KV。
> dummy run 的意义：①触发该 shape 的编译；②给 cudagraph 提供录制时机。

**D3.** `gpu_model_runner.py:1600` `capture_model`
```python
if not self.use_cuda_graph:
    logger.warning("Skipping CUDA graph capture. ...")   # :1602 enforce_eager 路径
    return
with graph_capture(device=self.device):                  # :1613 进入捕获上下文
    for num_tokens in reversed(self.cudagraph_batch_sizes):  # :1614 大→小
        for _ in range(...cudagraph_num_of_warmups):     # :1615-1617 warmup
            self._dummy_run(num_tokens)
        self._dummy_run(num_tokens)                       # :1618 这一次真正录制
```
- `:1614` **从大 size 到小 size**：让小 size graph 复用大 size 的显存池（共享 `global_graph_pool`）。
- `graph_capture`（`parallel_state.py:785`）把捕获放到独立 CUDA stream，并协调 TP/PP 的通信，避免后台 kernel 混入录制。

**D4.** capture 时每段的录制：`backends.py:604` `PiecewiseBackend.__call__`
```python
if not self.first_run_finished:           # :605 第一次（general shape）不 capture
    self.first_run_finished = True
    return self.compiled_graph_for_general_shape(*args)
runtime_shape = args[self.sym_shape_indices[0]]  # :610 取本次 batch size
if runtime_shape not in self.concrete_size_entries:  # :611 非捕获档 → 跑通用图
    return self.compiled_graph_for_general_shape(*args)
entry = self.concrete_size_entries[runtime_shape]    # :615
...
if entry.need_to_compile and not entry.compiled:     # :620 该 size 要 Inductor 静态编译
    entry.runnable = ...compile(..., runtime_shape=runtime_shape)
if not entry.use_cudagraph:                          # :637
    return entry.runnable(*args)
if entry.cudagraph is None:                          # :640 还没录 → 先 warmup 再录
    if entry.num_finished_warmup < cudagraph_num_of_warmups:  # :641
        entry.num_finished_warmup += 1
        return entry.runnable(*args)                 # :649 warmup 跑，不录
    input_addresses = [x.data_ptr() for x in args if ...]    # :658 记录输入地址
    entry.input_addresses = input_addresses
    cudagraph = torch.cuda.CUDAGraph()               # :662
    with ExitStack() as stack:
        if not self.is_first_graph:                  # :665 非首段禁 gc/empty_cache（否则 capture 极慢）
            stack.enter_context(patch("gc.collect", lambda: None))
            stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))
        with torch.cuda.graph(cudagraph, pool=self.graph_pool):  # :677 录制！
            output = entry.runnable(*args)
            if self.is_last_graph:
                output = weak_ref_tensors(output)    # :686 末段输出转弱引用省显存
    entry.output = weak_ref_tensors(output)          # :690
    entry.cudagraph = cudagraph                      # :691
    compilation_counter.num_cudagraph_caputured += 1 # :693
    return output                                    # :698 注意：返回真 output 不是弱引用
```
> 每个**静态段**单独录一张 `CUDAGraph`，绑定到本段本 size 的 `ConcreteSizeEntry`。一个模型 N 层就有 ~N+1 段，每段 × 每个 size 各一张图。

---

## 3. 运行期调用链：pad → replay

入口：[模块 00](../00-request-lifecycle/impl.md) 的 `EngineCore.step` → executor → `gpu_worker.py:242` → `gpu_model_runner.py:987` `execute_model`。

**R1.** `gpu_model_runner.py:559,568` `_prepare_inputs` 把本步 token 写进**持久 buffer**
```python
self.input_ids[:total_num_scheduled_tokens].copy_(
    self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)   # :559
self.positions[:total_num_scheduled_tokens].copy_(...)                    # :568
```
> 写进 §A6 那块固定地址 buffer，是 replay 能读到新数据的前提（地址不变、内容更新）。

**R2.** `gpu_model_runner.py:1001` 判定是否用 graph + **pad**
```python
if (self.use_cuda_graph
        and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):  # :1001 不超最大档
    num_input_tokens = self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)  # :1005
else:
    num_input_tokens = num_scheduled_tokens   # :1009 eager / 大 batch 不 pad
attn_metadata.num_input_tokens = num_input_tokens  # :1010
```

**R3.** `config.py:3704` `pad_for_cudagraph`：O(1) 查表
```python
return self.compilation_config.bs_to_padded_graph_size[batch_size]
```

**R4.** `gpu_model_runner.py:1040` 文本模型用 **token id** 作输入（而非 embedding）
```python
# For text-only models, we use token ids as input.
# ... it is not desirable for performance since then the embedding layer
# is not included in the CUDA graph.
input_ids = self.input_ids[:num_input_tokens]   # 取 pad 后长度的切片
```
> 故意用 token id：让 embedding 层也进 CUDA graph（多模态则被迫用 embedding，`:1021-1034`）。

**R5.** `gpu_model_runner.py:1062` `set_forward_context` + 调模型
```python
with set_forward_context(attn_metadata, self.vllm_config):
    hidden_states = self.model(input_ids=input_ids, positions=positions, ...)
```

**R6.** `forward_context.py:56` `set_forward_context`：把 `attn_metadata` 塞进全局
```python
_forward_context = ForwardContext(
    no_compile_layers=vllm_config.compilation_config.static_forward_context,  # :96
    virtual_engine=virtual_engine, attn_metadata=attn_metadata, ...)          # :98-100
```

**R7.** 模型 `forward` 内，逐段执行 = `静态段.replay()` 夹着 `attn(eager)`：

- **静态段 replay**：`backends.py:700` `PiecewiseBackend.__call__` 末尾
```python
if self.is_debugging_mode:                  # :700 debug 下校验输入地址不变
    assert new_input_addresses == entry.input_addresses
entry.cudagraph.replay()                    # :710 一条指令重放整段
return entry.output                          # :711
```
- **attention 段 eager**：`attention/layer.py:332` `unified_attention`
```python
forward_context = get_forward_context()          # :338 取全局上下文
attn_metadata = forward_context.attn_metadata     # :339 拿本步元数据
self = forward_context.no_compile_layers[layer_name]  # :340 据名反查层对象
kv_cache = self.kv_cache[forward_context.virtual_engine]
return self.impl.forward(self, query, key, value, kv_cache, attn_metadata)  # :342 真 attention
```
> attention 不在 graph 里，能自由读变长 KV、走 data-dependent 分支；它需要的 `attn_metadata` 走 forward context 旁路传入，而非 graph 输入。

**R8.** `gpu_model_runner.py:1073` 取回有效部分：`hidden_states = hidden_states[:num_scheduled_tokens]` —— **丢掉 pad 多算的尾部**，再 `compute_logits` / `sample`（详见 [模块 00](../00-request-lifecycle/impl.md) §F4、[模块 06](../06-sampling-structured-output/design.md)）。

---

## 4. 实现中的精妙之处（Tricks 与非显然设计）

> 每条格式：**现象/问题 → 代码位置 → 精妙之处**。这些是读代码时容易一眼滑过、理解了才算真懂 vLLM 编译栈的点。

### T1 · attention 做成 custom op，是为了给 Dynamo 一个"干净的切点"
- **代码**：`attention/layer.py:354,393` `direct_register_custom_op`；`config.py:3577` 把这两个 op 名设为 `splitting_ops`。
- **精妙之处**：Dynamo 默认会把 attention 也 trace 进图。把 attention 封成**注册过 fake 实现的 custom op**，Dynamo 就把它当成一个**不可见内部的黑盒节点**，`split_graph` 据 op 名精确切割。fake 实现（`unified_attention_fake`，`:345` 返回 `empty_like`）让 Dynamo 在 trace 期能推断 shape 而不真跑 attention。**切点的存在性完全由"是否注册为 custom op + 列入 splitting_ops"决定**——这是 piecewise 的物理基础。

### T2 · attention op 只收一个字符串 layer_name，运行期反查层对象
- **代码**：custom op 签名只有 tensors + `layer_name: str`（`attention/layer.py:336`）；`forward_context.no_compile_layers[layer_name]` 反查（`:340`）；登记在 `attention/layer.py:146-149`。
- **精妙之处**：Dynamo 不能把"一个 `nn.Module` 实例"当图输入/op 参数。于是只传**字符串名字**，真正的层对象、KV cache、`attn_metadata` 全走全局 `ForwardContext` 旁路。这样 attention op 的签名是纯张量+字符串，既能被 Dynamo 识别为干净 custom op，又能在图外拿到一切运行期状态。

### T3 · 持久 buffer 既是性能优化，又是 CUDA graph 的硬前提
- **代码**：`gpu_model_runner.py:203-208` 预分配；`:559/568` 每步 `copy_(..., non_blocking=True)` 写入；`:1040` 取切片。
- **精妙之处**：CUDA graph replay 按录制时的**输入地址**读数据。若每步新分配输入张量，地址会变、replay 读到错误内存。固定地址 buffer + 每步原地 copy_ 让"地址不变、内容更新"，graph 才能反复重放同一段。debug 模式还会断言地址一致（`backends.py:700-708`）。

### T4 · pad 到最近捕获档：用预计算数组把运行期开销压到 O(1)
- **代码**：`config.py:3559-3571` 构造 `bs_to_padded_graph_size`；`config.py:3704` `pad_for_cudagraph` 一次下标。
- **精妙之处**：本可以每步二分查找最近档，但热路径上连 `log n` 都要省。启动期一次性把 `[0, max_capture_size]` 每个 bs 的目标档**全部预计算成 list**（注释 `:3397-3400` 解释为何用 list 而非 dict——下标查比 hash 快）。运行期就是一次数组访问。

### T5 · 从大 size 到小 size 捕获，复用同一显存池
- **代码**：`gpu_model_runner.py:1614` `for num_tokens in reversed(...)`；`backends.py:313-319` 全局共享 `global_graph_pool`。
- **精妙之处**：CUDA graph 的中间激活占显存。先录最大 size（占最多显存），其显存池可被后续小 size 的 graph **复用**（小的需求一定 ≤ 大的）。反过来从小到大，则每个大 size 都要新开池子，显存浪费。`pool=self.graph_pool`（`backends.py:677`）让所有段/所有 size 共享一个池。

### T6 · 先 compile 再 capture，且 warmup 必须在 capture 之前
- **代码**：`gpu_worker.py:206-216` 先 `_dummy_run(warmup_sizes)` 再 `capture_model`；`backends.py:641-649` 每个 size capture 前先跑 `cudagraph_num_of_warmups` 次。
- **精妙之处**：CUDA graph 录的是"那一刻实际执行的 kernel 序列"。若没先编译，录到的是未优化 eager kernel；若没 warmup，cuBLAS/Inductor 的**惰性初始化、autotune、首次内存分配**会被错误地录进 graph，replay 时行为不可控。warmup 把这些一次性副作用"跑掉"，让录制只抓稳定的计算 kernel。

### T7 · 一张符号 shape 图通用所有 token 数（compile），但 cudagraph 必须每 size 一张
- **代码**：`decorators.py:188` `mark_dynamic`；`PiecewiseCompileInterpreter` 编译用 `runtime_shape=None`（`backends.py:268`）；cudagraph 按 `runtime_shape` 分 `ConcreteSizeEntry`（`backends.py:585`）。
- **精妙之处**：Inductor 能编译"动态符号 shape"图，一张图服务任意 size（注释 `config.py:3330-3338` 解释"为何 compile 与 cudagraph 用不同 size 集"）。但 CUDA graph 是 shape-frozen 的，**只能一个 size 一张**。所以 vLLM 把两者解耦：Inductor 编译一张通用图（可选再为少数 `compile_sizes` 静态编译），cudagraph 则为每个 `cudagraph_capture_sizes` 各录一张。

### T8 · custom_ops=["none"]：主动关掉手写 kernel，让 Inductor 融合
- **代码**：`config.py:3801`，注释 `:3796-3799`。
- **精妙之处**：vLLM 有很多手写 CUDA custom op（RMSNorm、量化等）。但 piecewise cudagraph 与某些手写 kernel 配合有问题，且手写 op 是 Inductor 的黑盒、无法参与融合。V1 默认 `custom_ops=["none"]`，把这些算子交还给 Inductor，使其能跨算子融合并稳定纳入 graph——以"放弃个别手写 kernel 的极致性能"换"整体可融合 + 可 cudagraph"。

### T9 · 末段输出转弱引用，省一份 cudagraph 显存
- **代码**：`backends.py:680-690` `if self.is_last_graph: output = weak_ref_tensors(output)`。
- **精妙之处**：cudagraph 的 output 张量住在 pytorch 的 cudagraph 内存池里，由池管理。对**最后一段**（其输出不会再被其他 graph 当输入）把 `output` 转弱引用，让原始强引用立即释放、显存回池。注释（`:681-686`）明确这只对末段安全。但 `return output`（`:698`）必须返回**真 output 而非弱引用**，否则 pytorch 在 capture 期无法正确管理内存——一个非常微妙的引用计数平衡。

### T10 · 录制非首段时禁用 gc 和 empty_cache，否则 capture 慢到不可用
- **代码**：`backends.py:664-674` 对 `not self.is_first_graph` 的段 `patch("gc.collect", lambda: None)` 和 `patch("torch.cuda.empty_cache", lambda: None)`。
- **精妙之处**：一次 forward 要录 ~N 段 graph（每层一段）。pytorch 在每次 `torch.cuda.graph` 进出时默认会 gc + empty_cache。逐段都跑一遍，会让捕获**慢上数量级**。于是只在**第一段**保留 gc（清理一次即可），其余段全部 patch 掉。这是把"5~20 秒捕获"压住的关键。

### T11 · debug 模式下断言 replay 输入地址不变
- **代码**：`backends.py:700-708`，仅 `VLLM_LOGGING_LEVEL=="DEBUG"` 时启用（`:581`）。
- **精妙之处**：CUDA graph 最隐蔽的 bug 是"输入地址悄悄变了"——replay 仍跑、但读的是旧地址的过期数据，结果**静默错误**。debug 模式把每次 replay 的 `data_ptr()` 和录制时的对比，第一时间炸出来。生产模式为省开销关掉。

### T12 · forward 期间禁止改写 nn.Module buffer，编译期直接拦截
- **代码**：`wrapper.py:110-115` bytecode hook 里，若 `use_cudagraph` 且新代码含 `"update"`，直接 `raise RuntimeError`。
- **精妙之处**：cudagraph 下，forward 里 in-place 改 module buffer 会造成"录制时改了、replay 时不改"的静默错误。vLLM 在 Dynamo 编译产出的字节码里**静态扫描** `update` 的痕迹，发现就报错并 depyf 反编译出可疑代码定位给用户。把一类极难调试的运行期错误提前到编译期。

### T13 · 编译缓存的 key 必须覆盖"所有影响图结构的因素"
- **代码**：`backends.py:359-394` 拼 env hash + vllm_config hash + **被 trace 的源码文件内容 hash** + compiler hash；trace 文件由 `decorators.py:221-234` hook `inline_call` 收集。
- **精妙之处**：编译产物缓存到磁盘，二次启动直接 load 省几十秒。但缓存命中错了会跑错模型。所以 key 不仅含配置，还含 **Dynamo 实际 trace 过的每个源码文件的内容**——改了任何一个模型/层的代码，hash 变、缓存自动失效。`compute_hash`（`config.py:3412`）注释也警告"新增影响图的字段必须加进 factors"。

### T14 · VllmBackend 每实例只能调一次，强约束编译的确定性
- **代码**：`backends.py:431` `assert not self._called, "VllmBackend can only be called once"`。
- **精妙之处**：vLLM 要**完全掌控**编译过程（切图、缓存、计数都依赖"这是第一次也是唯一一次"）。Dynamo 默认可能跨实例复用编译，所以 `decorators.py:213` 还主动 `remove_from_cache` 强制不复用。配合 DYNAMO_ONCE 的自定义 dispatcher（`wrapper.py:117` `dispatch_to_code`），保证"编译一次、之后直接派发到编译码，跳过 guard"。

### T15 · num_input_tokens 经 attn_metadata 回传，给 DP 对齐用
- **代码**：`gpu_model_runner.py:1010` `attn_metadata.num_input_tokens = num_input_tokens`；`forward_context.py:78-80` DP 下读 `attn_metadata.num_input_tokens` 做跨 DP all-reduce。
- **精妙之处**：pad 后的 token 数不仅决定取哪张 graph，还要让数据并行（DP）各 rank 对齐"这一步处理多少 token"（`forward_context.py:83-91` 把它 all-reduce 成 `cu_tokens_across_dp_cpu`）。pad 值（而非原始值）参与 DP 对齐，保证各 rank 的 cudagraph 选择一致，避免某 rank 走 graph、某 rank 走 eager 造成的集合通信错位。

---

## 5. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| `enforce_eager=True` | `config.py:3795`、`gpu_model_runner.py:187` | PIECEWISE 分支跳过，`use_cuda_graph=False`，全程 eager；`capture_model` 直接 warning return（`:1600-1605`） |
| batch > `max_capture_size` | `gpu_model_runner.py:1001-1009` | 不满足 `<= cudagraph_batch_sizes[-1]`，不 pad、不 replay，走通用编译图 / eager（常见于大 prefill） |
| batch 命中非捕获档但 ≤ 最大档 | `config.py:3704` + `backends.py:611-613` | pad 到最近捕获档跑 graph；若该 padded size 仍未在 entries（理论不会），`PiecewiseBackend` 回退通用图 |
| 多模态模型 | `gpu_model_runner.py:1021-1034` | 被迫用 embedding 作输入（要拼 vision soft token），embedding 层**不**进 cudagraph |
| M-RoPE 模型 | `gpu_model_runner.py:213-218`、`:1042-1043` | `mrope_positions` 故意多一维做成非连续，以兼容 torch.compile（PR #12128） |
| PP 中间段 rank | `gpu_model_runner.py:1069-1071` | 返回 hidden_states 而非采样；`intermediate_tensors` 也写进持久 buffer（`:1052-1057`） |
| 某 size 同时需编译+捕获 | `backends.py:620-635` | `PiecewiseBackend` 先 Inductor 静态编译该 size，再在其上录 cudagraph |
| forward 改写 module buffer | `wrapper.py:110-115` | 编译期 bytecode 扫到 `update` 直接 RuntimeError，附 depyf 反编译定位 |
| 编译缓存 stale 风险 | `backends.py:411-419` | `VLLM_DISABLE_COMPILE_CACHE` 可禁缓存；否则 key 含代码 hash 自动失效 |
| 不支持 dynamo 的环境 | `decorators.py:158` `not supports_dynamo()` | `do_not_compile=True`，退回 eager |

---

## 6. 一图速查：编译/捕获/重放调用链

```
启动期 ───────────────────────────────────────────────────────────────────────
config.py:3794 __post_init__ ─► level=PIECEWISE, splitting_ops=attention,
                                 _set_cudagraph_sizes ─► init_with_cudagraph_sizes
                                   └ bs_to_padded_graph_size 预计算 (config.py:3559)

gpu_model_runner.py:203  预分配持久 buffer input_ids/positions
load_model :1278 ─► @support_torch_compile (decorators.py:38)
                      └ wrapper.py:44 torch.compile(forward, backend=VllmBackend)  [惰性]

gpu_worker.py:202 compile_or_warm_up_model
  ├─ ① _dummy_run(general/compile_sizes)  ─► 首次 __call__ (decorators.py:167)
  │     └ mark_dynamic :188 ─► VllmBackend.__call__ (backends.py:350)
  │          └ split_graph(splitting_ops) :436 ── 整图切成 [静态段|attn|静态段|...]
  │             └ PiecewiseCompileInterpreter.run :455
  │                  └ 每静态段 Inductor 编译(general) + 替换成 PiecewiseBackend :270
  └─ ② capture_model :1600
        graph_capture():  for size in reversed(capture_sizes):   ← 大→小, 共享 graph_pool
          warmup ×k ; _dummy_run(size)
            └ PiecewiseBackend.__call__ (backends.py:604)
                 warmup(:641) → torch.cuda.graph 录制(:677) → entry.cudagraph/output

运行期 (每步) ─────────────────────────────────────────────────────────────────
execute_model (gpu_model_runner.py:987)
  ├ _prepare_inputs ─► input_ids[:n].copy_(...)  写持久 buffer (:559)
  ├ pad_for_cudagraph(n) ─► bs_to_padded_graph_size[n]  O(1) (config.py:3704)
  ├ input_ids = self.input_ids[:num_input_tokens]  文本模型用 token id (:1040)
  └ set_forward_context(attn_metadata) (:1062) ─► model(...)
        静态段0 ─► PiecewiseBackend.replay() (backends.py:710)
        attn0   ─► unified_attention (attention/layer.py:332, eager, 读 forward_context + paged KV)
        静态段1 ─► replay() ; attn1 ─► eager ; ...
  └ hidden_states[:num_scheduled_tokens]  丢弃 pad 尾部 (:1073) ─► logits ─► sample

enforce_eager=True ⇒ 上述编译/捕获/pad 全部跳过，纯 eager forward
```
