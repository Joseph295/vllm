# 模块 09 · LoRA 多适配器 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的 LoRA 实现细节。所有 `file:line` 基于仓库当前版本（路径以 `/Users/xujunhong/vllm` 为根），行号若与本地略有出入，以函数名 / 类名为准。
> 架构以 **V1** 为主（`vllm/v1/worker/lora_model_runner_mixin.py` 等）。LoRA 横切在模型执行（[模块 00](../00-request-lifecycle/impl.md) 阶段 F）与调度（[模块 01](../01-scheduler-batching/design.md)）之上。

---

## 1. 代码地图

```
lora/                                # LoRA 子系统核心
  ├─ request.py                      # LoRARequest：用户侧适配器标识（lora_int_id 全局唯一）
  ├─ peft_helper.py                  # PEFTHelper：解析 adapter_config.json + 校验 + 算 scaling
  ├─ lora.py                         # LoRALayerWeights / PackedLoRALayerWeights：单层 LoRA 权重
  ├─ models.py                       # LoRAModel（一个适配器）
  │                                  # LoRAModelManager / LRUCacheLoRAModelManager（GPU 槽位 + 双层 LRU）
  ├─ layers.py                       # BaseLayerWithLoRA 及各种包装层（Column/Merged/QKV/Row/Embedding/Logits）
  │                                  # LoRAMapping
  ├─ worker_manager.py               # WorkerLoRAManager / LRUCacheWorkerLoRAManager（worker 侧入口、读盘）
  ├─ fully_sharded_layers.py         # fully_sharded_loras=True 时的 TP 全分片变体
  ├─ utils.py                        # from_layer / replace_submodule / parse_fine_tuned_lora_name 等
  ├─ resolver.py                     # 动态解析 LoRA 来源（HF 等）
  ├─ punica_wrapper/
  │   ├─ punica_base.py              # PunicaWrapperABC / PunicaWrapperBase（mapping 元数据基类）
  │   ├─ punica_gpu.py               # PunicaWrapperGPU：add_lora_linear/add_shrink/add_expand → Triton
  │   ├─ punica_selector.py          # get_punica_wrapper：按平台选 wrapper
  │   └─ utils.py                    # convert_mapping（lora_int_id→slot）、compute_meta（SGMV 段元数据）
  └─ ops/
      ├─ triton_ops/
      │   ├─ lora_shrink.py          # Triton kernel：x @ A（hidden→rank）
      │   ├─ lora_expand.py          # Triton kernel：buffer @ B（rank→out，原地加）
      │   └─ lora_kernel_metadata.py # LoRAKernelMeta：kernel 元数据张量 + prepare_tensors
      └─ torch_ops/                  # 无 Triton 时的 torch 退化实现

V1 集成 v1/
  ├─ worker/lora_model_runner_mixin.py   # LoRAModelRunnerMixin：load_lora_model / set_active_loras
  ├─ worker/gpu_input_batch.py           # InputBatch：request_lora_mapping / make_lora_inputs
  ├─ worker/gpu_model_runner.py          # 调 load_lora_model（:1280）/ set_active_loras（:613）
  ├─ worker/gpu_worker.py                # Worker.add_lora/remove_lora/list_loras/pin_lora（:256-266）
  ├─ engine/core.py                      # EngineCore.add_lora/...（:277-287）转发到 executor
  └─ core/sched/scheduler.py             # max_loras 调度约束（:262-302）

config.py:2573                           # LoRAConfig
```

---

## 2. 端到端调用链

下面分四条链路：**A. 加载适配器** → **B. 把基座层包装成 LoRA 层（启动期一次性）** → **C. 每 batch 构建 mapping 并激活** → **D. forward 时 punica 按请求选 adapter**。

### 链路 A：加载 LoRA（add_lora）

> **这一步在干嘛**：用户调 `add_lora` 时，把一个适配器从磁盘/HF 读进来、校验合法（rank 没超限、不是不支持的 DoRA 等），转成 `LoRAModel` 先驻留 CPU 缓存，再 `copy_` 进一个空闲的 GPU 槽位。CPU/GPU 缓存满了就按 LRU 驱逐最旧的。

**A1.** `vllm/v1/engine/core.py:277` `EngineCore.add_lora()`
```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.model_executor.add_lora(lora_request)
```
用户 / server 侧的 `add_lora` 请求经 EngineCore 转发到 executor（再 `collective_rpc` 到每个 worker）。`remove_lora` / `list_loras` / `pin_lora` 同样（`:280-287`）。

**A2.** `vllm/v1/worker/gpu_worker.py:256` `Worker.add_lora()` → `self.model_runner.add_lora(lora_request)`
→ `vllm/v1/worker/lora_model_runner_mixin.py:131` `LoRAModelRunnerMixin.add_lora()`：
```python
return self.lora_manager.add_adapter(lora_request)
```

**A3.** `vllm/lora/worker_manager.py:229` `LRUCacheWorkerLoRAManager.add_adapter()` —— **双层缓存的入口**
```python
if lora_request.lora_int_id not in self.list_adapters():
    # 先加载新适配器以验证有效性，再驱逐旧的（可能短暂超过 max_cpu_loras）
    lora = self._load_adapter(lora_request)
    if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
        self._adapter_manager.remove_oldest_adapter()   # CPU 缓存 LRU 驱逐
    loaded = self._adapter_manager.add_adapter(lora)
else:
    loaded = self._adapter_manager.get_adapter(...) is not None  # touch 更新 LRU 顺序
self._adapter_manager.activate_adapter(lora_request.lora_int_id)  # 同步搬到 GPU 槽位
```
> 注意注释（`:231-234`）：**先 load 再驱逐**，所以加载瞬间 CPU 缓存可能短暂超过 `--max-cpu-loras`。

**A4.** `vllm/lora/worker_manager.py:85` `WorkerLoRAManager._load_adapter()` —— **读盘 + 校验**
- `:100` `get_adapter_absolute_path(lora_request.lora_path)`：解析路径（本地 / HF）。
- `:102` `PEFTHelper.from_local_dir(...)`：读 `adapter_config.json`。
- `:107` `peft_helper.validate_legal(self.lora_config)`：校验 rank ≤ `max_lora_rank`、bias、不支持 DoRA / modules_to_save（`peft_helper.py:101-115`）。
- `:117` `LoRAModel.from_local_checkpoint(...)`：读 `adapter_model.safetensors`，构建 `LoRAModel`，**device="cpu"**（先落 CPU 缓存）。

**A5.** `vllm/lora/models.py:186` `LoRAModel.from_local_checkpoint()` → `:112` `from_lora_tensors()`
- `:130` `parse_fine_tuned_lora_name`：从 tensor 名解析出 `module_name` + 是 `lora_a` 还是 `lora_b` 还是 bias。
- `:157` / `:163` 把 `lora_a` / `lora_b` `.t()` 转置后存进 `LoRALayerWeights`（注意 PEFT 存的是 `[out,in]`，vLLM 要 `[in,out]`）。
- `:178` `lora.optimize()`：把 `scaling` 折进 `lora_b`（见 T4）。

至此适配器以 `LoRAModel` 形式驻留 **CPU 缓存** `_registered_adapters`。

### 链路 B：把基座层包装成 LoRA 层（启动期一次性）

> **这一步在干嘛**：开机时一次性"改装"基座模型——把所有该挂 LoRA 的 Linear/Embedding/Logits 层换成 `*WithLoRA` 包装层，并按 `max_loras × max_lora_rank` 上限**预分配好 GPU stacked 权重张量**（全零待填）。之后激活适配器只往里 `copy_`、从不重新分配显存，这样地址固定、和 CUDA graph 兼容。

**B1.** `vllm/v1/worker/gpu_model_runner.py:1280` 在模型加载后调
`self.model = self.load_lora_model(self.model, model_config, scheduler_config, lora_config, device)`
→ `lora_model_runner_mixin.py:27` `LoRAModelRunnerMixin.load_lora_model()`：
- `:31` `assert supports_lora(model)`。
- `:47` 创建 `LRUCacheWorkerLoRAManager`（传入 `max_num_seqs`、`max_num_batched_tokens`、`vocab_size`、`lora_config` 等）。
- `:57` `return self.lora_manager.create_lora_manager(model)`。

**B2.** `worker_manager.py:200` → `models.py:782` `create_lora_manager()` → 构造 `LRUCacheLoRAModelManager`（`models.py:711`）。其 `__init__` 调父类 `LoRAModelManager.__init__`（`:307`），关键：
- `:332` `self.lora_index_to_id = [None] * self.lora_slots`：**GPU 槽位表**，长度 `max_loras`。
- `:335` `self.punica_wrapper = get_punica_wrapper(max_num_batched_tokens, max_batches=max_num_seqs, device=device, max_loras=max_loras)`。
- `:345` `get_supported_lora_modules(self.model)`：列出该模型可被 LoRA 的模块名。
- `:365` `self._create_lora_modules()`：**真正的层替换**。

**B3.** `vllm/lora/models.py:470` `LoRAModelManager._create_lora_modules()` —— **遍历模型、替换层**
```python
for module_name, module in self.model.named_modules(remove_duplicate=False):
    if not self._match_target_modules(module_name): continue       # 只替换 target 层
    if self._filter_unsupported_mm_module(module_name): continue   # 多模态：跳过 vision/connector
    new_module = replace_submodule(self.model, module_name,
        from_layer(module, self.lora_slots, self.lora_config, packed_moduled_lst, ...))
    ...
    self.register_module(module_name, new_module)
    new_module.set_mapping(self.punica_wrapper)   # 所有 LoRA 层共享同一个 punica_wrapper（按引用）
```
- `from_layer`（`lora/utils.py`）按基座层类型，把 `ColumnParallelLinear` 换成 `ColumnParallelLinearWithLoRA`、`QKVParallelLinear` 换成 `QKVParallelLinearWithLoRA` 等（`layers.py` 各类的 `can_replace_layer` 决定匹配）。
- `:522` **关键**：每个 LoRA 层 `set_mapping(self.punica_wrapper)`——全模型所有 LoRA 层**共享同一个 PunicaWrapper 实例**，于是 mapping 元数据只需更新一次，所有层都看得到。
- `:501` 特殊处理 `lm_head`：把 `logits_processor` 换成 `LogitsProcessorWithLoRA`。

**B4.** `vllm/lora/layers.py:299` `BaseLinearLayerWithLoRA.create_lora_weights()` —— **预分配 stacked GPU 张量**
```python
self.lora_a_stacked = tuple(torch.zeros(max_loras, 1, lora_a_out_size, self.input_size, ...)
                            for _ in range(self.n_slices))   # [max_loras, 1, rank, in]
self.lora_b_stacked = tuple(torch.zeros(max_loras, 1, lora_b_out_size, max_lora_rank, ...)
                            for _ in range(self.n_slices))   # [max_loras, 1, out, rank]
```
> 这里就把 GPU 显存按 **max_loras 个槽位 × max_lora_rank** 静态预分配好。后续激活只往里 `copy_`，从不重新分配（与 CUDA graph 兼容，见 design §3.4）。

至此基座模型的目标层已全部被 LoRA 包装层替换，GPU stacked 权重张量全零待填。

### 链路 C：每 batch 构建 mapping 并激活适配器

> **这一步在干嘛**：每一步组好 batch 后，要告诉 kernel"第 i 个 token 该用哪个 LoRA"。先把"每请求一个 LoRA id"用 `np.repeat` 展开成"每 token 一个 id"，再把全局 `lora_int_id` 翻译成 0..max_loras-1 的 GPU 槽位下标（无 LoRA 标 -1），最后按 LoRA 把 token 排序、算出分段元数据喂给 Punica kernel。

**C1.** `vllm/v1/worker/gpu_model_runner.py:613`（在 `_prepare_inputs` 内）
```python
if self.lora_config:
    self.set_active_loras(self.input_batch, num_scheduled_tokens)
```

**C2.** `vllm/v1/worker/lora_model_runner_mixin.py:74` `set_active_loras()`
```python
prompt_lora_mapping, token_lora_mapping, lora_requests = \
                    input_batch.make_lora_inputs(num_scheduled_tokens)
return self._set_active_loras(prompt_lora_mapping, token_lora_mapping, lora_requests)
```

**C3.** `vllm/v1/worker/gpu_input_batch.py:615` `InputBatch.make_lora_inputs()` —— **构建 token→lora mapping**
```python
req_lora_mapping = self.request_lora_mapping[:self.num_reqs]      # 每请求一个 lora_int_id
prompt_lora_mapping = tuple(req_lora_mapping)                     # 给 sampler/logits
token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))  # 展开到每 token
active_lora_requests = set(self.lora_id_to_lora_request.values())
```
> `request_lora_mapping[i]` 在请求加入 batch 时写入（`gpu_input_batch.py:347` `= lora_id`，无 LoRA 则 `:352` `= 0`）；移除请求时清零（`:378-384`）。`np.repeat` 把"每请求一个 id"按每请求被调度的 token 数展开成"每 token 一个 id"。

**C4.** `lora_model_runner_mixin.py:59` `_set_active_loras()`
```python
lora_mapping = LoRAMapping(token_lora_mapping, prompt_lora_mapping, is_prefill=True)
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```
> `is_prefill=True` 是刻意写死的（`:65-71` 注释）：CUDA 平台 prefill/decode 共用同一套 kernel，该 flag 基本被忽略；非 CUDA 平台靠它走 SGMV。
> **澄清**：这里的 `is_prefill` 是 LoRA / punica 底层算子层面的标志位（一个代码字段），只在非 CUDA 平台用于选 SGMV / BGMV kernel 路径、在 CUDA 上恒被忽略；它与调度器层面「取消 prefill/decode 阶段区分」（V1 调度无 prefill/decode 阶段之分，详见 [模块 01](../01-scheduler-batching/design.md)）不是同一回事。LoRA 路径并未复活调度层的阶段概念，读者不必把二者混为一谈。

**C5.** `worker_manager.py:165` `set_active_adapters()` → `set_active_adapters_worker`（`adapter_commons`）做两件事：
1. `_apply_adapters(requests)`（`worker_manager.py:216`）：保证本 batch 用到的每个 LoRA 都已激活到 GPU 槽位。`LRUCacheWorkerLoRAManager._apply_adapters` 先断言 `len(loras_map) <= lora_slots`（`:221`，**这是 max_loras 的最后一道防线**），再逐个 `add_adapter`（命中则 touch，未命中则 load+activate）。
2. `set_adapter_mapping(mapping)`：更新 punica 元数据（C6）。

**C6.** `models.py:381` `LoRAModelManager.activate_adapter()` —— **把 CPU 权重搬上 GPU 槽位**
```python
first_free_slot = next((i,id) for i,id in enumerate(self.lora_index_to_id) if id is None)
index, _ = first_free_slot
self.lora_index_to_id[index] = lora_model.id              # 占用槽位
for module_name, module in self.modules.items():
    module_lora = self._get_lora_layer_weights(lora_model, module_name)
    if module_lora:
        module.set_lora(index, module_lora.lora_a, module_lora.lora_b, ...)  # copy_ 进 stacked[index]
    else:
        module.reset_lora(index)                          # 该层无此 LoRA → 清零槽位
```
- 若无空槽，`LRUCacheLoRAModelManager.activate_adapter`（`models.py:743`）先 `self._active_adapters.remove_oldest()` **LRU 驱逐 GPU 上最旧的**，再激活。
- `set_lora`（`layers.py:365` / 打包层 `:691`）把 `lora_a.T` / `lora_b.T` `copy_(..., non_blocking=True)` 进 `lora_a_stacked[index]` / `lora_b_stacked[index]`。

**C7.** `models.py:689` `set_adapter_mapping()` → `:453` `_set_adapter_mapping()`
```python
self.punica_wrapper.update_metadata(mapping, self.lora_index_to_id,
                                    self.lora_slots + 1, self.vocab_size,
                                    self.lora_config.lora_extra_vocab_size, self.long_lora_context)
```
> 经 `set_adapter_mapping`（`models.py:689`）的 `_last_mapping` 缓存：**mapping 没变就不重算**（见 T6）。

**C8.** `punica_gpu.py:57` `PunicaWrapperGPU.update_metadata()`
```python
self._update_base_metadata(mapping, lora_index_to_id, ...)   # convert_mapping: int_id → slot
self.token_mapping_meta.prepare_tensors(self.token_lora_indices)   # 构造 SGMV 段元数据
self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)
```

**C8a.** `punica_wrapper/utils.py:44` `convert_mapping()` —— **lora_int_id → GPU slot 下标**
```python
lora_idx = lora_index_to_id.index(index_mapping_indices[i]) if id>0 else -1   # 翻译成槽位下标，无LoRA=-1
...
base_indices = indices[1]            # 每 token 的 slot 下标（token_lora_indices）
sampler_indices = prompt_mapping_tensor
```

**C8b.** `lora/ops/triton_ops/lora_kernel_metadata.py:80` `LoRAKernelMeta.prepare_tensors()` —— **构造分段元数据**
```python
no_lora = torch.all(token_lora_mapping == -1)
self.no_lora_flag_cpu[0] = no_lora
if no_lora: return                                    # 全无 LoRA：早退
_, idx_sorted = torch.sort(token_lora_mapping, stable=True)   # token 按 lora id 排序
self.token_indices_sorted_by_lora_ids[:n].copy_(idx_sorted)
lora_ids, num_tokens_per_lora = torch.unique(token_lora_mapping, sorted=True, return_counts=True)
self.active_lora_ids[...] = lora_ids                  # batch 内出现的 lora slot 集合
self.num_tokens_per_lora[...] = num_tokens_per_lora   # 每 lora 多少 token
lora_token_start_loc = torch.cumsum(num_tokens_per_lora) # 每 lora 段的起始 token 偏移
```
> 这套元数据正是 SGMV/BGMV kernel 的输入：`active_lora_ids` 告诉 kernel "本步有哪些 LoRA"，`num_tokens_per_lora` + `lora_token_start_loc` 划出每个 LoRA 的 token 段，`token_indices_sorted_by_lora_ids` 让 kernel gather 出每段的原始 token 行。

### 链路 D：forward 时 punica 按请求选 adapter

> **这一步在干嘛**：真正算的地方。每个被适配层先跑**基座大 GEMM**（全 batch 共享 `W`、只算一遍），再用 Punica kernel 分两步算 LoRA 增量——`shrink`（`x·A`，把 hidden 压到 rank）和 `expand`（`·B`，还原到 out 并**原地加回基座输出**）。kernel 在 grid 的一维上并行各 LoRA，无 LoRA 的 token（id=-1）直接早退，一次 launch 算完全 batch、全 LoRA。

**D1.** `gpu_model_runner.py` 模型 forward 中，每个被适配的 Linear 层走它的 `forward`。以 `ColumnParallelLinearWithLoRA.forward`（`layers.py:554`）为例：
```python
output_parallel = self.apply(input_, bias)   # 基座 + LoRA 增量
```

**D2.** `layers.py:401` `BaseLinearLayerWithLoRA.apply()`
```python
output = self.base_layer.quant_method.apply(self.base_layer, x, bias)   # ① 基座大 GEMM，全 batch 共享 W
...
self.punica_wrapper.add_lora_linear(output, x, self.lora_a_stacked,     # ② LoRA 增量，原地加到 output
                                    self.lora_b_stacked, self.lora_bias_stacked,
                                    1.0, self.output_slices)
return output
```
> 基座输出和 LoRA 增量都写进同一个 `output`，对应 design 的 `y = x·W + (x·A·B)·s`。

**D3.** `punica_gpu.py:181` `PunicaWrapperGPU.add_lora_linear()` —— **shrink + expand 两段**
```python
buffer = torch.zeros((len(output_slices), x.size(0), r), dtype=torch.float32, ...)  # [slices, tokens, rank]
self.add_shrink(buffer, x, lora_a_stacked, scale)   # buffer = (x @ A) * scale
self.add_expand(y, buffer, lora_b_stacked, None, output_slices, add_inputs=True)  # y += buffer @ B
```
> 中间 `buffer` 用 **float32**（`:227` 注释引 triton-lang/triton#1387，避免低精度累加误差）。

**D4.** `punica_gpu.py:76` `add_shrink()` → `lora_shrink(x, lora_a_stacked, y, *meta_args, scale)`
**D5.** `punica_gpu.py:102` `add_expand()` → `lora_expand(x, lora_b_stacked, y, *meta_args, ...)`
两者都把 `self.token_mapping_meta.meta_args(num_tokens)`（`lora_kernel_metadata.py:126`）展开成 kernel 的元数据参数：`(token_lora_mapping, token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, active_lora_ids, no_lora_flag_cpu)`。

**D6.** `lora/ops/triton_ops/lora_shrink.py:18` `_lora_shrink_kernel`（Triton）—— **分段并行的核心**
```python
lora_idx = tl.program_id(axis=2)          # grid 的一维专门跑"哪个 LoRA"
lora_id = tl.load(lora_ids + lora_idx)
if lora_id == -1: return                   # 无 LoRA 段，早退（不污染基座输出）
lora_m_size = tl.load(num_tokens_per_lora + lora_idx)   # 本 LoRA 有多少 token
...
lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
cta_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
ram = tl.load(cta_lora_seq_indices + offset_m)   # gather 出本 LoRA 该处理的原始 token 行下标
do_shrink_kernel(... lora_id ... ram ...)        # 用 lora_id 选权重、ram 选输入行，做 GEMM
```
> **这就是 SGMV 的精髓**：grid 的 `axis=2` 维度把"不同 LoRA"并行展开，每个 program block 负责一个 LoRA 的一段 token；`token_indices_sorted_by_lora_ids` 让属于同一 LoRA 的 token（即便在原 batch 里不连续）被 gather 到一起。一次 launch 算完全 batch、全 LoRA，且无 LoRA 的 token（`lora_id==-1`）直接早退。

**D7.** sampler / logits 侧：`LogitsProcessorWithLoRA`（`layers.py:957`）用 `add_lora_logits`（`punica_gpu.py:247`），走 `prompt_mapping_meta`（每请求一个 LoRA，而非每 token），同样 shrink+expand。Embedding 侧 `VocabParallelEmbeddingWithLoRA` 只需 expand（`add_lora_embedding`，`punica_gpu.py:153`）。

---

## 3. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。这些是读 LoRA 代码时容易一眼滑过、但理解了才算真懂的点。

### T1 · 不 merge ΔW 进 W，运行时分 shrink/expand 两步算增量

- **代码**：`punica_gpu.py:181` `add_lora_linear` → `add_shrink` + `add_expand`。
- **精妙之处**：如果把 `ΔW=AB` 加进 `W`，每个 LoRA 都会得到一份不同的满秩 `[in,out]` 权重，基座就无法共享、显存爆炸、也无法即时切换。vLLM 坚持**基座 `W` 全请求共享只算一遍**，LoRA 增量另算。而增量不直接算 `x·(AB)`（那是个 `[in,out]` 大矩阵乘），而是 `((x·A)·B)`——中间张量只有 `[num_tokens, rank]`，计算量从 `O(tokens·in·out)` 降到 `O(tokens·rank·(in+out))`。低秩在这里同时省了**显存**和**算力**。

### T2 · SGMV：用 grid 的一个维度并行"不同 LoRA"，靠排序后的 token 下标 gather

- **代码**：`lora_shrink.py:40` `lora_idx = tl.program_id(axis=2)`；`:59` `cta_lora_seq_indices = token_indices_sorted_by_lora_ids + start + offset`；元数据来自 `lora_kernel_metadata.py:106-124`。
- **精妙之处**：混合 batch 里 LoRA-1 的 token 和 LoRA-2 的 token 可能交错。`prepare_tensors` 先 `torch.sort(token_lora_mapping)` 把同 LoRA 的 token 下标聚到一起（`:106`），再 `torch.unique(return_counts=True)` 得到每 LoRA 的段长（`:113`）和 `cumsum` 段起点（`:122`）。kernel 于是能用 `program_id(axis=2)` 这一维把"每个 LoRA"并行展开，每个 program 通过 `start_loc + offset` 索引进排序后的下标数组、gather 出自己那段 token。**一次 launch 处理任意多个 LoRA，且各段大小可不同**——这正是 Punica/SGMV 解决"多租户单 batch"的关键。

### T3 · 全 batch 无 LoRA 时，用 CPU tensor flag 让 torch op 内部早退（绕开 torch.compile）

- **代码**：`lora_kernel_metadata.py:20-29` 的长注释 + `:62` `no_lora_flag_cpu`；`:92-97` 计算并早退；kernel 侧 `lora_shrink.py:135-136` `if no_lora_flag_cpu.item(): return`。
- **精妙之处**：V1 用 `torch.compile` trace forward。trace 有三个坑：① Python 标量被固化成常量；② 不能处理动态控制流；③ **不会 trace 进 torch op 内部**。于是 vLLM 故意把"本步是否完全没有 LoRA"伪装成一个 **CPU bool tensor** 传进 `lora_shrink` / `lora_expand` 这两个 torch op，利用第 ③ 点——在 op 内部读这个 CPU flag 做 `if` 早退，既能在"batch 全无 LoRA"时跳过整个 kernel，又不会让动态分支破坏 compile 出来的图。这是一个非常 hacky 但有效的"在编译图里偷塞运行时控制流"的手法。

### T4 · optimize()：把 scaling 折进 lora_b，且必须幂等

- **代码**：`lora.py:40` `LoRALayerWeights.optimize()`；`models.py:179` / `:402` 多处调用。
- **精妙之处**：`y += (x·A·B)·s`，与其每次前向多一次逐元素乘 `·s`，不如把 `s` 一次性乘进 `lora_b`（`self.lora_b *= self.scaling; self.scaling = 1`）。关键是它**用 `if self.scaling == 1: return self` 做幂等守卫**——`optimize` 在加载、激活、打包多处被调，没有这个守卫就会被重复乘多次、权重错乱。打包层 `PackedLoRALayerWeights.optimize`（`lora.py:179`）逐 slice 做同样的事。

### T5 · stacked 权重预分配 + copy_ 激活：显存恒定，CUDA graph 友好

- **代码**：`layers.py:325-342` `create_lora_weights` 预分配 `[max_loras, 1, rank, in]`；`layers.py:387-392` `set_lora` 往 `[index]` 切片 `copy_(non_blocking=True)`。
- **精妙之处**：GPU 上的 LoRA 权重张量**地址固定、大小固定**（按 `max_loras × max_lora_rank` 上限），激活适配器只是往某个 slot 切片里异步 `copy_`，**从不 malloc/free**。这让 ① 显存占用可预测（启动即定）；② 与 CUDA graph 的"固定地址 replay"要求兼容（[模块 07](../07-cuda-graph-compile/design.md)）——graph 捕获时这些张量地址不变，换 LoRA 只是改内容不改地址。实际 rank 比 `max_lora_rank` 小时，多出来的部分留零（`set_lora` 只 copy `:lora_a.shape` 那一片），靠零权重让多余维度不产生贡献。

### T6 · _last_mapping 缓存：mapping 没变就不重算元数据

- **代码**：`models.py:689` `set_adapter_mapping` → `set_adapter_mapping(mapping, self._last_mapping, self._set_adapter_mapping)`（`adapter_commons.utils`）。
- **精妙之处**：decode 阶段连续多步，如果 batch 组成不变，`LoRAMapping` 也不变。这里用 `_last_mapping` 比对，**相同就跳过 `update_metadata`**（跳过 convert_mapping + sort + unique + cumsum 这一串 CPU/GPU 操作）。只有 batch 增删请求导致 mapping 变化时才重算。这是 decode 热路径上的一个省 CPU 招。

### T7 · 双层 LRU：registered（CPU）与 active（GPU）两套独立缓存

- **代码**：`models.py:719-722`：`_registered_adapters = LoRALRUCache(capacity=max_cpu_loras)`，`_active_adapters = LoRALRUCache(lora_slots=max_loras)`。
- **精妙之处**：两层缓存容量不同、驱逐独立。`add_adapter`（`models.py:728`）管 CPU 层；`activate_adapter`（`models.py:743`）管 GPU 层——GPU 满了 `_active_adapters.remove_oldest()` 只把它**移出 GPU 槽位**（`_deactivate_adapter` 把 `lora_index_to_id[index]=None`，`models.py:420`），LoRAModel 仍在 CPU 缓存。下次再用直接从 CPU 激活，免读盘。CPU 层满了才真正丢弃（下次要用重新 `_load_adapter` 读盘）。对应 S-LoRA 的分层缓存思路。

### T8 · 先 load 再驱逐：保证不会因新适配器损坏而丢了旧的

- **代码**：`worker_manager.py:229-244`，注释 `:231-234`。
- **精妙之处**：`add_adapter` 先 `_load_adapter`（读盘 + 校验）**成功后**，才检查容量并 `remove_oldest_adapter`。如果反过来（先驱逐再加载），万一新适配器路径错 / 校验失败，旧的已经被丢了、新的又没加上，缓存白白损失一个。代价是加载瞬间缓存可能短暂 +1 超过 `max_cpu_loras`（注释明示可接受）。

### T9 · 调度层把 max_loras 约束前移，激活才不会失败

- **代码**：`scheduler.py:262-268`（统计 RUNNING 已用 LoRA + 断言）、`:293-302`（WAITING 请求若会超 max_loras 则跳过）、`:356-357`（调度成功则 `scheduled_loras.add`）。
- **精妙之处**：GPU 只有 `max_loras` 个激活槽位。调度器在组 batch 时就维护 `scheduled_loras` 集合，**保证送进 worker 的 batch 内不同 LoRA 数 ≤ max_loras**，于是 worker 侧 `activate_adapter` 永远能找到空槽。`worker_manager.py:221` 的 `RuntimeError` 是兜底，正常路径不会触发。被跳过的请求 `appendleft` 回 waiting 头部（`:301`），下一步再试——用排队换"激活永不失败"。

### T10 · 打包层（qkv_proj / gate_up_proj）：一个层多个子 LoRA 合成

- **代码**：`lora.py:121` `PackedLoRALayerWeights`；`models.py:631` `_create_merged_loras_inplace`；`layers.py:691` `MergedColumnParallelLinearWithLoRA.set_lora`（逐 slice copy）。
- **精妙之处**：vLLM 把 `q/k/v` 融成一个 `qkv_proj`、`gate/up` 融成 `gate_up_proj`（基座层融合是吞吐优化）。但 LoRA 是分别训练 `q_proj`、`k_proj`、`v_proj` 的，所以加载时要把这几个子 LoRA **打包**进一个 `PackedLoRALayerWeights`（`pack()`，`lora.py:151`），`lora_a_stacked` / `lora_b_stacked` 变成 `n_slices` 个张量的 tuple（`layers.py:638-647`），`set_lora` 逐 slice 把 `lora_a[i]` / `lora_b[i]` copy 进对应 slice。`add_expand` 的 `output_slices` 参数就是告诉 kernel 每个 slice 在输出里的偏移。某子模块没有 LoRA 时该 slice 为 `None`（跳过）。

### T11 · TP 下 LoRA 权重的切分方向：A 还是 B 切，取决于 Column/Row

- **代码**：`ColumnParallelLinearWithLoRA.slice_lora_b`（`layers.py:520`，切 B）；`RowParallelLinearWithLoRA.slice_lora_a`（`layers.py:888`，切 A）。
- **精妙之处**：基座 Linear 的 TP 切分（Column/Row Parallel、all-reduce/all-gather）属于 [模块 03](../03-distributed-parallel/design.md)；LoRA 必须沿用基座层完全一致的切分方向。ColumnParallel 按**输出维**切（每个 rank 算一段输出），所以 LoRA 的 `B`（`[r,out]`）按 out 切、`A` 不切；RowParallel 按**输入维**切（每个 rank 吃一段输入），所以 `A`（`[in,r]`）按 in 切、`B` 不切（结果靠 all-reduce 求和）。`MergedColumnParallel` 的 `slice_lora_b` 还要处理 gate/up 各半的偏移（`:523-532`）。这保证 LoRA 增量的并行切分和基座层完全一致，all-reduce/all-gather 时维度对得上。

### T12 · 共享同一个 PunicaWrapper：元数据更新一次，所有层受益

- **代码**：`models.py:522` `new_module.set_mapping(self.punica_wrapper)`（所有层传同一引用）；`layers.py:120` `set_mapping` 存引用。
- **精妙之处**：模型有几十上百个被适配的层，但 token→lora 的 mapping 对所有层是**同一份**。所以全模型共享一个 `PunicaWrapperGPU` 实例，`update_metadata`（C8）每步只调一次，构造好的 `token_mapping_meta` 被所有层的 `add_lora_linear` 复用（`meta_args` 只是切片）。省掉"每层各算一遍 mapping"的巨大浪费。

### T13 · dummy LoRA warmup：用最坏情况预热 kernel / CUDA graph

- **代码**：`lora_model_runner_mixin.py:86` `maybe_dummy_run_with_lora`；`:99-101` `prompt_lora_mapping = arange(num_reqs) % num_loras + 1`（循环分配 LoRA id 模拟最坏情况）；`:118-120` `add_dummy_lora`。
- **精妙之处**：CUDA graph 捕获 / kernel autotune 需要在真实请求前用 dummy batch 跑一遍。LoRA 的 dummy 必须模拟**最坏情况**——`num_reqs` 个请求循环用满 `max_loras` 个不同 LoRA（`% num_loras + 1`），这样捕获的 graph 覆盖"batch 内 LoRA 最多"的形状。dummy LoRA 是零权重（`create_dummy_lora`，`models.py:528`），`dummy_lora_cache`（`worker_manager.py:57`）让同一个 dummy 复用、不重复造，退出时 `remove_all_adapters` 清干净。

### T14 · no_lora 短路 + sort stable：正确性与性能的细节

- **代码**：`punica_wrapper/utils.py:37` `compute_meta` 的 `no_lora`（SGMV 路径，batch 全 -1 时不 launch kernel）；`lora_kernel_metadata.py:106` `torch.sort(..., stable=True)`。
- **精妙之处**：① `no_lora` 判定让"这一步根本没有请求带 LoRA"时彻底跳过 kernel launch（连 grid 都不起）。② `sort` 必须 `stable=True`——否则同一 LoRA 内 token 的相对顺序可能被打乱，虽然 LoRA 计算本身对段内顺序无关，但 `token_indices_sorted_by_lora_ids` 要能正确把结果 scatter 回原位置，稳定排序保证下标映射一致。

### T15 · LoRARequest 按 lora_name 哈希：跨引擎 / 跨进程一致

- **代码**：`request.py:81-97` `__eq__` / `__hash__` 都基于 `lora_name`（而非 `lora_int_id`）。
- **精妙之处**：`lora_int_id` 是引擎内部分配的、可能因进程不同而不同；但 `lora_name` 是用户给的稳定标识。用 name 做哈希/相等，使得 `active_lora_requests` 这种 set（`make_lora_inputs:633`）在跨 worker / 多引擎场景下能正确去重和识别"同一个适配器"，即便它们的内部 int id 不一致。

---

## 4. 关键边界情况与容错

| 情况 | 代码位置 | 行为 |
|---|---|---|
| batch 内 LoRA 种类将超 max_loras | `scheduler.py:295-302` | 跳过该 WAITING 请求，`appendleft` 回 waiting 头，下一步再试 |
| worker 侧请求 LoRA 数 > 槽位（兜底） | `worker_manager.py:221` | 抛 `RuntimeError("Number of requested LoRAs > GPU LoRA slots")` |
| GPU 无空槽要激活新 LoRA | `models.py:747-749` | `LRUCache` `remove_oldest()` 驱逐最旧的 GPU LoRA，再激活 |
| CPU 缓存满要加载新 LoRA | `worker_manager.py:239-242` | 先 load 成功，`remove_oldest_adapter()` 驱逐最旧的 registered LoRA |
| 本步整个 batch 无 LoRA | `lora_kernel_metadata.py:92-97` / kernel `:135` | `no_lora_flag_cpu=True`，kernel 内早退，不跑 GEMM |
| 某 token 不用 LoRA（id=0） | `convert_mapping utils.py:103` / 中标 -1 / `lora_shrink.py:43` | slot 下标记为 -1，kernel 见 -1 早退，该 token 只保留基座输出 |
| rank > max_lora_rank | `peft_helper.py:107-110` | `validate_legal` 抛 ValueError，加载失败 |
| 适配器目标模块不在模型里 | `models.py:241-247` / `:268-274` | `from_local_checkpoint` 报"unexpected modules" |
| DoRA / modules_to_save | `peft_helper.py:48-52` | `_validate_features` 报"vLLM does not yet support" |
| 多模态模型 vision/connector 层 | `models.py:605-616` `_filter_unsupported_mm_module` | 过滤掉，只对语言模型加 LoRA，warning |
| V1 + long LoRA | `config.py:2649-2652` `verify_lora_support` | 抛 ValueError，要求用 V0 |
| pin 一个未注册的 LoRA | `models.py:767-772` | `_pin_lora_in_cpu_cache` 抛 ValueError |
| 适配器找不到（路径/HF） | `worker_manager.py:130-138` | `FileNotFoundError` → 包装成 ValueError "No adapter found" |

---

## 5. 一图速查：LoRA 调用链主干

```
[add_lora 链路] EngineCore.add_lora core.py:277 ─► executor.add_lora ─► Worker.add_lora gpu_worker.py:256
        └─► LoRAModelRunnerMixin.add_lora mixin.py:131 ─► LRUCacheWorkerLoRAManager.add_adapter wm.py:229
              ├─ _load_adapter wm.py:85 ─► PEFTHelper 校验 + LoRAModel.from_local_checkpoint（→ CPU 缓存）
              ├─ remove_oldest_adapter（CPU 缓存 LRU 驱逐）
              └─ activate_adapter models.py:381 ─► module.set_lora(index, A, B)  copy_ 进 GPU stacked slot
                                              （GPU 满则 _active_adapters.remove_oldest 驱逐）

[启动期层替换] gpu_model_runner.py:1280 load_lora_model ─► create_lora_manager
        └─► _create_lora_modules models.py:470 ─► from_layer 把 Linear 换成 *WithLoRA 层
              ├─ create_lora_weights layers.py:299  预分配 lora_a_stacked/lora_b_stacked [max_loras,1,rank,in/out]
              └─ set_mapping(punica_wrapper)   全模型共享一个 PunicaWrapperGPU

[每 batch] gpu_model_runner.py:613 set_active_loras
        └─► InputBatch.make_lora_inputs input_batch.py:615
              request_lora_mapping ──np.repeat(num_scheduled_tokens)──► token_lora_mapping
        └─► _set_active_loras mixin.py:59 ─► LoRAMapping(token, prompt, is_prefill=True)
              └─► set_active_adapters wm.py:165 ─► _apply_adapters（激活/驱逐，断言 ≤ lora_slots）
                                              └─► set_adapter_mapping ─►（_last_mapping 缓存）
                    └─► punica.update_metadata punica_gpu.py:57
                          ├─ convert_mapping utils.py:44   lora_int_id ──lora_index_to_id.index──► slot, 无LoRA=-1
                          └─ LoRAKernelMeta.prepare_tensors metadata.py:80
                                sort by lora id → active_lora_ids / num_tokens_per_lora / lora_token_start_loc

[forward] *WithLoRA.forward layers.py:554 ─► apply layers.py:401
        ├─ ① base_layer.quant_method.apply(x)              基座大 GEMM（全 batch 共享 W，一遍）
        └─ ② punica.add_lora_linear punica_gpu.py:181
              ├─ add_shrink ─► lora_shrink kernel  buffer=(x@A)*scale   [tokens,rank], fp32
              │     kernel program_id(axis=2)=lora_idx 并行各 LoRA；lora_id==-1 早退
              └─ add_expand ─► lora_expand kernel  y += buffer@B        原地加到基座 output
        ▼
   y = x·W + (x·A·B)·s    一次 forward 完成全 batch、全 LoRA
```
