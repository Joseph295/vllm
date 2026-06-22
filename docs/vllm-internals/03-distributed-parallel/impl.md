# 模块 03 · 分布式并行（TP/PP/EP/DP）+ 分布式执行 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的分布式实现：进程组怎么初始化、各并行层在哪里调通信、Executor 怎么下发。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
> [模块 00](../00-request-lifecycle/impl.md) 已讲过 `MultiprocExecutor` 的 MessageQueue 广播与 `rank0_reply_only`（00 的 F2、T1），本文**不重复 IPC 细节**，聚焦并行策略本身的权重切分与通信调用点。

---

## 1. 代码地图

```
进程组 / 拓扑 distributed/
  ├─ parallel_state.py            # GroupCoordinator、init_distributed_environment、
  │                               #   initialize_model_parallel（world 拓扑）、get_tp/pp/dp_group
  ├─ communication_op.py          # tensor_model_parallel_all_reduce / all_gather / gather（薄封装）
  ├─ utils.py                     # get_pp_indices（PP 层分配）、StatelessProcessGroup
  └─ device_communicators/
       ├─ base_device_communicator.py  # DeviceCommunicatorBase：all_reduce/all_gather/gather/send/recv
       ├─ cuda_communicator.py         # CudaCommunicator：custom AR → pynccl → torch 三级分发
       ├─ custom_all_reduce.py         # CustomAllreduce：小消息 IPC all-reduce
       ├─ pynccl.py                    # PyNcclCommunicator：ctypes 直绑 NCCL
       └─ shm_broadcast.py             # MessageQueue（[模块 00](../00-request-lifecycle/impl.md) 已讲）

TP 层 model_executor/layers/
  ├─ linear.py                    # ColumnParallelLinear / RowParallelLinear / QKVParallelLinear / MergedColumn
  ├─ vocab_parallel_embedding.py  # VocabParallelEmbedding（mask + all-reduce）
  └─ logits_processor.py          # LogitsProcessor（gather / all-gather logits）

PP model_executor/models/
  ├─ utils.py                     # make_layers / PPMissingLayer / make_empty_intermediate_tensors_factory
  └─ llama.py                     # 典型模型的 PP forward（first/mid/last rank 分支）
  + sequence.py                   # IntermediateTensors
  + v1/worker/gpu_model_runner.py # execute_model 里收/返 IntermediateTensors
  + v1/engine/core.py             # step_with_batch_queue（PP 流水线）

EP model_executor/layers/fused_moe/
  ├─ layer.py                     # FusedMoE：ep_size/ep_rank/expert_map、forward_impl 通信
  ├─ moe_align_block_size.py      # expert_map 应用（global→local 专家 id）
  └─ fused_moe.py                 # fused_experts 融合 kernel

DP v1/engine/core.py             # DPEngineCoreProc：run_busy_loop / _has_global_unfinished_reqs / dummy batch
  + config.py                     # ParallelConfig：world_size / world_size_across_dp / has_unfinished_dp

Executor v1/executor/
  ├─ abstract.py                  # Executor 抽象
  ├─ multiproc_executor.py        # MultiprocExecutor（TP；V1 当前 world_size==tp_size）
  └─ ray_distributed_executor.py  # RayDistributedExecutor（支持 PP）
```

---

## 2. 端到端调用链

### 阶段 A：进程组初始化（一切并行的起点）

> **这一步在干嘛**：开跑前把所有 GPU（rank）编排成一张 `ExternalDP×DP×PP×TP` 的四维网格，并按维度建好 TP/PP/DP 三个通信组——之后哪几张卡在同一个并行组里、该往谁发消息，全由这里定下。

**A1.** `vllm/distributed/parallel_state.py:815` `init_distributed_environment()`
拿 `torch.distributed.init_process_group` 建立全局 WORLD。DP 时先把 rank 偏移、world size 放大：
```python
# :828-837
if config is not None and config.parallel_config.data_parallel_size > 1:
    rank = parallel_config.data_parallel_rank * world_size + rank      # DP rank 偏移进全局 rank
    world_size = parallel_config.world_size_across_dp                  # = TP×PP×DP
    ...
    distributed_init_method = f"tcp://{ip}:{port}"                     # 每个 DP 引擎独立 init port
```
解读：DP 的多个 EngineCore 进程**共享同一个全局 world**，靠 rank 偏移区分。`:861-864` 建 `_WORLD`（`use_device_communicator=False`，世界组不直接通信）。

**A2.** `parallel_state.py:870` `initialize_model_parallel()` —— **world 拓扑核心**
```python
# :919-921  把所有 rank reshape 成四维网格 ExternalDP × DP × PP × TP
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size)
```
- `:926` **TP group** = `all_ranks.view(-1, tp_size).unbind(0)`——最内层连续 `tp_size` 个 rank 一组。
- `:930-934` 建 `_TP`，**唯一带 `use_message_queue_broadcaster=True`** 的组（这就是 00 讲的广播通道）。
- `:940-942` **PP group** = `transpose(2,3).reshape(-1, pp_size)`。`:943-946` 建 `_PP`（无 MQ）。
- `:950-952` **DP group** = `transpose(1,3).reshape(-1, dp_size)`。`:954-957` 建 `_DP`。
- `:959-962` 打印 "rank X is assigned as DP rank.., PP rank.., TP rank.."。

**A3.** `parallel_state.py:161` `GroupCoordinator.__init__()` —— 每个组建两个 backend
```python
# :179-190
for ranks in group_ranks:
    device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)  # NCCL
    cpu_group    = torch.distributed.new_group(ranks, backend="gloo")                     # CPU
    if self.rank in ranks:
        self.ranks = ranks; self.world_size = len(ranks)
        self.rank_in_group = ranks.index(self.rank); ...
```
- `:208-216` 若 `world_size>1` 建 `device_communicator`（CUDA 上是 `CudaCommunicator`）。
- `:221-223` 仅 TP 组建 `mq_broadcaster`（共享内存 MessageQueue）。

---

### 阶段 B：TP —— 各层的权重切分与 forward 通信

> **这一步在干嘛**：张量并行的落地——每一层的权重矩阵被沿行/列切到不同卡上，各卡只算自己那片。靠"列并行→行并行"配对，中间激活全程切分、零通信，只在每个 Transformer block 末尾 all-reduce 两次（attn 后 + mlp 后）把部分和拼回完整结果。

**B1. 列并行** `vllm/model_executor/layers/linear.py:358` `ColumnParallelLinear`
- `:373-375` 切**输出维**：
  ```python
  self.tp_size = get_tensor_model_parallel_world_size()
  self.output_size_per_partition = divide(output_size, self.tp_size)
  ```
- `:419-457` `weight_loader`：按 rank narrow 出自己那片输出列（`:445-449` `start_idx = tp_rank * shard_size`）。
- `:467-483` **forward**：核心是**默认不通信**：
  ```python
  output_parallel = self.quant_method.apply(self, input_, bias)   # :474 各 rank 算自己的输出列
  if self.gather_output:                                          # :475 仅显式要求时才 all-gather
      output = tensor_model_parallel_all_gather(output_parallel)  # :477
  else:
      output = output_parallel                                    # :479 直接返回切片，交给下游行并行
  ```

**B2. 行并行** `linear.py:1110` `RowParallelLinear`
- `:1152-1153` 切**输入维**：`input_size_per_partition = divide(input_size, tp_size)`。
- `:1242-1270` **forward**：核心是**末尾一次 all-reduce**：
  ```python
  # :1257  bias 只在 rank0 加，避免 all-reduce 后被加 tp_size 次
  bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
  output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)  # :1258 算部分和
  if self.reduce_results and self.tp_size > 1:
      output = tensor_model_parallel_all_reduce(output_parallel)               # :1262 求和
  ```
  解读：列并行(B1)输出已是切片，直接喂行并行 `input_is_parallel=True`（`:1245`），中间**零通信**；只有 `down_proj`/`o_proj` 末尾这一次 all-reduce。**每个 block 共 2 次**（attn 后 + mlp 后）。

**B3. QKV 切分** `linear.py:768` `QKVParallelLinear`
- `:816-823` GQA/MQA 处理：
  ```python
  self.num_heads = divide(self.total_num_heads, tp_size)            # query head 全切
  if tp_size >= self.total_num_kv_heads:
      self.num_kv_heads = 1
      self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)  # KV head 复制
  else:
      self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
  ```
  解读：query head 数多、直接均分；KV head 少（GQA），当 rank 数超过 KV head 数时**复制 KV** 让每 rank 都有一份。

**B4. 词表并行** `vocab_parallel_embedding.py:198` `VocabParallelEmbedding`
- `:252` `num_embeddings_per_partition = divide(num_embeddings_padded, tp_size)`。
- `:403-422` **forward**：mask 掉不属于本 rank 的 token，查表后 all-reduce：
  ```python
  if self.tp_size > 1:
      masked_input, input_mask = get_masked_input_and_mask(...)     # :405 本 rank 词表外的 token 置 0
  output_parallel = self.quant_method.embedding(self, masked_input)
  if self.tp_size > 1:
      output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)     # :419 越界 token embedding 清零
  output = tensor_model_parallel_all_reduce(output_parallel)        # :421 各 rank 的部分拼成完整
  ```

**B5. logits gather** `logits_processor.py:87` `_gather_logits`
```python
if self.use_all_gather:                                  # :89 TPU/严格 SPMD
    logits = tensor_model_parallel_all_gather(logits)
else:                                                    # :96 其他设备
    logits = tensor_model_parallel_gather(logits)        # 非 rank0 返回 None
```
解读：lm_head 也按 vocab 切，算完各 rank 的部分 logits 后 gather 回完整词表再采样；`:117` `logits[..., :self.org_vocab_size]` 去掉 vocab padding。

**B6. 通信原语薄封装** `communication_op.py:11-26`
```python
def tensor_model_parallel_all_reduce(input_):  return get_tp_group().all_reduce(input_)   # :11-13
def tensor_model_parallel_all_gather(input_, dim=-1): return get_tp_group().all_gather(input_, dim)  # :16-19
def tensor_model_parallel_gather(input_, dst=0, dim=-1): return get_tp_group().gather(...)  # :22-26
```
全部转发到 `get_tp_group()`，即只在 TP group 内通信。

---

### 阶段 C：all-reduce 的三级分发（性能关键）

> **这一步在干嘛**：上一步要做的那次 all-reduce，到底用哪种实现？这里按消息大小和卡间拓扑择优——小消息走 vLLM 自研的 custom all-reduce（最低延迟），否则 pynccl，再不行退回 torch.distributed。decode 阶段 all-reduce 既小又高频，选对实现对性能影响极大。

**C1.** `parallel_state.py:292` `GroupCoordinator.all_reduce()`
```python
if self.world_size == 1: return input_                    # :308 单卡直通
if self.use_custom_op_call:                               # :311 CUDA/TPU 走 torch custom op
    return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)  # :312
else:
    return self._all_reduce_out_place(input_)             # :315
```
解读：custom op（`:109-114` 的 `all_reduce(tensor, group_name)`）只传 `group_name` 字符串、内部按名查回 coordinator——为了让 `torch.compile` 能捕获 all-reduce（Dynamo 不能传 `self`）。

**C2.** `cuda_communicator.py:52` `CudaCommunicator.all_reduce()` —— **三级回退**
```python
ca_comm = self.ca_comm
if ca_comm is not None and not ca_comm.disabled and ca_comm.should_custom_ar(input_):  # :56-57
    out = ca_comm.custom_all_reduce(input_)               # :58 ① custom AR（小消息）
    return out
pynccl_comm = self.pynccl_comm
out = pynccl_comm.all_reduce(input_)                      # :63 ② pynccl
if out is None:                                           # :64
    out = input_.clone()
    torch.distributed.all_reduce(out, group=self.device_group)  # :70 ③ torch 兜底
```

**C3.** `custom_all_reduce.py:213` `should_custom_ar()` —— **何时用 custom AR**
```python
inp_size = inp.numel() * inp.element_size()
if inp_size % 16 != 0: return False                       # :218 需 16 字节对齐
if not is_weak_contiguous(inp): return False              # :220
if self.world_size == 2 or self.fully_connected:          # :224 仅 2卡 或 NVLink 全连接
    return inp_size < self.max_size                        # :225 且 < 8MB
return False                                              # :226 PCIe-only 4+卡：不用 custom AR
```
配合 `:50` `_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]` 与 `:56` `max_size=8192*1024`（8MB）。`__init__`（`:53-152`）层层判 `disabled`：缺库(`:70`)/跨节点(`:82`)/单卡(`:92`)/world 不支持(`:96`)/PCIe-only 多卡(`:135`)/无 P2P(`:145`)，全过才 `:152 self.disabled = False`。

**C4.** `pynccl.py:109` `PyNcclCommunicator.all_reduce()`
```python
if self.disabled: return None                             # :113-114 回退信号
out_tensor = torch.empty_like(in_tensor)
self.nccl.ncclAllReduce(buffer_type(in_tensor.data_ptr()), ...)  # :126 ctypes 直调 NCCL
```
解读：直绑 NCCL 而非走 `torch.distributed`，可控制 stream、可入 CUDA graph（见 [模块 07](../07-cuda-graph-compile/design.md)）。

---

### 阶段 D：PP —— 中间张量在段间流动

> **这一步在干嘛**：流水线并行的落地——模型按层切成若干段，每段放一组卡。一条请求的 hidden_states 像在流水线上流动：首段算 embedding+前若干层，把中间张量 send 给下一段，末段才算 norm+logits+采样。段间只传一次激活，通信量远小于 TP。

**D1.** `sequence.py:1130` `IntermediateTensors` —— 段间载荷
```python
class IntermediateTensors:
    tensors: dict[str, torch.Tensor]   # 通常 {"hidden_states", "residual"}
```

**D2.** `model_executor/models/utils.py` 的层分配
- `make_layers`（按 `get_pp_indices` 把 `[start_layer, end_layer)` 给本 rank，其余填 `PPMissingLayer()` 占位）。
- `PPMissingLayer`（`nn.Identity` 子类，forward 直接透传输入）——让所有 rank 的 `ModuleList` 长度一致、索引对齐。
- `make_empty_intermediate_tensors_factory(["hidden_states","residual"], hidden_size)` 造空中间张量。
- `get_pp_indices`（`distributed/utils.py`）：层均分，不整除时余数给**非最后段**（最后段还要算 norm，留点 compute 余量）。

**D3.** 模型 forward 的三分支 `model_executor/models/llama.py`
```python
def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
    if get_pp_group().is_first_rank:                       # 首段：算 embedding
        hidden_states = self.get_input_embeddings(input_ids); residual = None
    else:                                                  # 中/末段：从上游取
        hidden_states = intermediate_tensors["hidden_states"]
        residual      = intermediate_tensors["residual"]
    for layer in self.layers[self.start_layer:self.end_layer]:   # 只算本段的层
        hidden_states, residual = layer(positions, hidden_states, residual)
    if not get_pp_group().is_last_rank:                    # 非末段：打包返回中间张量
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
    hidden_states, _ = self.norm(hidden_states, residual)  # 末段：norm → 交给 compute_logits
    return hidden_states
```
（`embed_tokens`/`norm` 在非对应 rank 上也被替换成 `PPMissingLayer`，模型构造时按 `is_first_rank`/`is_last_rank` 决定。）

**D4.** `v1/worker/gpu_model_runner.py:987` `execute_model()` 的 PP 收/返
- 非首段：把上游传来的 `intermediate_tensors` 拷进本地持久 buffer（`gpu_model_runner.py:1047-1058` 一带，按 `is_first_rank` 分支）。
- `gpu_model_runner.py:1069-1071` **非末段直接返回 hidden_states**（不采样），交由 PP 通信送下一段（[模块 00](../00-request-lifecycle/impl.md) 的边界表已记此点）。

**D5.** PP 流水线在 EngineCore 进程层 `v1/engine/core.py:212` `step_with_batch_queue()`（详见 [模块 00](../00-request-lifecycle/impl.md) 的 T10）
```python
# :115  batch_queue_size = self.model_executor.max_concurrent_batches
if (self.scheduler.get_num_unscheduled_requests() > 0 and not self.batch_queue.full()):
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output)   # 提交但不等
    self.batch_queue.put_nowait((future, scheduler_output))
if not scheduled_batch and not self.batch_queue.empty():
    future, scheduler_output = self.batch_queue.get_nowait()
    model_output = future.result()                                 # 队列满/无新批 才阻塞取
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```
`ray_distributed_executor.py` 的 `max_concurrent_batches = pipeline_parallel_size`，PP 时 `execute_model` 返回 `FutureWrapper` 让调度器去排下一批——这就是填气泡的机制。

---

### 阶段 E：EP —— MoE 专家分散

> **这一步在干嘛**：专家并行的落地——当 MoE 专家多到单卡放不下时，把整个专家分到不同卡（每卡只持有一部分专家的完整权重）。router 决定每个 token 去哪些专家后，把 token dispatch 到持有对应专家的卡上算、再 combine 回来；`expert_map` 负责把"全局专家 id → 本地专家槽位"，不在本卡的专家标记为 -1 跳过。

**E1.** `fused_moe/layer.py:437` `FusedMoE.__init__` 决定 TP 还是 EP
```python
use_ep = (vllm_config.parallel_config.enable_expert_parallel
          and self.tp_size * self.dp_size > 1)              # :439-440
if use_ep:
    self.ep_rank = tp_rank + self.tp_size * self.dp_rank    # :454
    self.tp_size = 1; self.ep_size = self.tp_size * self.dp_size  # :456-457
    self.local_num_experts, self.expert_map = determine_expert_map(  # :459 算本 rank 持哪些专家
        ep_size=self.ep_size, ep_rank=self.ep_rank, global_num_experts=self.global_num_experts)
else:                                                       # TP-on-MoE：切每个专家的矩阵
    self.tp_size = self.tp_size * self.dp_size; self.ep_size = 1; self.expert_map = None  # :467-470
```
解读：**EP 把 tp_size 归 1、ep_size 设为 tp×dp**——即同样的 GPU，从"切专家矩阵"切换到"分专家"。

**E2.** `determine_expert_map()`（`layer.py:332` 一带）造 global→local 映射
```python
local_num_experts = global_num_experts // ep_size
expert_map = torch.full((global_num_experts,), -1, ...)    # 默认 -1 = 不在本 rank
expert_map[ep_rank*local : (ep_rank+1)*local] = arange(0, local_num_experts)  # 本 rank 的专家映到 [0..local)
```
余数给最后一个 rank。`-1` 在 kernel 里表示"跳过这个专家"。

**E3.** `fused_moe/layer.py:847` `forward_impl()` 的通信
```python
if self.dp_size > 1:
    hidden_states  = self.naive_multicast(hidden_states, cu_tokens_across_dp_cpu)   # :855 跨DP凑齐token
    router_logits  = self.naive_multicast(router_logits, cu_tokens_across_dp_cpu)   # :857
final_hidden_states = self.quant_method.apply(..., expert_map=self.expert_map, ...) # :861-877 只算本地专家
if self.dp_size > 1:
    all_hidden_states = get_dp_group().all_reduce(final_hidden_states)              # :884 combine
    final_hidden_states = all_hidden_states[start:end, :]                          # :885 取回本rank的slice
if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)     # :889 EP/TP 求和
```

**E4.** `fused_moe/moe_align_block_size.py` 应用 expert_map
```python
if expert_map is not None:
    expert_ids = expert_map[expert_ids]    # global expert id → local id（-1 跳过）
```
解读：路由产出的是全局专家 id，这一步翻译成本 rank 的本地专家槽位，不在本 rank 的 token 被 -1 标记跳过。

---

### 阶段 F：DP —— 多引擎协调

> **这一步在干嘛**：数据并行的落地——把整个引擎复制多份，各自独立调度、独立 forward，吞不同的请求流。绝大多数时候互不通信，但有两个强同步点：① 每 16 步 all-reduce 一次判断"是否全局都没活了"才能一起暂停；② 某 rank 本地没活但别人还在跑时，它必须跑一个空 forward（dummy batch）参与隐含的集合通信，否则别的 rank 会卡死。

**F1.** `v1/engine/core.py:558` `DPEngineCoreProc`（继承 `EngineCoreProc`）
- `:594` `self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()`——建独立的 DP 通信组（无状态，进程级）。
- `:602` `self.counter = 0`——finish 同步的步数计数器。

**F2.** `core.py:609` `DPEngineCoreProc.run_busy_loop()` —— DP 专用主循环
```python
while True:
    self._process_input_queue()                                 # :615
    local_unfinished_reqs = self.scheduler.has_unfinished_requests()  # :617
    if local_unfinished_reqs:
        self._process_engine_step()                             # :621 有本地活：正常 step
        local_unfinished_reqs = self.scheduler.has_unfinished_requests()
    else:
        if self.scheduler.has_finished_requests():
            self._process_engine_step()                         # :633 仅 flush finished，不 forward
        if not self.global_unfinished_reqs:
            continue                                            # :637 全局都空：歇着
        self.execute_dummy_batch()                              # :641 别人还在跑→我必须空跑，否则死锁
    self.global_unfinished_reqs = self._has_global_unfinished_reqs(local_unfinished_reqs)  # :644
    if not self.global_unfinished_reqs:
        self.output_queue.put_nowait(ENGINE_PAUSED_OUTPUTS)     # :649 通知前端暂停
```

**F3.** `core.py:651` `_has_global_unfinished_reqs()` —— 每 16 步才 all-reduce
```python
self.counter += 1
if self.counter != 16: return True       # :655-656 前15步乐观假设"还有活"
self.counter = 0
return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)  # :659
```

**F4.** `config.py:1635` `ParallelConfig.has_unfinished_dp()` —— 用 MAX 做逻辑 OR
```python
tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)  # :1645 任一rank有活→全局有活
return bool(tensor.item())
```

**F5.** `core.py:274-275` `execute_dummy_batch()` → `collective_rpc("execute_dummy_batch")` → worker `_dummy_run(1)`（`gpu_model_runner.py:1390` 一带的 `_dummy_run`）。
解读：空 forward 让该 rank 仍参与隐含的集合通信（如 MoE 的 all-reduce），不掉队。

**F6.** world size 派生 `config.py`
```python
self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size  # :1663-1664 一个模型实例的GPU数
self.world_size_across_dp = self.world_size * self.data_parallel_size      # :1678 含DP总GPU数
```

---

### 阶段 G：Executor 下发

> **这一步在干嘛**：上层 `step()` 只调一次 `execute_model`，由 Executor 把这步执行**下发**给底下的 Worker——单卡直接本进程调用，多进程 TP 用共享内存广播给所有 worker，PP 走 Ray compiled DAG 流过各 stage。这层就是让"单卡/多卡/多机"对调度完全透明的接缝。

**G1.** `v1/executor/multiproc_executor.py:44` `MultiprocExecutor`
- `:54-59` **V1 现状断言**：
  ```python
  self.world_size = self.parallel_config.world_size
  assert self.world_size == self.parallel_config.tensor_parallel_size, (
      "... Pipeline parallelism is not yet implemented in v1")   # MultiprocExecutor 只做 TP
  ```
- `:142-150` `execute_model` → `collective_rpc("execute_model", rank0_reply_only=True)`。
- `:152-191` `collective_rpc`：`:173` `rpc_broadcast_mq.enqueue(...)` 广播；`:176` `rank0_reply_only` 时只等 worker0（[模块 00](../00-request-lifecycle/impl.md) 的 F2 已详述）。

**G2.** `v1/executor/ray_distributed_executor.py` `RayDistributedExecutor`（支持 PP）
- `max_concurrent_batches = pipeline_parallel_size`。
- `execute_model` 用 Ray compiled DAG 把 scheduler_output 流过各 PP stage；`max_concurrent_batches > 1`（即 PP）时返回 `FutureWrapper(refs[0])` 让调度器去排下一批，否则 `refs[0].get()` 阻塞。

---

## 3. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。下面每条都是读代码时容易滑过、理解了才算真懂分布式部分的点。

### T1 · 列并行 + 行并行配对：整个 MLP/Attn 只 all-reduce 一次
- **代码**：列并行 forward `linear.py:475-479`（默认 `gather_output=False`，**不通信**）；行并行 forward `linear.py:1261-1262`（**末尾一次 all-reduce**）。
- **精妙之处**：`up_proj`（列并行）输出天然按 rank 切片，直接作为 `down_proj`（行并行，`input_is_parallel=True`）的输入，**中间激活全程切分、零通信**。一个 Transformer block 因此恰好 2 次 all-reduce（attn 后 + mlp 后），而非每个 matmul 都通信。这是 Megatron 的核心洞见，也是 TP 能用的根本原因。

### T2 · 行并行的 bias 只在 rank0 加
- **代码**：`linear.py:1257` `bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias`。
- **精妙之处**：行并行末尾要 all-reduce 求和。若每个 rank 都把完整 bias 加进部分和，all-reduce 后 bias 会被加 `tp_size` 次。所以**只 rank0 加 bias**，其余 rank 不加，求和后正好一份。一个一行的边界，错了结果就偏。

### T3 · GQA 下 KV head 不够分就复制
- **代码**：`linear.py:817-823` 的 `num_kv_head_replicas`。
- **精妙之处**：query head 多、好均分；但 GQA 模型 KV head 很少（如 8 个），TP=16 时不够每 rank 一份。于是当 `tp_size >= num_kv_heads` 时**复制 KV head**（`num_kv_head_replicas = tp_size // num_kv_heads`），保证每 rank 都有完整 KV 可算 attention，代价是 KV 权重冗余存储。weight_loader 据此把同一份 KV 权重加载到多个 rank（`:1077-1080` 的 `shard_id = tp_rank // num_kv_head_replicas`）。

### T4 · 词表并行：越界 token 先 mask 再 all-reduce
- **代码**：`vocab_parallel_embedding.py:405`（mask）→ `:419`（清零）→ `:421`（all-reduce）。
- **精妙之处**：词表按 rank 切后，一个 token 只属于某一个 rank。各 rank 把**不属于自己的 token id 映射到本地范围内**（避免越界），查表后**把这些位置的 embedding 强制清零**，再 all-reduce——求和时只有"真正拥有该 token"的 rank 贡献非零值，恰好拼出完整 embedding。用一次 all-reduce + mask 替代了 all-gather，省带宽。

### T5 · all-reduce 包成 torch custom op，只传 group_name 字符串
- **代码**：`parallel_state.py:311-313` `torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)`；查表 `:109-114`。
- **精妙之处**：`torch.compile`/Dynamo **不能把 `self`（GroupCoordinator 对象）作为参数传进编译图**。于是只传可序列化的 `group_name` 字符串，在 op 实现里用 `_groups[group_name]()`（weakref）查回 coordinator。这样 all-reduce 才能被 compile 捕获、与计算图融合（[模块 07](../07-cuda-graph-compile/design.md)）。`all_reduce_fake`（`:117`）提供 fake tensor 让 Dynamo 能 trace 形状。

### T6 · custom all-reduce 凭什么比 NCCL 快（且只在小消息用）
- **代码**：`custom_all_reduce.py:213-226` `should_custom_ar`；`:241-245` 的 `ops.all_reduce`。
- **精妙之处**：custom AR 用 **IPC 把各 rank 的 buffer 互相映射进对方地址空间**，一个 kernel 直接读写 peer 显存完成归约，省掉 NCCL ring/tree 协议的握手与多段传输——这在**小张量、超高频**的 decode all-reduce 上把延迟压到最低。但它只在 `<8MB`、world∈{2,4,6,8}、单节点、NVLink 全连接时启用（`:224-226` + `__init__` 的层层 disable）；超出就退回 NCCL，因为大消息 NCCL 的带宽优势反超。`registered=True`（CUDA graph 内）走预注册 buffer 免拷贝，`registered=False`（eager）多一次 cudaMemcpy 但 `<=1% 延迟`（`:261-263` 注释）。这套 **GPU IPC handle 交换、NVLink P2P 可达性探测、one-shot 与 two-shot 归约 kernel 的 csrc/ 实现**见 [模块 11](../11-gpu-kernels-memory/design.md)。

### T7 · NCCL barrier "有毒"，同步走 CPU group
- **代码**：`parallel_state.py:673-680` `barrier()` 用 `cpu_group`；注释明示原因。
- **精妙之处**：NCCL 的 barrier 内部是个 broadcast，会**偷偷创建 GPU tensor**，容易搞乱"当前 device"导致难查的 bug。所以 vLLM 所有 barrier、object 广播、metadata 传输都走 gloo 的 `cpu_group`，只有真正的 tensor 通信才用 NCCL 的 `device_group`。每个 GroupCoordinator 同时持两个 group 正是为此（`:180-184`）。

### T8 · PP 层分配：余数给前面的段，不给最后一段
- **代码**：`distributed/utils.py` 的 `get_pp_indices`（"remaining layers are evenly distributed across all but the last partition"）。
- **精妙之处**：层数不整除段数时，把多出来的层分给**非最后段**。因为最后一段除了 transformer 层还要算 `norm + lm_head + 采样`，已经更重；把余数层避开它，让各段 compute 更均衡、气泡更小。还支持 `VLLM_PP_LAYER_PARTITION` 环境变量手动指定不均分（应对异构层）。

### T9 · PPMissingLayer 占位，保持所有 rank 的 ModuleList 索引对齐
- **代码**：`make_layers` 用 `[PPMissingLayer()]*start + 真实层 + [PPMissingLayer()]*tail`；`PPMissingLayer` 是 `nn.Identity` 子类，forward 透传。
- **精妙之处**：每个 PP rank 只持有 `[start_layer, end_layer)` 的真实层，但若直接建短 ModuleList，各 rank 的层索引会错位、权重名对不上。用占位层把 `ModuleList` 撑到全长，**层名/索引在所有 rank 间一致**，权重加载、checkpoint 命名都能复用单卡逻辑。`embed_tokens`/`norm` 同理，非对应 rank 换成 `PPMissingLayer`。

### T10 · EP 是"把 tp_size 归 1、专家分散"，与 TP-on-MoE 互斥
- **代码**：`fused_moe/layer.py:451-470`。
- **精妙之处**：同样一批 GPU，`enable_expert_parallel` 决定它们是**切每个专家的矩阵**（TP-on-MoE：`tp_size *= dp`，`ep_size=1`，`expert_map=None`）还是**分不同专家**（EP：`tp_size=1`，`ep_size = tp×dp`，建 `expert_map`）。一个布尔开关切换两种完全不同的并行语义，上层模型代码无感。`expert_map` 里 `-1` 标记"不在本 rank 的专家"，融合 kernel（`moe_align_block_size`）据此跳过。

### T11 · DP 空闲 rank 必须跑 dummy batch，否则集合通信死锁
- **代码**：`core.py:635-641`（`execute_dummy_batch`）。
- **精妙之处**：DP 各引擎本应独立，但 **MoE+DP 时 forward 内部有跨 DP 的 all-reduce/multicast**（`layer.py:855/884`）。若某 rank 本地没请求就直接歇着，它不进入那次集合通信，**其他还在跑的 rank 会在 all-reduce 处永久阻塞**。所以只要"全局还有活"（`global_unfinished_reqs`），本地没活的 rank 也要跑一个空 forward 参与通信。这是 DP 最隐蔽的正确性陷阱。

### T12 · finish 同步摊销：每 16 步才真 all-reduce，用 MAX 当逻辑 OR
- **代码**：`core.py:651-660` + `config.py:1635-1647`。
- **精妙之处**：判断"是否所有 DP rank 都空闲"需要集合通信，每步做太贵。于是用 `counter` **每 16 步才 all-reduce 一次**（`:655`），其余步乐观返回"还有活"。聚合用 `ReduceOp.MAX`——整数版的逻辑 OR：任一 rank `has_unfinished=1` → 全局 `=1`（`config.py:1641-1645` 注释画了真值表）。代价是最多多空跑 15 步才发现全局停机，换来集合通信开销降到 1/16。

### T13 · 拓扑用"转置 + reshape + unbind"一行取出每个维度的分组
- **代码**：`parallel_state.py:919-957`。
- **精妙之处**：把 `world_size` 个 rank reshape 成 `[ExternalDP, DP, PP, TP]` 四维张量后，**要取哪个维度的组，就把那一维转置到最后再 reshape+unbind**——TP 直接 `view(-1, tp_size)`，PP 是 `transpose(2,3)`，DP 是 `transpose(1,3)`。把"哪些 rank 在同一个并行组"这件复杂的事压成几行张量操作，且天然保证 TP 组是最内层连续 rank（应放同一 NVLink 域）。

### T14 · TP group 独享 MessageQueue 广播，PP/DP 不挂
- **代码**：`parallel_state.py:930-934`（仅 TP `use_message_queue_broadcaster=True`）；`:221-223` 建 mq。
- **精妙之处**：[模块 00](../00-request-lifecycle/design.md) 讲的"共享内存广播 scheduler_output"只发生在 **TP group 内**（同节点、需要把同一份输入喂给所有 TP worker）。PP 段间是点对点 send/recv（传不同的中间张量，不是广播）、DP 各引擎独立，都不需要这个广播器。所以只有 TP coordinator 挂 `mq_broadcaster`，省内存也省 mq 创建开销。

### T15 · 模型实例数 vs 总 GPU 数：world_size 与 world_size_across_dp 分离
- **代码**：`config.py:1663-1664`（`world_size = pp*tp`）与 `:1678`（`world_size_across_dp = world_size*dp`）。
- **精妙之处**：`MultiprocExecutor` 按 `world_size`（= 一个模型实例的 GPU 数）创建 Worker 进程；而全局分布式 world 用 `world_size_across_dp`（含 DP 复制）。两者刻意分开：Executor 只管"一个实例"的 worker，DP 复制由多个 EngineCore 进程在更上层实现（`init_distributed_environment` 的 rank 偏移）。这让 Executor 完全不感知 DP 的存在。

---

## 4. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| 单卡（world_size==1） | `parallel_state.py:308`、`linear.py` 各 forward | all-reduce/all-gather 直接返回输入，不通信 |
| V1 多进程想用 PP | `multiproc_executor.py:56-59` | 断言失败 "PP not yet implemented in v1"——V1 的 PP 必须走 Ray Executor |
| custom AR 消息 ≥ 8MB | `custom_all_reduce.py:225` | `should_custom_ar` 返回 False → 回退 pynccl |
| custom AR world∉{2,4,6,8} | `custom_all_reduce.py:96-102` | `disabled=True`，全程用 pynccl |
| PCIe-only 且 >2 卡 | `custom_all_reduce.py:135-140` | custom AR 禁用（NCCL 在此场景已够好） |
| pynccl `disabled`（如缺库） | `pynccl.py:113-114` | `all_reduce` 返回 None → CudaCommunicator 退回 torch.distributed |
| PP 非末段 rank | `gpu_model_runner.py:1069-1071` | 返回 hidden_states 而非采样结果 |
| GQA：tp_size ≥ num_kv_heads | `linear.py:817-820` | KV head 复制（`num_kv_head_replicas`） |
| EP：global_num_experts 不整除 ep_size | `determine_expert_map`（`layer.py:332` 一带） | 余数专家全给最后一个 ep rank |
| EP：token 路由到不在本 rank 的专家 | `moe_align_block_size.py` | expert_map 映射为 -1，kernel 跳过 |
| DP rank 本地无请求但全局有活 | `core.py:639-641` | 跑 `execute_dummy_batch` 参与隐含集合通信，防死锁 |
| DP 全局空闲 | `core.py:647-649` | 发 `ENGINE_PAUSED_OUTPUTS` 通知前端暂停 |
| NCCL barrier 需求 | `parallel_state.py:673-680` | 改用 `cpu_group`（gloo）做 barrier，避免 NCCL 偷建 GPU tensor |

---

## 5. 一图速查：并行调用链主干

```
[初始化] init_distributed_environment :815  (DP rank 偏移 / world_size_across_dp)
         └─ initialize_model_parallel :870  reshape 成 ExternalDP×DP×PP×TP
              ├─ _TP = init_model_parallel_group(..., use_message_queue_broadcaster=True) :930
              ├─ _PP = ... :943      └─ _DP = ... :954
              └─ GroupCoordinator.__init__ :161  每组建 device_group(NCCL)+cpu_group(gloo)

[每步 step] EngineCore.step / step_with_batch_queue(PP) :195/:212
   └─ executor.execute_model
        ├─ MultiprocExecutor :142  collective_rpc(rank0_reply_only) ─► rpc_broadcast_mq 广播（仅TP）
        └─ RayDistributedExecutor   compiled DAG 流过 PP stage（PP 时返回 Future 排下一批）
             │
             ▼  Worker.execute_model ─► GPUModelRunner.execute_model :987
                   model.forward:
                     [PP] first_rank 算 embedding → mid 透传 → last 算 norm；非末段 return IntermediateTensors :1069
                     [TP] 每 block:
                          ColumnParallelLinear.forward :467   (qkv/up, 无通信)
                          → attention →
                          RowParallelLinear.forward :1242 ─► tensor_model_parallel_all_reduce :1262
                               └─ GroupCoordinator.all_reduce :292 ─► torch.ops.vllm.all_reduce :312
                                    └─ CudaCommunicator.all_reduce :52
                                         ① custom AR (should_custom_ar<8MB,NVLink) :56
                                         ② pynccl.all_reduce :63    ③ torch :70
                     [EP] FusedMoE.forward_impl :847
                          naive_multicast(跨DP) :855 → apply(expert_map) :861 → dp all_reduce :884 → tp all_reduce :889
                     [vocab] VocabParallelEmbedding.forward :403 (mask→all_reduce :421)
                             LogitsProcessor._gather_logits :87 (gather/all_gather)

[DP 协调] DPEngineCoreProc.run_busy_loop :609
   ├─ 有本地活 → _process_engine_step :621
   ├─ 无本地活但全局有活 → execute_dummy_batch :641 (防死锁)
   └─ _has_global_unfinished_reqs :651 (每16步 all_reduce MAX) → 全空闲发 ENGINE_PAUSED :649
```
