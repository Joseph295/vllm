# 模块 03 · 分布式并行（TP/PP/EP/DP）+ 分布式执行 —— 设计文档

> 范围：当一个模型**单卡放不下**或**单卡算不快**时，vLLM 如何把权重、计算、请求切分到多卡多机上，以及上层 `step()` 如何通过 Executor 抽象对这一切**完全无感**。
> 架构：**V1 为主**。[模块 00](../00-request-lifecycle/design.md) 已讲过 `MultiprocExecutor` 用共享内存 `MessageQueue` 广播 `scheduler_output` + `rank0_reply_only` 的进程模型（见 [模块 00](../00-request-lifecycle/impl.md) 的 §3.5、T1、F2）；本模块**不重复 IPC 细节**，而是深入**四种并行策略本身**：权重怎么切、通信原语在哪里调、world 拓扑怎么排。
> 阅读前提：先读 [模块 00](../00-request-lifecycle/design.md) 建立"前端 / EngineCore 进程 / Worker 三层 busy loop"的总骨架。

---

## 1. 背景与定位

> 如果你还不清楚 **TP/PP/EP/DP 各切什么、Executor 与 Worker 的分工**，可先扫一眼 [PRIMER §2.4](../PRIMER.md#24-易混概念对照区别与联系专治听过但分不清) 的并行维度对照表，这里默认你已经有了那套心智模型。

### 1.1 这个模块在系统里的位置

本模块解决的是"**一张卡放不下/算不快时怎么办**"——把模型切到多卡多机，并让上层调度毫无感知：

- **上游**：调度器（[模块 01](../01-scheduler-batching/design.md)）每步产出一个 `SchedulerOutput`，只管"算什么"，不管"在几张卡上算"。
- **本模块**：通过 **Executor** 把这一步执行**下发**到一个或多个 **Worker**；同时实现四种并行策略——把权重（TP）、层（PP）、MoE 专家（EP）切开，或把整个引擎复制多份（DP），并在各处插入所需的集合通信（all-reduce / send-recv / all-to-all）。
- **下游**：每个 Worker 在自己那张卡上跑模型分片的 forward（[模块 00](../00-request-lifecycle/impl.md) 的 GPUModelRunner），结果汇总回后端。

一句话：**它是把"一个 `step()`"透明地扩展到单卡 / 多卡 / 多机的"接缝层"——换并行策略 = 换 Executor，调度与模型代码不动。**

### 1.2 关键文件与类各干什么（先认人）

| 文件 / 类 | 主要作用（一句话） |
|---|---|
| `distributed/parallel_state.py` 的 `GroupCoordinator` | 一个进程组的**通信门面**：持 NCCL group + gloo group + 通信器，封装 all-reduce/all-gather/send/recv |
| `distributed/parallel_state.py` 的 `initialize_model_parallel` | 把 world 个 rank reshape 成 `ExternalDP×DP×PP×TP` 四维网格，建 `_TP/_PP/_DP` 组 |
| `model_executor/layers/linear.py` 的 `ColumnParallelLinear`/`RowParallelLinear`/`QKVParallelLinear` | TP 的**列并行/行并行/QKV 切分**线性层（权重沿不同维切、在必要处通信） |
| `model_executor/layers/fused_moe/layer.py` 的 `FusedMoE` | MoE 层，含 `ep_size`/`expert_map`，在 **TP-on-MoE 与 EP** 间切换 |
| `sequence.py` 的 `IntermediateTensors` | PP 段间传递的 `{hidden_states, residual}` 载荷 |
| `v1/executor/multiproc_executor.py` 的 `MultiprocExecutor` | 多进程 TP 执行后端（V1 当前 `world_size==tp_size`） |
| `v1/executor/ray_distributed_executor.py` 的 `RayDistributedExecutor` | 支持 PP 的执行后端（用 Ray compiled DAG 流过各 stage） |
| `v1/engine/core.py` 的 `DPEngineCoreProc` | DP 专用的 EngineCore 进程：finish 同步、空闲时跑 dummy batch |

### 1.3 和相似模块 / 概念的区别与联系（专治"听过但分不清"）

- **TP vs PP vs EP vs DP**：四个**正交**维度，可任意叠加。TP=切每一层的矩阵（高频 all-reduce、最吃带宽）；PP=按层切成段（段间只传一次中间张量、对跨节点带宽不敏感）；EP=把 MoE 的整个专家分到不同卡；DP=把整个引擎复制多份各喂不同请求。**前三者切的是"一个模型实例"，DP 复制的是"整个实例"**（见 §3、[PRIMER §2.4](../PRIMER.md#24-易混概念对照区别与联系专治听过但分不清)）。
- **Executor vs Worker**：**下发者 vs 干活者**。`Executor`（本进程或 Ray）把"执行一步"广播/路由给若干 `Worker`；每个 `Worker` 在一张卡上跑模型分片的 forward。换单卡/多卡/多机 = 换 Executor 实现，Worker 侧逻辑基本不变。
- **调度 vs 执行**：调度（[模块 01](../01-scheduler-batching/design.md)）只**动脑**产出 `SchedulerOutput`，不碰 GPU、不知道有几张卡；本模块的 Executor 才**动手**把它下发执行。正因调度产物是纯数据、与并行无关，才能"换并行不改调度"。
- **EP vs TP-on-MoE**：同一批 GPU 上 MoE 的两种切法。TP-on-MoE=**切每个专家的矩阵**（所有 rank 都参与每个专家）；EP=**把不同专家放到不同 rank**（每 rank 只算自己的专家）。靠 `enable_expert_parallel` 一个开关切换（§3.3）。

🔗 **回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)**：那个 `[11,7,4]` 的例子是**单卡**跑的，不涉及任何并行通信。但可借它想象 TP=2 时会发生什么：例子里有 2 个注意力头，TP=2 下 QKVParallelLinear 会把它们按 head 切到两张卡，每卡只算 1 个头、只存这一半的 KV；attention 算完后经一次 all-reduce 把两卡的部分输出加回完整结果。其余 prefill/decode/采样的数据流与单卡完全一致。

---

### 1.4 这个模块要解决的问题

LLM 推理在"单个 GPU"上会撞到两堵墙：

1. **显存墙（放不下）**：一个 70B 模型 fp16 权重就要 ~140GB，加上 KV cache、激活，远超单卡 80GB。必须把**权重本身**切到多张卡上。
2. **算力/带宽墙（算不快）**：即使放得下，单卡的 FLOPs 和显存带宽也限制了单请求的 decode 速度（TPOT）与可容纳的并发；要用更多卡**并行算**同一层、或**流水**不同层、或**复制**整个引擎吞更多请求。

针对这两堵墙，业界沉淀出**四种正交的并行维度**，vLLM 全部支持，且可任意组合：

| 维度 | 切什么 | 主要解决 | 通信代价 |
|---|---|---|---|
| **TP**（Tensor Parallel，张量并行） | **每一层的权重矩阵**沿行/列切 | 显存 + 单层算力 | 每层 1~2 次 all-reduce（**高频、在关键路径**） |
| **PP**（Pipeline Parallel，流水线并行） | **按层**切，每段 GPU 放连续若干层 | 显存（跨段摊薄） | 段间 1 次 send/recv（**低频、点对点**） |
| **EP**（Expert Parallel，专家并行） | MoE 的**整个专家**分到不同卡（MoE 机制本身见 [模块 12](../12-moe/design.md)） | MoE 显存（专家极多时） | 每个 MoE 层 1 次 all-to-all / all-reduce |
| **DP**（Data Parallel，数据并行） | **整个引擎复制多份**，各喂不同请求 | 吞吐（放得下但要更多并发） | 仅 finish 同步 + MoE 时的 token 重分布 |

核心思想是：**TP/PP/EP 切的是"一个模型实例"，DP 复制的是"整个模型实例"**。前三者降低单实例的显存/延迟，DP 提高实例数从而提高吞吐。world 拓扑 `ExternalDP × DP × PP × TP` 把它们排成一个四维网格（见 §4.4）。

---

## 2. 设计目标与约束

在把模型切到多卡多机时，vLLM 的分布式层围绕以下几条目标与约束展开（均为本文档后续各节归纳出的取向）：

**目标**

- **用最小通信换显存/算力扩展**：每种并行维度都把"切了什么"与"为此付出多少通信"明确配对 —— TP 切每层矩阵但只在 block 边界 all-reduce（§3.1），PP 按层切段但段间只传一次中间张量（§3.2），EP 分散专家、DP 复制实例。选择哪种、怎么组合，本质是在"通信代价"与"显存/延迟收益"之间取舍（见 §6）。
- **对上层 `step()` 完全透明**：并行策略全部藏在 Executor 抽象之后，`schedule → execute → update` 三段式对"底下是单卡还是多机多卡"毫不知情（见 §6）。换并行策略 = 换 Executor，调度与模型代码无需改动。
- **四种并行可正交组合**：TP/PP/EP/DP 是四个相互独立的维度，可任意叠加成 `ExternalDP × DP × PP × TP` 的四维拓扑（§4.4）；典型组合是"节点内 TP 吃 NVLink + 节点间 PP 省带宽 + DP 复制提吞吐 + 超大 MoE 叠 EP"。
- **通信原语按场景择优**：同一个 all-reduce 在不同消息大小/拓扑下走不同实现 —— custom all-reduce 专为小消息低延迟优化（decode 高频 all-reduce 受益最大），超出范围再回退 pynccl / torch（§4.2）。
- **可被 `torch.compile` 捕获**：通信原语包成 PyTorch custom op，让 all-reduce 能进入编译图、与计算融合（见 §4.2 与 [模块 07](../07-cuda-graph-compile/design.md)）。

**约束 / 当前实现边界**

- **V1 多进程 Executor 暂不支持 PP**：`MultiprocExecutor` 当前断言 `world_size == tensor_parallel_size`，V1 的 PP 必须走 **Ray Executor**（详见 §4.3、§7 与 impl 阶段 G）。
- **TP 强依赖高带宽、通常不跨节点**：TP 的 all-reduce 高频且在关键路径，需要 NVLink 级别的卡间带宽；跨节点 TP 几乎不可行。
- **custom all-reduce 有严格启用条件**：仅在单节点、NVLink 全连接、消息 `<8MB`、`world_size∈{2,4,6,8}` 时启用，否则回退 NCCL。
- **DP 的隐含同步不可省**：DP 各实例多数时候独立，但 MoE+DP 组合下 forward 内部有跨 DP 集合通信，空闲 rank 必须跑 dummy batch 参与，否则死锁（§4.4 与 impl 的 T11）。

---

## 3. 四种并行各自的设计与适用场景

### 3.1 TP（张量并行）—— 把每一层矩阵切开，只在必要处 all-reduce

**思想（Megatron-LM）**：Transformer 的两大算子（MLP 的两层、Attention 的 QKV+O）都可以写成"列并行 → 行并行"的配对，让中间激活天然按 TP rank 切分，**整段只在末尾做一次 all-reduce**。

- **ColumnParallelLinear（列并行）**：权重 `W` 沿**输出维**切成 `W = [W_1, W_2, ..., W_n]`，每个 rank 持 `W_i`，算出 `Y_i = X W_i`。各 rank 拿到**输出的不同列**，**无需通信**（除非 `gather_output=True` 才 all-gather）。
- **RowParallelLinear（行并行）**：权重 `W` 沿**输入维**切成 `W = [W_1; W_2; ...; W_n]`，输入 `X` 也按列切（恰好是上游列并行的输出），各 rank 算 `Y_i = X_i W_i` 得到**部分和**，**末尾一次 all-reduce** 求和。

把 MLP 的 `up_proj`（列并行）→ 激活 → `down_proj`（行并行）串起来，**中间激活全程切分、不通信，只在 `down_proj` 后 all-reduce 一次**。Attention 同理：QKV 列并行（每 rank 算一部分 head）→ attention → O 行并行 all-reduce。**每个 Transformer block 因此只有 2 次 all-reduce**（attn 后 + mlp 后）。

- **QKVParallelLinear**：列并行的特例，按 **head** 切。对 GQA/MQA（`num_kv_heads < num_heads`），当 `tp_size >= num_kv_heads` 时 KV head 被**复制**到多个 rank（`num_kv_head_replicas`），保证每 rank 至少有一份 KV。
- **VocabParallelEmbedding / LogitsProcessor**：词表沿 vocab 维切，embedding 时各 rank **mask 掉不属于自己的 token**、查表后 all-reduce；logits 则 gather（或 all-gather）回完整词表再采样。

**适用**：几乎所有多卡场景的基础维度，尤其单层大、追求低 decode 延迟时。**代价**：all-reduce 高频且在关键路径，强依赖卡间高带宽（NVLink）。通常**不跨节点做 TP**。

> 设计动机出处：Shoeybi et al., *"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"* (2019, arXiv:1909.08053)；列/行并行配对消除中间通信的洞见出自该文 §3。vLLM 的 `parallel_state.py` 文件头注明 "Adapted from NVIDIA/Megatron-LM"。

### 3.2 PP（流水线并行）—— 按层切段，段间只传一次中间张量

**思想（GPipe / PipeDream）**：把 L 层模型切成 P 段，每段放在一组 GPU 上。一条请求的 hidden_states 像在流水线上流动：第 0 段算 embedding + 前若干层 → 把 `IntermediateTensors`（hidden_states + residual）**send 给下一段** → ... → 最后一段算 norm + logits + 采样。

- 段间通信是**点对点 send/recv**，每段边界**只传一次**激活张量，通信量远小于 TP 的 all-reduce。
- 朴素 PP 有**流水线气泡**（bubble）：当只有一批数据时，某一时刻只有一段 GPU 在干活。vLLM 用 **batch queue** 把多批"叠流水"来填气泡（见 §4.3、引用 [模块 00](../00-request-lifecycle/impl.md) 的 T10）——这正是 **1F1B**（one-forward-one-backward，推理只有 forward）思想在推理侧的体现：在 stage 之间始终塞着多个 in-flight 的 micro-batch。

**适用**：模型层数很多、显存是主瓶颈、且要**跨节点**扩展时（PP 段间通信少，对跨节点带宽不敏感）。**代价**：有气泡（需 batch queue 摊薄）；单请求延迟会因为穿过 P 段而增加。

> 注意 V1 实现现状：`MultiprocExecutor` 当前断言 `world_size == tensor_parallel_size`，**V1 的多进程 Executor 尚未支持 PP**（`multiproc_executor.py:56-59`）；V1 的 PP 走 **Ray Executor**（`max_concurrent_batches = pipeline_parallel_size`）。模型层与通信代码（`make_layers` / `IntermediateTensors`）是通用的。

> 设计动机出处：Huang et al., *"GPipe"* (2019, arXiv:1811.06965)；Narayanan et al., *"PipeDream"* (SOSP 2019) 与 Megatron 的 **1F1B** 调度（arXiv:2104.04473）。

### 3.3 EP（专家并行）—— MoE 把整个专家分散到不同卡

**思想（GShard / Switch Transformer）**：MoE 层有 `E` 个专家，每个 token 只激活 top-k 个。当 `E` 很大（如 256 个专家）时，把**整个专家**（而非专家内的矩阵）分到不同 rank：rank `i` 只持有 `E / ep_size` 个专家的完整权重。

- 与 TP-on-MoE 的区别：**TP 把每个专家的矩阵切开**（所有 rank 都参与每个专家的计算）；**EP 把不同专家放到不同 rank**（每个 rank 只算自己的专家）。vLLM 用 `enable_expert_parallel` 在二者间切换。
- 通信：router 决定每个 token 去哪些专家后，需要把 token **dispatch** 到持有对应专家的 rank（all-to-all / multicast），算完再 **combine** 回来（all-reduce）。
- `expert_map`（global expert id → local expert id 的查表，不在本 rank 的专家映射为 -1）让每个 rank 在融合 kernel 里**只处理自己的专家**。

**适用**：超大 MoE 模型（DeepSeek/Mixtral 级别），专家数多到单卡放不下全部专家。**代价**：dispatch/combine 的 all-to-all 是新的通信瓶颈；负载可能不均（热门专家集中在某些 rank）。

> 设计动机出处：Lepikhin et al., *"GShard"* (2020, arXiv:2006.16668)；Fedus et al., *"Switch Transformer"* (2021, arXiv:2101.03961)。

### 3.4 DP（数据并行）—— 复制整个引擎，各喂不同请求

**思想**：当模型**放得下**单卡（或单个 TP/PP 组），但要更高吞吐时，把**整个 EngineCore + 它的 Worker 群**复制 `dp_size` 份，每份独立调度、独立 forward，各自处理不同的请求流。前端用 ROUTER socket 按 identity 把请求分发到不同 DP 引擎（见 [模块 00](../00-request-lifecycle/impl.md) 的 T1）。

- 不同于 TP/PP 共享一个模型实例，**DP 的每个 rank 是一个完整独立的实例**，绝大多数时候**互不通信**。
- 唯一的强同步点：①**全局 finish 同步**——必须知道"是否所有 DP rank 都没活了"才能一起暂停（每 16 步 all-reduce 一次，见 §4.4 与 impl T12）；② **MoE + DP 组合**时，专家计算需要跨 DP 把 token 凑齐（`naive_multicast` + all-reduce）。
- **死锁陷阱**：某 DP rank 本地没请求、但别的 rank 还在跑，它必须跑一个 **dummy batch**（空 forward），否则它不参与那些隐含的集合通信，会让其他 rank 卡死。

**适用**：模型不大、追求高 QPS 时；或与 TP 组合（每个 DP 实例内部再 TP）。**代价**：显存成倍（每份都是完整副本）；finish 同步的集合通信。

> 设计动机出处：DP 在 vLLM 中主要服务于 online serving 的吞吐扩展与 **MoE 的 EP+DP 协同**（vLLM 文档 *"Data Parallel Deployment"*）。

---

## 4. 核心设计思想

### 4.1 GroupCoordinator：把 PyTorch ProcessGroup 包成"会通信的对象"

vLLM 不直接用裸 `torch.distributed`，而是用 **`GroupCoordinator`**（`parallel_state.py:132`）统一封装一个进程组的所有通信。它的关键设计：

- **一个 coordinator 同时持有两个 group**：`device_group`（NCCL，跑 GPU tensor）+ `cpu_group`（gloo，跑 CPU 上的 metadata / object，`parallel_state.py:180-184`）。很多操作（broadcast object、barrier）刻意走 CPU group——注释明确说 NCCL 的 barrier "terrible"，会偷偷建 GPU tensor 搞乱当前 device（`:673-680`）。
- **device_communicator 抽象**：真正的 all_reduce/all_gather/send/recv 委托给一个 `DeviceCommunicatorBase`（CUDA 上是 `CudaCommunicator`），由它决定用 custom all-reduce 还是 pynccl 还是 torch（见 §4.2）。
- **rank 的多重身份**：`rank`（全局）、`rank_in_group`（组内）、`local_rank`（用于选 device）三者分离（`:142-154` 的注释表给了跨节点的例子）。
- **TP group 额外挂一个 `mq_broadcaster`**（共享内存 MessageQueue，`:221-223`），这就是 [模块 00](../00-request-lifecycle/design.md) 讲的广播 `scheduler_output` 的通道——**只在 TP group 启用**（`initialize_model_parallel` 里 `use_message_queue_broadcaster=True`，`:933`）。

四个全局 coordinator：`_WORLD` / `_TP` / `_PP` / `_DP`，通过 `get_tp_group()` / `get_pp_group()` / `get_dp_group()` 取用（`parallel_state.py:748/766/761`）。

### 4.2 通信原语：custom all-reduce → pynccl → torch 三级回退

CUDA 上 all-reduce 的分发逻辑（`cuda_communicator.py:52-71`）是理解性能的关键：

```
always try custom all-reduce first  → 否则 pynccl  → 否则 torch.distributed
```

- **custom all-reduce**（`custom_all_reduce.py`）：vLLM 自研的一次性内核，专为**小消息、单节点、NVLink 全连接**优化。`should_custom_ar`（`:213-226`）只在 **`inp_size < max_size`（默认 8MB）** 且 **world_size==2 或 GPU 全连接** 时启用。
- **pynccl**：vLLM 直接 ctypes 绑定 NCCL（`pynccl.py`），绕过 PyTorch 的 NCCL 封装，可在 CUDA graph 内被捕获、可控制 stream（见 [模块 07](../07-cuda-graph-compile/design.md)）。
- **torch.distributed**：兜底（主要测试场景）。

> **为什么 custom all-reduce 在小消息上比 NCCL 快**：NCCL 的 ring/tree all-reduce 在小张量上**延迟受启动开销和多次 kernel launch 主导**，吞吐优势发挥不出来。vLLM 的 custom kernel 用 **IPC 把各 rank 的 buffer 直接映射进彼此地址空间**，一次性 kernel 读写 peer 显存完成归约，省掉 NCCL 的协议握手与多段传输——在 decode 阶段（每步激活只有几 KB~几 MB、且 all-reduce 极其高频）收益最大。一旦消息超过 `max_size` 或跨节点/PCIe-only，就退回 NCCL。custom all-reduce 的 **GPU IPC 显存映射、NVLink P2P 探测、one-shot / two-shot kernel 的 GPU 视角实现**见 [模块 11](../11-gpu-kernels-memory/design.md)。
> 设计动机出处：vLLM PR #2192 / #2760（custom all-reduce 引入与多 GPU 支持），动机即"小消息低延迟 all-reduce"。

`GroupCoordinator.all_reduce`（`:292-318`）还包了一层 **PyTorch custom op**（`torch.ops.vllm.all_reduce`，`:312`）：因为 Dynamo/torch.compile 不能把 `self` 这种任意对象传进图，于是只传 `group_name` 字符串，再在 op 内部按名字查回 coordinator（`:109-114`）——这是为了让 all-reduce 能被 `torch.compile` 正确捕获（见 [模块 07](../07-cuda-graph-compile/design.md)）。

### 4.3 PP 与 batch queue：用流水线填气泡

PP 的核心配合在 EngineCore 进程层（已在 [模块 00](../00-request-lifecycle/impl.md) 的 T10 详述，这里给设计要点）：`step_with_batch_queue`（`core.py:212`）在队列没满且有未调度请求时**只调度+提交下一批、立即返回不等结果**，只有"无新批可调度"或"队列满"时才 `future.result()` 阻塞取最早一批。`batch_queue_size = max_concurrent_batches`（`core.py:115`），而 Ray Executor 把它设成 `pipeline_parallel_size`（`ray_distributed_executor.py`）。于是有 P 段就有 P 批在流水线里同时飞，气泡被填满——这正是 1F1B 的推理版。

### 4.4 world 拓扑：ExternalDP × DP × PP × TP

`initialize_model_parallel`（`parallel_state.py:870`）用一行 reshape 把 `world_size` 个 rank 排成四维网格（`:919-921`）：

```python
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size)
#         ExternalDP        DP                PP                      TP
```

然后通过 **转置 + reshape + unbind** 取出每个维度的分组：
- **TP group**：`view(-1, tp_size)`——最内层连续的 `tp_size` 个 rank 是一个 TP 组（`:926`）。相邻 rank 同组 → 应放同一 DGX/NVLink 域。
- **PP group**：`transpose(2,3).reshape(-1, pp_size)`（`:940-941`）——跨 PP 维取。
- **DP group**：`transpose(1,3).reshape(-1, dp_size)`（`:950-952`）。

显存与进程数的换算（`config.py`）：
- `world_size = pipeline_parallel_size * tensor_parallel_size`（`config.py:1663-1664`）——**一个模型实例**需要的 GPU 数 = 创建多少 Worker。
- `world_size_across_dp = world_size * data_parallel_size`（`config.py:1678`）——含 DP 的**总** GPU 数。

`init_distributed_environment`（`:828-837`）据此把 DP rank 偏移进全局 rank、调整总 world size，并为每个 DP 引擎分配独立的 init port（`get_next_dp_init_port`）。

注释（`:910-916`）讲清了 DP 的两类语义：**内部 DP**（同一 DP 组必须**同步** generate，否则隐含集合通信死锁）vs **ExternalDP**（每个 DP rank 完全独立，如 verl RL 训练集成）。

---

## 5. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `GroupCoordinator` | `parallel_state.py:132` | 一个进程组的通信门面，持 device_group(NCCL)+cpu_group(gloo)+device_communicator+可选 mq_broadcaster |
| `_WORLD/_TP/_PP/_DP` | `parallel_state.py:708/745/756/758` | 四个全局 coordinator 单例 |
| `DeviceCommunicatorBase` | `device_communicators/base_device_communicator.py:17` | all_reduce/all_gather/gather/send/recv 的抽象基类 |
| `CudaCommunicator` | `device_communicators/cuda_communicator.py:11` | CUDA 实现，内含 `pynccl_comm` + `ca_comm`（custom AR）并做三级分发 |
| `CustomAllreduce` | `device_communicators/custom_all_reduce.py:48` | 小消息 IPC all-reduce；`_SUPPORTED_WORLD_SIZES=[2,4,6,8]`、`max_size=8MB` |
| `PyNcclCommunicator` | `device_communicators/pynccl.py` | ctypes 直绑 NCCL，可入 CUDA graph |
| `IntermediateTensors` | `sequence.py:1130` | PP 段间传递的 `{hidden_states, residual}` 字典 |
| `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` | `model_executor/layers/linear.py:358/1110/768` | TP 的列并行/行并行/QKV 切分线性层 |
| `VocabParallelEmbedding` | `model_executor/layers/vocab_parallel_embedding.py:198` | 词表并行 embedding（mask + all-reduce） |
| `FusedMoE` | `model_executor/layers/fused_moe/layer.py` | MoE 层，含 `ep_size/ep_rank/expert_map`、TP↔EP 切换 |
| `ParallelConfig` | `config.py:1538` | tp/pp/dp size、`enable_expert_parallel`、world_size 派生 |
| `MultiprocExecutor` / `RayDistributedExecutor` / `UniprocExecutor` | `v1/executor/` | 把 `execute_model` 下发到 1/N 个 Worker 的执行后端 |

---

## 6. 权衡取舍（通信量 vs 显存 vs 延迟）

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| **TP**（列+行并行配对） | 单层显存与算力线性下降；中间激活不通信 | 每 block 2 次 all-reduce，**高频在关键路径**，强依赖 NVLink；跨节点 TP 几乎不可行 |
| **PP**（按层切段） | 段间通信极少，**对跨节点带宽不敏感**；显存跨段摊薄 | 流水线气泡（需 batch queue）；单请求延迟随段数增加；V1 仅 Ray Executor 支持 |
| **EP**（专家分散） | 超大 MoE 显存可扩展；每 rank 只算部分专家 | dispatch/combine 的 all-to-all 成新瓶颈；专家负载不均 |
| **DP**（复制实例） | 吞吐近线性扩展；实例间几乎不通信 | 显存成倍；finish 同步集合通信；空闲 rank 必须跑 dummy batch 否则死锁 |
| **custom all-reduce 优先** | 小消息低延迟，decode 高频 all-reduce 受益大 | 仅单节点、NVLink 全连接、`<8MB`、world∈{2,4,6,8} 时启用；否则退回 NCCL |
| **GroupCoordinator 包 custom op** | all-reduce 可被 torch.compile 捕获 | 多一层"按 group_name 查表"的间接 |
| **finish 同步每 16 步才 all-reduce** | 摊销 DP 集合通信开销 | 最多多跑 15 步才发现全局空闲；中间乐观假设"还有活" |

> 一般组合策略：**节点内 TP（吃 NVLink）+ 节点间 PP（省带宽）+ DP 复制提吞吐 + 超大 MoE 叠 EP**。这正是 `ExternalDP × DP × PP × TP` 拓扑要表达的四维空间。

---

## 7. Executor 抽象：让并行对上层透明

上层 `step()` 只调 `model_executor.execute_model(scheduler_output)`，并行策略完全藏在 Executor 后面：

- **`UniprocExecutor`**：单卡，本进程直接调 Worker，无广播。
- **`MultiprocExecutor`**：多进程 TP（V1 当前 `world_size == tp_size`），用共享内存 MessageQueue 广播 + `rank0_reply_only`（[模块 00](../00-request-lifecycle/design.md) §3.5、F2）。
- **`RayDistributedExecutor`**：支持 PP（`max_concurrent_batches = pp_size`），用 Ray compiled DAG 把 scheduler_output 流过各 stage，PP 时返回 `FutureWrapper` 让调度器去排下一批。

这层抽象是整个分布式系统的"接缝"：换并行策略 = 换 Executor，`schedule → execute → update` 三段式毫不知情。

---

## 8. 一图速览：并行维度与通信

```
                         ┌──────────────────── world ranks (ExternalDP × DP × PP × TP) ───────────────────┐
                         │                                                                                 │
   DP（复制整个实例）─────┼──► DP rank0 引擎          DP rank1 引擎          ... （各自独立调度，前端 ROUTER 分发）
                         │      │                       │
                         │   ┌──┴───────────┐        ┌──┴───────────┐
   PP（按层切段）─────────┼─► │ PP stage0    │ send   │ PP stage0    │   段间只传 IntermediateTensors
                         │   │  layers 0..k │ ─────► │              │   {hidden_states, residual}
                         │   │ PP stage1    │ recv   │ PP stage1    │
                         │   │  layers k..L │        │              │
                         │   └──┬───────────┘        └──────────────┘
                         │      │
   TP（切每层矩阵）───────┼─►  ┌─┴─────────────────────────────────────────┐
                         │    │  rank0      rank1      rank2      rank3     │  每个 Transformer block:
                         │    │  W_0        W_1        W_2        W_3       │   col-parallel(无通信)
                         │    │   └──── all-reduce (attn 后) ────┘          │   → row-parallel + all-reduce
                         │    │   └──── all-reduce (mlp 后)  ────┘          │   custom AR(<8MB,NVLink) / pynccl
                         │    └────────────────────────────────────────────┘
   EP（MoE 专家分散）─────┼─►  rank0: experts[0..e)   rank1: experts[e..2e) ...
                         │    dispatch(all-to-all/multicast) → 本地专家计算 → combine(all-reduce)
                         └──────────────────────────────────────────────────────────────────────────────┘

   通信频率/代价：  TP all-reduce（每层2次，最高频）> EP all-to-all（每MoE层）> PP send/recv（每段1次）> DP finish-sync（每16步）
```

> 逐行调用链、`file:line` 对照、实现精妙之处与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 9. 设计背后的考量与历史教训

> 本节比 §6 的"权衡取舍"表更进一步：挖掘**为什么是现在这个设计**（否决了什么、被什么约束倒逼、随哪些重构演进），以及从真实 bugfix 里读出的设计教训。所有 PR 号均来自本仓库 `git log` 真实记录。

### 9.1 设计背后的考量（深化"为什么"）

1. **为什么不直接用裸 `torch.distributed`，而要包一层 `GroupCoordinator`**：裸 `ProcessGroup` 只有"一个 group 一种 backend"的概念，但 vLLM 需要在**同一个逻辑组**里同时跑 GPU tensor（NCCL）和 CPU metadata/object（gloo），还要在小消息上换成 custom all-reduce、在 CUDA graph 内换成 pynccl。于是 `GroupCoordinator` 把"device_group + cpu_group + device_communicator + 可选 mq_broadcaster"绑成一个门面对象（§4.1）。代价是多一层抽象，但换来了"按消息大小/拓扑/是否在编译图内择优"的自由——这正是 §4.2 三级回退能存在的前提。注释里"NCCL barrier terrible"那句（`:673-680`）是这一选择的活证据：连最简单的 barrier 都刻意绕开 NCCL 走 CPU group。

2. **`world` 拓扑为什么演进成四维 `ExternalDP × DP × PP × TP`**：早期 reshape 只有 `DP × PP × TP` 三维，用一个 `has_external_dp` 布尔 + "world_size 对不上就当外部 DP"的启发式来兼容 verl 这类 RL 训练集成（外部框架自己复制实例、各 rank 独立 generate）。**#15355（`[distributed] fix dp group`）把这个易错的启发式删掉**，改成显式的第四维 `ExternalDP`（reshape 第一维从 `data_parallel_size` 变成 `-1`），并在注释里钉死两类 DP 的根本区别：**内部 DP 必须同步 generate（否则隐含集合通信死锁），ExternalDP 各 rank 完全独立**（§4.4）。这条演进说明：当"用一个 flag 区分两种语义"开始出 bug 时，正确的做法是把语义提升成拓扑里的一个**显式维度**。

3. **被 `torch.compile` 倒逼的"按 group_name 查表"设计**：all-reduce 本可以直接 `self.device_communicator.all_reduce(x)`，但 Dynamo 无法把 `self`（任意 Python 对象）作为参数 trace 进图。约束倒逼出 §4.2 的方案：通信原语包成 `torch.ops.vllm.all_reduce`，**只传 `group_name` 字符串**，op 内部再按名字查回 coordinator（`:109-114`）。这是"为了能被编译图捕获"而牺牲一点间接性的典型取舍——不是为了优雅，而是为了让 all-reduce 能和计算融合进同一张图（见 [模块 07](../07-cuda-graph-compile/design.md)）。

4. **custom all-reduce 为什么甘愿被严格条件锁死**：理论上自研 all-reduce 越通用越好，但 §4.2 的 `should_custom_ar` 只在"单节点 + NVLink 全连接 + `<8MB` + world∈{2,4,6,8}"时启用。这不是实现没做完，而是清醒的取舍：custom kernel 靠 **GPU IPC 把 peer 显存映射进本地地址空间**一次性归约，这套机制只在 NVLink P2P 可达、消息小到不值得走 NCCL 多段协议时才赢。超出范围硬上只会更慢甚至错。于是设计选择"窄而快的快路径 + NCCL 兜底"，而非"宽而平庸的通用实现"。

5. **EP vs TP-on-MoE 为什么做成可切换而非二选一**：MoE 既可以"把每个专家的矩阵 TP 切开"（所有 rank 参与每个专家），也可以"把不同专家分到不同 rank"（EP）。二者通信模式（all-reduce vs all-to-all）、负载特征（均匀 vs 可能倾斜）完全不同，没有一种在所有规模下都最优。所以 vLLM 用 `enable_expert_parallel` 开关让二者共存于同一个 `FusedMoE`（§3.3），靠 `expert_map` 把"全局专家 id → 本地专家 id"的查表统一掉。这层灵活性的代价，恰恰是 9.2 里 #13784 那类"global vs local 专家数没区分清"的 bug 温床。

6. **Executor 抽象为什么是整个系统的"接缝"**：把四种并行全藏在 `execute_model` 之后（§7），换来的是"调度与模型代码对底下是单卡还是多机多卡毫不知情"。但这层接缝也定义了**当前的实现边界**：V1 的 `MultiprocExecutor` 断言 `world_size == tensor_parallel_size`，于是 V1 的 PP 只能走 Ray Executor（§4.3、§7）。这不是疏忽，而是"先把 TP 这条最高频路径的进程模型做扎实，PP 这类低频跨节点场景交给 Ray 的 compiled DAG"的优先级选择。

### 9.2 重要 bug 修复（真实、精选）

> 下列 PR 均可在本仓库 `git log` 查到；每条点出它**暴露的设计点 / 教训**。

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| **#15355** `[distributed] fix dp group` | 旧实现用 `has_external_dp` 布尔 +"world_size 对不上就当外部 DP"的启发式拼 world 拓扑，DP 分组在外部 DP 场景下算错 | "内部 DP / 外部 DP"是**两种根本不同的同步语义**，不该用一个 flag 兜。修复把它提升为显式的第四维 `ExternalDP`（§4.4）。教训：当一个布尔开始承载两种语义且出 bug，把它升成拓扑维度 |
| **#16161** `fix use-ep bug to enable ep by dp/tp size > 1` | `use_ep` 的判定写成 `enable_expert_parallel and self.tp_size > 1`，导致**纯 DP（tp_size==1）下 EP 开不起来** | EP 的并行度来自 `tp_size * dp_size` 而非只看 TP。一行 `tp_size > 1` → `tp_size * dp_size > 1` 的修复，暴露了 §3.3 里"EP/DP/TP 三维如何共同决定专家分布"这个易错耦合点 |
| **#13784** `Fix FP8 + EP` | `FusedMoE` 里 `num_experts` 一个字段同时被当"全局专家数"和"本地专家数"用，FP8+EP 组合下权重切分错位 | EP 下必须严格区分 `global_num_experts` 与 `local_num_experts = global // ep_size`。修复把这俩拆成两个字段。教训：凡是被并行切分的"数量"，全局值与本地值必须在命名上就分开（§3.3 的 `expert_map` 同源） |
| **#11233** `Use correct device to initialize GPU data during CUDA-graph-capture` | `graph_capture()` 不接收 device 参数，捕获 CUDA graph 时在**错误的 device** 上建 stream / 初始化 GPU 数据，多卡下串味 | 多卡进程里"当前 device"是隐式全局状态，捕获 CUDA graph 这种对 device/stream 极敏感的操作必须**显式传 device**（修复把 `graph_capture(device)` 显式化）。呼应 §4.2 通信原语要能干净进编译图/CUDA graph 的约束 |
| **#14053** `Fix shutting_down flag checking in V1 MultiprocExecutor` | `shutdown()` 里把判定写反：`if getattr(self,'shutting_down',False)` 导致正常情况下**根本不执行清理**（只有已经在关闭时才清理，逻辑颠倒） | 多进程 Executor 的关闭路径（回收 worker、释放共享内存 MessageQueue）一旦写反，资源泄漏 / 卡死且难复现。教训：守卫"只清理一次"的 flag，判定方向（`not shutting_down` 才进入）是关键一字之差 |
| **#12934** `respect distributed_executor_backend in world_size=1` | `world_size==1` 时强制走 Uniproc，**忽略**用户显式指定的 `distributed_executor_backend`（如想用 Ray/外部 executor 做单卡调试） | §7 的"换并行=换 Executor"抽象要彻底：即便单卡也应尊重用户选的后端，否则抽象在边界条件上漏风。暴露了"world_size 派生逻辑"与"executor 选择逻辑"耦合处的边界遗漏 |
