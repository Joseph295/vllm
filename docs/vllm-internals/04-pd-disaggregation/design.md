# 模块 04 · PD 分离（Prefill-Decode Disaggregation）—— 设计文档

> 范围：把一条请求的 **prefill 阶段**（吃满计算）与 **decode 阶段**（吃满访存）拆到**两个独立的 vLLM 实例**上跑，prefill 实例算完 KV cache 后，把 KV **跨实例搬运**给 decode 实例续算。本模块聚焦 `vllm/distributed/kv_transfer/` 这套"KV 跨实例传输"基础设施。
>
> **架构现状（务必先读）**：本仓库快照里，PD 分离的 KV 传输代码**全部位于 `vllm/distributed/kv_transfer/`，且只接在 V0 的 worker（`vllm/worker/model_runner.py`）上**。`vllm/v1/` 下**没有任何** `kv_transfer` / `KVConnector` 引用（grep 为空），也**不存在** `kv_connector/v1/` 目录。更关键的是：`vllm/engine/arg_utils.py:1526-1530` 明确把 `--kv-transfer-config` 列为"V1 尚不支持"的特性 —— **一旦设置该参数，引擎会从 V1 回退（fallback）到 V0**。所以本模块以 V0 集成路径为唯一事实来源讲解，并在结尾说明 V1 现状。整套机制官方也标注为 **experimental**（`examples/online_serving/disaggregated_prefill.sh:8`）。

---

## 1. 这个模块解决什么问题

LLM 自回归推理天然分成两个**计算特征截然相反**的阶段：

| 阶段 | 一次 forward 处理的 token | 算术强度 | 瓶颈 | 关键延迟指标 |
|---|---|---|---|---|
| **Prefill**（处理 prompt） | 整段 prompt（几十~几千个） | 高，是**计算密集**（compute-bound）的大矩阵乘 | GPU 算力（FLOPs / Tensor Core） | **TTFT**（Time-To-First-Token） |
| **Decode**（逐 token 生成） | 每步 1 个（投机解码 k+1 个） | 低，是**访存密集**（memory-bound），每步都要把全部权重 + KV 从显存搬一遍 | 显存带宽（HBM bandwidth） | **TPOT**（Time-Per-Output-Token） |

把两者**混在同一个实例、同一个 batch 里跑**（vLLM 默认的 continuous batching + chunked prefill，见 [模块 00](../00-request-lifecycle/design.md) / [模块 01](../01-scheduler-batching/design.md)），会带来几组互相拉扯的矛盾：

1. **互相干扰**：一个长 prompt 的 prefill 会"霸占" GPU 算力一整步，正在 decode 的请求被拖慢 —— TPOT 抖动、长尾延迟恶化（head-of-line blocking）。反过来，为了照顾 decode 的低延迟而频繁打断 prefill，又拉低吞吐。
2. **TTFT vs TPOT 无法分别优化**：两阶段对资源的需求不同，但混跑时只能用**同一套** batch size、并行度（TP/PP）、调度策略，无法针对性调参。例如 decode 希望大 batch 摊薄权重读取、prefill 希望小 batch 控制 TTFT —— 单实例只能折中。
3. **硬件配比僵化**：prefill 缺算力、decode 缺带宽，混跑时无法按阶段用不同型号/数量的 GPU。

**PD 分离的核心主张**：既然两阶段特征相反，就**物理拆开** —— 用一组实例专做 prefill（KV producer），另一组专做 decode（KV consumer）。每组可独立选择并行度、batch 策略、GPU 配比、甚至独立弹性扩缩容。代价是：prefill 算出的 **KV cache 必须跨实例搬运**到 decode 实例。本模块就是这套搬运基础设施。

> 设计动机出处见 §6。一句话：DistServe（OSDI 2024）与 Splitwise（ISCA 2024）用实验证明了"分离后 TTFT 和 TPOT 可各自满足 SLO，且总吞吐/成本更优"；Mooncake（Moonshot/Kimi 生产系统）把它落地为以 KV cache 为中心的分离式架构。

---

## 2. 设计目标与约束

**目标**

- **可插拔的传输层抽象**：vLLM 核心不绑死某一种传输介质（NCCL / RDMA / 第三方 KV store）。新增一种传输后端只需实现一个 `KVConnector`，注册进工厂即可（`kv_connector/factory.py`）。
- **producer / consumer 角色清晰**：同一份 worker 代码，靠配置 `kv_role` 决定本实例是发送方还是接收方（`config.py:3173`），发送/接收逻辑完全对称地挂在 model forward 的前后。
- **bypass 模型前向**：consumer 收到 KV + hidden states 后，**直接跳过 prefill 前向**，省掉 decode 实例上的重复计算（`base.py:97-99` 的 `bypass_model_exec` 语义）。
- **乱序容错**：prefill 与 decode 实例处理请求的顺序可能不同（高 QPS 下尤甚），传输层不能假设 FIFO 严格对齐。
- **背压（backpressure）**：decode 慢时要能反压 prefill，避免 KV 在传输 buffer 里无限堆积撑爆显存。

**约束 / 当前实现边界**

- **只在 V0 worker 上接通**；V1 设置该 config 会回退到 V0（`arg_utils.py:1527`）。
- **仅支持 1P1D（1 prefill + 1 decode）**：`kv_rank` 注释明确写"Currently only 1P1D is supported"（`config.py:3176-3177`）。多 P 多 D 的路由/编排尚未在本层实现。
- **请求级编排由外部代理（proxy）完成**：vLLM 本身不决定"先发 prefill 再发 decode"，这套编排在一个外置 proxy server 里（`benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py`）。vLLM 只负责"算到 / 收到 KV"。
- **SimpleConnector 假设整批都是 prefill**：`simple_connector.py:170`、`:234` 的 `FIXME(Kuntai): This assume that all requests are prefill`。因此需关掉 chunked prefill 并对齐 batch 尺寸（`:244-246` 的告警）。

---

## 3. 核心设计思想

### 3.1 拆成 prefill 实例与 decode 实例，KV 跨实例传输

整体拓扑是**两个完全独立的 vLLM 进程组** + 一个**外置编排代理**：

```
                    ┌─────────────────────────────────────────────┐
   client ───────►  │  Proxy server (quart)                       │
                    │  1) max_tokens=1 → 发给 Prefill 实例          │
                    │  2) prefill 完成后 → 把原请求发给 Decode 实例  │
                    └───────────────┬───────────────┬─────────────┘
                                    │               │
                       (HTTP) ◄─────┘               └─────► (HTTP)
                          ▼                                   ▼
                 ┌──────────────────┐   KV cache    ┌──────────────────┐
                 │ Prefill 实例      │ ───────────►  │ Decode 实例       │
                 │ kv_role=producer │  + hidden     │ kv_role=consumer │
                 │ kv_rank=0        │   states      │ kv_rank=1        │
                 └──────────────────┘               └──────────────────┘
```

- Proxy 给 prefill 实例发请求时把 `max_tokens` 改成 1（`disagg_prefill_proxy_server.py` 的 `prefill_request['max_tokens'] = 1`），让它**只做 prefill**、不真正生成；prefill 实例在 forward 末尾把 KV + hidden states **发**出去。
- Proxy 随后把**原请求**发给 decode 实例；decode 实例在 forward 开头**收**到 KV，`bypass_model_exec=True` 跳过 prefill 前向，直接拿 hidden states 算第一个 token 并继续 decode。

> 关键认知：**vLLM 实例之间不互相知道对方**。它们只通过共享的 KV 传输管道（按 `kv_ip:kv_port` + `kv_rank` 建立的 stateless process group）对接。"哪条请求先 prefill 后 decode"的时序由 proxy 保证。

### 3.2 三层可插拔抽象：Connector / LookupBuffer / Pipe

`vllm/distributed/kv_transfer/README.md` 把传输栈分成三层，自顶向下：

```
┌─────────────────────────────────────────────────────────────┐
│ KVConnector  (kv_connector/base.py)                          │
│   贴着 vLLM model runner 的两个钩子：                          │
│   send_kv_caches_and_hidden_states / recv_...                │
│   职责：从 paged KV 里"按 token 抽出本请求的 KV"、决定 bypass  │
├─────────────────────────────────────────────────────────────┤
│ KVLookupBuffer  (kv_lookup_buffer/base.py)                   │
│   把 FIFO 管道封装成"按 token 查找"的 buffer                   │
│   insert(tokens, roi, k, v, hidden) / drop_select(tokens,roi)│
│   职责：解决 prefill/decode 请求顺序不一致（见 3.4）           │
├─────────────────────────────────────────────────────────────┤
│ KVPipe  (kv_pipe/base.py)                                    │
│   最底层 FIFO 张量管道：send_tensor / recv_tensor             │
│   实现：PyNcclPipe（NCCL，GPU）/ MooncakePipe（RDMA）         │
└─────────────────────────────────────────────────────────────┘
```

**可插拔的两个维度**：

1. **整条栈可替换**：通过 `kv_connector` 字符串选择不同 connector（`factory.py:42-60` 注册了 4 个）。
2. **下面两层可旁路（bypass）**：README 明确两条旁路规则 ——
   - 若底层通信服务**自带 key-value 查找**（Redis / RDMA database），可跳过 KVPipe 层。
   - 若想**改变 vLLM 的执行流**（如只对部分 token 收 KV、其余重算），可同时跳过 Pipe + Buffer，直接在 Connector 层实现（README 第 17-21 行；代价是 vLLM model input 一变就可能失效）。

四个内建 connector 的定位（`factory.py:42-60`）：

| connector | 底层 | 定位 / 适用场景 |
|---|---|---|
| `PyNcclConnector` → `SimpleConnector` | PyNcclPipe + SimpleBuffer | **参考实现 / 单机或同集群 1P1D**。用 NCCL 在两实例间点对点传 KV，自带 lookup buffer 处理乱序与背压。 |
| `MooncakeConnector` → `SimpleConnector` | MooncakePipe（RDMA）+ SimpleBuffer | 同 SimpleConnector 的逻辑，但底层管道换成 Mooncake 的 **RDMA 传输引擎**，跨机高带宽低延迟。需 `MOONCAKE_CONFIG_PATH`（`simple_connector.py:51-57`）。 |
| `LMCacheConnector` | LMCache 引擎 | 委托给外部 **LMCache** 项目：不仅做 PD 间传输，还做 **KV offload / 跨请求共享**（`lmcache_connector.py:5-7`）。vLLM 侧只是薄适配层。 |
| `MooncakeStoreConnector` | MooncakeStore（KVStoreBufferBase，put/get） | 把 KV 当成**数据库式 KV store**：用 `blake2b(tokens)` 当 key 存取（`mooncake_store_connector.py:89,147,196`），天然支持乱序、解耦 producer/consumer 在线状态。 |

### 3.3 KV 如何按 token / layer 切分传输

> 本节涉及的 paged KV cache、物理 slot、`reshape_and_cache` 等概念属 [模块 02](../02-paged-attention-kvcache/design.md)；按 layer 切分对应的 PP 范围（`start_layer..end_layer`）属 [模块 03](../03-distributed-parallel/design.md)。

发送端 `SimpleConnector.send_kv_caches_and_hidden_states`（`simple_connector.py:151`）的切分逻辑：

- **按请求（按 token 段）切**：用 `attn_metadata.seq_lens` 把 batch 里每条请求的 token 段 `[start_pos:end_pos]` 切出来（`:171-173`）。
- **按 layer 切**：对每条请求，遍历本 worker 负责的 `start_layer..end_layer`（PP 切分），用该请求的 `slot_mapping` 从 paged KV cache 里**按物理 slot 抽出 key/value**（`:187-198`）。即 `key_cache[current_slot_mapping]` —— 只取属于这条请求这些 token 的 KV，不传整块 paged memory。
- **打包发送**：把每条请求的 `(tokens, roi, keys[layer×token], values[layer×token], hidden[token])` 五元组 `insert` 进 producer buffer（`:200-203`）。

> 这里有个非显然点：传的不是"逻辑 token 顺序的 KV"，而是**按 paged slot 抽取后重新堆叠**的紧凑张量。接收端再用自己的 `slot_mapping` 把它们 `reshape_and_cache` 回**本地的** paged 显存（`utils.py:61-90`），因为两实例的 block 分配是各自独立的。

接收端 `recv_kv_caches_and_hidden_states`（`simple_connector.py:207`）逐请求 `drop_select` 查询、把命中的 KV 写回本地 paged cache，并拼接 hidden states；**只有所有请求都完整命中**才 `bypass_model_exec=True`，否则回退到正常前向重算（`:300-317`）。

### 3.4 为什么需要 LookupBuffer：FIFO 管道 ≠ 正确传输

README（第 15 行）点出核心难点：**prefill 和 decode 实例处理请求的顺序可能不同**。高 QPS 下，prefill 按 A→B→C 算完，decode 可能先要 C 的 KV。裸 FIFO 管道无法支持这种"按内容查找"。

`SimpleBuffer` 的解法（`simple_buffer.py`）：

- **producer 侧**：`insert` 把 KV 入本地 deque（`:214-225`），并起一个**后台 `drop_select_handler` 线程**（`:135`）监听 consumer 的查询请求。
- **consumer 侧**：`drop_select` 把"我要哪些 token 的 KV"（input_tokens + roi）经管道发给 producer（`:185-212`）。
- **producer 后台线程**：收到查询后，在 deque 里做 **O(n) 前缀匹配**（`_matches`，`:53-80`；`is_buffer_available` 轮转匹配 `:153-165`），命中就把整条 KV `popleft` 出来经管道送回（`:174-177`）。

这就把"FIFO 管道"翻译成了"按 token 查找的 buffer"，解决乱序。同时 buffer 用 `Condition` 变量做**背压**：buffer 满则 `insert` 阻塞等待（`:120-130`），decode 慢时自动反压 prefill。

### 3.5 传输与计算重叠

- **发送非阻塞**：model runner 注释明确"the send operation is non-blocking"（`model_runner.py:1785`）。`PyNcclPipe.send_tensor` 把实际发送丢进一个单线程 `ThreadPoolExecutor`（`pynccl_pipe.py:233-247`），主推理线程提交后立刻返回，下一步 forward 与上一步 KV 发送重叠。
- **接收阻塞但可与监听解耦**：`recv` 是阻塞的（`model_runner.py:1740` "The receive operation is blocking"）。`SimpleBuffer` 特意把**信号管道放 CPU**（`simple_buffer.py:30-39`）：注释说明"on-device recv 会阻塞进程内所有线程，使 producer 无法在传输 KV 时监听新请求；而 CPU recv 只阻塞当前线程"，因此用 CPU 信号管道监听、GPU 数据管道传 KV，二者解耦。
- **buffer 自带 size 记账与背压**（`pynccl_pipe.py:216-223` `block_if_full`），把传输与显存占用解耦。

### 3.6 何时分离才划算

PD 分离不是免费的 —— 它引入了**KV 跨实例搬运的网络开销**和**至少 2 倍的实例数**。划算与否取决于：

- **prompt 越长、生成越短**，prefill 占比越高，分离收益越大（DistServe / Splitwise 的核心观察）。
- **网络带宽要足够**：KV cache 大小 ≈ `2 × layers × kv_heads × head_dim × seq_len × dtype_bytes`。带宽不够时，搬运 KV 的时间会吃掉省下的计算时间。这正是 Mooncake 用 RDMA、本模块预留 `MooncakePipe` 的原因。
- **SLO 分离需求**：当 TTFT 和 TPOT 有**各自独立**的 SLO、且混跑无法同时满足时，分离让你能分别加机器满足。
- **规模要够大**：1P1D 在低负载下纯属浪费（两台机器干一台的活）。收益来自高负载下"prefill 池"和"decode 池"各自打满、互不干扰。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `KVTransferConfig` | `vllm/config.py:3157` | PD 分离的全部配置：`kv_connector` 选哪种 connector、`kv_role`（producer/consumer/both）、`kv_rank`、`kv_parallel_size`、`kv_ip/kv_port`、`kv_buffer_size`、`kv_buffer_device`、`kv_connector_extra_config`。 |
| `KVTransferAgent` | `kv_transfer_agent.py:24` | 进程级单例总控（`_KV_TRANSFER`，`parallel_state.py:775`）。薄 shim：把 send/recv 转给底层 connector。 |
| `KVConnectorBase` | `kv_connector/base.py:22` | 抽象基类。两个核心抽象方法：`send_kv_caches_and_hidden_states` / `recv_kv_caches_and_hidden_states`（返回 `(hidden, bypass_model_exec, model_input)`）。 |
| `KVLookupBufferBase` | `kv_lookup_buffer/base.py:39` | lookup buffer 抽象：`insert(tokens, roi, key, value, hidden)` / `drop_select(tokens, roi)`（SQL 式语义）。 |
| `KVStoreBufferBase` | `kv_lookup_buffer/base.py:122` | KV store 抽象（put/get），供 MooncakeStore 这类数据库式后端用。 |
| `KVPipeBase` | `kv_pipe/base.py` | 最底层 FIFO 张量管道：`send_tensor` / `recv_tensor`。 |
| `SimpleBuffer` | `kv_lookup_buffer/simple_buffer.py:26` | LookupBuffer 参考实现，deque + Condition 背压 + 后台 drop_select 线程。 |
| `PyNcclPipe` | `kv_pipe/pynccl_pipe.py:41` | NCCL 管道。GPU 走 PyNCCL send/recv，CPU 走 stateless group 的 obj send/recv（传控制信号）。 |
| **`roi`**（region of interest） | `kv_lookup_buffer/base.py:46-54` | input_tokens 上的**二值掩码**，标记"哪些 token 的 KV 实际可用"。当前只用全 1（`simple_connector.py:201`），但抽象上预留了"部分 token 命中"和"TP/PP 下只持有部分 KV"的扩展。 |

**`KVConnectorBase.recv` 的返回三元组**（`base.py:108-119`）是整个 bypass 机制的关键契约：
- `hidden_or_intermediate_states`：全部命中时是拼好的 hidden states，否则 `None`。
- `bypass_model_exec`：`True` = 可跳过模型前向；`False` = 至少一条请求缺 KV，需回退重算。
- `model_input`：可被调整后的输入（供"只对缺失 token 重算"的高级场景）。

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| **物理拆分 prefill / decode 实例** | TTFT 与 TPOT 可分别优化；两池独立选并行度/batch/GPU 配比、独立扩缩容；消除 prefill 对 decode 的干扰 | 实例数翻倍；引入 KV 跨实例搬运的网络开销；低负载下纯浪费 |
| **三层抽象（Connector/Buffer/Pipe）+ 工厂注册** | 传输后端可插拔；NCCL/RDMA/第三方 store 各自实现；下两层可旁路 | 抽象层多、调用链长；`SimpleConnector` 仍把 send/recv 逻辑写死了"全 prefill"假设 |
| **传 KV + hidden states（而非只传 KV）** | consumer 可直接 bypass prefill 前向，连第一个 token 的 hidden 都不用重算 | 传输量更大（多一份 hidden states）；buffer 注释里坦承"未来应同时传 sampler outputs"（`base.py:77`） |
| **LookupBuffer 做按 token 匹配** | 容忍 prefill/decode 请求乱序；buffer 满则背压 | 匹配是 **O(n) 线性扫描**（`simple_buffer.py:157` FIXME 承认非 O(1)）；buffer 只能小，不能扩太大 |
| **send 非阻塞 / recv 阻塞 + CPU 信号管道** | 发送与下一步 forward 重叠；producer 传输时仍能监听新请求 | 实现上要小心 GPU recv 阻塞全进程线程的坑（`simple_buffer.py:32-37` 专门规避） |
| **请求编排放在外置 proxy** | vLLM 核心保持简单，不管多实例时序 | 需要额外部署 proxy；当前 1P1D 写死、API 标注 "subject to change"（`disaggregated_prefill.sh:77-78`） |
| **只接 V0，V1 直接 fallback** | 不阻塞 V1 主线开发；复用 V0 成熟的 model_runner 钩子 | V1 用户开启 PD 分离会被强制退回 V0，享受不到 V1 的进程模型/调度收益（`arg_utils.py:1527`） |

---

## 6. 设计动机出处

> 以下为**网络/论文**来源，用于补充"为什么这么设计"的动机；具体实现以仓库代码为准。

- **DistServe**（Zhong et al., *OSDI 2024*，"DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving"）：系统性论证了"prefill 与 decode 干扰会同时损害 TTFT 和 TPOT"，提出**分离两阶段到不同 GPU/实例**，按各自 SLO 独立调度与配比资源，从而最大化 **goodput**。本模块"拆 producer/consumer 实例"的核心思想与之一致。
- **Splitwise**（Patel et al., *ISCA 2024*，Microsoft）："prompt"（prefill）与"token generation"（decode）的资源画像不同，把它们 split 到不同机器池，并在池间**搬运 KV cache**，可在同等成本下提升吞吐 / 降低功耗。本模块的"prefill 池 → KV 传输 → decode 池"拓扑与之同构。
- **Mooncake**（Moonshot AI / Kimi，"Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"）：生产级落地。以 **KVCache 为中心**、用 **RDMA** 高速搬运、并把 KV 当作可池化/可复用的资源。本模块的 `MooncakeConnector`（RDMA pipe）与 `MooncakeStoreConnector`（KV store）即对接其传输引擎与 store。
- **TetriServe / 后续工作**：进一步研究分离式下的请求路由、SLO 感知调度与多 P 多 D 编排 —— 对应本模块当前 "仅 1P1D、编排在外置 proxy" 的局限，是未来扩展方向。
- **LMCache**：开源 KV cache 层，做 KV **offload（下放到 CPU/磁盘）与跨请求共享**，不仅服务 PD 分离。`LMCacheConnector` 是 vLLM 对它的薄适配（`lmcache_connector.py`）。
- **vLLM 官方 README/示例**：`vllm/distributed/kv_transfer/README.md` 给出三层抽象的设计纲领与旁路规则；`examples/online_serving/disaggregated_prefill.sh` 给出 1P1D 的可运行配置，并标注 experimental。

---

## 7. 一图速查：PD 分离数据流

```
                         ┌──────────────── Proxy (quart) ─────────────────┐
   client request ─────► │ ① 复制请求, max_tokens=1 → POST :8100 (prefill) │
                         │ ② 等 prefill 返回                               │
                         │ ③ 原请求 → POST :8200 (decode), 流式回 client    │
                         └───────┬─────────────────────────┬──────────────┘
                                 │                          │
        ┌────────────────────────▼──────┐       ┌───────────▼─────────────────────┐
        │   PREFILL 实例 (V0 worker)      │       │   DECODE 实例 (V0 worker)         │
        │   kv_role=kv_producer, rank=0   │       │   kv_role=kv_consumer, rank=1    │
        │                                 │       │                                  │
        │  model_runner.execute_model     │       │  model_runner.execute_model      │
        │   │                             │       │   │                              │
        │   │ (forward: 计算 prefill KV)   │       │   │ need_recv_kv? ──► recv_kv_...│
        │   ▼                             │       │   │   drop_select(tokens,roi)    │
        │  need_send_kv? ──► send_kv_...  │       │   │   把 KV 写回本地 paged cache  │
        │   按 seq_lens 切请求             │       │   │   bypass_model_exec=True     │
        │   按 layer×slot 抽 K/V/hidden    │       │   ▼   (跳过 prefill forward)     │
        │   buffer.insert(...)            │       │  用 hidden 直接 sample 首 token   │
        │     │                           │       │  继续 decode (本实例 paged KV)    │
        │     ▼                           │       │                                  │
        │  ┌───────── SimpleBuffer ───────┴───────┴────────┐                         │
        │  │ producer deque  ◄── drop_select 查询(O(n)前缀) │                         │
        │  │  后台 drop_select_handler 线程                  │                         │
        │  └──────────────┬──────────────────┬─────────────┘                         │
        └─────────────────┼──────────────────┼────────────────────────────────────┘
                          │ signal pipe(CPU)  │ data pipe(GPU)
                          ▼                   ▼
                 ┌──────────────────────────────────────┐
                 │  KVPipe: PyNcclPipe (NCCL) /          │
                 │          MooncakePipe (RDMA)          │
                 │  send_tensor(非阻塞) / recv_tensor     │
                 │  StatelessProcessGroup @ kv_ip:kv_port │
                 │  (kv_rank 0 ↔ 1, kv_parallel_size=2)  │
                 └──────────────────────────────────────┘
```

> 实现层面的逐行调用链、`file:line` 对照与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 8. 设计背后的考量与历史教训

> 本节深化 §5 "权衡取舍"表背后的"为什么"，并从真实 bugfix 里读出教训。**重要前提**：如 §0 / §2 所述，本模块的 KV 传输代码**只接在 V0 worker 上、官方标注 experimental、仅 1P1D**，且 `SimpleConnector` 写死"整批都是 prefill"的假设。这意味着它的 bug 大多集中在 `SimpleConnector` 这一个参考实现的边界条件上——下面的修复列表如实反映了这一点（几乎全部围绕 `simple_connector.py` / `simple_buffer.py`），并非全功能成熟模块的均匀分布。

### 8.1 设计背后的考量（深化"为什么"）

1. **为什么是三层（Connector / LookupBuffer / Pipe）而非一层传输 API**：最朴素的设计是"给我一个 `send_kv(req)` / `recv_kv(req)`"。但 §2 列出的三条约束——可插拔后端、乱序容错、背压——各自落在不同层：**后端可替换**落在 Pipe（NCCL/RDMA/store 各实现一个），**乱序容错**落在 LookupBuffer（把 FIFO 翻译成"按 token 查找"，§3.4），**bypass 决策**落在 Connector（从 paged KV 抽本请求的 KV、决定是否跳过前向）。把它们压成一层会让"换 RDMA 后端"被迫重写乱序逻辑。三层的代价是调用链长（§5 已坦承），但每层只对一个约束负责，且 README 明确允许"下两层旁路"（Redis/RDMA 自带查找时跳过 Pipe，§3.2）。

2. **为什么传输编排放在外置 proxy，而不是让两个 vLLM 实例互相感知**：一个看似自然的设计是让 prefill 实例算完直接"推"给某个 decode 实例。但这会把"哪条请求先 prefill 后 decode、发给哪个 decode 池成员"的调度逻辑塞进 vLLM 核心，并要求实例间维护彼此的在线状态。vLLM 选择**让实例之间互不知道对方**（§3.1 的关键认知），只通过 `kv_ip:kv_port + kv_rank` 建的 stateless process group 对接，时序由外置 proxy 保证。代价是要额外部署 proxy、且当前 1P1D 写死；收益是 vLLM 核心保持简单、两池可各自独立扩缩容（§5 末行）。

3. **为什么 KV 不能"按逻辑 token 顺序"直接传，而要先按 paged slot 抽取再重堆叠**：两个实例的 block 分配**各自独立**——同一条请求在 prefill 实例和 decode 实例占的物理 slot 完全不同。所以发送端必须用本地 `slot_mapping` 从 paged 显存里把本请求这些 token 的 K/V **抽成紧凑张量**再传，接收端再用**自己的** `slot_mapping` `reshape_and_cache` 回本地 paged 显存（§3.3 的非显然点）。这条约束直接解释了为什么传的是"按 slot 抽取后的紧凑张量"而非"整块 paged memory"。

4. **为什么信号管道放 CPU、数据管道放 GPU（一个被注释反复强调的设计）**：producer 在传输 KV 时还要能监听 consumer 的新查询。但 **on-device（GPU）recv 会阻塞进程内所有线程**，使 producer 一边传 KV 就没法监听；而 CPU recv 只阻塞当前线程。于是 `SimpleBuffer` 刻意把**信号管道放 CPU、数据管道放 GPU**（§3.5、`simple_buffer.py:30-39`），用 CPU 信号解耦"监听"与"传输"。这是被"GPU recv 阻塞全进程"这一硬约束倒逼出来的非对称设计。

5. **为什么传 KV 还要捎带 hidden states**：只传 KV 的话，decode 实例还得自己跑一遍 prefill 的最后一层才能拿到第一个 token 的 hidden。捎带 hidden states 让 consumer 能**彻底 bypass prefill 前向**（§3.1、`base.py` 的 `bypass_model_exec`），连首 token 的 hidden 都不用重算。代价是传输量更大（§5），且 `base.py:77` 注释坦承"未来应连 sampler outputs 一起传"——说明这是个仍在演进的取舍点。

### 8.2 重要 bug 修复（真实、精选）

> 下列 PR 均可在本仓库 `git log` 查到。这些修复高度集中在 `SimpleConnector` / `SimpleBuffer`，恰好印证了"参考实现 + 1P1D + 整批 prefill 假设"这套边界条件最易出问题。

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| **#13987** `patch the inflight batching on the decode node in SimpleConnector to avoid hangs in SimpleBuffer` | decode 节点做 inflight batching 时，batch 里**混进了 decode token**（不全是 prefill），`SimpleConnector` 仍按"全 prefill"逐 `seq_lens` 切，导致与 producer 的 NCCL 传输错位、**SimpleBuffer 挂死** | 直接暴露 §2 / §5 那条"`SimpleConnector` 假设整批都是 prefill"。修复加了 `start_pos >= num_prefill_tokens` 的判断：一旦越界就 `bypass=False` 回退正常前向并告警"应关 chunked prefill"。教训：基于 FIFO+NCCL 的同步传输对"batch 构成"极敏感，假设一旦被打破就是 hang 而非报错 |
| **#12723** `Fix disagg hang caused by the prefill and decode communication issues` | prefill / decode 间通信时序问题导致整体挂死（围绕 `SimpleBuffer` 的信号/数据管道与锁） | 跨实例同步传输的死锁难复现、难定位。佐证了 §3.5 为何要把信号管道放 CPU、并用 `Condition` 做背压——这套并发结构是 PD 分离正确性的脆弱点 |
| **#14369** `Add a check in send_kv_caches_and_hidden_states and fix the reshape of the KVCache` | `head_size` 直接用 `hidden_size // num_attention_heads` 算，对带独立 `head_dim` 的模型（如 DeepSeek）**算错**，reshape KV 时形状不对；且缺少对 inflight batching 的越界检查 | §3.3 "按 layer×slot 抽 K/V 再 reshape"依赖正确的 `(num_heads, head_size)`。修复改用 `getattr(model_config, "head_dim", ...)`。教训：KV 形状参数不能从 hidden_size 反推，必须尊重模型自带的 `head_dim` |
| **#12074** `Fix num_heads value for simple connector when tp enabled` | `num_heads` 直接取 `model_config.num_key_value_heads`，**没除以 tp_size**，TP 开启时每个 worker 只持有 `kv_heads/tp` 份 KV，却按全量 reshape | 跨实例传 KV 时，发送端持有的是**本 TP rank 切分后**的 KV，形状必须按 `num_key_value_heads / tp_size` 算。暴露了"PD 分离 × TP"组合下，每 rank 只传自己那份 KV 的切分契约（§3.3 提到的 PP `start_layer..end_layer` 同理） |
| **#11058** `Fix value unpack error of simple connector for KVCache transfer` | 原代码从 `kv_cache[0].shape` 解包 `_, _, num_heads, head_size`，依赖 KV cache 张量的特定 4 维布局，布局一变就 unpack 失败 | 不该从 KV cache 张量形状里"猜" `num_heads/head_size`，而应从 `model_config` 显式取。这是 #12074 / #14369 同一处代码连续被修的起点——说明"从张量形状反推语义"是反复出 bug 的设计气味 |
| **#14367** `Add timeout configuration for the torch.store and add KVTransferConfig.kv_connector_extra_config` | 底层 stateless process group 建连用 `torch.distributed` 的 store，**超时写死**，跨机/慢网络下建连卡住无法配置；且 connector 想加自定义参数无处可放 | 暴露 `KVTransferConfig` 早期把配置项写死的局限。修复加了 `kv_connector_extra_config` 这个开放字典 + `get_from_extra_config`，让"新 connector 需要的额外参数"无需改 config 类即可注入——呼应 §2 "可插拔后端"目标 |
