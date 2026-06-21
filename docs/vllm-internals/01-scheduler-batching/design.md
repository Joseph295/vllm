# 模块 01 · 调度器：Continuous Batching + Chunked Prefill —— 设计文档

> 范围：EngineCore 每一步 `step()` 里的 **①调度** 环节——决定"这一步对哪些请求、各算多少 token、用哪些 KV block"。它是吞吐与延迟的总阀门，也是 chunked prefill / prefix caching / 投机解码 / 多模态在调度层面的交汇点。
> 架构：**V1 为主（`vllm/v1/core/sched/scheduler.py`）**，关键处对比 V0（`vllm/core/scheduler.py`）说明重构动机。
> 上游背景见 [模块 00](../00-request-lifecycle/design.md)（请求全生命周期），KV/block 细节见 [模块 02](../02-paged-attention-kvcache/design.md)，投机解码见 [模块 05](../05-speculative-decoding/design.md)。

---

## 1. 这个模块解决什么问题

LLM 推理有一个根本的不对称：**prefill 是计算密集（compute-bound）**——一次把整个 prompt 的所有 token 并行算完，GPU 算力打满；**decode 是访存密集（memory-bound）**——每步只算 1 个新 token，但要把全部 KV cache 从显存搬一遍，算力大量闲置。

朴素地"一条请求跑完再跑下一条"会让 GPU 在 decode 期间严重空转。**连续批处理（continuous batching）** 的思路是：在 **iteration（一次 forward）粒度**上动态拼批——每完成一步就重新决定下一步谁进 batch，一旦某条请求结束就立刻让新请求补位，不必等整个 batch 对齐。这把 GPU 利用率从"按最慢请求对齐"提升到"逐步填满"。

但连续批处理本身还留了一个坑：一条 **长 prompt 的 prefill**（比如 32K token）会占满一整步的算力预算，导致这一步**没法夹带其他请求的 decode**——正在流式输出的请求被卡住，inter-token latency（ITL）飙升，这就是 **head-of-line blocking**。**Chunked prefill** 的思路是：把长 prefill **切成多个 chunk**，每步只算一部分 prompt token，把省下来的预算让给其他请求的 decode token，让"长 prompt 进来"和"已有请求平稳出 token"能并存。

这个模块要同时满足几组互相拉扯的诉求：

1. **高吞吐**：每一步都尽量把 token 预算（`max_num_batched_tokens`）填满，GPU 不空转。
2. **低 ITL / 公平性**：长 prefill 不能独占整步、饿死正在 decode 的请求。
3. **显存安全**：KV cache 是有限的；放不下时要有抢占（preemption）兜底，且不能死锁。
4. **统一覆盖多种推理模式**：同一套调度逻辑要能同时支持 chunked prefill、prefix caching、投机解码、多模态 encoder，而不是为每种模式写一套分支。

> 设计动机出处：
> - **连续批处理**：Orca, *"Orca: A Distributed Serving System for Transformer-Based Generative Models"*（Yu et al., OSDI 2022）提出 iteration-level scheduling，是 vLLM continuous batching 的理论原型。
> - **Chunked prefill**：Sarathi / Sarathi-Serve, *"SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills"*（Agrawal et al., 2023）与后续 *"Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"*（OSDI 2024）提出 chunked-prefill + decode-piggybacking。
> - **V1 重写动机**：vLLM 官方博客 *"vLLM V1: A Major Upgrade to vLLM's Core Architecture"*（blog.vllm.ai，2025-01）。

---

## 2. 设计目标与约束

- **iteration-level，一步一调度**：`schedule()` 由 busy loop 反复调用，每次只决定"下一个 forward 算什么"（`interface.py:17-39` 的 docstring 是纲领）。
- **统一的 token 推进抽象**：不区分 prefill / decode 阶段，每条请求只有 `num_computed_tokens` 去追 `num_tokens_with_spec`（`scheduler.py:122-132`）。
- **token 预算混合填充**：一个 batch 里可以同时包含某请求的 prefill chunk 和另一些请求的 decode token，二者共享同一个 `token_budget`。
- **双重约束**：每步同时受 `max_num_batched_tokens`（token 上限）和 `max_num_seqs`（并发请求数上限）约束。
- **抢占而非交换**：V1 显存不足时只做"重计算式抢占"（preempt-by-recompute），**取消了 V0 的 swap-to-CPU**。
- **FCFS 为主，可选 priority**：默认先到先服务（`policy="fcfs"`），保证公平不饿死；少数约束（encoder budget、LoRA 上限）下会有限地放松严格 FCFS。
- **调度产物是纯数据**：`schedule()` 只产出一个 `SchedulerOutput`（描述算什么），不碰 GPU；执行与回填是另外两段。这让调度可单测、可与执行解耦（PP 流水线化）。

---

## 3. 核心设计思想（含 V0 → V1 对比）

### 3.1 "没有 prefill 阶段，也没有 decode 阶段"的统一抽象（V1 最关键的简化）

**V0 痛点**：V0 调度器内部把请求按阶段分流到三个队列 `waiting` / `running` / `swapped`（`vllm/core/scheduler.py:469-475`），并有两套顶层策略 `_schedule_default`（`scheduler.py:1219`，prefill 优先、不混 decode）与 `_schedule_chunked_prefill`（`scheduler.py:1325`）。`_schedule_default` 的注释直白写着 *"Don't schedule decodes if prefills are scheduled"*（`scheduler.py:1255`）。这套"阶段分流"导致：① 逻辑分散在 `_schedule_prefills` / `_schedule_running` / `_schedule_swapped` 多个方法里，组合爆炸难维护；② prefill 和 decode 默认互斥，长 prefill 会阻塞 decode。

**V1 方案**：V1 调度器**彻底取消阶段概念**。`schedule()` 开头的注释（`scheduler.py:123-132`）是整个模块的设计纲领：

> 调度器里既没有 "decoding phase" 也没有 "prefill phase"。每条请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`。每一步调度就是给请求分配一些 token，让它的 `num_computed_tokens` 去追 `num_tokens_with_spec`。这个抽象足够通用，能同时覆盖 chunked prefill、prefix caching、speculative decoding，乃至未来的 jump decoding。

为什么一个抽象能覆盖这么多模式？因为它们本质上都是"给请求推进若干 token"的特例：

| 模式 | 在统一抽象里的表现 |
|---|---|
| **首次 prefill** | `num_new_tokens = num_prompt_tokens - 0`，一步算一大段 prompt token |
| **Chunked prefill** | 同上，但 `num_new_tokens` 被 `token_budget` / `long_prefill_token_threshold` 截断，分多步算完 |
| **Decode** | `num_new_tokens = 1`（`num_tokens_with_spec` 只比 `num_computed_tokens` 多 1）——是 chunked prefill 的退化情形 |
| **投机解码（spec decode）** | `num_new_tokens = k+1`（k 个 draft token + 1），`num_tokens_with_spec` 把 `spec_token_ids` 也计入（`request.py:113-114`） |
| **Prefix caching** | 命中的前缀直接计入 `num_computed_tokens`（`get_computed_blocks`，`scheduler.py:305`），跳过计算 |
| **混合批** | 上述任意组合同处一个 batch，共享一个 `token_budget` |

于是同一段循环逻辑同时处理了所有情况——**这正是 V1 调度器代码量比 V0 小得多的根本原因**。

### 3.2 token 预算如何在一个 batch 里混合 prefill + decode

调度的核心变量是 `token_budget`，初值 = `max_num_batched_tokens`（`scheduler.py:149`）。`schedule()` 分两段消费这个预算：

```
token_budget = max_num_batched_tokens          # 一步能算的总 token 上限
  │
  ├─ 第一段：遍历 RUNNING 队列（scheduler.py:161）
  │     每条请求 num_new_tokens = num_tokens_with_spec - num_computed_tokens
  │       · 正在 decode 的请求：num_new_tokens = 1（或 spec 的 k+1）
  │       · 正在 chunked-prefill 的请求：num_new_tokens = 一大段，被 token_budget 截断
  │     token_budget -= num_new_tokens          # 边调度边扣预算
  │
  └─ 第二段：遍历 WAITING 队列（scheduler.py:276，仅当本步无抢占时）
        新请求 / 被抢占恢复的请求的 prefill，继续消费剩余 token_budget
```

关键在：**decode 和 prefill 用的是同一个 `token_budget`，没有任何"这部分预算只给 decode"的隔离**。先调度 RUNNING（其中大量是 decode，每条只吃 1 个 token），再用剩余预算去 WAITING 队列拉新 prefill。这样一个 batch 自然就变成"一堆 decode token + 一段（或几段）prefill chunk"的混合——**正是 Sarathi 的 decode-piggybacking**：让 memory-bound 的 decode 搭着 compute-bound 的 prefill 一起跑，把 GPU 的算力和带宽同时吃满。

**chunked prefill 的截断点**有两处：
1. `num_new_tokens = min(num_new_tokens, token_budget)`（`scheduler.py:174`、`316`）——预算用完就这一步只算这么多，下一步接着算。
2. `long_prefill_token_threshold`（`scheduler.py:170-173`、`312-315`）——单条长 prompt 一步最多算这么多 token，防止一条超长 prompt 把整个预算吞掉、挤占其他请求。

### 3.3 RUNNING 优先于 WAITING：保证"出 token"不被"进新请求"卡住

调度顺序是刻意的：**先把所有 RUNNING 请求推进（第一段），再考虑 WAITING 新请求（第二段）**。这保证了正在生成的请求每一步都拿到 decode 名额、平稳出 token，新请求只能用"剩饭"预算——这是低 ITL 的结构性保证。

与之配套的一个关键约束：**第二段只在本步没有发生抢占时才执行**（`scheduler.py:275` `if not preempted_reqs:`）。一旦本步已经因显存不足抢占了请求，就说明显存正紧张，此时再拉新请求进来只会加剧抖动，所以本步干脆不收新活。

### 3.4 抢占 vs 重计算：V1 砍掉了 swap

**V0 痛点**：V0 显存不足时有两种抢占模式（`PreemptionMode`，`vllm/core/scheduler.py:34`）：`RECOMPUTE`（丢弃 KV、之后重算）和 `SWAP`（把 KV 换出到 CPU 内存，之后换回）。`SWAP` 需要一条专门的 `swapped` 队列、`_swap_in`/`_swap_out`、`blocks_to_swap_in/out`，还因为多序列组（beam search）不支持重算而强制走 swap（`scheduler.py:1758-1761`），整套机制相当复杂。

**V1 方案**：V1 **只保留 recompute 一种**。抢占逻辑浓缩成 RUNNING 调度循环里的一个内嵌 `while True`（`scheduler.py:196-223`）：当 `allocate_slots` 返回 `None`（显存不足），就 `self.running.pop()` 抢占**优先级最低**（队尾）的请求，`free` 它的 KV，状态置 `PREEMPTED`、`num_computed_tokens = 0`，塞回 `waiting` 队头（`appendleft`），然后重试分配。被抢占的请求之后会作为一次新的（带 prefix cache 命中的）prefill 重新调度。

为什么敢砍掉 swap？因为 V1 默认开启 **prefix caching**——被抢占请求重算时，它的 prompt 前缀大概率还在 prefix cache 里命中，重算成本远没有看起来那么高；而 swap 的 PCIe 来回搬运 KV 在大模型下往往比重算还慢。**recompute + prefix caching 让 swap 变得不划算**，于是被整体移除，换来调度器和 block manager 的大幅简化。同时 V1 把请求抽象从 V0 的 `SequenceGroup`（可含多序列）降为单一 `Request`，beam search 那条"必须 swap"的理由也不复存在。

### 3.5 公平性与"有限放松 FCFS"

默认 FCFS（`policy="fcfs"`）按到达顺序服务，天然不饿死。但有两处**故意用 `continue` 而非 `break` 来放松严格 FCFS**：

- **encoder budget 不足**（`scheduler.py:186-191`）：多模态请求若因 encoder 计算预算/缓存耗尽无法调度，用 `continue` 跳过它去看下一条，而非 `break` 卡住整个队列——注释明示这是"intentionally relax the strict FCFS"。
- **LoRA 上限**（`scheduler.py:295-302`）：若再加一个新 LoRA 会超过 `max_loras`，把该请求 `popleft` 后暂存到 `skipped_waiting_requests`，跳过它继续看后面能复用已加载 LoRA 的请求；循环结束再把跳过的请求放回队头（`scheduler.py:375-377`），不改变它们的相对顺序。

这两处的共同精神：**当队头请求被"非显存"的约束卡住时，不连坐后面的请求**，但又通过"放回队头"保持整体的到达序，是吞吐与公平的折中。

### 3.6 双重记账：先把 token 记成"已算"，错了再退回

`schedule()` 在还没真正 forward 之前，就把 `num_computed_tokens` 加上本步调度的 token 数（`scheduler.py:455-456`）。这是"乐观推进"：

> 提前推进，使得**下一步 `schedule()` 能立刻继续这条请求的下一个 chunk**（chunked prefill 连续推进 / PP 流水线提前排下一批），不必等本步执行结果回来。

如果本步是投机解码、且部分 draft token 被 reject，`update_from_output` 再用 `len(spec)+1 - len(generated)` 精确**回退** `num_computed_tokens`（`scheduler.py:605-607`）。这套"乐观推进 + 事后纠正"是调度能与执行解耦、支持 PP 批队列流水线的基础（见 [模块 00](../00-request-lifecycle/impl.md) 的 T4、T10）。

### 3.7 cascade attention 的公共前缀探测

调度尾声会算一个 `num_common_prefix_blocks`（`scheduler.py:390-397`）：取 RUNNING 队列任一请求，沿它的 block 列表往前数，凡是 `ref_cnt == num_running_requests`（被所有 running 请求共享）的 block 都算公共前缀（`kv_cache_manager.py:368-377`）。这个数被放进 `SchedulerOutput`，供 attention backend 决定是否启用 **cascade attention**——当多个请求共享一段长公共前缀（如同一个 system prompt）时，对前缀只做一次 attention 再"级联"各自的后缀，省显存带宽。是否真正启用由 backend 的启发式 `use_cascade_attention`（`flash_attn.py:553`）决定（前缀 < 256 token、请求数 < 8、或带 ALiBi/滑窗则不用）。调度器只负责"算出公共前缀有多长"，用不用交给执行层。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `Scheduler` | `v1/core/sched/scheduler.py:33` | 调度器主体，持有所有队列与 KV/encoder cache manager |
| `self.waiting` | `scheduler.py:79`（`deque[Request]`） | 等待队列：新请求 + 被抢占请求（队头优先） |
| `self.running` | `scheduler.py:80`（`list[Request]`） | 运行队列：已分配 KV、正在推进的请求；**队尾 = 最低优先级**（抢占从队尾取） |
| `self.scheduled_req_ids` | `scheduler.py:83`（`set[str]`） | 本步已调度的请求 id，避免一步内重复调度 |
| `self.finished_req_ids` | `scheduler.py:89`（`set[str]`） | 上一步到本步之间结束的请求，随 `SchedulerOutput` 下发让 worker 清缓存 |
| `self._cached_reqs_data` | `scheduler.py:94`（`dict[str, CachedRequestData]`） | **复用池**：缓存 `CachedRequestData` 对象，避免每步为每条请求新建 |
| `Request` | `v1/request.py:18` | 请求状态机：`num_computed_tokens`、`spec_token_ids`、`status`、token 列表 |
| `RequestStatus` | `v1/request.py:149` | `WAITING` / `WAITING_FOR_FSM` / `RUNNING` / `PREEMPTED` / `FINISHED_*`（`> PREEMPTED` 即 finished，`:164`） |
| `SchedulerOutput` | `v1/core/sched/output.py:81` | **调度产物**：新请求/缓存请求数据、`num_scheduled_tokens`、spec token、encoder 输入、公共前缀块数、grammar bitmask |
| `NewRequestData` | `output.py:19` | 首次调度的请求的完整数据（prompt token、mm 输入、block_ids 等），worker 收到后缓存 |
| `CachedRequestData` | `output.py:53` | 已调度过的请求只发**增量**（新 token、新 block、`num_computed_tokens`），省 IPC |

**`Request` 的两个核心计数器**（统一抽象的载体）：
- `num_computed_tokens`：已经算进 KV cache 的 token 数（含 prefix cache 命中）。
- `num_tokens_with_spec`（`request.py:113`）= `len(all_token_ids) + len(spec_token_ids)` = prompt + 已生成 output + 待验证的 draft token。

每步调度的本质就是一行：`num_new_tokens = num_tokens_with_spec - num_computed_tokens`，再被各种预算截断。

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 统一 token 抽象（无 prefill/decode 之分） | chunked prefill / prefix cache / spec decode / 多模态共用一套循环；代码量大降 | 正确性高度依赖 `num_computed_tokens` 双重记账；单步逻辑要兼顾所有模式 |
| chunked prefill 默认常开（V1，`arg_utils.py:1652`） | 长 prompt 不阻塞 decode，ITL 平稳；prefill/decode 混批提升 GPU 利用 | 长 prompt 总 TTFT 略增（被拆成多步）；chunk 边界需正确处理 mm/attention |
| 只保留 recompute 抢占，砍掉 swap | 调度器/block manager 大幅简化；配合 prefix cache 重算成本低 | 重算理论上比 swap 多算一遍（但 prefix cache 命中后差距很小）；无 CPU offload 兜底 |
| RUNNING 先于 WAITING，且抢占步不收新活 | decode 平稳出 token；显存紧张时不雪上加霜 | 极端高压下新请求 TTFT 可能被推迟 |
| FCFS + 有限放松（encoder/LoRA 用 continue） | 公平、不饿死；又不让队头被非显存约束连坐 | 放松点是局部 hack，需配合"放回队头"维持到达序 |
| 乐观推进 `num_computed_tokens` | 下一步可立即续排；解耦调度与执行（支持 PP 流水线） | spec decode 需事后精确回退，记账出错即静默错误 |
| 复用 `CachedRequestData` + 只发增量 | running 上千条时省大量小对象分配与 IPC 带宽 | 对象状态被原地改写，逻辑上需保证字段全量覆盖 |
| 公共前缀块数交给执行层决策 | 调度器不背 attention 性能模型；cascade 用不用由 backend 启发式定 | 存在"有公共前缀但因未调度的 running 请求而算成 0"的边界（`kv_cache_manager.py:353-357`） |

---

## 6. V1 相比 V0 的净收益（小结）

- **公平性 / 延迟**：取消 prefill/decode 分离 + 默认 chunked prefill，长 prompt 不再独占整步、阻塞 decode，ITL 显著平稳。
- **可维护性**：三个队列（waiting/running/swapped）+ 两套策略 + `SchedulingBudget` 类，简化为两个队列（waiting/running）+ 一段统一循环 + 一个 `token_budget` 整数。
- **吞吐**：decode-piggybacking 让 memory-bound 的 decode 搭 compute-bound 的 prefill，GPU 算力与带宽同时吃满。
- **简化的显存模型**：只 recompute、不 swap，配合默认 prefix caching，去掉了一整套 CPU-GPU 搬运逻辑。

> 实现层面的逐行调用链、`file:line` 对照、Tricks 与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 7. 一图速查：调度一步的 ASCII 架构图

```
                          Scheduler.schedule()   (一次 forward = 一次调度)
                          token_budget = max_num_batched_tokens
                          ┌───────────────────────────────────────────────┐
   self.running (list)    │  ① 先调度 RUNNING（出 token 优先）              │
   [decode, decode,       │     for req in running, while token_budget>0:  │
    chunked-prefill, ...] │       num_new = num_tokens_with_spec           │
        队尾=最低优先级    │                 - num_computed_tokens          │
            │             │       num_new = min(num_new, budget,           │
            │             │                long_prefill_token_threshold)   │
            │             │       allocate_slots(req, num_new) ──┐         │
            │             │         └ None? → 抢占队尾请求 ◄──────┘         │
            │             │              while True: running.pop()→free    │
            ▼             │                   →PREEMPTED→waiting队头        │
   self.waiting (deque)   │       budget -= num_new                        │
   [new, preempted, ...]  │                                                │
   队头=最高优先级 ───────►│  ② 若本步无抢占，再调度 WAITING（拉新 prefill） │
                          │     while waiting and budget>0                 │
                          │            and len(running)<max_num_seqs:      │
                          │       get_computed_blocks()  ← prefix cache    │
                          │       num_new = num_tokens - num_computed      │
                          │       allocate_slots(); 放不下→break           │
                          │       waiting.popleft()→running.append()       │
                          │       budget -= num_new                        │
                          ├───────────────────────────────────────────────┤
                          │  ③ 收尾：                                       │
                          │     num_common_prefix_blocks（cascade attn）   │
                          │     grammar_bitmask（structured output）       │
                          │     打包 SchedulerOutput                       │
                          │     num_computed_tokens += num_scheduled（乐观）│
                          └───────────────────────────────────────────────┘
                                              │
                                              ▼
                          SchedulerOutput  ──►  executor.execute_model()
                          { scheduled_new_reqs : [NewRequestData]   (全量)
                            scheduled_cached_reqs:[CachedRequestData](增量)
                            num_scheduled_tokens : {req_id: n}
                            scheduled_spec_decode_tokens
                            num_common_prefix_blocks
                            grammar_bitmask, ... }
                                              │
                                              ▼  (GPU forward + sample)
                          update_from_output() ── append token / check_stop
                                              └─ spec reject → 回退 num_computed_tokens
```

---

## 8. 设计背后的考量与历史教训

### 8.1 设计背后的考量（动机与历史）

1. **"取消 prefill/decode 阶段"否决了 V0 的"两套顶层策略"路线**。V0 用 `_schedule_default`（prefill 优先、明文 *"Don't schedule decodes if prefills are scheduled"*）和 `_schedule_chunked_prefill` 两套策略 + 三个队列（waiting/running/swapped）应对不同模式，逻辑分散在 `_schedule_prefills`/`_schedule_running`/`_schedule_swapped`，组合爆炸。V1 赌的是"所有模式都是给请求推进若干 token 的特例"这一抽象足够通用，于是用**一段统一循环 + 一个 `token_budget` 整数**替掉整套分流。代价是单步逻辑要兼顾全部模式、正确性全押在 `num_computed_tokens` 记账上——这也是为什么后面一长串 bug 都集中在"chunk 边界""prefix 命中边界"这类记账角落。

2. **砍掉 swap-to-CPU 是"recompute + 默认 prefix caching"组合拳算过账后的结论，不是简单删功能**。V0 显存不足有 RECOMPUTE 和 SWAP 两种抢占，SWAP 还因 beam search 不支持重算而被强制使用，拖着一整套 `swapped` 队列与 `_swap_in/out`。V1 默认开 prefix caching 后，被抢占请求重算时前缀大概率命中、重算成本骤降，而 swap 的 PCIe 来回搬 KV 在大模型下往往比重算还慢；同时请求抽象从 `SequenceGroup` 降为单一 `Request`，beam search "必须 swap"的理由也消失。于是 swap 被整体移除，换来调度器与 block manager 的大幅简化。

3. **chunked prefill 的两个截断点是对 Sarathi 思想的工程化补强**。仅靠 `min(num_new_tokens, token_budget)` 还不够——一条超长 prompt 即使在预算内也会吞掉整步、挤死其他请求的 decode 名额。`#15419` 引入 `long_prefill_token_threshold`，给单条长 prompt 单步可算 token 数再加一道闸。这说明"decode-piggybacking"在生产里要的不只是"能混批"，还要"防单条独占"的公平阀门。

4. **"RUNNING 优先 + 抢占步不收新活"是延迟与稳定性的结构性保证，而非性能优化**。先推进所有 RUNNING（其中大量是每条只吃 1 token 的 decode）再用剩饭预算拉 WAITING，保证了正在流式输出的请求每步都拿到名额；而 `if not preempted_reqs:` 让"本步一旦抢占就不再收新请求"，避免显存已经吃紧时雪上加霜。这两条是刻意写死的调度顺序约束，体现"低 ITL 要靠结构保证、不能靠调参"。

5. **演进脉络：抽象先立、记账后收敛、模块再拆分**。`#9289`（V1 1/N 落地统一抽象）→ `#12608`（统一 prefill/decode 的 slot 分配）→ `#12003`/`#15307`（把 `num_computed_tokens` 的算账逻辑收进 KVCacheManager、再做一次重构）→ `#15250`（抽出 `SchedulerInterface`）→ `#14466`（可插拔调度器）。这条线说明 V1 调度器是"先押注一个激进抽象，再用一连串记账修复把它焊牢，最后才抽接口对外开放"，顺序不可颠倒。

### 8.2 重要 bug 修复（真实、精选）

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| `#11186` | prompt 长度恰为 block_size 整数倍且全部命中 prefix cache 时，`num_new_tokens==0`，无 token 可调度 | 暴露核心不变量 **`num_computed_tokens` 必须是 block_size 整数倍**。修复是回退**整整一个 block**（`num_computed_tokens -= block_size`）强制重算最后一块，而非只退 1 个 token——记账不变量比省那点算力更重要 |
| `#12065` | VLM 调度：因 prefix caching，`num_computed_tokens` 可能已 **超过** encoder 输入的 `start_pos`，旧逻辑算出负的 `num_new_tokens` | "统一 token 抽象"与"多模态 encoder 输入必须整块处理"在 prefix 命中时打架；修复显式处理 `num_computed_tokens >= start_pos` 时只能本步调度 0 token。教训：每引入一种新模式，都要重新核对它与 prefix caching 的边界 |
| `#12674` | 对 partial（chunked prefill 中途）请求有多余的调度约束 | 早期为安全给 partial 请求加了限制，反而阻碍混批；移除后才真正释放 chunked prefill + decode 同批的吞吐。说明保守约束会悄悄吃掉抽象本该带来的收益 |
| `#12545` | 请求 abort 后，其 encoder cache 未释放 | 调度器持有的 encoder cache 也是需要按生命周期回收的资源；修复把 `free` 拆成 `free_encoder_input`（单个）与 `free`（整请求），abort 路径调用整请求版本。多模态让"资源回收"的面变大了 |
| `#13169` | preemption 未被指标系统正确记账 | 抢占是 V1 唯一的显存兜底手段，却长期对可观测性"隐身"；补上 preemption 计数后，生产中"显存抖动"才变得可诊断。教训：核心兜底路径必须可观测 |
