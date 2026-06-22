# 模块 00 · 请求全生命周期 —— 设计文档

> 范围：一条请求从进入 vLLM（offline `LLM.generate` 或 online OpenAI server）到逐 token 返回的**端到端主干**。本模块是后续所有专题模块（调度、KV、并行、采样……）的"骨架"，这里只讲整体编排与进程模型，专题细节留给对应模块。
> 架构：**V1 为主，关键处对比 V0**。

---

## 1. 背景与定位

> 如果你还不清楚 **tokenize/detokenize、前端/后端进程、continuous batching** 这些前置概念，请先读 [PRIMER](../PRIMER.md#第一部分llm-推理服务-101先建立心智模型)，这里默认你已经有了那套心智模型。

### 1.1 这个模块在系统里的位置

请求生命周期是整个引擎的**总骨架**——它不解决某一个具体技术点，而是把"一条请求从进门到出门"的编排串起来：

- **上游**：用户从 HTTP（OpenAI server）或 `LLM.generate`（offline）把文本+采样参数丢进来。
- **本模块**：前端进程把文本 **tokenize** 成 token id、打包成 `EngineCoreRequest`，跨进程送进 EngineCore 后端进程；后端跑 `schedule → execute → update` 主循环逐 token 生成；生成的 token 再跨进程回前端 **detokenize** 成文本。
- **下游**：把每个新 token 流式（SSE）吐回客户端，直到命中结束符或长度上限。

一句话：**它是把"收请求 → 调度（[模块 01](../01-scheduler-batching/design.md)）→ 管显存（[模块 02](../02-paged-attention-kvcache/design.md)）→ 分布式执行（[模块 03](../03-distributed-parallel/design.md)）→ 采样（[模块 06](../06-sampling-structured-output/design.md)）→ 出 token"这一整条流水线编排起来的进程与数据流模型。**

### 1.2 关键文件与类各干什么（先认人）

| 文件 / 类 | 主要作用（一句话） |
|---|---|
| `entrypoints/openai/serving_completion.py` / `entrypoints/llm.py` | 两个入口：online（异步 HTTP）/ offline（同步 `LLM.generate`） |
| `v1/engine/async_llm.py` 的 `AsyncLLM` | **异步前端引擎**：收请求、跑唯一的 `output_handler` 后台 loop、把结果流式吐回 |
| `v1/engine/llm_engine.py` 的 `LLMEngine` | **同步前端引擎**（offline 用）：主线程同步 step 循环，无 asyncio |
| `v1/engine/processor.py` 的 `Processor` | 输入侧 CPU 重活：文本 **tokenize**、多模态预处理、参数校验 → 产出 `EngineCoreRequest` |
| `v1/engine/output_processor.py` 的 `OutputProcessor` | 输出侧 CPU 重活：token **detokenize** 成文本、按 request_id 分发、stop string 检测 |
| `v1/engine/core.py` 的 `EngineCore` / `EngineCoreProc` | **后端引擎进程**：独立进程里跑纯同步 `while True: step()` 主循环，独占 GPU 关键路径 |
| `v1/engine/core_client.py` 的 `EngineCoreClient` | 前端侧的 IPC 门面：ZMQ socket + msgpack 序列化收发 |

### 1.3 和相似模块 / 概念的区别与联系（专治"听过但分不清"）

- **`AsyncLLM`（前端）vs `EngineCore`（后端）**：两者**在不同进程**。`AsyncLLM` 在 API server 进程里做 CPU 活（tokenize/detokenize/序列化），`EngineCore` 在独立后台进程里独占 GPU 关键路径——这是 V1 把"CPU 后处理"与"GPU 推理"进程级隔离、让 GPU 永不空转的核心结构（见 §3.1）。
- **`Processor`（tokenize）vs `OutputProcessor`（detokenize）**：二者**互为反向**。`Processor` 把"用户文本 → token id"（进门方向），`OutputProcessor` 把"采样出的 token id → 文本"（出门方向）；两者都跑在前端进程，是关键路径之外的 CPU 活。
- **同步 `LLMEngine` vs 异步 `AsyncLLM`**：offline 批量场景用 `LLMEngine`（主线程一个同步 step 循环），online 服务用 `AsyncLLM`（asyncio + 单一 `output_handler` 多路复用 N 条请求）。**二者共用同一套 `process_inputs → EngineCore.step → process_outputs` 三段**，只是驱动方式不同（见 [`impl.md`](impl.md) §3）。
- **调度 vs 执行**：本模块的 `step()` 三段式里，`schedule()` 只**动脑**（决定算什么，见 [模块 01](../01-scheduler-batching/design.md)），`execute_model()` 才**动手**（去 GPU 算，见 [模块 03](../03-distributed-parallel/design.md)）。

🔗 **回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)**：例子里 prompt `[11,7,4]` 就是 `Processor` 在前端 tokenize 出来的 token id，打包进 `EngineCoreRequest` 过界送进 EngineCore；最后采样出的 token id=9 又经 `OutputProcessor` detokenize 成文本、流式吐回客户端——这一进一出正是本模块编排的两端。

---

### 1.4 这个模块要解决的问题

一个 LLM 推理服务在"请求编排"层面要同时满足几组互相拉扯的诉求：

1. **高吞吐**：GPU 要尽量满载，不能因为单条请求的 tokenize、detokenize、Python 逻辑而空转。
2. **低延迟 / 流式**：每生成一个 token 要尽快回吐给客户端（SSE 流式），不能等整条序列生成完。
3. **前端不阻塞后端**：HTTP 框架（FastAPI/uvicorn）的 asyncio 事件循环、JSON 序列化、tokenizer 这些 CPU 活，不能挤占 GPU 推理的关键路径。
4. **可扩展到分布式**：同一套编排逻辑要能跑在单卡，也要能跨 TP/PP/DP 多进程多机。

V1 的请求生命周期设计，本质就是围绕"**把 CPU 活和 GPU 活拆开、让 GPU 永不空转**"来组织进程与数据流。

---

## 2. 设计目标与约束

- **GPU 永远有活干**：调度器在每一步都尽量把 token 预算（`max_num_batched_tokens`）填满，prefill 与 decode 混在同一个 batch（见 [模块 01](../01-scheduler-batching/design.md)）。
- **CPU 与 GPU 重叠**：tokenize/detokenize、ZMQ 序列化、Prometheus 统计等 CPU 工作必须能与 GPU forward 并行，不进入关键路径。
- **前后端进程级解耦**：API server 进程崩溃/卡顿不应拖垮推理；推理进程的 GIL 不应被 HTTP 框架抢占。
- **统一的请求状态抽象**：不再区分"prefill 请求"和"decode 请求"，用统一的 `num_computed_tokens` 推进，天然兼容 chunked prefill（见 [模块 01](../01-scheduler-batching/design.md)）、prefix caching（见 [模块 02](../02-paged-attention-kvcache/design.md)）、投机解码（见 [模块 05](../05-speculative-decoding/design.md)）。
- **per-request 流式输出**：每条请求有独立的输出队列，支持 `n>1` 并行采样、提前 abort、客户端断连即取消。

---

## 3. 核心设计思想（含 V0 → V1 对比）

### 3.1 进程模型：EngineCore 进程（V1 最大的结构性改动）

**V0 的痛点**：`LLMEngine` 与 API server 在同一个 Python 进程里。tokenize、detokenize、HTTP 处理、调度、采样后处理全部抢同一个 GIL。在高并发下，CPU 侧的后处理会周期性"卡住"GPU，吞吐和 TTFT 都受损。

**V1 的方案**：把引擎核心拆成一个独立的 **EngineCore 进程**（引擎核心进程，`vllm/v1/engine/core.py`），通过 **ZMQ socket + msgpack** 与前端通信：

```
┌────────────── API server 进程（前端，asyncio）──────────────┐
│  OpenAI server / LLM                                         │
│    ├─ Processor        : 文本→token、多模态预处理（CPU）      │
│    ├─ OutputProcessor  : token→文本 detokenize（CPU）        │
│    └─ EngineCoreClient : ZMQ 收发 + msgpack 序列化            │
└──────────────────────────┬──────────────────────────────────┘
                  input_socket │ ▲ output_socket   (ZMQ, msgpack)
                              ▼ │
┌────────────── EngineCore 进程（后端，纯同步 busy loop）──────┐
│  run_busy_loop:  while True: 取输入 → step()                 │
│    step() = scheduler.schedule()                            │
│             → executor.execute_model()                      │
│             → scheduler.update_from_output()                │
│    ├─ 输入/输出各一个后台线程跑 ZMQ IO（与 GPU 重叠）        │
│    └─ Executor → Worker(s) → GPUModelRunner（GPU forward）   │
└─────────────────────────────────────────────────────────────┘
```

这样：
- **GPU 关键路径独占一个进程**，busy loop 是纯同步代码，没有 asyncio 调度抖动。
- **CPU 活（tokenize/detokenize/序列化）留在前端进程**，且 EngineCore 内部用独立 IO 线程跑 ZMQ（线程在 socket IO 时释放 GIL），实现 **ZMQ IO 与 GPU forward 重叠**。
- 单进程调试场景仍保留 `InprocClient`（`core_client.py`），EngineCore 退化为同进程对象，不走 ZMQ。

> 设计动机出处：vLLM 官方 V1 博客 *"vLLM V1: A Major Upgrade to vLLM's Core Architecture"*（blog.vllm.ai，2025-01）将"isolated EngineCore process + multiprocessing 化前端"列为 V1 的核心目标之一，明确动机是消除 CPU 后处理对 GPU 的阻塞。代码侧对应 `EngineCoreProc`（`core.py:308`）。

#### 进程与线程全景（务必记住的并发结构）

V1 的性能很大程度来自"把不同性质的工作放到不同的执行体里，让它们重叠"。一条请求在飞行途中会穿过下面这些并发实体：

```
进程 1：API server（前端，asyncio 事件循环）
  ├─ 协程：每条请求一个 generate() 消费协程（从自己的队列取→yield）
  └─ 协程：唯一的 output_handler 后台任务（拉 EngineCore 输出→detokenize→分发）
        └─ 任务：process_outputs_socket（ZMQ PULL 收输出，core_client.py:659）

   ↕  input_socket (ROUTER↔DEALER) / output_socket (PUSH→PULL)，msgpack 零拷贝

进程 2：EngineCore（后端，纯同步 busy loop，独占 GPU 关键路径）
  ├─ 主线程：run_busy_loop → step()（schedule / execute / update）
  ├─ IO 线程 A：process_input_socket（收请求、msgpack 解码、入 input_queue）
  └─ IO 线程 B：process_output_socket（出 output_queue、msgpack 编码、发送）

   ↕  collective_rpc，共享内存 MessageQueue 广播（TP/PP 时）

进程 3..N：Worker（每个 TP/PP rank 一个进程）
  └─ worker_busy_loop：从广播队列 dequeue → GPUModelRunner.execute_model（GPU forward）
```

**三层 busy loop**（前端 output_handler、后端 run_busy_loop、worker_busy_loop）各自独立推进，靠队列/ socket 解耦——这是理解 V1 并发模型的总纲。数据并行（DP）则是把"进程 2 + 它的 Worker 群"整组复制多份（`DPEngineCoreProc`），前端用 ROUTER 按 identity 寻址。

### 3.2 全异步前端 + 单一后台 output_handler

前端 `AsyncLLM`（`vllm/v1/engine/async_llm.py`）不为每条请求各开一个轮询循环，而是：

- 每条请求持有一个 `RequestOutputCollector` 队列（`async_llm.py:205`）。
- **全局只有一个**后台任务 `output_handler`（`async_llm.py:340`）不停从 EngineCore 拉 `EngineCoreOutputs`，detokenize 后**按 request_id 分发**到各自队列。
- `generate()` 协程只负责从自己的队列 `get()` 并 `yield`（`async_llm.py:290-298`）。

这把"N 条请求"的输出多路复用收敛成"1 个拉取循环 + N 个队列"，避免 N 个协程各自 await EngineCore 造成的惊群与任务切换开销。`output_handler` 还会把大批输出按 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 切片（`async_llm.py:353`），在切片之间 `await asyncio.sleep(0)` 让出事件循环，避免一次性 detokenize 阻塞太久。

### 3.3 "没有 prefill 阶段，也没有 decode 阶段"的统一调度抽象

V0 的调度器内部区分 prefill batch 和 decode batch，二者分开跑，导致 prefill 长请求会"独占"一整步、阻塞正在 decode 的请求（head-of-line blocking）。

V1 调度器取消了这个区分（`scheduler.py:122-132` 的注释是设计纲领）：

> 每条请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`；每步调度就是给请求分配一些 token，让 `num_computed_tokens` 去追 `num_tokens_with_spec`。

这一个抽象同时覆盖了：
- **Chunked prefill**：长 prompt 一步只算一部分 token（受 `token_budget` 约束）。
- **Decode**：一步算 1 个（或投机解码的 k+1 个）token，本质是 chunked prefill 的退化情形。
- **Prefix caching**：已命中的前缀直接算进 `num_computed_tokens`，跳过计算。
- **混合批**：prefill 的部分 token 和其他请求的 decode token 同处一个 batch，填满 GPU。

详见模块 01。本模块只需知道：**调度的产物是一个 `SchedulerOutput`**，描述"这一步每条请求算多少 token、用哪些 block"。

### 3.4 三段式 step：schedule → execute → update

EngineCore 的最小工作单元是 `step()`（`core.py:195-210`）：

```python
scheduler_output  = self.scheduler.schedule()                       # ① 决定这步算什么
output            = self.model_executor.execute_model(scheduler_output)  # ② GPU forward + sample
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, output)  # ③ 回填状态、检测 stop
```

这三步职责清晰、可独立测试，也是后续 PP（pipeline 并行）做"批队列"异步化的接缝（`step_with_batch_queue`，`core.py:212`）。

### 3.5 分布式执行：Executor 抽象 + MessageQueue 广播

`execute_model` 不直接调模型，而是经 **Executor**（`v1/executor/abstract.py`）下发到一个或多个 **Worker**：

- 单进程 → `UniprocExecutor`；多进程 TP/PP → `MultiprocExecutor`（`multiproc_executor.py`）；Ray → `RayDistributedExecutor`。
- `MultiprocExecutor` 用一个 **共享内存 MessageQueue** 把 `scheduler_output` **广播**给所有 worker（`multiproc_executor.py:173`），各 worker 跑自己那一份模型分片，只有 rank0 把 `ModelRunnerOutput` 回传（`rank0_reply_only`）。

这层抽象让"单卡 / 多卡 / 多机"对上层 `step()` 完全透明。详见 [模块 03](../03-distributed-parallel/design.md)。

---

## 4. 关键数据结构

沿着生命周期，请求在不同阶段换不同的"外壳"，每次跨进程边界都对应一次 msgpack 序列化：

| 数据结构 | 定义位置 | 角色 | 跨进程？ |
|---|---|---|---|
| `PromptType` / `SamplingParams` | `inputs/`、`sampling_params.py` | 用户原始输入 | 否（前端内） |
| **`EngineCoreRequest`** | `v1/engine/__init__.py` | Processor 产出；已 tokenize、含采样参数。**前端→EngineCore 的过界载荷** | ✅ 入向 |
| `Request` | `v1/request.py` | EngineCore 内部的请求状态机（`num_computed_tokens`、`status`、block 等） | 否（后端内） |
| **`SchedulerOutput`** | `v1/core/sched/output.py` | 一步调度的产物：每条请求算多少 token、block 映射、spec token、grammar bitmask 等 | 后端→worker（广播） |
| **`ModelRunnerOutput`** | `v1/outputs.py` | 一步 forward 的产物：采样出的 token id、logprobs | worker rank0→后端 |
| **`EngineCoreOutputs`** | `v1/engine/__init__.py` | `update_from_output` 产物：新 token、finish_reason、调度统计。**EngineCore→前端的过界载荷** | ✅ 出向 |
| `RequestOutput` | `outputs.py` | OutputProcessor detokenize 后，返回给用户的最终对象 | 否（前端内） |

`Request` 的状态机（`v1/request.py` 的 `RequestStatus`）：`WAITING`（含 `WAITING_FOR_FSM`）→ `RUNNING` ⇄ `PREEMPTED` → `FINISHED_*`。状态迁移由调度器驱动（见 [模块 01](../01-scheduler-batching/design.md)）。

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| EngineCore 独立进程 + ZMQ | GPU 不被 CPU 后处理阻塞；前后端容错隔离 | 多一次 msgpack 序列化与 IPC 拷贝；调试链路变长（保留 `InprocClient` 兜底） |
| 单 `output_handler` 多路复用 | 避免 N 协程惊群；输出分发集中可控 | 单点：handler 异常会 `propagate_error` 给所有在途请求（`async_llm.py:387`） |
| 统一 token 推进抽象（无 prefill/decode 之分） | chunked prefill / prefix cache / spec decode 共用一套逻辑 | 调度器单步逻辑更复杂；正确性高度依赖 `num_computed_tokens` 记账 |
| 同步 busy loop 跑 EngineCore | 无 asyncio 抖动，关键路径确定性强 | 输入/输出需额外 IO 线程才能与 GPU 重叠（`core.py:341-348`） |
| msgpack + 共享内存 MQ | 低开销跨进程/跨 worker 传输 | 所有过界类型需可被 `MsgpackEncoder` 序列化（`v1/serial_utils.py`） |

---

## 6. V1 相比 V0 的净收益（小结）

- **吞吐/延迟**：GPU 关键路径与 CPU 后处理进程级隔离，消除周期性卡顿。
- **公平性**：取消 prefill/decode 分离，长 prompt 不再独占整步、阻塞 decode。
- **可维护性**：`schedule / execute / update` 三段式 + Executor 抽象，把"调度逻辑、并行策略、模型执行"解耦，单元更小更易测。
- **可扩展性**：DP（数据并行）通过多个 `DPEngineCoreProc` 实现（`core.py:558`），PP 通过 batch queue 异步化以消除流水线气泡（`core.py:212`），都建立在同一套 step 抽象上。**注意**：V1 的多进程 `MultiprocExecutor` 目前断言 `world_size == tensor_parallel_size`，即多进程后端尚不支持 PP；V1 下的 PP 需经 Ray Executor（详见 [模块 03](../03-distributed-parallel/design.md)）。

> 实现层面的逐行调用链、`file:line` 对照与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 7. 设计背后的考量与历史教训

### 7.1 设计背后的考量（动机与历史）

1. **"独立 EngineCore 进程"否决了"单进程多线程"和"协程化引擎"两条路**。最自然的省事方案是把引擎仍留在 API server 进程，靠线程池或 asyncio 把 CPU 后处理挪开。但只要还在同一个进程，GIL 就让 tokenize/detokenize/序列化和 GPU 调度抢同一把锁，高并发下周期性卡顿无法根除。V1 最终选择**进程级隔离 + ZMQ/msgpack**，宁可多一次 IPC 拷贝，也要让 GPU busy loop 独占一个纯同步进程、没有事件循环抖动。这是"用确定性换灵活性"的典型取舍——也是为什么 EngineCore 内部反而**不用** asyncio，而退回最朴素的 `while True: step()` 加两个 IO 线程。

2. **"单一后台 output_handler 多路复用"是被 N 协程惊群逼出来的**。早期实现倾向于每条请求各开一个轮询协程直接 await EngineCore，但在上千并发下，N 个协程各自唤醒、各自反序列化会造成严重的任务切换开销与延迟长尾。`#15156`（*Simpler request output queues*）把它收敛成"1 个拉取循环 + 每请求一个 `RequestOutputCollector` 队列"，并在切片之间 `await asyncio.sleep(0)` 主动让出——本质是把"多消费者直连"改成"单生产者扇出"，用集中调度换可控延迟。

3. **shutdown / 错误传播被反复打磨，因为"进程拆分"把容错变成了一等问题**。一旦前后端跨进程，任意一端崩溃、ZMQ context 卡在 `term()`、worker 启动即失败等都会变成挂死或难懂的 traceback。`#13298`、`#13869`、`#16137`、`#15367` 这一串提交把"显式关闭 socket→先停 EngineCore→再 term context""启动期失败要让父进程干净退出""把 model_executor 放进后台进程后补全相关 RPC"逐一补齐。教训是：**进程化带来的吞吐收益，必须用同等投入的生命周期/容错代码来兑现**，否则隔离反而制造新的挂死面。

4. **统一 `num_computed_tokens` 推进抽象是为"让一套编排覆盖所有推理模式"而设**。否决的替代是"为 prefill / decode / spec / prefix-cache 各写一条流水线分支"。代价是正确性高度押注在记账上——任何一处少加/多加 token 都会变成静默错误（详见模块 01 的乐观推进回退），但换来的是 chunked prefill、prefix caching、投机解码共用同一段 step 逻辑。

5. **演进脉络：从"单体 LLMEngine"到"可水平复制的 EngineCore 群"**。`#9826`（AsyncLLM 雏形）→ `#9856`（多进程 TP）→ `#13923`（AsyncLLM 数据并行）→ `#15906`（输入队列改用 ZMQ ROUTER/DEALER 为 DP scale-out 铺路）这条线说明：进程模型一旦拆对，DP 就退化为"把 EngineCore + 其 Worker 群整组复制 N 份、前端按 identity 寻址"，几乎不动核心 step 逻辑。这正是当初坚持进程隔离的长期回报。

### 7.2 重要 bug 修复（真实、精选）

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| `#15043` | 重复 `request_id` 导致 EngineCore 崩溃 | 跨进程加请求时，前端无法保证 id 全局唯一；引擎核心必须对"脏输入"鲁棒。修复方向是**去掉 step() 里对 `total_num_scheduled_tokens==0` 的特判 early-return**，让重复 id 的请求走正常清理路径而非触发不变量崩溃 |
| `#14512` | 并行采样（`n>1`）的 finish/abort 记账错乱 | `n>1` 在前端被拆成多个子请求再聚合，父子生命周期与 abort 传播是 OutputProcessor 的薄弱点；说明"per-request 输出队列"抽象在 fan-out 场景下需要额外的聚合层（`parallel_sampling.py`） |
| `#13298` / `#13869` | EngineCoreClient 关停时 ZMQ `context.term()` 挂死 | `context.destroy()` 名义上会关 socket，实际必须**显式先关 socket、再 term**，否则卡死。进程化的代价：关停顺序成为正确性的一部分 |
| `#16137` | EngineCore 启动期失败时父进程不退出、空等 | 后台进程的**启动失败**必须显式上报前端并触发干净退出，否则表现为静默挂起——拆进程后"失败也要能传回来" |
| `#15367` | 把 `model_executor` 放进后台进程后，依赖它的功能（如 sharded state 存取）失效 | 进程边界把"原本同进程可直接调用的对象"切断了，需要补 `collective_rpc` 一类的过界通道——重构进程模型时要系统排查所有跨界调用 |
| `#9629` | 请求 abort 后资源未清理 | 客户端断连/提前取消是常态而非异常；EngineCore 必须把 abort 当一等事件，主动清 `Request` 状态与缓存，否则泄漏累积 |
