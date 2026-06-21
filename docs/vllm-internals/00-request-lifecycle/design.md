# 模块 00 · 请求全生命周期 —— 设计文档

> 范围：一条请求从进入 vLLM（offline `LLM.generate` 或 online OpenAI server）到逐 token 返回的**端到端主干**。本模块是后续所有专题模块（调度、KV、并行、采样……）的"骨架"，这里只讲整体编排与进程模型，专题细节留给对应模块。
> 架构：**V1 为主，关键处对比 V0**。

---

## 1. 这个模块解决什么问题

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
