# 模块 00 · 请求全生命周期 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的端到端调用链。所有 `file:line` 基于仓库当前版本，行号若与你本地略有出入，以函数名/逻辑为准。
> 阅读方式：左侧是调用链的"主干步骤"，每步标注 `文件:行号 函数名`，并摘录关键代码与解读。

---

## 1. 代码地图（参与请求生命周期的文件）

```
入口层 entrypoints/
  ├─ llm.py                      # offline：LLM.generate（同步）
  └─ openai/serving_completion.py# online：调 engine_client.generate（异步）

前端引擎层 v1/engine/
  ├─ async_llm.py                # AsyncLLM：异步前端 + output_handler 后台 loop
  ├─ llm_engine.py               # LLMEngine：同步前端（offline 用）
  ├─ processor.py                # Processor：Inputs → EngineCoreRequest（tokenize/校验/多模态）
  ├─ output_processor.py         # OutputProcessor：EngineCoreOutputs → RequestOutput（detokenize）
  ├─ core_client.py              # EngineCoreClient：ZMQ + msgpack IPC（前端侧）
  └─ core.py                     # EngineCore / EngineCoreProc：EngineCore 进程 busy loop + step()

调度层 v1/core/sched/
  └─ scheduler.py                # Scheduler：schedule() / update_from_output()

执行层 v1/executor/
  ├─ abstract.py                 # Executor 抽象
  └─ multiproc_executor.py       # MultiprocExecutor：MessageQueue 广播到 workers

Worker 层 v1/worker/
  ├─ gpu_worker.py               # Worker：execute_model 转发
  └─ gpu_model_runner.py         # GPUModelRunner：真正的 GPU forward + sample
```

---

## 2. 端到端调用链（一条请求的完整旅程）

下面以 **online（OpenAI server，异步）** 为主线；offline 差异在 §3 单列。

### 阶段 A：入口 —— 接收请求

**A1.** `vllm/entrypoints/openai/serving_completion.py:162`
HTTP handler 对每个 prompt 调用引擎客户端并拿到一个异步生成器：
```python
generator = self.engine_client.generate(engine_prompt, sampling_params, request_id, ...)
```
`engine_client` 即 `AsyncLLM`（实现了 `EngineClient` 协议）。

---

### 阶段 B：前端 —— 输入处理与入队（API server 进程内）

**B1.** `vllm/v1/engine/async_llm.py:246` `AsyncLLM.generate()`
- `:275` 首次调用时启动后台 `output_handler`（懒启动，便于 OpenAI server 优雅处理启动失败）。
- `:277` `q = await self.add_request(...)` 拿到该请求的输出队列。
- `:290-298` 进入消费循环：`out = q.get_nowait() or await q.get()` → `yield out`，直到 `out.finished`。
  > `get_nowait()` 优先：高负载下尽量不 await，减少任务切换（`:292` 注释）。
- `:302` 客户端断连会触发 `asyncio.CancelledError` → `await self.abort(request_id)`。

**B2.** `async_llm.py:185` `AsyncLLM.add_request()`
- `:205` 建立 per-request 队列 `RequestOutputCollector(output_kind=...)`。
- `:208` **调 `Processor.process_inputs(...)` 把用户输入转成 `EngineCoreRequest`**（见 B3）。
- `:214-226` 处理 `n>1`：用 `ParentRequest` fan-out 出 n 个 child request，逐个 `_add_request`。

**B3.** `vllm/v1/engine/processor.py:194` `Processor.process_inputs()` —— **CPU 重活，在前端进程做**
- `:208-215` 参数校验：LoRA、采样参数（`_validate_params`，含 logprobs/logit_bias/structured output 合法性，`:58-192`）。
- `:225` `self.input_preprocessor.preprocess(...)`：**tokenize 文本**；多模态模型在此预处理图像/音频并把 prompt token 扩展到含占位符。
- `:241` `split_enc_dec_inputs`：拆 encoder/decoder（V1 暂不支持 encoder-decoder，`:244` 直接 `NotImplementedError`）。
- `:251-258` 补全采样参数：`max_tokens` 缺省填到 `max_model_len - len(prompt)`；并从 generation_config / tokenizer 合并 eos 等。
- `:264-306` 多模态：合并并排序 mm 占位符/hash/kwargs，命中 `mm_input_cache` 则复用。
- `:308` **产出 `EngineCoreRequest`**（已 tokenize 的 prompt_token_ids + 采样参数 + mm 输入）。

**B4.** `async_llm.py:228` `AsyncLLM._add_request()`
- `:233` `self.output_processor.add_request(...)`：在**前端**登记该请求（detokenizer 状态、parent/child 关系、输出队列）。
- `:236` `await self.engine_core.add_request_async(request)`：**把 `EngineCoreRequest` 推过 ZMQ 送进 EngineCore 进程**（见阶段 C）。

---

### 阶段 C：过界 —— ZMQ + msgpack 把请求送进 EngineCore 进程

**C1.** `vllm/v1/engine/core_client.py:730` `AsyncMPClient.add_request_async()`
将请求 msgpack 编码后，经 `input_socket.send_multipart` 发送（`:712` `_send_input`）。请求类型枚举为 `EngineCoreRequestType.ADD`。

**C2.** `vllm/v1/engine/core.py:502` `EngineCoreProc.process_input_socket()`（**EngineCore 进程的输入 IO 线程**）
- `:506` 用 `MsgpackDecoder(EngineCoreRequest)` 解码。
- `:520-521` `recv_multipart` 收到 `(type_frame, *data_frames)`，解析出 `request_type`。
- `:530` `self.input_queue.put_nowait((request_type, request))` 把请求塞进进程内队列。
  > 关键：socket IO 在独立线程（`core.py:341`），收发时释放 GIL，与主 busy loop 的 GPU forward 重叠。

---

### 阶段 D：后端 —— EngineCore busy loop

**D1.** `core.py:403` `EngineCoreProc.run_busy_loop()`（后端主循环，纯同步）
```python
while True:
    self._process_input_queue()   # 取输入：没活时阻塞等，有活时排空队列
    self._process_engine_step()   # 干活：step() 并把输出塞进 output_queue
```
- `:413` `_process_input_queue`：无请求时 `input_queue.get()` 阻塞；有请求时把 ADD/ABORT/UTILITY 等分派给 `_handle_client_request`（`:445`）。
- `:449-450` `EngineCoreRequestType.ADD` → `self.add_request(request)`。

**D2.** `core.py:166` `EngineCore.add_request()`
- `:169-177` 若有多模态 hash，从 server 侧 `mm_input_cache_server` 取/存。
- `:179` `Request.from_engine_core_request(request)`：把过界的 `EngineCoreRequest` 转成内部状态机对象 `Request`。
- `:180-182` 若用结构化输出，异步启动 grammar 编译（`structured_output_manager.grammar_init`，见 [模块 06](../06-sampling-structured-output/design.md)）。
- `:184` `self.scheduler.add_request(req)` → `scheduler.py:695`：**追加到 `waiting` deque**，登记进 `self.requests`。

**D3.** `core.py:436` `_process_engine_step()` → `core.py:195` `EngineCore.step()`（**三段式**）
```python
scheduler_output    = self.scheduler.schedule()                          # ① 调度
output              = self.model_executor.execute_model(scheduler_output) # ② 执行
engine_core_outputs = self.scheduler.update_from_output(...)             # ③ 回填
```

---

### 阶段 E：① 调度 —— 决定这一步算什么

**E1.** `vllm/v1/core/sched/scheduler.py:122` `Scheduler.schedule()`
- `:161-260` **先调度 RUNNING 队列**：每条请求 `num_new_tokens = num_tokens_with_spec - num_computed_tokens`，受 `token_budget`（=`max_num_batched_tokens`）与 `long_prefill_token_threshold` 截断 → 即 **chunked prefill**。
  - `:197` `kv_cache_manager.allocate_slots(...)` 分配 block；返回 `None` 表示显存不足 → `:204` **抢占**最低优先级请求（`running.pop()` → `free` → 回 `waiting`）。
- `:274-373` **再调度 WAITING 队列**（仅当本步无抢占时）：
  - `:305` `get_computed_blocks(request)` 查 **prefix cache** 命中的前缀（见 [模块 02](../02-paged-attention-kvcache/design.md)），命中部分直接计入 `num_computed_tokens`。
  - `:332` `allocate_slots` 分配新 block；放不下就 `break`。
  - `:348-363` 状态从 `WAITING`/`PREEMPTED` 迁到 `RUNNING`，记账 `token_budget`。
- `:404-444` 打包 `SchedulerOutput`：`scheduled_new_reqs` / `scheduled_cached_reqs` / `num_scheduled_tokens` / `scheduled_spec_decode_tokens` / `grammar_bitmask` / block 映射等。
- `:455-456` **关键记账**：调度后立即 `num_computed_tokens += num_scheduled_token`，使下一步能立刻继续推进该请求（若后续 token 被 reject，再在 update 阶段回退）。

> 调度算法的完整细节（预算分配、抢占策略、cascade attention 公共前缀）见 [模块 01](../01-scheduler-batching/design.md)。

---

### 阶段 F：② 执行 —— GPU forward + 采样

**F1.** `core.py:206` `self.model_executor.execute_model(scheduler_output)`

**F2.** `vllm/v1/executor/multiproc_executor.py:142` `MultiprocExecutor.execute_model()`
- `:146` `collective_rpc("execute_model", args=(scheduler_output,), rank0_reply_only=True)`。
- `:173` `self.rpc_broadcast_mq.enqueue(...)`：把 `(method, args, kwargs, rank0_only)` 经**共享内存 MessageQueue 广播**给所有 worker。
- `:178-189` 仅等待 rank0 的回复（`rank0_reply_only`），其余 rank 各自算分片不回传。
  > 单卡时是 `UniprocExecutor`，直接本进程调用，无广播。

**F3.** worker 侧 `multiproc_executor.py:458` `WorkerProc` 循环 `dequeue` 到 `execute_model` 后转发：
`vllm/v1/worker/gpu_worker.py:237` `Worker.execute_model()` → `:242` `self.model_runner.execute_model(scheduler_output)`，且 `:243` 仅 driver(rank0) 返回 output。

**F4.** `vllm/v1/worker/gpu_model_runner.py:987` `GPUModelRunner.execute_model()` —— **真正的 GPU 工作**
- `:992` `_update_states(scheduler_output)`：把调度结果同步进 `InputBatch`（增删请求、更新 block table、token）。
- `:993-995` 若本步没有 token 要算（如仅 flush finished），返回 `EMPTY_MODEL_RUNNER_OUTPUT`。
- `:998` `_prepare_inputs(...)`：构造 `attn_metadata`、`logits_indices`、`spec_decode_metadata`，并可能 reorder batch。
- `:1001-1010` 若启用 CUDA graph 且 token 数在捕获范围内，按 `pad_for_cudagraph` 把 batch 补齐到捕获尺寸（见 [模块 07](../07-cuda-graph-compile/design.md)）。
- `:1014-1045` 多模态：跑 encoder、聚合 mm embedding；文本模型直接用 token id（保留 embedding 层在 CUDA graph 内）。
- `:1062-1068` **`with set_forward_context(attn_metadata, ...)`: `hidden_states = self.model(...)`** —— 模型前向。`forward_context` 把 attention 元数据传给各层 attention backend（见 [模块 02](../02-paged-attention-kvcache/design.md) / [模块 07](../07-cuda-graph-compile/design.md)）。
- `:1069-1071` PP 非末段 rank 直接返回 hidden_states（中间张量经 PP 通信传给下一段）。
- `:1075` `compute_logits`；`:1078-1079` 若有 `grammar_bitmask` 则应用结构化输出掩码（[模块 06](../06-sampling-structured-output/design.md)）。
- `:1082-1096` `self.model.sample(logits, sampling_metadata)`：采样下一个 token（含投机解码的 bonus token 处理，见 [模块 05](../05-speculative-decoding/design.md) / [模块 06](../06-sampling-structured-output/design.md)）。
- 最终打包 `ModelRunnerOutput`（采样 token id、logprobs 等）回传 rank0 → Executor → EngineCore。

---

### 阶段 G：③ 回填 —— 状态更新与 stop 检测

**G1.** `scheduler.py:569` `Scheduler.update_from_output(scheduler_output, model_runner_output)`
- `:585` 遍历 `self.running`（注释 `:582` 提醒此循环可能上千次，避免重操作）。
- `:594` 取出该请求本步采样出的 `generated_token_ids`。
- `:596-611` **投机解码回退**：若有 spec token 被 reject，按 `len(spec)+1-len(generated)` 回退 `num_computed_tokens`，并记 spec 统计。
- `:638-647` 逐个 append 输出 token，每个都 `check_stop(request, max_model_len)`（`sched/utils.py`）检测 EOS / stop string / max_tokens；命中即 `_free_request`（释放 KV/encoder cache）。
- `:655-660` 结构化输出：把新 token 喂回 grammar 推进 FSM。
- `:664-674` 为有新 token 的请求生成 `EngineCoreOutput`（new_token_ids、finish_reason、logprobs、stop_reason）。
- `:683-687` 更新 `self.running`，打包成 `EngineCoreOutputs`（含 `scheduler_stats`）返回。

**G2.** `core.py:442-443` `_process_engine_step`：`self.output_queue.put_nowait(outputs)`。

---

### 阶段 H：过界回程 —— 输出经 ZMQ 回到前端

**H1.** `core.py:532` `EngineCoreProc.process_output_socket()`（**输出 IO 线程**）
- `:545-552` 从 `output_queue` 取 `EngineCoreOutputs`，msgpack `encode_into`，经 `output_socket`（ZMQ PUSH）`send_multipart`。

**H2.** `core_client.py:655-686` `AsyncMPClient`：后台从 `output_socket.recv_multipart` 收帧、解码，`get_output_async()` 返回 `EngineCoreOutputs`。

---

### 阶段 I：前端 —— detokenize 与分发回客户端

**I1.** `vllm/v1/engine/async_llm.py:340` `output_handler()`（**全局唯一后台 loop**）
- `:344` `outputs = await engine_core.get_output_async()` 拉一批输出。
- `:353-358` 按 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 切片，避免一次 detokenize 阻塞事件循环太久。
- `:362` `output_processor.process_outputs(slice, ...)`：**detokenize**，并把结果 `RequestOutput` **push 进每条请求的 `RequestOutputCollector` 队列**（按 request_id 分发）。
- `:369` 切片之间 `await asyncio.sleep(0)` 让出事件循环。
- `:372` `await engine_core.abort_requests_async(reqs_to_abort)`：因 stop string 等需中止的请求，反向通知 EngineCore。
- `:378-384` 记录 Prometheus/日志统计。

**I2.** 回到 **B1** 的 `generate()` 消费循环：`q.get()` 拿到 detokenize 后的 `RequestOutput` → `yield` → OpenAI server 转成 SSE chunk 发回客户端。`out.finished=True` 时循环结束，请求生命周期完成。

---

## 3. offline（同步 `LLM`）路径差异

`vllm/entrypoints/llm.py` 的 `LLM.generate()`：

- **B/C 阶段**：`llm.py:462` `_validate_and_add_requests` → 内部对每个 prompt 调 `LLMEngine.add_request`（`v1/engine/llm_engine.py:174`），同样经 `Processor.process_inputs` 产出 `EngineCoreRequest` 后 `engine_core.add_request`（`llm_engine.py:198`）。
- **驱动方式**：`llm.py:470` `_run_engine` 是一个**同步循环**，反复调 `LLMEngine.step()`（`llm_engine.py:214`）：
  ```python
  outputs = self.engine_core.get_output()      # 同步拉
  ... output_processor.process_outputs(...)    # detokenize
  self.engine_core.abort_requests(...)         # 处理 stop
  ```
- **IPC 形态**：offline 默认仍可用多进程 EngineCore（`SyncMPClient`，`core_client.py:472`）；若 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 等场景，则用 `InprocClient`（`core_client.py:185`）把 EngineCore 退化为同进程对象，`step()` 直接本地调用，无 ZMQ。

一句话：offline 把"前端异步 output_handler"换成"主线程同步 step 循环"，其余 `process_inputs → EngineCore.step → process_outputs` 三段完全一致。

---

## 4. 关键边界情况与容错

| 情况 | 代码位置 | 行为 |
|---|---|---|
| 客户端断连 | `async_llm.py:302` | `generate()` 收到 `CancelledError` → `abort(request_id)` → 通知 OutputProcessor + EngineCore 释放 |
| stop string 触发的中止 | `async_llm.py:371-373` | output_handler 收集 `reqs_to_abort`，反向 `abort_requests_async` |
| EngineCore 进程死亡 | `core.py:490` `_send_engine_dead` → 前端 `engine_dead` | `AsyncLLM.errored=True`，后续请求抛 `EngineDeadError`（`async_llm.py:198`） |
| output_handler 自身异常 | `async_llm.py:385-387` | `propagate_error(e)`：把异常广播给所有在途请求队列 |
| 显存不足 | `scheduler.py:201-217` | 抢占最低优先级请求（回退到 `waiting`，`num_computed_tokens=0`），下次重新 prefill |
| 本步无 token 可算 | `gpu_model_runner.py:993` | 返回 `EMPTY_MODEL_RUNNER_OUTPUT`，跳过 forward |
| PP 中间段 rank | `gpu_model_runner.py:1069` | 返回 hidden_states 而非采样结果，交给下一段 |
| 结构化输出 grammar 未编译完 | `scheduler.py:284-291` | 请求处于 `WAITING_FOR_FSM`，本步跳过，待编译完成再调度 |

---

## 5. 实现中的精妙之处（Tricks 与非显然设计）

> 这一节是"心中有数"的关键。下面每条都是**读代码时容易一眼滑过、但理解了才算真懂 vLLM** 的点。格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · ZMQ socket 拓扑：为什么是 ROUTER/DEALER + PUSH/PULL，还有 READY 握手

- **现象**：前后端两条 socket，方向和类型刻意选得不一样。
- **代码**：前端 `core_client.py:380` `input_socket = make_zmq_socket(..., zmq.ROUTER, bind=True)`；后端 `core.py:510` `zmq.DEALER, identity=engine_index`；输出侧后端 `core.py:542` `zmq.PUSH, linger=4000`，前端 `core_client.py:655` `zmq.PULL`。
- **精妙之处**：
  - **输入用 ROUTER↔DEALER 而非 PUSH/PULL**：因为 DP（数据并行）下有多个 EngineCore 进程，前端要能**按 identity 精确寻址**到某个引擎——`_send_input_message`（`core_client.py:711`）在消息最前面塞 `engine.identity`，ROUTER 据此路由。单引擎只是这套寻址的退化情形。
  - **输出用 PUSH/PULL**：输出是**单向流**，EngineCore 源源不断推，前端不需要逐条回 ACK，PUSH/PULL 比 REQ/REP 省一个往返。
  - **`linger=4000`**：保证进程退出前，`ENGINE_CORE_DEAD` 这条"遗言"能发出去（`core.py:494`），让前端把卡住的请求标记为 `engine_dead` 而不是永久 hang。
  - **READY 握手**：每个 EngineCore 启动后先发 `b'READY'`（`core.py:516`），前端 `_wait_for_engine_startup`（`core_client.py:403`）**同时 poll socket 和子进程句柄**——这样进程若在发 READY 前就崩了，能立刻报错而非死等（`:419`）。

### T2 · msgpack 零拷贝：大张量不进 JSON，而是走独立 ZMQ frame

- **现象**：跨进程传 tensor/ndarray（KV 相关、多模态、采样结果）若走常规序列化会产生大量内存拷贝。
- **代码**：`vllm/v1/serial_utils.py` 的 `MsgpackEncoder`。
- **精妙之处**：
  - 编码时维护一个 `aux_buffers` 列表（`:57`）。小数组（`< VLLM_MSGPACK_ZERO_COPY_THRESHOLD`，默认 256B）**内联**进主 buffer（`CUSTOM_TYPE_RAW_VIEW`，`:125`）；大数组**只把"第几个 buffer"的索引写进主消息**，真实数据 append 到 `aux_buffers`（`:128-129`）。
  - 返回的是 `bufs` 列表，正好对应 ZMQ 的 **multipart frames**——每个大张量是一帧，**全程零拷贝**直达 socket。
  - 解码时 `np.ndarray(buffer=self.aux_buffers[data], ...)`（`:204-206`）直接在收到的帧上**建视图**，`torch.from_numpy` 再共享同一块内存。
  - 输出线程用 `encode_into(outputs, buffer)`（`core.py:551`）**复用同一个 `bytearray`**（`core.py:538` `buffer = bytearray()`），避免每步重新分配主 buffer。
  - 代价：`aux_buffers` 是实例状态，所以 encoder/decoder **非线程安全**（`:44` 注释明示），每个 IO 线程各持一个。

### T3 · IO 线程与 GPU 重叠：busy loop 是同步的，但 socket IO 在另开的线程

- **现象**：EngineCore 是纯同步 busy loop（无 asyncio），但 ZMQ 收发不会阻塞 GPU。
- **代码**：`core.py:341-348` 起两个 daemon 线程跑 `process_input_socket` / `process_output_socket`；主线程跑 `run_busy_loop`（`:403`）。
- **精妙之处**：Python 的 socket `recv/send` 在 IO 时**释放 GIL**。于是"输入线程解码下一批请求""输出线程编码并发送上一批结果"可以和主线程的"GPU forward launch + Python 调度逻辑"**真正并行**。三者通过两个进程内 `queue.Queue` 解耦（`input_queue` / `output_queue`），这是 V1 把"序列化/反序列化"挪出关键路径的关键一招。

### T4 · 调度的"双重记账"：先把 token 记成已算，错了再退回

- **现象**：`schedule()` 还没真正 forward，就把 `num_computed_tokens` 加上去了。
- **代码**：`scheduler.py:455-456` 调度后立即 `num_computed_tokens += num_scheduled_token`；`scheduler.py:605-607` 在 `update_from_output` 里按拒绝数回退。
- **精妙之处**：提前推进，使得**下一步 `schedule()` 能立刻继续这条请求的下一个 chunk**（chunked prefill 连续推进 / PP 流水线提前排下一批），不必等本步结果回来。投机解码若有 token 被 reject，再用 `len(spec)+1 - len(generated)` 精确回退（`:605`）。这套"乐观推进 + 事后纠正"是 V1 调度能和执行解耦、支持流水线的基础。

### T5 · 抢占是个会"自环退出"的 while 循环

- **现象**：显存不足时要抢占，但抢占谁、抢到什么时候停，藏着一个边界。
- **代码**：`scheduler.py:196-223`。
- **精妙之处**：`while True` 里不断 `self.running.pop()` 抢占**最低优先级**请求并 `free` 其 KV，直到 `allocate_slots` 成功。关键边界在 `:214` `if preempted_req == request:`——当被抢占的恰好是**当前正想调度的这条请求自己**时，说明"已经没有更低优先级的可抢了"，此时 `can_schedule=False` 跳出，放弃调度它。少了这个自环判断就会死循环或越界。

### T6 · `get_nowait() or await get()`：高负载下少一次任务切换

- **代码**：`async_llm.py:293` `out = q.get_nowait() or await q.get()`。
- **精妙之处**：队列里已有数据时走**同步**的 `get_nowait()`，避免 `await` 触发的协程挂起/恢复；只有队列空了才真正 `await`。高 QPS 下每条请求每个 token 都省一次事件循环调度，累积收益可观。

### T7 · 刻意打破循环引用，否则后台任务永不被 GC

- **代码**：`async_llm.py:333-338`（output_handler 闭包只捕获 `engine_core`/`output_processor` 等**局部变量**而非 `self`）；`core_client.py:653` `_self_ref = weakref.ref(self)`。
- **精妙之处**：后台 asyncio 任务若直接引用 `self`（AsyncLLM / client），会和对象形成循环引用，导致即使外部不再持有引擎、它也不会被回收、后台 loop 永不退出。这里用"只引用局部变量 + weakref"让引擎可被正常 GC，从而 `shutdown` 时任务能干净结束。`_run_output_handler` 注释（`:333`）专门点明了这一点。

### T8 · 持久化 InputBatch：每步只打"增量补丁"，不重建

- **现象**：每个调度步都要把请求状态搬上 GPU，若每步重建整个 batch 会很慢。
- **代码**：`gpu_model_runner.py:275` `_update_states`；`:493-548` 的向量化下标计算；`:559` `copy_(..., non_blocking=True)`。
- **精妙之处**：
  - **增量维护**：只移除 finished / 未调度的请求、追加 new 请求（`:286-349`），其余原地保留。注释（`:317-320`）坦承这是"假设相邻批次请求高度重叠"的优化——若负载在两组请求间反复横跳，这个优化会退化。
  - **纯 numpy 向量化构造输入**：用 `np.repeat` / `np.cumsum` 一次算出所有 token 的 `req_indices`、`positions`、`slot_mapping`（`:493-548`），避免 Python 逐请求循环。注释（`:527`）还特意选 `torch.index_select` 而非 `np.take`（大张量更快）。
  - **pinned CPU 缓冲 + 异步 H2D**：先写进固定地址的 CPU staging buffer，再 `non_blocking` 拷到 GPU，与后续 CPU 计算重叠。
  - 这些持久 buffer 同时是 **CUDA graph 能 replay 的前提**（见 T14）。

### T9 · 复用 `CachedRequestData` 对象，省掉每步的小对象分配

- **代码**：`scheduler.py:469-487` `_make_cached_request_data`，命中 `self._cached_reqs_data` 就原地改字段而非新建。
- **精妙之处**：running 队列可能上千条，每步都为每条新建一个 dataclass 会制造大量 GC 压力。这里把对象按 req_id 缓存复用，只更新 `new_token_ids` / `new_block_ids` 等少数字段。`update_from_output` 的循环注释（`:582`）也反复强调"这个循环是热点，避免里面做重操作"。

### T10 · PP 用"批队列"消除流水线气泡

- **现象**：朴素实现里，调度→执行→更新串行，PP 各段会有气泡（bubble）。
- **代码**：`core.py:212` `step_with_batch_queue`，配 `self.batch_queue`（`core.py:116-121`，大小 = `max_concurrent_batches`）。
- **精妙之处**：只要有未调度请求且队列没满，就**先调度并提交下一批**（拿到一个 `Future`），**不立即等结果**；只有当"无新批可调度"或"队列已满"时，才 `future.result()` 阻塞取最早一批的结果再 `update_from_output`。这样 GPU 在 PP 各 stage 间始终有活，把气泡填上。注释（`:216-225`）写得很清楚。

### T11 · DP：finish 同步每 16 步才 all-reduce，空闲引擎跑 dummy batch

- **现象**：多个 EngineCore 数据并行时，要知道"是否全局都没活了"才能一起暂停，但每步都做集合通信太贵。
- **代码**：`core.py:651-660` `_has_global_unfinished_reqs`（计数到 16 才真正 all-reduce）；`core.py:641` 空闲时 `execute_dummy_batch()`；`core.py:649` 全空闲时发 `ENGINE_PAUSED_OUTPUTS`。
- **精妙之处**：
  - **摊销集合通信**：用一个 `counter` 每 16 步才 `has_unfinished_dp` 一次（`:654-657`），其余步乐观地认为"还有活"。
  - **dummy forward 保持步调一致**：某个 DP rank 本地没请求、但别的 rank 还在跑时，它必须跑一个**空 forward**（`:641`），否则它不参与那些隐含的集合通信（如某些 all-reduce）会让其他 rank 卡死。这是数据并行里非常隐蔽但致命的同步点。

### T12 · `n>1` 并行采样：最后一个 child 直接复用父请求对象

- **代码**：`async_llm.py:222` `child_request = request if idx == params.n - 1 else copy(request)`。
- **精妙之处**：fan-out 出 n 个 child 时，前 n-1 个 `copy()`，**最后一个直接复用原 `request` 对象**（反正原对象之后不再用），省一次深拷贝。小处见功夫。

### T13 · 多模态/文本都不重复过界：镜像缓存 + 把 prompt 文本置空

- **现象**：图像张量很大，原始 prompt 文本在后端用不到，都不该反复走 IPC。
- **代码**：`processor.py:303` `get_and_update_p0`（前端侧）与 `core.py:176` `get_and_update_p1`（后端侧）构成**镜像缓存**；`core_client.py:733` `request.prompt = None`。
- **精妙之处**：
  - 前后端各持一份按 hash 索引的 mm 缓存，且**保持镜像**。一旦某图像已缓存，后续请求**只把 hash 过界**，后端凭 hash 命中本地缓存还原，避免重复传大张量（`core.py:169-177` 注释明示"有 hash 必有 HIT"）。
  - `add_request_async` 发送前把 `request.prompt = None`（`core_client.py:731-733`）——文本已 tokenize 成 `prompt_token_ids`，原文后端用不上，置空省带宽。

### T14 · CUDA graph：padding 到捕获尺寸 + 在持久 buffer 上 replay

- **代码**：`gpu_model_runner.py:1001-1010`，`num_input_tokens = self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)`。
- **精妙之处**：CUDA graph 只能 replay 固定 shape。于是把实际 token 数**向上 pad 到最近的捕获尺寸**，并始终在 T8 那些**固定地址的持久 buffer**（`self.input_ids[:num_input_tokens]` 等）上跑，graph 才能稳定 replay。文本模型刻意用 token id 而非 embedding 作输入（`:1036-1040` 注释），就是为了把 embedding 层也纳入 graph。详见 [模块 07](../07-cuda-graph-compile/design.md)。

### T15 · Utility RPC 的参数类型自动纠偏

- **代码**：`core.py:476` `_convert_msgspec_args`。
- **精妙之处**：前端通过通用 utility 通道调后端任意方法时，参数经过 msgpack 往返可能丢了精确类型。这里按目标方法的签名注解，把需要的参数**自动 `msgspec.convert` 回原类型**，让"远程方法调用"用起来像本地调用。

---

## 6. 一图速查：调用链主干

```
[OpenAI server] serving_completion.py:162  engine_client.generate()
      │
[前端] AsyncLLM.generate :246 ─► add_request :185 ─► Processor.process_inputs :194  (tokenize → EngineCoreRequest)
      │                                         └─► _add_request :228 ─► OutputProcessor.add_request
      │                                                              └─► engine_core.add_request_async  (ZMQ/msgpack)
      ▼ ───────────────────────────── 进程边界 ─────────────────────────────
[后端] EngineCoreProc.process_input_socket :502  (IO 线程, 解码入队)
       run_busy_loop :403 ─► _process_input_queue :413 ─► add_request :166 ─► scheduler.add_request (→ waiting)
                          └─► _process_engine_step :436 ─► step :195
                                   ① scheduler.schedule :122        ─► SchedulerOutput
                                   ② executor.execute_model         ─► MultiprocExecutor :142
                                        └─ rpc_broadcast_mq 广播 ─► Worker.execute_model :237
                                              └─ GPUModelRunner.execute_model :987  (forward + sample) ─► ModelRunnerOutput
                                   ③ scheduler.update_from_output :569  (append token / check_stop) ─► EngineCoreOutputs
                                   └─► output_queue
       process_output_socket :532  (IO 线程, 编码发送)
      ▲ ───────────────────────────── 进程边界 ─────────────────────────────
[前端] AsyncLLM.output_handler :340  get_output_async ─► OutputProcessor.process_outputs (detokenize)
                                    └─► 按 request_id push 到 RequestOutputCollector 队列
       generate :290  q.get() ─► yield RequestOutput ─► SSE 回客户端
```
