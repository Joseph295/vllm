# 模块 01 · 调度器：Continuous Batching + Chunked Prefill —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的调度调用链。所有 `file:line` 基于仓库当前版本（行号若与本地略有出入，以函数名/逻辑为准）。
> 主战场是单文件 `vllm/v1/core/sched/scheduler.py`，约 780 行——这正是 V1 统一抽象带来的简洁。

---

## 1. 代码地图

```
调度核心 v1/core/sched/
  ├─ scheduler.py        # Scheduler：schedule() / update_from_output() —— 本模块主体
  ├─ output.py           # SchedulerOutput / NewRequestData / CachedRequestData（调度产物的数据壳）
  ├─ interface.py        # SchedulerInterface 抽象基类（schedule/update/add_request 契约）
  └─ utils.py            # check_stop()：EOS / stop token / max_tokens / max_model_len 检测

请求状态机 v1/request.py
  ├─ Request             # num_computed_tokens / num_tokens_with_spec / spec_token_ids / status
  └─ RequestStatus       # WAITING / WAITING_FOR_FSM / RUNNING / PREEMPTED / FINISHED_*

依赖的 manager（细节见对应模块）
  ├─ v1/core/kv_cache_manager.py       # allocate_slots / get_computed_blocks /
  │                                      get_num_common_prefix_blocks（[模块 02](../02-paged-attention-kvcache/design.md)）
  └─ v1/core/encoder_cache_manager.py  # 多模态 encoder 预算与缓存（[模块 10](../10-multimodal/design.md)）

配置 vllm/config.py
  └─ SchedulerConfig:1774  # max_num_batched_tokens / max_num_seqs /
                           # long_prefill_token_threshold / enable_chunked_prefill /
                           # disable_chunked_mm_input / policy
驱动方
  └─ v1/engine/core.py:195 step() / :212 step_with_batch_queue()  # 反复调 schedule→execute→update

V0 对照（理解重构动机用）
  └─ vllm/core/scheduler.py  # 三队列 + 两套策略 + SchedulingBudget + swap
```

---

## 2. 端到端调用链（一步调度的完整旅程）

`EngineCore.step()`（`core.py:205-208`）把调度夹在执行前后：

```python
scheduler_output    = self.scheduler.schedule()                          # ① 本文 §2.A~§2.F
output              = self.model_executor.execute_model(scheduler_output) # ② 执行（[模块 00](../00-request-lifecycle/impl.md) / [模块 03](../03-distributed-parallel/design.md)）
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, output)  # ③ 本文 §2.G
```

### 阶段 A：初始化预算与累加器

**A1.** `scheduler.py:122` `Scheduler.schedule()` 开头先备好本步的累加器与预算：
```python
token_budget = self.max_num_scheduled_tokens          # :149  = max_num_batched_tokens
encoder_budget = self.max_num_encoder_input_tokens    # :152  多模态 encoder 预算
scheduled_new_reqs / scheduled_resumed_reqs / scheduled_running_reqs / preempted_reqs = []
num_scheduled_tokens: dict[str, int] = {}             # :148  req_id -> 本步算多少 token
```
> `max_num_scheduled_tokens` 在 `__init__` 绑定到 `scheduler_config.max_num_batched_tokens`（`scheduler.py:63-64`）；`max_num_running_reqs` 绑定到 `max_num_seqs`（`:62`）。这两个就是 batch 的"双重约束"。

---

### 阶段 B：① 调度 RUNNING 队列（出 token 优先 + chunked prefill 推进）

**B1.** `scheduler.py:161` 主循环 `while req_index < len(self.running) and token_budget > 0:`
- `:163-166` 若该请求已在 `scheduled_req_ids`（比如这一步早先已被处理），`req_index += 1` 跳过。

**B2.** `scheduler.py:168-175` 计算本步给它算多少 token —— **统一抽象的那一行**：
```python
num_new_tokens = (request.num_tokens_with_spec - request.num_computed_tokens)
if (0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens):
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold   # 长 prompt 单步截断
num_new_tokens = min(num_new_tokens, token_budget)                        # 预算截断 → chunked prefill
assert num_new_tokens > 0
```
> 解读：decode 请求这里 `num_new_tokens` 通常 = 1（或 spec 的 k+1）；正在 prefill 的请求是一大段，被两道截断切成 chunk。`assert > 0` 是个不变量——能进这个循环的 running 请求一定还有 token 要算。

**B3.** `scheduler.py:177-194` 多模态 encoder 输入调度（`_try_schedule_encoder_inputs`，`:489`）。若 encoder 预算/缓存耗尽导致 `num_new_tokens == 0`，**用 `continue` 而非 `break`**（`:186-191`）跳过本请求看下一条——故意放松严格 FCFS（注释明示）。

**B4.** `scheduler.py:196-223` **抢占 while 循环**（V1 抢占的全部逻辑）：
```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
    if new_blocks is None:                       # 显存不足
        preempted_req = self.running.pop()       # 抢占队尾（最低优先级）
        self.kv_cache_manager.free(preempted_req)
        preempted_req.status = RequestStatus.PREEMPTED
        preempted_req.num_computed_tokens = 0    # 整条作废，之后重算
        self.waiting.appendleft(preempted_req)   # 塞回 waiting 队头
        preempted_reqs.append(preempted_req)
        if preempted_req == request:             # ← 关键边界：把自己都抢没了
            can_schedule = False
            break
    else:
        can_schedule = True
        break
if not can_schedule:
    break                                        # RUNNING 段到此为止
```
> 解读：`allocate_slots` 返回 `None` 即显存放不下（`kv_cache_manager.py:232`）。抢占永远从 `running` **队尾** pop（list 末尾 = 最低优先级）。`if preempted_req == request` 是防死循环的命门——当被抢的恰好是当前想调度的这条，说明没有更低优先级可抢了，放弃它并跳出。

**B5.** `scheduler.py:226-260` 调度成功后的记账：
- `:227` `scheduled_running_reqs.append(request)`；`:228` 加入 `scheduled_req_ids`。
- `:235-238` 记 block_ids、`num_scheduled_tokens[req_id] = num_new_tokens`；`:239` `token_budget -= num_new_tokens`。
- `:242-251` **spec decode 裁剪**：算出本步实际能调度的 draft token 数 `num_scheduled_spec_tokens`，把 `request.spec_token_ids` 截到这个长度（多出的 draft 这步算不下）。

---

### 阶段 C：② 调度 WAITING 队列（拉新 prefill / 恢复被抢占请求）

**C1.** `scheduler.py:275` `if not preempted_reqs:` —— **本步发生过抢占就整段跳过**（显存紧张时不收新活）。

**C2.** `scheduler.py:276-278` 循环条件三连：`while self.waiting and token_budget > 0` 且 `len(self.running) < max_num_running_reqs`（`max_num_seqs` 约束在此生效）。

**C3.** `scheduler.py:284-302` 两道"跳过但不连坐"的检查：
- `:284-291` `WAITING_FOR_FSM`：结构化输出 grammar 还没编译完，则 `popleft` 后暂存到 `skipped_waiting_requests`，`continue`（编译好的会就地翻成 `WAITING` 继续）。
- `:295-302` LoRA 上限：再加新 LoRA 会超 `max_loras`，同样暂存跳过。

**C4.** `scheduler.py:304-317` **prefix cache 命中 + num_new_tokens 计算**：
```python
computed_blocks, num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
num_new_tokens = request.num_tokens - num_computed_tokens   # 用 num_tokens 兼顾被抢占恢复的请求
# 同样的 long_prefill_token_threshold / token_budget 两道截断
```
> 解读：`get_computed_blocks` 返回 prefix cache 命中的前缀块及其 token 数，命中部分**直接当作已算**，只需调度剩下的 `num_new_tokens`。用 `request.num_tokens`（含已生成 output）而非 `num_prompt_tokens`，是为了正确处理"被抢占后恢复"的请求——它已经有 output token 了。

**C5.** `scheduler.py:332-336` `allocate_slots(request, num_new_tokens, computed_blocks)`，返回 `None` 直接 `break`（WAITING 段放不下就停，不抢占）。

**C6.** `scheduler.py:338-373` 入运行态：`waiting.popleft()` → `running.append()`；按 `request.status` 分流到 `scheduled_new_reqs`（`WAITING`）或 `scheduled_resumed_reqs`（`PREEMPTED`，`:348-354`）；`:363-364` 置 `RUNNING` 并写 `num_computed_tokens = num_computed_tokens`（含 prefix 命中）。

**C7.** `scheduler.py:375-377` 把 `skipped_waiting_requests` 用 `extendleft` 放回 `waiting` 队头，维持到达序。

---

### 阶段 D：收尾断言与公共前缀

**D1.** `scheduler.py:379-388` 一组不变量断言：`total_num_scheduled_tokens <= max_num_scheduled_tokens`、`token_budget >= 0`、`len(running) <= max_num_running_reqs`、调度请求数 ≤ running 数。

**D2.** `scheduler.py:390-397` **cascade attention 公共前缀**：
```python
if self.running:
    any_request = self.running[0]
    num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(
        any_request, len(self.running))
```
> `get_num_common_prefix_blocks`（`kv_cache_manager.py:331`）沿任一 running 请求的 block 往前数，`ref_cnt == num_running_requests` 的块即公共前缀。注意它用 `len(self.running)` 而非"本步调度数"，可能因未调度的 running 请求不共享前缀而算成 0（`kv_cache_manager.py:353-357` 的边界说明）。

**D3.** `scheduler.py:399-403` `grammar_bitmask`：结构化输出的 token 掩码（[模块 06](../06-sampling-structured-output/design.md)）。

---

### 阶段 E：打包 SchedulerOutput

**E1.** `scheduler.py:405-409` `scheduled_new_reqs` 用 `NewRequestData.from_request` 打成**全量**数据（worker 首见此请求，需 prompt token / mm 输入 / block 全发）。

**E2.** `scheduler.py:410-427` `scheduled_resumed_reqs` + `scheduled_running_reqs` 用 `_make_cached_request_data` 打成**增量**（worker 已缓存过，只发 diff）。

**E3.** `scheduler.py:461` `_make_cached_request_data`：**对象复用**——
```python
new_token_ids = request.all_token_ids[num_computed:num_computed + num_regular_tokens]
req_data = self._cached_reqs_data.get(request.request_id)
if req_data is not None:                 # 命中缓存：原地改字段
    req_data.new_token_ids = new_token_ids
    req_data.new_block_ids = new_block_ids
    req_data.num_computed_tokens = num_computed_tokens
else:                                    # 首次：建对象并入缓存池
    req_data = CachedRequestData.from_request(...)
    self._cached_reqs_data[request.request_id] = req_data
```
> `num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens`（`:472`）——只把"真实新 token"切给 worker，spec token 单独走 `scheduled_spec_decode_tokens`。

**E4.** `scheduler.py:428-444` 组装 `SchedulerOutput`（见 `output.py:81`）：含 `num_scheduled_tokens`、`scheduled_spec_decode_tokens`、`num_common_prefix_blocks`、`finished_req_ids`、`free_encoder_input_ids`、`grammar_bitmask` 等。

---

### 阶段 F：乐观推进 `num_computed_tokens`

**F1.** `scheduler.py:446-456`：
```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    self.requests[req_id].num_computed_tokens += num_scheduled_token
```
> **在 forward 之前**就把 token 记成已算（注释 `:446-454` 三点解释）：① `SchedulerOutput` 已保留原始调度数以构造 input；② 提前推进让下一步能立刻续排该请求；③ 若 spec token 被拒，`update_from_output` 再退回。`:458` 清空 `finished_req_ids`（已随本 output 下发）。

---

### 阶段 G：③ update_from_output —— 回填、stop 检测、spec 回退

**G1.** `scheduler.py:567` `update_from_output(scheduler_output, model_runner_output)`。`:585` 遍历 `self.running`（注释 `:582-584` 警告此循环可能上千次，循环体内避免重操作）。

**G2.** `scheduler.py:587-591` 本步没调度的请求（`num_tokens_scheduled == 0`）直接进 `new_running` 跳过。

**G3.** `scheduler.py:593-611` **spec decode 回退**：
```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = sampled_token_ids[req_index]
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
if scheduled_spec_token_ids:
    num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 - len(generated_token_ids))
    request.num_computed_tokens -= num_tokens_rejected      # 退回被拒 draft 的记账
```
> `+1` 是因为 verify 模型在 k 个 draft 之外总会多产出 1 个 bonus token；接受了 `len(generated)` 个，其余 `len(spec)+1-len(generated)` 被拒、退回。

**G4.** `scheduler.py:638-647` **逐 token append + stop 检测**：
```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)   # utils.py:5
    if stopped:
        self._free_request(request)
        del new_token_ids[num_new:]   # 命中 stop，截掉后续 token
        break
```
> `check_stop`（`utils.py:5-22`）依次查：`num_tokens >= max_model_len` 或 `num_output_tokens >= max_tokens`（→ `FINISHED_LENGTH_CAPPED`）；EOS（→ `FINISHED_STOPPED`）；`stop_token_ids`（→ `FINISHED_STOPPED` 并记 `stop_reason`）。注意 **stop string 不在这里**——那是前端 detokenize 后才能判断的，走 `finish_requests` 反向 abort（见 [模块 00](../00-request-lifecycle/impl.md)）。

**G5.** `scheduler.py:655-674` 结构化输出推进 grammar（`accept_tokens`）；为有新 token 的请求生成 `EngineCoreOutput`（new_token_ids、finish_reason、logprobs、stop_reason）。`:675-677` 不变量：prefill 中途的请求返回空 token，断言此时无 prompt_logprobs。

**G6.** `scheduler.py:679-693` `scheduled_req_ids.remove(req_id)`；未 stop 的进 `new_running`；`self.running = new_running`；打包 `EngineCoreOutputs`（含 `scheduler_stats`）返回。

---

## 3. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。专挑读代码容易一眼滑过、理解了才算真懂调度器的点。

### T1 · 一行算尽千般模式：`num_tokens_with_spec - num_computed_tokens`

- **现象**：整个调度器没有任何 `if is_prefill / if is_decode` 的分支。
- **代码**：`scheduler.py:168-169`、配套 `request.py:113` `num_tokens_with_spec`。
- **精妙之处**：prefill / decode / chunked / spec / prefix-cache 全部被归一成"`num_computed_tokens` 追 `num_tokens_with_spec`"。decode 是这个差为 1 的退化，chunked prefill 是这个差很大被 budget 截断，spec 是把 `spec_token_ids` 也计进被追目标。**正因没有阶段分支，一段循环就同时是 continuous batching 和 chunked prefill 的实现**——对比 V0 的 `_schedule_prefills` / `_schedule_running` / `_schedule_swapped` 三方法两策略（`vllm/core/scheduler.py:1032/649/821`），简化是结构性的。

### T2 · 抢占 while 循环的"自环退出"判断

- **现象**：显存不足要抢占，但抢到什么时候停，藏着一个会死循环的边界。
- **代码**：`scheduler.py:214` `if preempted_req == request:`。
- **精妙之处**：`while True` 不断 `running.pop()` 抢最低优先级请求直到 `allocate_slots` 成功。但若一路抢到把**当前正想调度的这条请求本身**都抢了（说明它就是全场最低优先级、已无更低可抢），就 `can_schedule = False` 跳出、放弃调度它。少了这个判断，循环会在空 `running` 上继续 `pop()` 而越界，或对自己反复抢占。

### T3 · 抢占只发生在 RUNNING 段，WAITING 段放不下就 break

- **现象**：同样是 `allocate_slots` 返回 `None`，两段处理截然不同。
- **代码**：RUNNING 段 `:201` 触发抢占；WAITING 段 `:334-336` 直接 `break`。
- **精妙之处**：RUNNING 请求是"已经在跑、必须继续"的，放不下就牺牲更低优先级者保住它；WAITING 请求是"还没开始"的，放不下就停止收新活、把显存留给在跑的。一个抢占、一个止步，体现"先保住已有进度"的优先级哲学。配合 `:275 if not preempted_reqs` ——抢占过的步连 WAITING 段都不进。

### T4 · 双重记账：schedule 里先加，update 里按需减

- **现象**：`num_computed_tokens` 在还没 forward 时就被加上去了。
- **代码**：`scheduler.py:455-456`（加）↔ `scheduler.py:605-607`（spec 拒绝时减）。
- **精妙之处**：乐观推进让"下一步 schedule 能立刻续排这条请求的下一个 chunk / 下一批 PP"，不必等执行结果回流。代价是 spec decode 必须在 update 里用 `len(spec)+1-len(generated)` 精确回退被拒 token——多减或少减都会让 KV 与 token 计数错位、静默产生错误输出。这是"调度与执行解耦"换来的记账复杂度。

### T5 · `continue` vs `break`：故意打破严格 FCFS 的两处放松

- **现象**：队头请求被卡住时，到底是连坐后面所有请求，还是越过它？
- **代码**：encoder 预算 `scheduler.py:186-191`（`continue`）；LoRA 上限 `:295-302`（暂存跳过）；对比 WAITING 段显存不足 `:336`（`break`）。
- **精妙之处**：被"非显存"约束（encoder/LoRA）卡住时用 `continue`/暂存，让低优先级请求能见缝插针，注释明确写 *"intentionally relax the strict FCFS"*；而被"显存"卡住时用 `break`——因为显存是全局资源，后面的请求大概率也放不下，越过它无意义。**约束的性质决定了越过还是连坐**。

### T6 · skipped 队列 + extendleft：放松 FCFS 又不打乱到达序

- **现象**：跳过的 WAITING 请求若直接丢弃会乱序，留在原地又会反复触发同一检查。
- **代码**：`scheduler.py:272` `skipped_waiting_requests`；`:289-290`/`:300-301` `popleft`+`appendleft` 暂存；`:375-377` `extendleft` 放回队头。
- **精妙之处**：把"本步无法调度但下步可能可以"的请求暂时移出主循环（避免每轮都重判 grammar/LoRA），循环结束再 `extendleft` 整体塞回队头——因为暂存时用的是 `appendleft`，两次反转正好**还原原始相对顺序**。既不饿死它们，也不破坏 FCFS。

### T7 · `request.num_tokens` 而非 `num_prompt_tokens`：兼容被抢占恢复的请求

- **现象**：WAITING 段算 `num_new_tokens` 时用的是 `num_tokens` 而不是 prompt 长度。
- **代码**：`scheduler.py:308-311`，注释明示。
- **精妙之处**：被抢占（PREEMPTED）的请求回到 waiting 时已经生成过若干 output token，`num_computed_tokens` 被清零要从头重算。若用 `num_prompt_tokens` 就会漏掉那些 output token。用 `num_tokens`（= prompt + output 全部）才能让恢复的请求把"已生成部分"也重新算进 KV——这是 recompute 抢占能正确恢复的前提。

### T8 · 复用 `CachedRequestData` + 只发增量，省 IPC 与 GC

- **现象**：running 上千条，每步都为每条新建 dataclass + 全量序列化会很贵。
- **代码**：`scheduler.py:469-487` `_make_cached_request_data`（命中 `_cached_reqs_data` 就原地改字段）；`output.py:87-91` 注释解释 new=全量 / cached=增量。
- **精妙之处**：首次调度发 `NewRequestData`（全量，worker 缓存住），之后只发 `CachedRequestData`（new_token_ids / new_block_ids / num_computed_tokens 三个增量字段），且对象本身从缓存池复用、不重新分配。两层省钱：跨进程 IPC 带宽 + Python 小对象 GC 压力。`_free_request` 里会 `_cached_reqs_data.pop`（`:736`）清理。

### T9 · 切给 worker 的只有"真实 token"，spec token 单走一路

- **现象**：`new_token_ids` 的切片长度不等于 `num_scheduled_tokens`。
- **代码**：`scheduler.py:472` `num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens`。
- **精妙之处**：一步里调度的 token = 已确定的真实 token + 待验证的 spec token。真实 token 通过 `CachedRequestData.new_token_ids` 走，spec token 通过 `SchedulerOutput.scheduled_spec_decode_tokens` 单独走（`:250-251` 还会把 `spec_token_ids` 列表 `del` 截断到能调度的长度）。两条路分开，让 worker 端能区分"要算 KV 的确定 token"和"要 verify 的草稿 token"。

### T10 · `num_computed_tokens` 在 RUNNING/WAITING 两段的"双重身份"

- **现象**：同一个字段，RUNNING 段是"读出来算差"，WAITING 段是"被 prefix cache 重写"。
- **代码**:RUNNING 段 `:169` 读 `request.num_computed_tokens`；WAITING 段 `:364` 写 `request.num_computed_tokens = num_computed_tokens`（来自 `get_computed_blocks`）。
- **精妙之处**：新请求/恢复请求进入 running 前，`num_computed_tokens` 被 prefix cache 命中数**直接重写**（命中的前缀视同已算，跳过计算）；之后它就只增不重写，由 decode/chunk 一步步推进。理解这个字段在"入场被 prefix-cache 校准"和"在场被乐观推进"两个相位，才能读懂 prefix caching 如何无缝接入统一抽象。

### T11 · 公共前缀用 `len(self.running)` 而非"本步调度数"，并接受算成 0

- **现象**：`get_num_common_prefix_blocks` 传的是整个 running 长度，可能低估公共前缀。
- **代码**：`scheduler.py:393-397` + `kv_cache_manager.py:343-377`。
- **精妙之处**：判定"公共块"的标准是 `block.ref_cnt == num_running_requests`。但 running 里可能有本步**未被调度**的请求，它们不一定共享前缀，于是即便所有"被调度"的请求都共享前缀，`ref_cnt` 也凑不齐 `num_running_requests`，函数保守返回 0（注释 `kv_cache_manager.py:353-357` 坦承此边界"目前难以廉价检测"）。调度器只给执行层一个"安全下界"，宁可不启用 cascade 也不算错。

### T12 · 调度器只算公共前缀长度，用不用 cascade 交给 backend 启发式

- **现象**：`num_common_prefix_blocks > 0` 不代表真会跑 cascade attention。
- **代码**：调度侧 `scheduler.py:390-397`；决策侧 `flash_attn.py:553 use_cascade_attention`（前缀 < 256 token、请求 < 8、ALiBi/滑窗 → False；否则用一个粗性能模型比 cascade vs FlashDecoding）。
- **精妙之处**：职责切得很干净——**调度器不背 attention 性能模型**，只产出"公共前缀有多长"这个事实；要不要把它变成 cascade attention 由 backend 按硬件（SM 数）、head 配置、query 数现场启发式决定。换 backend 不影响调度逻辑。

### T13 · 多模态 encoder 输入的"对齐式"chunk 截断

- **现象**：chunked prefill 切到一半正好切进一张图的 token 中间，会出问题。
- **代码**：`scheduler.py:489 _try_schedule_encoder_inputs`，尤其 `:538-543`（`disable_chunked_mm_input`）与 `:545-561`（预算不足时回退到 mm 项之前）。
- **精妙之处**：encoder 通常用双向 attention，一张图的 token 必须**整体**算，不能跨 chunk 切。所以当本步预算只能覆盖某个 mm 项的一部分时，把 `num_new_tokens` 回退到"刚好停在这个 mm 项之前"（`:542 num_new_tokens = start_pos - num_computed_tokens`），让这张图留到下一步整体处理。prefix caching 下还有个刁钻情形：`num_computed_tokens > start_pos` 但 encoder 输出还没算，此时只能 `num_new_tokens = 0` 本步啥也不调度（`:555-560`）。

### T14 · stop 命中后 `del new_token_ids[num_new:]`：截断越界 token

- **现象**：一步可能 append 多个 token（spec decode），但中途某个就触发了 stop。
- **代码**：`scheduler.py:646` `del new_token_ids[num_new:]`。
- **精妙之处**：spec decode 一步能产出多个 token，若第 `num_new` 个就命中 EOS/stop，后面那些 token 在语义上**不该存在**（请求已结束）。这里就地把列表截断到 stop 处，保证回吐给用户的 `new_token_ids` 不包含"停止符之后"的多余 token。少了这一刀，流式输出会多吐几个 token。

### T15 · stop string 不在调度器判定，而是前端反向 abort

- **现象**：`check_stop` 只查 EOS / stop_token_ids / 长度，没有 stop **string**。
- **代码**：`utils.py:5-22`（无 stop string）；对比 [模块 00](../00-request-lifecycle/impl.md) 中前端 `output_handler` 收集 `reqs_to_abort` → `abort_requests_async` → `finish_requests`（`scheduler.py:701`）。
- **精妙之处**：stop string 是字符层面的（如 `"\n\n"`），只有 detokenize 之后才能判断，而 detokenize 在**前端进程**。所以后端调度器只判 token 级停止，字符级停止由前端检测后**反向**通知 EngineCore 调 `finish_requests` 释放。这条边界解释了为什么 stop 检测被拆成两处。

### T16 · `RequestStatus > PREEMPTED` 即 finished：用枚举序省判断

- **代码**：`request.py:149-164`，`is_finished` = `status > RequestStatus.PREEMPTED`。
- **精妙之处**：枚举值刻意排成 `WAITING(1) < WAITING_FOR_FSM < RUNNING < PREEMPTED < FINISHED_*`，于是"是否结束"退化成一次整数比较，不用枚举集合查表。`finish_requests`（`:701`）据此分流：RUNNING 从 `running` 移除、否则从 `waiting` 移除，再统一 `_free_request`。

---

## 4. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| 显存不足（RUNNING 段） | `scheduler.py:201-217` | 抢占队尾最低优先级请求 → `free` KV → `PREEMPTED`、`num_computed_tokens=0` → 回 waiting 队头 |
| 抢到自己头上（无更低可抢） | `scheduler.py:214-217` | `can_schedule=False`，放弃调度当前请求并 `break` 出 RUNNING 段 |
| 显存不足（WAITING 段） | `scheduler.py:334-336` | 直接 `break`，不抢占——把显存留给已在跑的请求 |
| 本步已抢占过 | `scheduler.py:275` | 跳过整个 WAITING 段，本步不收新请求 |
| 达到 `max_num_seqs` | `scheduler.py:277-278` | WAITING 段 `break`，不再新增并发请求 |
| grammar 未编译完 | `scheduler.py:284-291` | `WAITING_FOR_FSM`，暂存 `skipped_waiting_requests` 跳过，编译好下步再调度 |
| LoRA 超 `max_loras` | `scheduler.py:295-302` | 暂存跳过该请求，去调度能复用已加载 LoRA 的后续请求 |
| 多模态项跨 chunk | `scheduler.py:538-561` | 回退 `num_new_tokens` 到 mm 项之前，整张图留到下一步；prefix 命中且 encoder 未算时 `num_new_tokens=0` |
| spec token 部分被拒 | `scheduler.py:605-607` | `num_computed_tokens -= len(spec)+1-len(generated)`，退回被拒 token 的记账 |
| 一步内中途 stop | `scheduler.py:644-647` | `_free_request` + `del new_token_ids[num_new:]` 截断越界 token |
| prefill 中途的请求 | `scheduler.py:638`/`675-677` | model runner 对未完成 prefill 返回空 token；断言此时无 prompt_logprobs |
| 请求已在本步调度过 | `scheduler.py:163-166` | `req_index += 1` 跳过，避免一步内重复调度 |
| 公共前缀因未调度 running 请求算成 0 | `kv_cache_manager.py:353-357` | 保守返回 0，本步不启用 cascade（安全下界） |

---

## 5. 一图速查：schedule() 主干

```
Scheduler.schedule()  scheduler.py:122
  token_budget = max_num_batched_tokens                                 :149
  │
  ├─ ① RUNNING 段  while req_index<len(running) and budget>0            :161
  │     num_new = num_tokens_with_spec - num_computed_tokens            :168  ← 统一抽象
  │     num_new = min(num_new, long_prefill_threshold, budget)          :170-174  ← chunked prefill
  │     [mm] _try_schedule_encoder_inputs (continue if 0)               :177-191
  │     while True: allocate_slots()                                    :196
  │        None → running.pop() 抢占 → free → PREEMPTED → waiting队头    :201-213
  │               if preempted_req == request: 放弃, break              :214  ← 自环退出
  │     scheduled_running_reqs += req;  budget -= num_new               :227-239
  │     spec: 裁剪 spec_token_ids                                       :242-251
  │
  ├─ ② WAITING 段  if not preempted_reqs:  while waiting and budget>0   :275-278
  │     and len(running) < max_num_seqs                                 :277
  │     WAITING_FOR_FSM / LoRA超限 → 暂存 skipped, continue             :284-302  ← 放松 FCFS
  │     get_computed_blocks() ← prefix cache 命中                       :305
  │     num_new = num_tokens - num_computed; 同样两道截断               :311-316
  │     allocate_slots(); None → break (不抢占)                         :332-336
  │     waiting.popleft() → running.append(); RUNNING                   :338-363
  │     budget -= num_new                                               :362
  │  extendleft(skipped) 放回队头                                       :375-377
  │
  ├─ ③ 收尾                                                              :379
  │     断言 budget>=0, running<=max_num_seqs                           :381-388
  │     get_num_common_prefix_blocks (cascade attn)                     :390-397
  │     grammar_bitmask                                                 :399-403
  │     SchedulerOutput(new=全量 NewRequestData,                        :405-444
  │                     cached=增量 CachedRequestData[复用池])
  │     num_computed_tokens += num_scheduled  (乐观推进)                :455-456  ← 双重记账
  ▼
SchedulerOutput ─► execute_model ─► update_from_output  scheduler.py:567
                                       append token + check_stop        :638-647
                                       spec reject → 回退 num_computed   :605-607
                                       → EngineCoreOutputs               :684-693
```
