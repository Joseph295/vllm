# 模块 05 · Speculative Decoding 投机解码 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的投机解码调用链：`proposer 提议 draft → target 一次验证 → rejection sampling → 调度器回填`。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数/类名与逻辑为准。
> 前置：本模块叠加在 [模块 00](../00-request-lifecycle/design.md)（请求生命周期 `step()` 三段式）、[模块 01](../01-scheduler-batching/design.md)（调度器 `schedule/update_from_output`）、[模块 02](../02-paged-attention-kvcache/design.md)（KV `allocate_slots`）、[模块 06](../06-sampling-structured-output/design.md)（采样）之上，建议先读那几篇。

---

## 1. 代码地图

```
V1 投机解码核心  vllm/v1/spec_decode/
  ├─ ngram_proposer.py     # NgramProposer：prompt lookup（KMP 找重复 n-gram），无模型
  ├─ eagle.py              # EagleProposer：轻量 draft head，自回归提议 k 个 token + prepare_inputs
  ├─ metadata.py           # SpecDecodeMetadata：验证所需的全部索引/张量
  ├─ metrics.py            # SpecDecodingStats / SpecDecodingMetrics：接受率统计与打印
  └─ utils.py              # is_spec_decode_supported：哪些采样特性不兼容投机解码

V1 验证/采样   vllm/v1/sample/
  └─ rejection_sampler.py  # RejectionSampler + 4 个 Triton 内核（greedy/random 验证、recovered、expand）

V1 调度集成    vllm/v1/core/sched/
  └─ scheduler.py          # num_lookahead_tokens / scheduled_spec_decode_tokens / 拒绝回退记账

V1 ModelRunner 集成  vllm/v1/worker/
  └─ gpu_model_runner.py   # drafter 初始化 / _calc_spec_decode_metadata / execute_model 验证 + propose

配置          vllm/config.py
  └─ SpeculativeConfig     # method / num_speculative_tokens / prompt_lookup_* / acceptance_method

V0 对比（已被 V1 重写）  vllm/spec_decode/
  ├─ spec_decode_worker.py # SpecDecodeWorker：独立 worker 编排 propose→score→verify
  ├─ batch_expansion.py    # BatchExpansionTop1Scorer：B·(k+1) 展开（V1 已弃）
  ├─ mqa_scorer.py         # MQAScorer：多 query 一次 forward（V1 原位验证的前身）
  ├─ ngram_worker.py / multi_step_worker.py / top1_proposer.py  # V0 proposer 体系
```

---

## 2. 端到端调用链（投机解码一步的完整旅程）

下面跟随 EngineCore 进程的一个 `step()`（见 [模块 00](../00-request-lifecycle/impl.md) 阶段 E/F/G），聚焦投机解码相关分支。

### 阶段 0：初始化 —— 选 proposer、建 rejection sampler

> **这一步在干嘛**：引擎启动时，按配置选好"猜测者"（ngram 还是 eagle 等）并建好"验证者"（拒绝采样器）。猜测者只在能拿到 target 最后 hidden state 的那段 GPU（PP 最后一段 rank）上建。

**0a.** `vllm/v1/worker/gpu_model_runner.py:160` `GPUModelRunner.__init__`（投机解码 setup）
```python
self.use_spec_decode = False
if self.speculative_config:
    self.use_spec_decode = True
    if get_pp_group().is_last_rank:
        if self.speculative_config.method == "ngram":
            self.drafter = NgramProposer(self.vllm_config)
        elif self.speculative_config.method == "eagle":
            self.drafter = EagleProposer(self.vllm_config, self.device)
        ...
        self.rejection_sampler = RejectionSampler()
```
> 解读：proposer 只在 **PP 最后一段 rank** 上建（draft 需要 target 的最后 hidden state / 最终 token）。`RejectionSampler` 是一个 `nn.Module`，但本质是几个 Triton 内核的包装。

**0b.** `config.py:2081` `SpeculativeConfig`：`method` 决定 proposer（`:2091-2107`），`num_speculative_tokens` 即 `k`，ngram 额外需要 `prompt_lookup_min/max`（`:2098-2103`）。`acceptance_method`（`:2108-2125`）默认 `rejection_sampler`（无损），可选 `typical_acceptance_sampler`（有损但更快）。

---

### 阶段 ①：调度 —— 把 draft 当"待验证 token"推进

> **这一步在干嘛**：上一步猜出的 `k` 个 draft token，被调度器**当成普通 token 一起推进**（无需特判），并为它们额外预留 KV slot。token 预算不够就把 draft 截断、只验证装得下的部分。这里还"乐观地"把 draft 也记成"已算"，万一之后被拒再回退（阶段⑥）。

**1a.** `scheduler.py:117` 构造时确定 lookahead：
```python
self.num_lookahead_tokens = 0
if speculative_config and speculative_config.method == "eagle":
    self.num_lookahead_tokens = speculative_config.num_speculative_tokens
```
> 解读：eagle 的 draft token 也要占 KV slot，所以调度分配 block 时要多预留 `k` 个。ngram 走另一条路（draft 不需要预先 forward，slot 由后续推进自然覆盖）。

**1b.** `scheduler.py:168` 算 `num_new_tokens`（统一抽象，draft 自动算进来）：
```python
num_new_tokens = (request.num_tokens_with_spec - request.num_computed_tokens)
```
> `num_tokens_with_spec = len(prompt)+len(output)+len(spec_token_ids)`（`scheduler.py:126-127`）。上一步 proposer 写进 `request.spec_token_ids` 的 `k` 个 draft，让 `num_new_tokens` 自然变成 `1 个真 token + k 个 draft`，无需特判。

**1c.** `scheduler.py:196-200` 分配 KV slot，预留 lookahead：
```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
```

**1d.** `scheduler.py:243-251` 记录本步实际验证的 draft（关键）：
```python
if request.spec_token_ids:
    num_scheduled_spec_tokens = (num_new_tokens + request.num_computed_tokens
                                 - request.num_tokens)
    if num_scheduled_spec_tokens > 0:
        # Trim spec_token_ids list to num_scheduled_spec_tokens.
        del request.spec_token_ids[num_scheduled_spec_tokens:]
        scheduled_spec_decode_tokens[request.request_id] = request.spec_token_ids
```
> 解读：`num_tokens` 是不含 spec 的 token 数。若 token budget 不足以验证全部 `k` 个 draft，这里把 `spec_token_ids` **截断**到本步实际能容纳的数量 —— draft 可以"部分验证"。结果存进 `scheduled_spec_decode_tokens`，随 `SchedulerOutput` 下发（`:433`）。

**1e.** `scheduler.py:455-456` 乐观推进 `num_computed_tokens`（含 draft）：
```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    self.requests[req_id].num_computed_tokens += num_scheduled_token
```
> 解读：连 draft token 都先记成"已算"。注释（`:446-454`）明说："若 spec token 之后被拒，会在 `update_from_output` 里调整"。这是 [模块 00](../00-request-lifecycle/impl.md) T4"乐观推进 + 事后纠正"在投机解码上的体现。

---

### 阶段 ②：构造验证元数据 —— `_prepare_inputs` 里算索引

> **这一步在干嘛**：一次 forward 会算出一长串扁平 logits，里头混着"要验证的 draft 位置"和"白送的 bonus 位置"。这一步用纯 numpy 一次性算好三组下标 —— 哪些位置要喂拒绝采样器、哪些是 bonus、draft token 本身是哪几个 —— 这样后面才能从扁平 logits 里"原地切片"取对，不用复制 batch。

**2a.** `gpu_model_runner.py:587` `_prepare_inputs` 判断本步是否有投机解码：
```python
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
if not use_spec_decode:
    ... logits_indices = query_start_loc[1:] - 1   # 普通 decode：每请求最后一个位置
    spec_decode_metadata = None
else:
    num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
    for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
        ... num_draft_tokens[req_idx] = len(draft_token_ids)
    spec_decode_metadata = self._calc_spec_decode_metadata(num_draft_tokens, cu_num_tokens)
    logits_indices = spec_decode_metadata.logits_indices
```

**2b.** `gpu_model_runner.py:753` `_calc_spec_decode_metadata` —— **纯 numpy 向量化算三组索引**（投机解码实现最精巧的一处）。代码自带的小例子（`:758-766`）：
```
cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]   # 每请求本步算几个 token 的前缀和
num_draft_tokens:         [  3,   0,   2,   0,   1]    # 每请求几个 draft
# 输出三组索引（都是进扁平 logits 的下标）：
logits_indices:           [0,1,2,3, 103,104,105,106, 206,207,208]  # 全部要采样的位置 (k+1 个/请求)
target_logits_indices:    [0,1,2, 5,6, 9]             # k 个 draft 位置（喂 rejection sampler）
bonus_logits_indices:     [3,4, 7,8, 10]              # 每请求的 bonus 位置（全 accept 时白送）
```
> 解读：`num_sampled = num_draft + 1`（`:770`），用 `np.repeat`+`np.cumsum`+`arange` 三连一次性把每请求 `k+1` 个位置展平成扁平下标（`:771-783`）。`bonus_logits_indices = cu_num_sampled_tokens - 1`（`:786`，每请求最后一个）。`target_logits_indices` 单独再算（`:790-801`）。最后 `:815-816` 用 `draft_token_ids = self.input_ids[logits_indices][target_logits_indices+1]` 把 draft token 本身取出来 —— **draft token 就是上一步写进 input_ids 的那些**。

**2c.** 打包成 `SpecDecodeMetadata`（`gpu_model_runner.py:818-826` → `metadata.py:8`）。`__post_init__` 算 `max_spec_len = max(num_draft_tokens)`（`metadata.py:24-25`），供 rejection sampler 断言 `<= MAX_SPEC_LEN(32)`。

---

### 阶段 ③：Target 一次 forward + 切 logits

> **这一步在干嘛**：target 模型跑**一次** forward，因为 draft token 已经拼进输入序列，所以这一次就同时算出了所有 draft 位置和 bonus 位置的 logits。然后按阶段②的索引把 logits 切成两份：bonus 位置交给完整采样器（可用 top_p/top_k），draft 位置交给拒绝采样器去验证。

**3a.** `gpu_model_runner.py:998` `execute_model` 拿到元数据：
```python
attn_metadata, logits_indices, spec_decode_metadata = self._prepare_inputs(scheduler_output)
```
随后正常跑 `self.model(...)` 一次 forward（`:1062` 附近，见 [模块 00](../00-request-lifecycle/impl.md) F4），得到 `hidden_states` 与 `logits`。draft token 已经拼在输入序列里，所以这**一次** forward 同时算出了 draft 位置和 bonus 位置的 logits。

**3b.** `gpu_model_runner.py:1083-1103` 按有无投机解码分流：
```python
if spec_decode_metadata is None:
    sampler_output = self.model.sample(logits=logits, sampling_metadata=...)   # 普通路径
else:
    # bonus 位置：单独用完整采样器（可 top_p/top_k）
    bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
    sampler_output = self.model.sample(logits=bonus_logits, sampling_metadata=...)
    bonus_token_ids = sampler_output.sampled_token_ids
    # draft 位置：切出 target_logits 喂 rejection sampler
    target_logits = logits[spec_decode_metadata.target_logits_indices]
    output_token_ids = self.rejection_sampler(
        spec_decode_metadata, None /*draft_probs*/, target_logits,
        bonus_token_ids, sampling_metadata)
    sampler_output.sampled_token_ids = output_token_ids
```
> 解读：注释（`:1089-1102`）特意说明 —— 用 tensor 索引 `logits[indices]` 会**新建独立存储**，所以 `bonus_logits`/`target_logits` 是新张量，rejection sampler 可以**原地改** `target_logits`（省显存）而不污染原 `logits`。`draft_probs` 传 `None`（`:1106`）：V1 当前不缓存上一步 draft 的概率分布，rejection sampler 会按 ngram 模式把 draft prob 当 1（见 §阶段④，`TODO` 在 `:1229-1231`）。

---

### 阶段 ④：Rejection Sampling —— 验证内核

> **这一步在干嘛**：这是投机解码的"裁判"。逐个 draft 位置比较 target 概率 `p` 和 draft 概率 `q`：按 `min(1, p/q)` 的概率接受 draft，一旦拒绝就从修正分布 `max(0, p−q)` 重采一个 token 并丢弃后面所有 draft；若 `k` 个全接受则追加 bonus token。这条规则的数学保证就是"无论 draft 多差，输出都服从 target 分布"（design §3.3）。

**4a.** `rejection_sampler.py:46` `RejectionSampler.forward`：
```python
target_probs = compute_probs(target_logits, metadata.cu_num_draft_tokens, sampling_metadata)
output_token_ids = rejection_sample(metadata.draft_token_ids, metadata.num_draft_tokens,
    metadata.max_spec_len, metadata.cu_num_draft_tokens, draft_probs, target_probs,
    bonus_token_ids, sampling_metadata)
```
**4b.** `compute_probs`（`:235-293`）：对 target logits 做 temperature 缩放 + top_k/top_p + softmax。greedy 请求直接返回 logits（`:259-260`，argmax 不需要归一化）。temperature/top_k/top_p 用 `expand_batch_to_tokens`（`:296`）把 per-request 标量**展开到 per-token**（因为一个请求有多个 draft 位置）。

**4c.** `rejection_sample`（`:135`）分两类请求：

- **greedy**（`:178-192`）：`target_argmax = target_probs.argmax(-1)`，调 `rejection_greedy_sample_kernel`。内核（`:433`）逐位置：`store(target_argmax)`，若 `draft_token_id != target_argmax_id` 则 `rejected=True`（`:467`）后续不再写。全 accept 则在 `num_draft_tokens` 位置写 bonus（`:471-476`）。
- **random**（`:194-231`）：先 `generate_uniform_probs`（`:337`，每 draft 位置一个 `r~U(0,1)`，支持 per-request seed）；再 `sample_recovered_tokens`（`:387`，预采好每位置的 recovered token）；最后 `rejection_random_sample_kernel`。

**4d.** `rejection_random_sample_kernel`（`:481-539`）—— **拒绝采样规则的内核实现**：
```python
for pos in range(num_draft_tokens):
    if not rejected:
        draft_token_id = load(draft_token_ids[start+pos])
        draft_prob  = 1 if IS_NGRAM else load(draft_probs[start+pos, draft_token_id])
        target_prob = load(target_probs[start+pos, draft_token_id])
        uniform_prob = load(uniform_probs[start+pos])
        if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
            token_id = draft_token_id            # 接受
        else:
            rejected = True
            token_id = load(recovered_token_ids[start+pos])   # 拒绝→recovered
        store(output[req, pos], token_id)
if not rejected:                                  # 全 accept→追加 bonus
    store(output[req, num_draft_tokens], bonus_token_id)
```
> 解读：这就是 design §3.3 那条 `accept iff p/q ≥ r` 规则。一旦 reject（`:529`），`rejected=True` 让循环后续位置不再写 —— **被拒位置之后的 draft 全部丢弃**（它们建立在被推翻的前缀上）。`draft_prob>0` 的检查（`:522-524`）防 NaN。

**4e.** recovered token 来自修正分布 `max(0, p−q)`：`sample_recovered_tokens_kernel`（`:568-631`）。非 ngram 时 `prob = tl.maximum(target_prob - draft_prob, 0)`（`:617`），用 `q ~ Exponential` 做 Gumbel-max：`recovered_id = argmax(prob / q)`（`:624`）。ngram 时没有 draft 分布，改为"把 draft token 的 target 概率临时置 0 再 argmax"（`:601-607`），算完再恢复（`:627-631`）—— 等价于 `p − q` 在 `q` 是 one-hot 时的情形。

**4f.** `parse_output`（`:107-132`）：输出 shape `[batch, max_spec_len+1]`，reject 处是 `PLACEHOLDER_TOKEN_ID(-1)`，这里用 mask 过滤掉 `-1` 和越界 id，得到每请求变长的 `valid_sampled_token_ids`（`gpu_model_runner.py:1151`）。

---

### 阶段 ⑤：为下一步提议新 draft

> **这一步在干嘛**：本步验证刚结束，立刻为**下一步**再猜 `k` 个 draft，好让流水线连续转起来。ngram 就拿刚采的 token 跑 KMP 找重复；eagle 则先剔除本轮被拒的位置，再让 draft head 在"被接受的前缀"上自回归续写 `k` 个。

回到 `gpu_model_runner.py:1159`，验证完后立刻为**下一步**提议 draft：

**5a. ngram**（`:1162-1165` → `generate_draft_token_ids`，`:1242`）：
```python
for i, sampled_ids in enumerate(sampled_token_ids):
    if not sampled_ids: draft_token_ids.append([]); continue
    if not is_spec_decode_supported(req_id, self.input_batch):  # min_p/penalty/logprobs 不支持
        draft_token_ids.append([]); continue
    # 把刚采的 token 写进 CPU token 缓冲，再跑 KMP
    self.input_batch.token_ids_cpu[i, start:end] = sampled_ids
    drafter_output = self.drafter.propose(self.input_batch.token_ids_cpu[i, :end])
    draft_token_ids.append(drafter_output.tolist() if drafter_output else [])
```
> `NgramProposer.propose`（`ngram_proposer.py:25`）：从 `max_n` 到 `min_n` 依次尝试，用 `_find_subarray_kmp`（`:89`，KMP 算法）在历史里找最后 `n` 个 token 的出现，返回其后 `k` 个 token。找不到返回 `None`。纯 numba JIT，CPU 上跑。

**5b. eagle**（`:1166-1231`）：
```python
next_token_ids = [token_ids[-1] for ...]   # 每请求本步最后一个被接受的 token（:1169-1185）
if spec_decode_metadata is None:           # 上一步是纯 prefill，无 draft
    target_token_ids/positions/hidden_states = ... [:num_scheduled_tokens]
else:                                       # 上一步有 draft：要剔除被拒位置
    num_rejected_tokens = [n+1-len(valid_sampled[i]) if n>0 else 0 for ...]   # :1200-1203
    cu_num_tokens, token_indices = self.drafter.prepare_inputs(
        attn_metadata.query_start_loc, num_rejected_tokens)                   # :1209
    target_token_ids/positions/hidden_states = ...[token_indices]            # :1213-1216
draft_token_ids, draft_probs = self.drafter.propose(...)                     # :1218
spec_token_ids = draft_token_ids.tolist()
del draft_probs                            # :1231 当前不缓存（TODO）
```
> `EagleProposer.propose`（`eagle.py:34`）：把 input_ids **移位一格**（`:56-62`，用 `next_token_ids` 替换每序列最后一个），喂进 draft head 跑一次 forward，采第 1 个 draft（`:86-95`）；若 `k>1` 则把 hidden_states 喂回自己，自回归再跑 `k-1` 步（`:111-136`），每步手动更新 `slot_mapping`（`:117-123`）。`prepare_inputs`（`:144`）用 Triton 内核剔除被拒 token 的位置（`:249` `prepare_input_kernel`），保证 draft head 只在"被接受的前缀"上续写。

**5c.** `gpu_model_runner.py:1233-1240` 打包 `ModelRunnerOutput`，`spec_token_ids` 随之回传 EngineCore。

---

### 阶段 ⑥：调度器回填 —— 按拒绝数回退记账

> **这一步在干嘛**：调度器收到结果，发现实际只生成了 `len(generated)` 个 token，而阶段①乐观记账时把 `k+1` 个都记成"已算"，于是把多记的 `(k+1) − len(generated)` 个**退回去**（被拒的 draft 不该占 KV 记账）；同时统计接受率，并把阶段⑤新猜的 draft 写回请求供下一步用。

**6a.** `scheduler.py:569` `update_from_output`（见 [模块 00](../00-request-lifecycle/impl.md) 阶段 G），投机解码分支：
```python
generated_token_ids = sampled_token_ids[req_index]           # :594
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
if scheduled_spec_token_ids:                                 # :598
    # num_computed_tokens 已在 schedule() 乐观加了 (1 真 + k draft)
    # 实际只生成 len(generated) 个 → 回退被拒的
    num_tokens_rejected = len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)  # :605-606
    request.num_computed_tokens -= num_tokens_rejected        # :607
    spec_decoding_stats = self.make_spec_decoding_stats(
        spec_decoding_stats,
        num_draft_tokens=len(scheduled_spec_token_ids),
        num_accepted_tokens=len(generated_token_ids) - 1)     # :608-611  (-1 去掉 bonus/真token)
```
> 解读：这是 design §3.1 的记账闭环。`schedule()` 乐观地把 `k` 个 draft 都记成"已算"（`:455-456`）；这里发现实际只产出 `len(generated)` 个 token，于是回退 `(k+1) - len(generated)` 个。若全 accept：`len(generated)=k+1`，回退 0；若第一个就被拒：`len(generated)=1`，回退 `k`。

**6b.** `scheduler.py:628-629` 把本步 proposer 新产的 draft 写回请求，供下一步调度：
```python
if spec_token_ids is not None:
    request.spec_token_ids = spec_token_ids[req_index]
```

**6c.** 接受率统计：`make_spec_decoding_stats`（`:767-779`）→ `SpecDecodingStats.observe`（`metrics.py:27-29`）累加；`SpecDecodingMetrics.log`（`metrics.py:46-62`）周期打印 `Draft acceptance rate = num_accepted / num_draft`。

---

## 3. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。这些是读代码时一眼滑过、但理解了才算真懂 V1 投机解码的点。

### T1 · draft 故意"只用 temperature"，把分布纠正全交给拒绝采样
- **代码**：`eagle.py:230-234`（注释）；`compute_probs_and_sample_next_token`（`eagle.py:214`）只用 temperature。
- **精妙**：draft 采样时**忽略 top_p/top_k/penalty 等几乎所有采样参数**，只保留 temperature。注释直言："这可能降低接受率，但不影响拒绝采样后的输出分布"。这是 design §3.3 数学保证的工程兑现 —— draft 只管"速度（接受率）"，"正确性（分布）"由 rejection sampler 兜底。于是 draft 路径可以极简、极快。

### T2 · bonus token 走"另一条采样器"，专门为了能用 top_p/top_k
- **代码**：`gpu_model_runner.py:1093-1098`（单独 `model.sample(bonus_logits)`）；`rejection_sampler.py:33-40`（注释）。
- **精妙**：拒绝采样内核**不支持** top_p/top_k（它要的是完整概率分布做 `p/q` 比较）。但全 accept 时那个 bonus token 是"纯 target 分布"采的，完全可以享受高级采样策略。所以 vLLM 把 bonus **拆出来**用标准采样器先采好，再当 `bonus_token_ids` 传进内核。一个 token 的灵活性，单独开一条路。

### T3 · 三组 logits 索引全用 numpy 向量化一次算出，零 Python 循环
- **代码**：`gpu_model_runner.py:768-801` `_calc_spec_decode_metadata`。
- **精妙**：每请求 draft 数不同（变长），朴素实现要 Python 逐请求拼下标。这里用 `np.repeat(cu - num, num) + (arange - cumsums_offsets)` 的组合，在 CPU 上**一次**算出 `logits_indices`/`target_logits_indices`/`bonus_logits_indices` 三组扁平下标，再一把 `non_blocking` 拷到 GPU（`:804-811`）。注释里贴的具体数值例子（`:758-766`）是读懂这段的钥匙。

### T4 · "draft token 不用重新算，就是上一步写进 input_ids 的那些"
- **代码**：`gpu_model_runner.py:813-816`：`draft_token_ids = self.input_ids[logits_indices][target_logits_indices+1]`。
- **精妙**：验证需要知道"每个 draft 位置提议的是哪个 token"。但这些 token 上一步已经作为输入拼进了 `input_ids`（调度阶段 `scheduler.py:243-251` 把它们当 token 调度了）。所以**不需要单独传 draft token**，直接从 `input_ids` 里按 `target_logits_indices+1`（draft 位置的下一格就是该 draft token）取出来即可。复用已有数据，零额外搬运。

### T5 · rejection sampler 原地改 `target_logits` 省显存 —— 但前提是它是"索引出来的新张量"
- **代码**：`gpu_model_runner.py:1100-1103`（注释）；`rejection_sampler.py:70`、`compute_probs` 的 `logits.div_`（`:271`）。
- **精妙**：`target_logits = logits[target_logits_indices]` 用 tensor 索引会**新建独立存储**，所以对它做 `div_`（temperature）、softmax 等原地操作不会污染原始 `logits`（原始 logits 里 bonus 位置还要给 T2 用）。代码注释专门点明"这就是为什么 in-place 是安全的"。一个 PyTorch 索引语义的精确利用。

### T6 · ngram 没有 draft 概率，内核把它"假装成 1"，修正分布退化为置零再 argmax
- **代码**：`rejection_sampler.py:512-513`（`draft_prob = 1`）、`:594-607`（ngram 修正分布）。
- **精妙**：标准拒绝采样需要 `q(x̃)`，但 ngram 是确定性复制、没有概率分布。代码用 `IS_NGRAM` 编译期常量分流：accept 判据退化为 `target_prob >= uniform_prob`（`draft_prob=1`），修正分布 `max(0, p−q)` 退化为"把 draft token 的 target 概率临时置 0 再 argmax"（置零=减去那个"概率 1 的 one-hot"），算完 `:627-631` 再恢复原值。同一套内核优雅兼容"有分布"和"无分布"两类 proposer。

### T7 · greedy 与 random 用两个独立内核，且都能"提前退出不属于自己的请求"
- **代码**：`rejection_sampler.py:178-192`（greedy）与 `:217-231`（random）；内核内 `:445-451`、`:496-499` 的 early exit。
- **精妙**：一个 batch 里可能 greedy/random 请求混杂。greedy 内核对非 greedy 请求 `return`（`:449-451`），random 内核对 greedy 请求 `return`（`:497-499`）。`is_greedy` 用 `temperature == -1`（`GREEDY_TEMPERATURE`，`:177`）编码。两个内核各扫一遍 batch、各管各的，避免在单个内核里写复杂分支。`all_greedy` 时甚至直接跳过 random 内核（`:191-192`）。

### T8 · 调度的"乐观推进 + 拒绝回退"，让 draft 套进统一 token 抽象
- **代码**：`scheduler.py:455-456`（乐观加）↔ `scheduler.py:605-607`（回退）。
- **精妙**：投机解码没有给调度器加新状态机，而是复用了 [模块 00](../00-request-lifecycle/impl.md) T4 的"先记成已算，错了再退"。draft 在 `schedule()` 里被当普通 token 推进 `num_computed_tokens`；`update_from_output` 按 `(k+1)-len(generated)` 精确回退。`num_tokens_with_spec`（含 spec）与 `num_tokens`（不含）两个属性的并存，让"chunked prefill / decode / spec decode"共用一条 `num_new_tokens` 公式（`:168`）。

### T9 · 调度按 token budget "部分验证" draft —— 装不下就截断
- **代码**：`scheduler.py:243-251`，`del request.spec_token_ids[num_scheduled_spec_tokens:]`。
- **精妙**：proposer 提议了 `k` 个，但本步 token budget 可能不够验证全部。这里算出本步实际能容纳的 spec 数 `num_scheduled_spec_tokens`，把超出的 draft **截断丢弃**。投机解码因此能优雅地与 chunked prefill / 预算约束共存，而不是"要么全验要么不验"。

### T10 · eagle 自回归提议时手动维护 slot_mapping 和 seq_lens
- **代码**：`eagle.py:111-136`，尤其 `:117-123` 的 `block_numbers // block_size` 计算。
- **精妙**：draft head 跑第 2..k 步时是"一个 token 进、一个 token 出"的纯 decode，但它不在主 attention metadata 的管理范围内。所以每步手动 `positions += 1`、`seq_lens += 1`、`max_seq_len += 1`，并用 block_table gather 重算 `slot_mapping`（`:118-123`），把新 draft 的 KV 写到正确的 paged 位置。这是"draft head 复用 target KV 空间"的底层细节。

### T11 · eagle 复用 target 的 lm_head（甚至 embedding），这是它"轻"的关键
- **代码**：`eagle.py:209` `self.model.lm_head = target_model.lm_head`。
- **精妙**：lm_head（vocab × hidden 的大矩阵）是模型里很重的一块。EAGLE 的 draft head **直接共享 target 的 lm_head**，自己只学一个小 transformer。这既省显存又让 draft 的输出空间天然和 target 对齐（提升接受率）。这正是 EAGLE 论文"在 feature 层预测、复用输出头"思想的落地。

### T12 · partial prefill 时丢弃采样 token，并回退 RNG 偏移以保证可复现
- **代码**：`gpu_model_runner.py:1115-1129`。
- **精妙**：一个请求还在 prefill 中途（`seq_len < num_tokens`）时，本步采的 token 是无效的（prompt 还没喂完）。代码不仅把它标记进 `discard_sampled_tokens_req_indices`（`:1129`）事后清空，还**回退该请求 generator 的 offset**（`:1124-1126` `set_offset(offset-4)`）—— 让 RNG 状态"假装没采过"，保证 seed 复现性。投机解码下 prefill 与 spec 混批时这个边界尤其重要。

### T13 · eagle 提议下一轮 draft 前，必须先剔除本轮被拒的位置
- **代码**：`gpu_model_runner.py:1199-1216`，`num_rejected_tokens = [n+1-len(valid[i]) ...]` → `prepare_inputs` → `[token_indices]`。
- **精妙**：draft head 要在"被接受的前缀"上续写下一轮 draft。但本轮可能有 draft 被拒，那些位置的 hidden_states 不能作为续写基础。`prepare_inputs`（`eagle.py:144`）用 Triton 内核算出"剔除被拒 token 后"的 `token_indices`，再用它切 `target_token_ids/positions/hidden_states/slot_mapping`（`:1213-1216`）。少了这步，draft 会基于被推翻的 token 续写，接受率崩盘。

### T14 · `MAX_SPEC_LEN=32` 与 `do_not_specialize` —— 防 Triton 反复重编译
- **代码**：`rejection_sampler.py:20`（`MAX_SPEC_LEN=32`）、`:331`（`MAX_NUM_TOKENS=MAX_SPEC_LEN` 固定）、`:432`/`:480`（`@triton.jit(do_not_specialize=["max_spec_len"])`）。
- **精妙**：Triton 会对不同的常量参数值分别编译。若 `max_spec_len` 每步变化就会反复 JIT。这里把它标为 `do_not_specialize`，并把 `expand_kernel` 的循环上界固定成常量 `MAX_SPEC_LEN`（`:331` 注释 "To avoid recompilation"），用一次编译覆盖所有 `k <= 32` 的情形。

### T15 · is_greedy_ptr 在 profiling 与 runtime 间的重编译陷阱（已知 FIXME）
- **代码**：`rejection_sampler.py:443-448`（FIXME 注释）。
- **精妙（也是坑）**：`rejection_greedy_sample_kernel` 的 `is_greedy_ptr` 在 profiling run 时非 None、runtime 可能为 None，导致一次额外重编译。注释把这个已知缺陷标了出来 —— 读代码时看到 `if is_greedy_ptr is None: is_greedy=True` 这种"看似冗余"的分支，背后是 Triton 编译特化的现实约束。

### T16 · ngram 在 `__init__` 里先跑一次 dummy 触发 numba JIT
- **代码**：`ngram_proposer.py:21-23`，`self.propose(np.zeros(1024, dtype=np.int32))`。
- **精妙**：`_find_subarray_kmp` 用 `@jit(nopython=True)` numba 编译，首次调用会有秒级 JIT 延迟。构造时先用 dummy 输入跑一次，把编译成本挪到初始化阶段（注释 "usually takes less than 1 second"），避免第一条真实请求被 JIT 卡住。

---

## 4. 关键边界情况与容错

| 情况 | 代码位置 | 行为 |
|---|---|---|
| 第一个 draft 就被拒 | `rejection_random_sample_kernel:529` | `rejected=True`，只输出 1 个 recovered token，等价普通 decode 一步；调度回退 `k` 个（`scheduler.py:607`） |
| k 个 draft 全 accept | `rejection_sampler.py:534-539` | 追加 bonus token，输出 `k+1` 个，回退 0 |
| ngram 找不到匹配 | `ngram_proposer.py:58` 返回 None → `gpu_model_runner.py:1268-1269` | 该请求本步不提议 draft（`spec_token_ids=[]`），下一步退化为普通 decode |
| 请求用了 min_p / penalty / per-token logprobs | `utils.py:5-18` `is_spec_decode_supported` | ngram 路径直接不提议（`gpu_model_runner.py:1258-1259`），fallback 普通 decode |
| 请求仍在 partial prefill | `gpu_model_runner.py:1120-1129` | 丢弃采样 token + 回退 RNG offset；eagle 则从 req_state 取真实 next token（`:1174-1181`） |
| token budget 不足以验证全部 draft | `scheduler.py:243-251` | 截断 `spec_token_ids`，只验证装得下的部分 |
| `k > MAX_SPEC_LEN(32)` | `rejection_sampler.py:85` `assert metadata.max_spec_len <= MAX_SPEC_LEN` | 断言失败（配置层应已限制） |
| draft_prob 恰为 0 | `rejection_sampler.py:522-524` | 视为拒绝（防 `p/q` 出 NaN/Inf） |
| 全 batch 都是 greedy | `rejection_sampler.py:191-192` | 跑完 greedy 内核直接 return，跳过 random 内核与 recovered 采样 |
| PP 非最后一段 rank | `gpu_model_runner.py:164` `get_pp_group().is_last_rank` | 不建 drafter（draft 需要最后 hidden state） |
| 上一步是纯 prefill（无 draft） | `gpu_model_runner.py:1187-1196` | eagle 用整段 `[:num_scheduled_tokens]` 作 target 输入，无需剔除被拒位置 |

---

## 5. 一图速查：投机解码调用链主干

```
[gpu_model_runner.__init__ :160]  drafter = Ngram/EagleProposer ; rejection_sampler = RejectionSampler()
        │
[① 调度] scheduler.schedule :122
        num_new_tokens = num_tokens_with_spec - num_computed_tokens          (:168  含 k 个 draft)
        allocate_slots(num_lookahead_tokens=k)                               (:200)
        scheduled_spec_decode_tokens[req] = spec_token_ids[:budget]          (:243-251  可截断)
        num_computed_tokens += num_scheduled  (乐观，含 draft)                (:455-456)
        ▼
[② 元数据] gpu_model_runner._prepare_inputs :587 → _calc_spec_decode_metadata :753
        numpy 向量化算 logits_indices / target_logits_indices / bonus_logits_indices  (:768-801)
        draft_token_ids = input_ids[logits_indices][target_logits_indices+1]          (:815)
        ▼
[③ Target forward] execute_model :998 → self.model(...) 一次 forward → logits
        bonus_logits = logits[bonus_logits_indices] → model.sample → bonus_token_ids  (:1093-1098)
        target_logits = logits[target_logits_indices]                                 (:1103)
        ▼
[④ 验证] rejection_sampler.forward :46
        compute_probs(target_logits) → target_probs                                   (:89)
        ├─ greedy:  rejection_greedy_sample_kernel  (draft==argmax 才 accept)         (:433)
        └─ random:  uniform r → sample_recovered_tokens (max(0,p−q)) → 
                    rejection_random_sample_kernel  (p/q ≥ r 才 accept)               (:481)
        output_token_ids [batch, k+1] (reject 填 -1) → parse_output 过滤              (:107)
        ▼
[⑤ 下一轮 draft] gpu_model_runner :1159
        ngram:  token_ids_cpu 写回 → NgramProposer.propose (KMP)                      (:1242 / ngram_proposer.py:25)
        eagle:  剔除被拒位置 prepare_inputs → EagleProposer.propose (自回归 k 步)      (:1209 / eagle.py:34)
        ModelRunnerOutput.spec_token_ids ──► 回传 EngineCore
        ▼
[⑥ 回填] scheduler.update_from_output :569
        num_rejected = (k+1) - len(generated) ; num_computed_tokens -= num_rejected   (:605-607)
        request.spec_token_ids = spec_token_ids[req_index]  (下一步用)                 (:628-629)
        observe(num_draft=k, num_accepted=len(generated)-1) → 接受率统计              (:608-611 / metrics.py)
```

> 对照 V0：上面 ②~④ 在 V0 里是独立的 `SpecDecodeWorker` 调 `BatchExpansionTop1Scorer`（把 batch 膨胀 `(k+1)` 倍）或 `MQAScorer` 来打分，再用 PyTorch `RejectionSampler` module 验证。V1 把这一切**整合进 `GPUModelRunner` 的一次 forward + 一组索引 + Triton 内核**，彻底去掉了 batch expansion 与独立 worker。
