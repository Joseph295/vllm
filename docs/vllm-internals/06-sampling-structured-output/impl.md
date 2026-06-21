# 模块 06 · Sampling + Structured/Guided Output —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的两条调用链：(a) 采样 logits→token；(b) 结构化输出 grammar 编译→bitmask→约束→推进 FSM。
> 所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。

---

## 1. 代码地图

```
采样 v1/sample/
  ├─ sampler.py                  # Sampler.forward：从 logits 到 token 的算子流水线
  ├─ metadata.py                 # SamplingMetadata：batch 级采样输入包
  ├─ rejection_sampler.py        # RejectionSampler：投机解码的 verify（[模块 05](../05-speculative-decoding/design.md)）
  └─ ops/
       ├─ topk_topp_sampler.py   # TopKTopPSampler + random_sample + flashinfer_sample
       ├─ penalties.py           # presence/frequency/repetition + min_token 惩罚
       └─ bad_words.py           # bad_words 前缀匹配置 -inf

采样参数 vllm/
  └─ sampling_params.py          # SamplingParams / GuidedDecodingParams / SamplingType

结构化输出 v1/structured_output/
  ├─ __init__.py                 # StructuredOutputManager：选后端/异步编译/生成 bitmask
  ├─ backend_types.py            # StructuredOutputBackend / Grammar / Options 抽象
  ├─ backend_xgrammar.py         # XgrammarBackend + XgrammarGrammar（默认后端）
  ├─ backend_guidance.py         # GuidanceBackend + GuidanceGrammar（llguidance）
  ├─ request.py                  # StructuredOutputRequest：持 Future[Grammar] / Grammar
  └─ utils.py                    # lark→ebnf、choice→grammar 等转换

集成点
  ├─ v1/engine/processor.py      # _validate_structured_output：后端选择/校验（请求入口）
  ├─ v1/engine/core.py           # add_request → grammar_init（异步编译启动）
  ├─ v1/core/sched/scheduler.py  # schedule()：WAITING_FOR_FSM、grammar_bitmask、index 分配
  │                              # update_from_output()：accept_tokens 推进 FSM
  ├─ v1/worker/gpu_input_batch.py# InputBatch._make_sampling_metadata：建 SamplingMetadata
  └─ v1/worker/gpu_model_runner.py# execute_model：compute_logits→apply_grammar_bitmask→sample
                                  # apply_grammar_bitmask：切片+重排+xgr in-place 掩码
```

---

## 2. 调用链 (a)：采样 —— logits → token

### A0. 准备 —— 参数如何变成 batch 级张量

**A0-1.** `vllm/sampling_params.py:510` `SamplingParams.sampling_type`
温度与 seed 决定采样类型：
```python
def sampling_type(self) -> SamplingType:
    if self.temperature < _SAMPLING_EPS:
        return SamplingType.GREEDY
    if self.seed is not None:
        return SamplingType.RANDOM_SEED
    return SamplingType.RANDOM
```
> `temperature < 1e-5` 即贪心；带 seed 即 RANDOM_SEED（需 per-request generator）。

**A0-2.** `vllm/v1/worker/gpu_input_batch.py:269-304` `InputBatch.add_request()`
请求加入持久化 batch 时，把每个标量参数写进对应 `_cpu` 数组的某一行：
```python
if sampling_params.sampling_type == SamplingType.GREEDY:
    self.temperature_cpu[req_index] = -1.0      # 贪心：占位 -1 避免后续除零
    self.greedy_reqs.add(req_id)
else:
    self.temperature_cpu[req_index] = sampling_params.temperature
    self.random_reqs.add(req_id)
...
self.top_p_cpu[req_index] = sampling_params.top_p
...
```
> `greedy_reqs` / `random_reqs` 两个集合决定后面的 `all_greedy` / `all_random` fast-path（`:643-647`）。

**A0-3.** `gpu_model_runner.py:331-335` —— **per-request generator 的诞生**
```python
if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
    generator = torch.Generator(device=self.device)
    generator.manual_seed(sampling_params.seed)
else:
    generator = None
```
> 只有带 seed 的请求才建独立 `torch.Generator`。它随 `CachedRequestState` 保存，进 `InputBatch.generators[req_index]`（`gpu_input_batch.py:308-309`）。**不带 seed 的请求不进这个 dict**——这是 `random_sample` 批处理优化的前提（见 A4）。

**A0-4.** `gpu_input_batch.py:537` `InputBatch._make_sampling_metadata()`
每次 batch 变动后重建 `SamplingMetadata`，把 `_cpu` 数组 `copy_slice` 到 GPU 张量（且仅当该特性被用到时才拷，省同步）：
```python
if not self.all_greedy:
    temperature = copy_slice(self.temperature_cpu_tensor, self.temperature, num_reqs)
else:
    temperature = None
...
if not self.no_penalties:
    copy_slice(...frequency...); copy_slice(...presence...); copy_slice(...repetition...)
    prompt_token_ids = self._make_prompt_token_ids_tensor()
else:
    prompt_token_ids = None
...
return SamplingMetadata(temperature=temperature, all_greedy=..., generators=self.generators, ...)
```
> 关键：`top_p=None if self.no_top_p else ...`（`:580`）。**整个 batch 没人用 top_p，张量就是 None，算子被整段跳过。**

### A1. 入口 —— `Sampler.forward`

**A1-1.** `vllm/v1/sample/sampler.py:23` `Sampler.forward(logits, sampling_metadata)`
```python
num_logprobs = sampling_metadata.max_num_logprobs
if num_logprobs is not None:
    raw_logprobs = self.compute_logprobs(logits)   # ← 惩罚/温度之前先存 logprobs
logits = logits.to(torch.float32)
logits = self.apply_allowed_token_ids(logits, sampling_metadata)
logits = self.apply_bad_words(logits, sampling_metadata)
logits = self.apply_logits_bias(logits, sampling_metadata)
logits = self.apply_penalties(logits, sampling_metadata)
sampled = self.sample(logits, sampling_metadata)
sampled = sampled.long()
```
> 注释（`:28-33`）点明：logprobs 用**原始 logits** 算，与 V0 不同。这样返回给用户的 logprobs 是模型真实分布，不被温度/惩罚扭曲。

### A2. logits 预处理算子（按 batch 索引就地修改）

**A2-1.** `sampler.py:249` `apply_allowed_token_ids`
```python
if sampling_metadata.allowed_token_ids_mask is not None:
    logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))
```
> `allowed_token_ids_mask` 是 `[batch, vocab]` 的 bool 张量，True 处即"不允许"，直接 `masked_fill_` 成 -inf。

**A2-2.** `sampler.py:259` `apply_bad_words` → `ops/bad_words.py:31`
```python
for i, bad_words_ids in bad_words_token_ids.items():
    _apply_bad_words_single_batch(logits[i], bad_words_ids, past_tokens_ids[i])
```
`_apply_bad_words_single_batch`（`bad_words.py:8`）对每个 bad word 序列：**只有当已生成的尾部 token 恰好等于该 bad word 的前缀时**，才把最后一个 token 置 -inf（`:27-28`）——即"只在会真正组成 bad word 时才禁掉那一步"。

**A2-3.** `sampler.py:225` `apply_logits_bias`
```python
for i, logit_bias in enumerate(sampling_metadata.logit_bias):
    if logit_bias:
        for token_id, bias in logit_bias.items():
            logits[i, token_id] += bias
```
> 注释（`:230`）自承"极低效"，是逐请求逐 token 的 Python 循环，留待优化。

**A2-4.** `sampler.py:182` `apply_penalties` → `ops/penalties.py`
```python
if sampling_metadata.min_tokens:
    apply_min_token_penalties(logits, output_token_ids, min_tokens)
if not sampling_metadata.no_penalties:
    logits = apply_all_penalties(logits, prompt_token_ids, presence, frequency, repetition, output_token_ids)
```
- `apply_min_token_penalties`（`penalties.py:9`）：还没生成够 `min_tokens` 的请求，把其 stop/eos token 置 -inf，**阻止提前结束**。
- `apply_all_penalties`（`penalties.py:25`）→ `model_executor/layers/utils.py:25` `apply_penalties`：用 `scatter_add_` 统计 prompt/output 的 token bin counts（`get_token_bin_counts_and_mask`，`utils.py:8`），然后：
  ```python
  logits[logits > 0]  /= where(mask, repetition_penalties, 1.0)[logits > 0]
  logits[logits <= 0] *= where(mask, repetition_penalties, 1.0)[logits <= 0]
  logits -= frequency_penalties.unsqueeze(1) * output_bin_counts
  logits -= presence_penalties.unsqueeze(1)  * output_mask
  ```
  > repetition 对正/负 logits 分别做除/乘（保持方向），frequency 按出现次数线性扣分，presence 按是否出现扣分——全 batch 向量化，遵循 OpenAI API 定义（`utils.py:54`）。

### A3. `sample` —— 贪心 / 随机分支

**A3-1.** `sampler.py:85` `Sampler.sample()`
```python
assert not (all_greedy and all_random)
if sampling_metadata.all_random:
    greedy_sampled = None
else:
    greedy_sampled = self.greedy_sample(logits)     # argmax
    if sampling_metadata.all_greedy:
        return greedy_sampled                        # 全贪心 fast-path，直接返回

logits = self.apply_temperature(logits, sampling_metadata.temperature)  # /temp
if sampling_metadata.min_p is not None:
    logits = self.apply_min_p(logits, sampling_metadata.min_p)
random_sampled = self.topk_topp_sampler(logits, generators, top_k, top_p)

if greedy_sampled is None:
    return random_sampled                            # 全随机 fast-path

sampled = torch.where(
    sampling_metadata.temperature < _SAMPLING_EPS,
    greedy_sampled, random_sampled, out=greedy_sampled)  # 逐行二选一，复用张量
```
- **`greedy_sample`**（`:82`）：`logits.argmax(dim=-1)`。
- **`apply_temperature`**（`:74`）：`logits.div_(temp.unsqueeze(1))`，in-place 除。
- **混批选择**（`:125`）：`torch.where` 按每行 `temperature < 1e-5` 在贪心/随机结果间选，`out=greedy_sampled` 复用张量省显存。

**A3-2.** `apply_min_p`（`sampler.py:203`）
```python
probability_values = softmax(logits, dim=-1)
max_probabilities = amax(probability_values, dim=-1, keepdim=True)
adjusted_min_p = min_p.unsqueeze(1) * max_probabilities   # 阈值 = min_p × 峰值
valid_token_mask = probability_values >= adjusted_min_p
logits[~valid_token_mask] = -float('inf')
```

### A4. top-k / top-p + 随机采样 —— `TopKTopPSampler`

**A4-1.** `ops/topk_topp_sampler.py:29` `TopKTopPSampler.__init__` —— **实现选择**
构造时根据平台/库可用性把 `self.forward` 绑到不同实现：
- CUDA + FlashInfer 可用且 `<0.2.3` 且未禁用 → `forward_cuda`（`:60`）。
- FlashInfer `>=0.2.3` → 因 API 不兼容**回退** `forward_native`（`:44-49`）。
- TPU → `forward_tpu`（`:82`）；其余 → `forward_native`。

**A4-2.** `forward_native`（`:86`）
```python
logits = apply_top_k_top_p(logits, k, p)
probs = logits.softmax(dim=-1, dtype=torch.float32)
return random_sample(probs, generators)
```

**A4-3.** `apply_top_k_top_p`（`:167`）—— top-k/top-p 掩码
```python
if p is None:
    if k is None: return logits
    return apply_top_k_only(logits, k)         # 只 top-k，不排序整个 vocab
logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
if k is not None:                              # top-k
    top_k_mask = logits_sort.gather(1, (size - k).unsqueeze(1))
    logits_sort.masked_fill_(logits_sort < top_k_mask, -inf)
if p is not None:                              # top-p（核采样）
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum  = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(1)
    top_p_mask[:, -1] = False                  # 至少保留一个 token
    logits_sort.masked_fill_(top_p_mask, -inf)
logits = logits_sort.scatter(-1, logits_idx, logits_sort)  # 还原原顺序
```
> `apply_top_k_only`（`:210`）用 `logits.topk(max_top_k).values.gather(...)` 取第 k 大值作阈值，**避免对整个 vocab 排序**——只有 top-k 时走这条快路径。

**A4-4.** `random_sample`（`:235`）—— **Gumbel-max / 无同步随机采样**
```python
q = torch.empty_like(probs)
if len(generators) != probs.shape[0]:          # 不是"每个请求都有 seed"
    q.exponential_()                           # 整批一次性填 Exp(1)
if generators:
    for i, generator in generators.items():    # 仅带 seed 的请求逐行覆盖
        q[i].exponential_(generator=generator)
return probs.div_(q).argmax(dim=-1).view(-1)   # argmax(p / Exp(1)) ≡ 多项式采样
```
> 注释（`:241`）：刻意不用 `torch.multinomial`（会 CPU-GPU 同步）。批处理优化（`:245-248`）：先假设**无人带 seed**，整批 `q.exponential_()`；再只为带 seed 的请求逐行用其 generator 重填——这就是为什么 A0-3 里**不带 seed 的请求不进 generators dict**。

**A4-5.** `forward_cuda` → `flashinfer_sample`（`:259`）—— **拒绝采样，避免排序**
```python
probs = logits.softmax(dim=-1, dtype=torch.float32)
if k is None and p is None:
    return random_sample(probs, generators)    # 无 k/p 时仍走无同步路径
return flashinfer_sample(probs, k, p, generators)
```
`flashinfer_sample` 用 `flashinfer.sampling.top_k_top_p_sampling_from_probs(...)`（`:301`）做拒绝采样，**不排序 vocab**；末尾检查 `success.all()`（**此处 CPU-GPU 同步**，`:304-305`），失败则 `top_k_renorm_prob` / `top_p_renorm_prob` 后重采（`:307-311`）。

### A5. 收尾 —— logprobs 与打包

**A5-1.** `sampler.py:54-71`
```python
sampled = sampled.long()
logprobs_tensors = None if num_logprobs is None else \
    self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)
sampled = sampled.to(torch.int32)
return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=logprobs_tensors)
```
- `gather_logprobs`（`:136`）：对 `raw_logprobs` 取 topk + 拼上采样 token 自己的 logprob，并算 rank（`token_ranks = (logprobs >= token_logprob).sum(-1)`，`:171`）。
- `sampled` 转 int32 减小张量（`:62`）；FlashInfer 返回 int32 而 torch 返回 int64，故前面统一 `.long()`（`:50-54` 注释）。

### A6. runner 串接 + spec decode 衔接

**A6-1.** `gpu_model_runner.py:1075-1097` `execute_model`
```python
logits = self.model.compute_logits(sample_hidden_states, None)
if scheduler_output.grammar_bitmask is not None:
    self.apply_grammar_bitmask(scheduler_output, logits)   # ← 链 (b)，在采样前
sampling_metadata = self.input_batch.sampling_metadata
if spec_decode_metadata is None:
    sampler_output = self.model.sample(logits=logits, sampling_metadata=sampling_metadata)
else:
    bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
    sampler_output = self.model.sample(logits=bonus_logits, sampling_metadata=sampling_metadata)
    target_logits = logits[spec_decode_metadata.target_logits_indices]
    output_token_ids = self.rejection_sampler(spec_decode_metadata, None, target_logits, bonus_token_ids, sampling_metadata)
    sampler_output.sampled_token_ids = output_token_ids
```
> 无投机解码时直接 `sample`；有则 `RejectionSampler` 接管（[模块 05](../05-speculative-decoding/design.md)）。

**A6-2.** chunked prefill 时丢弃中间采样 token + **回退 generator**（`:1115-1129`）
```python
if seq_len < req_state.num_tokens:               # 这一步只是部分 prefill，不该产 token
    generator = self.input_batch.generators.get(i)
    if generator is not None:
        generator.set_offset(generator.get_offset() - 4)   # 倒回随机数游标
    discard_sampled_tokens_req_indices.append(i)
```
> 精妙：partial prefill 步采样出的 token 要丢弃，但随机数已被"消费"。这里 `set_offset(-4)` **倒回 generator 游标**，让下一步采样得到与"从未采过"完全一致的随机序列，保证 seed 复现不被 chunked prefill 破坏。

---

## 3. 调用链 (b)：结构化输出 —— grammar 编译 → bitmask → 约束 → 推进 FSM

### B1. 请求入口 —— 后端选择与校验

**B1-1.** `vllm/v1/engine/processor.py:79` `_validate_sampling_params` → `:144` `_validate_structured_output`
```python
supported_backends = ["xgrammar", "xgrammar:disable-any-whitespace",
                      "guidance", "guidance:disable-any-whitespace", "auto"]
engine_level_backend = self.decoding_config.guided_decoding_backend
...
if engine_level_backend.startswith("xgrammar"):
    validate_xgrammar_grammar(params)               # 解析校验（不编译）
elif engine_level_backend == "auto":
    try:
        validate_xgrammar_grammar(params)
        params.guided_decoding.backend = "xgrammar"
    except ValueError:
        params.guided_decoding.backend = "guidance" # 命中 xgrammar 不支持特性 → 回退
if engine_level_backend.startswith("guidance"):
    validate_guidance_grammar(params, tokenizer=None)
```
> `auto` 的回退靠 `validate_xgrammar_grammar` 抛 `ValueError`（来自 `has_xgrammar_unsupported_json_features`，`backend_xgrammar.py:166`：pattern / 数值范围 / minItems / format 等）。**请求级选后端已被禁止**（`:156-164`），只能用引擎级。

**B1-2.** `vllm/v1/request.py:41-43` —— 初始状态
```python
self.status = (RequestStatus.WAITING_FOR_FSM
               if sampling_params.guided_decoding is not None else
               RequestStatus.WAITING)
```
> 结构化请求一进来就是 `WAITING_FOR_FSM`（`RequestStatus` 枚举 `:152`），等编译完成。

### B2. 异步编译 —— `grammar_init`

**B2-1.** `vllm/v1/engine/core.py:180-182` `EngineCore.add_request`
```python
if req.use_structured_output:
    self.structured_output_manager.grammar_init(req)   # 启动异步编译
```

**B2-2.** `vllm/v1/structured_output/__init__.py:39` `StructuredOutputManager.grammar_init`
```python
if self.backend is None:                               # 首次用到才初始化后端
    backend_name = request.sampling_params.guided_decoding.backend_name
    if backend_name == "xgrammar":
        self.backend = XgrammarBackend(self.vllm_config)
    elif backend_name == "guidance":
        self.backend = GuidanceBackend(self.vllm_config)
grammar = self.executor.submit(self._async_create_grammar, request)  # 丢线程池
request.structured_output_request.grammar = grammar    # 存 Future
```
> `executor` 是 `ThreadPoolExecutor`，`max_workers = (cpu_count+1)//2`（`:36`，注释解释：grammar 编译是 CPU-bound，不能用默认的 cpu*5）。`_async_create_grammar`（`:63`）取 `structured_output_key`（`(request_type, grammar_spec)`）调 `backend.compile_grammar`。

**B2-3.** `vllm/v1/structured_output/backend_xgrammar.py:92` `XgrammarBackend.compile_grammar`
```python
if request_type == StructuredOutputOptions.JSON:
    ctx = self.compiler.compile_json_schema(grammar_spec, any_whitespace=...)
elif request_type == StructuredOutputOptions.REGEX:
    ctx = self.compiler.compile_regex(grammar_spec)
elif request_type == StructuredOutputOptions.GRAMMAR:
    ctx = self.compiler.compile_grammar(grammar_spec)
...
return XgrammarGrammar(matcher=xgr.GrammarMatcher(ctx), vocab_size=self.vocab_size, ctx=ctx)
```
> `compiler` 是 `xgr.GrammarCompiler`（`:85`），带缓存（`cache_enabled=True`，`cache_limit_bytes` 由 `VLLM_XGRAMMAR_CACHE_MB` 控制）。每个请求得到一个独立的 `GrammarMatcher`（FSM 实例，持自己的状态）。

### B3. 调度时检查编译完成 + 分配 bitmask index

**B3-1.** `vllm/v1/core/sched/scheduler.py:284-291` —— `WAITING_FOR_FSM` 门控
```python
if request.status == RequestStatus.WAITING_FOR_FSM:
    structured_output_req = request.structured_output_request
    if structured_output_req and structured_output_req.grammar:   # ← 触发 ready 检查
        request.status = RequestStatus.WAITING
    else:
        self.waiting.popleft()
        skipped_waiting_requests.appendleft(request)               # 编译没好，本步跳过
        continue
```
> `.grammar` 属性（`structured_output/request.py:42`）内部调 `_check_grammar_completion`（`:24`）：`self._grammar.result(timeout=0.0001)` **非阻塞**地探 Future。未完成抛 `TimeoutError` 返 None → 请求继续等；完成则替换为编译好的 grammar 并返回。**这就是 grammar 异步编译与调度的衔接点。**

**B3-2.** 给结构化请求分配 batch index（running 与 waiting 两处）
- running 队列：`scheduler.py:229-234`
  ```python
  if request.use_structured_output:
      structured_output_request_ids[request.request_id] = req_index
  ```
- waiting 队列：`scheduler.py:339-341`（同理）。
> `structured_output_request_ids` 把 `req_id → 调度顺序 index` 记下，注释（`:139-144`）说明用途：切 bitmask、只对结构化请求 apply mask。

### B4. 生成 bitmask

**B4-1.** `scheduler.py:399-403` —— schedule() 末尾调用
```python
grammar_bitmask = self.structured_output_manager.grammar_bitmask(
    self.requests, structured_output_request_ids, len(self.running))
...
scheduler_output = SchedulerOutput(..., structured_output_request_ids=..., grammar_bitmask=grammar_bitmask)
```

**B4-2.** `vllm/v1/structured_output/__init__.py:79` `StructuredOutputManager.grammar_bitmask`
```python
if not structured_output_request_ids:
    return None                                         # 本步无结构化请求，省掉一切
if self._grammar_bitmask is None:
    self._grammar_bitmask = self.backend.allocate_token_bitmask(max_num_seqs)  # 复用分配
bitmask_tensor = self._grammar_bitmask
for req_id, batch_index in structured_output_request_ids.items():
    request = requests[req_id].structured_output_request
    if not request.grammar.is_terminated():
        request.grammar.fill_bitmask(bitmask_tensor, batch_index)   # 填该请求合法 token 位图
if batch_len < self._grammar_bitmask.shape[0]:
    bitmask_tensor = self._grammar_bitmask[:batch_len]              # 切到 batch 大小
return bitmask_tensor.numpy()                                       # np.ndarray 过界更高效
```
- `allocate_token_bitmask`（xgrammar `:118`）：`xgr.allocate_token_bitmask(max_num_seqs, vocab_size)`，形状 `[max_num_seqs, ceil(vocab/32)]`，每个 int32 压 32 个 token 的合法位。**只分配一次复用**。
- `fill_bitmask`（`backend_xgrammar.py:155`）：`self.matcher.fill_next_token_bitmask(bitmask, idx)`——把"当前 FSM 状态下合法的 token"写进第 `idx` 行。
- **已 terminated 的 grammar 跳过填充**（`:101`）——它不再约束（终止后该位置保持上一次/全零，后续逻辑不再 apply）。

### B5. 应用 bitmask 到 logits（GPU runner）

**B5-1.** `gpu_model_runner.py:1077-1079` execute_model 中
```python
if scheduler_output.grammar_bitmask is not None:
    self.apply_grammar_bitmask(scheduler_output, logits)
```

**B5-2.** `gpu_model_runner.py:939` `apply_grammar_bitmask` —— **按 running index 切片 + 重排对齐**
```python
grammar_bitmask = scheduler_output.grammar_bitmask
if grammar_bitmask is None: return
struct_out_req_batch_indices = {}
indices_match = True
for req_id in self.input_batch.req_ids:
    mask_index = scheduler_output.structured_output_request_ids.get(req_id)
    if mask_index is None: continue                    # 非结构化请求，跳过
    batch_index = self.input_batch.req_id_to_index[req_id]
    if batch_index != mask_index:
        indices_match = False                          # scheduler 顺序 ≠ batch 顺序
    struct_out_req_batch_indices[req_id] = batch_index
if not indices_match:                                  # 重排 bitmask 行对齐 batch
    sorted_bitmask = np.zeros_like(grammar_bitmask)
    for req_id, batch_index in struct_out_req_batch_indices.items():
        orig_index = scheduler_output.structured_output_request_ids[req_id]
        sorted_bitmask[batch_index] = grammar_bitmask[orig_index]
    grammar_bitmask = sorted_bitmask
grammar_bitmask = torch.from_numpy(grammar_bitmask)
xgr.apply_token_bitmask_inplace(                       # ← 非法 token logits = -inf
    logits,
    grammar_bitmask.to(self.device, non_blocking=True),
    indices=list(struct_out_req_batch_indices.values()))
```
> 关键边界（注释 `:950-954`）：scheduler 按调度顺序填 bitmask，但 `InputBatch` 因持久化/swap 排列可能不同，故须**逐行重排**对齐，否则会把 A 请求的合法集合错套到 B 请求上。`apply_token_bitmask_inplace` 是 xgrammar 的 CUDA kernel，对 `indices` 指定的行把 bitmask 为 0 的 token 置 -inf。`# TODO: compatibility with spec decode`（`:979`）——目前 bitmask 与投机解码不兼容。

### B6. 采样后推进 FSM

**B6-1.** `vllm/v1/core/sched/scheduler.py:655-660` `update_from_output` 中
```python
if new_token_ids and request.use_structured_output:
    request.structured_output_request.grammar.accept_tokens(req_id, new_token_ids)
```

**B6-2.** `backend_xgrammar.py:140` `XgrammarGrammar.accept_tokens`
```python
for token in tokens:
    if not self.matcher.accept_token(token):
        logger.error("Failed to advance FSM for request %s ...", request_id, token)
        return False
    self.num_processed_tokens += 1
return True
```
> 把刚采样出的 token 喂回 matcher 推进状态。理论上因为 bitmask 已约束过，token 必然合法；若失败说明 bug（记 error）。下一步 `fill_bitmask` 就会基于新状态给出新的合法集合。`is_terminated`（`:158`）= `matcher.is_terminated()`，用于 B4-2 跳过填充与 B6 之后的停止。

**B6-3.** guidance 后端对应实现（`backend_guidance.py:92` `accept_tokens` / `:118` `fill_bitmask`）：
```python
def accept_tokens(self, request_id, tokens):
    if self.ll_tokenizer.eos_token in tokens:
        self.terminated = True
    if self.ll_matcher.is_stopped(): return True
    r = self.ll_matcher.consume_tokens(tokens)
    return r
def fill_bitmask(self, bitmask, idx):
    llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
```
> 接口与 xgrammar 完全对称（`StructuredOutputGrammar` 抽象，`backend_types.py:20`），故 §B4/§B6 的调用点对两个后端是同一套代码。

---

## 4. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · 随机采样不用 `multinomial`，用 `argmax(p / Exp(1))`
- `topk_topp_sampler.py:235-256` `random_sample`。
- `torch.multinomial` 要把分布读回 CPU 做累加，触发 **CPU-GPU 同步**，在每步热路径上代价高。`probs.div_(q).argmax()`（`q ~ Exp(1)`）是 Gumbel-max trick 的指数变体，统计上等价于多项式采样，**全程在 GPU、零同步**。这是 V1 采样吞吐的核心一招。

### T2 · "先假设无人带 seed"的批处理优化
- `topk_topp_sampler.py:249-255`。
- 常见情况是**整批没人带 seed**。所以先 `q.exponential_()` 整批一次性填随机数（向量化），再**只为带 seed 的请求**逐行 `q[i].exponential_(generator=...)` 覆盖。代价是带 seed 的请求要逐个处理（`:253` 注释承认慢），但换来了"无 seed 多数请求"的全向量化。配套前提：A0-3 让无 seed 请求**不进** `generators` dict。

### T3 · per-request generator 只覆盖自己那一行 → batch 无关复现
- `gpu_model_runner.py:331-333` 建 generator；`topk_topp_sampler.py:254-255` 只覆盖 `q[i]`。
- 带 seed 的请求若用全局随机源，结果会随"同 batch 来了哪些别的请求"变化，无法复现。这里每请求独立 `torch.Generator(seed)` 只决定**自己那一行**的指数随机数，因此**同一 seed + 同一 prompt 必得同一序列**，与 batch 组成无关。

### T4 · chunked prefill 倒回 generator 游标
- `gpu_model_runner.py:1120-1126`，`generator.set_offset(generator.get_offset() - 4)`。
- partial prefill 步会"顺手"采样一个 token 但必须丢弃。问题是随机数游标已前进。这里**把 generator 游标倒回 4**（依赖 CUDA generator 内部实现细节），使得真正该采样的那一步拿到与"从未采过"一致的随机数——否则 chunked prefill 会悄悄破坏 seed 复现。

### T5 · logprobs 用"惩罚/温度之前"的原始 logits
- `sampler.py:28-36`，`raw_logprobs = self.compute_logprobs(logits)` 在所有算子之前。
- V0 返回的是"采样用的 logits（含温度/惩罚）"的 logprobs，会被扭曲。V1 刻意先存原始分布的 logprobs，**返回给用户的概率是模型真实的**，与采样实际用的分布解耦。注释里还附了讨论链接说明这是有意为之。

### T6 · `all_greedy` / `all_random` / `no_*` 全是 batch 级 fast-path
- `sampler.py:96-103`；`gpu_input_batch.py:537-595`。
- batch 全贪心则 `sample` 直接 `argmax` 返回，不进随机分支；全随机则不算 argmax。更进一步，`SamplingMetadata` 里只要 batch 没人用某特性，对应张量就是 `None`（top_p/top_k/min_p/penalties），整段算子被 `if ... is not None` 跳过。**"不用的特性零开销"** 贯穿整个采样链。

### T7 · 混批用 `torch.where(..., out=greedy_sampled)` 复用张量
- `sampler.py:125-130`。
- 贪心与随机请求混在一个 batch 时，两条分支都算，再用 `torch.where` 按每行 temperature 选。`out=greedy_sampled` 让结果写回已有张量，**省一次分配**。小处见功夫，但在每步都跑的热路径上累积可观。

### T8 · top-k only 不排序整个 vocab
- `topk_topp_sampler.py:210` `apply_top_k_only`。
- top-p 必须排序（`logits.sort`，对大 vocab 慢）。但**只有 top-k 时**用 `logits.topk(max_top_k).values.gather(...)` 取第 k 大值作阈值即可，避免全排序。`apply_top_k_top_p` 在 `p is None` 时刻意走这条快路径（`:179-184`）。

### T9 · FlashInfer 拒绝采样避免排序，但代价是一次同步
- `topk_topp_sampler.py:259-312` `flashinfer_sample`。
- FlashInfer 用拒绝采样直接在 probs 上采，**完全不排序 vocab**，对超大 vocab 更快。代价是末尾 `success.all()` 检查引入 CPU-GPU 同步（`:304`），所以注释强调"放在 forward 末尾以摊薄同步开销"，且 `k is None and p is None` 时退回无同步的 `random_sample`（`:111-115`）。失败回退到 renorm + 重采保证正确性。

### T10 · bad_words 只在"会真正组成 bad word"那一步禁
- `ops/bad_words.py:8-28`。
- 不是无脑禁掉 bad word 的所有 token，而是**仅当已生成尾部恰好等于 bad word 的前缀**时，才把其最后一个 token 置 -inf。这样 bad word 的子串若出现在别处仍合法，只精确阻断"正在组成完整 bad word"的那一刻。

### T11 · 共享 bitmask 张量只分配一次，按 max_num_seqs 预留
- `structured_output/__init__.py:89-92`，`allocate_token_bitmask(max_num_seqs)`。
- bitmask 是 `[max_num_seqs, vocab/32]` 的大张量。`StructuredOutputManager` **只在首次需要时分配一次**并缓存进 `self._grammar_bitmask`，之后每步复用、按 `batch_len` 切片（`:103-104`）。避免每步重新分配大张量。

### T12 · bitmask 用 `np.ndarray` 过界，不用 tensor
- `structured_output/__init__.py:106-109`（转 `.numpy()`）；`gpu_model_runner.py:944-946`（注释）。
- bitmask 在 scheduler 进程（CPU）生成，要随 `SchedulerOutput` 序列化广播给 worker。注释明示 **`np.ndarray` 的序列化比 tensor 高效得多**，故在 CPU 端就转 numpy，到 GPU 才 `torch.from_numpy(...).to(device, non_blocking=True)`。

### T13 · bitmask 行重排对齐 batch 顺序
- `gpu_model_runner.py:955-975`。
- scheduler 按"调度顺序"填 bitmask 行（`structured_output_request_ids` 的 index），但 GPU runner 的 `InputBatch` 因持久化/swap 顺序可能不同。runner 先比对 index 是否一致，不一致则**逐行重排** `sorted_bitmask[batch_index] = grammar_bitmask[orig_index]`。少了这步会把约束套错请求。

### T14 · `WAITING_FOR_FSM` 是"异步编译"与"GPU 不空转"的接缝
- `request.py:41`、`scheduler.py:284-291`、`structured_output/request.py:24-35`。
- 文法编译是 CPU 重活，同步编译会让 GPU 等。解法：请求先进 `WAITING_FOR_FSM`，编译丢线程池返回 Future。调度器每步用 `result(timeout=0.0001)` **非阻塞探测**，没好就把请求放回 waiting 头部跳过本步，好了才转 `WAITING` 正常调度。这样 GPU 永远去算"文法已就绪"的请求，**编译与推理重叠**。

### T15 · `auto` 后端靠"校验抛异常"做回退
- `processor.py:173-184` + `backend_xgrammar.py:166` `has_xgrammar_unsupported_json_features`。
- `auto` 不是跑时择优，而是在**请求入口**先用 `validate_xgrammar_grammar` 校验：若 JSON Schema 含 xgrammar 不支持的特性（pattern/数值范围/minItems/format…）会抛 `ValueError`，`except` 分支即把后端定为 guidance。把"特性兼容性判断"前移到入口，避免编译期才失败。

### T16 · terminated 的 grammar 跳过填充
- `structured_output/__init__.py:101` `if not request.grammar.is_terminated():`。
- 一旦文法到达终止态（如 JSON 闭合），它不再需要约束后续 token（请求很可能也该停了）。跳过 `fill_bitmask` 既省 CPU，也避免对已完成的结构化请求继续施加无意义的掩码。

---

## 5. 关键边界情况与容错

| 情况 | 代码位置 | 行为 |
|---|---|---|
| 全贪心 batch | `sampler.py:102-103` | `sample` 直接 `argmax` 返回，跳过温度/top-k/top-p/随机分支 |
| 全随机 batch | `sampler.py:98-99,122-123` | 不算 `greedy_sample`，`random_sampled` 直接返回 |
| 某采样特性全 batch 无人用 | `gpu_input_batch.py:580-582` | 对应张量为 `None`，`Sampler` 中整段 `if not None` 算子被跳过 |
| temperature < 1e-5 | `sampling_params.py:510-512`；`gpu_input_batch.py:270-272` | 判为 GREEDY；InputBatch 中 temperature 置 -1.0 避免除零 |
| 带 seed 请求 | `gpu_model_runner.py:331-333` | 建 per-request `torch.Generator`，进 `generators` dict |
| chunked prefill 的中间步 | `gpu_model_runner.py:1120-1129` | 丢弃采样 token，`set_offset(-4)` 倒回 generator |
| top-p/top-k 把所有 token 砍光 | `topk_topp_sampler.py:202`、`:157` | `top_p_mask[:, -1] = False`，强制至少保留一个 token |
| FlashInfer 采样失败 | `topk_topp_sampler.py:305-311` | renorm 概率后用 `sampling_from_probs` 重采 |
| FlashInfer ≥ 0.2.3 | `topk_topp_sampler.py:34-49` | API 不兼容，**默认禁用**，回退 PyTorch-native |
| logit_bias token 越界 | `sampler.py:241-245` | 抛 `ValueError`（out-of-vocab token id） |
| min_tokens 未达 | `ops/penalties.py:17-22` | stop/eos token 置 -inf，阻止提前结束 |
| grammar 编译未完成 | `scheduler.py:284-291` | 请求留在 `WAITING_FOR_FSM`，本步跳过，下次再探 Future |
| grammar 已 terminated | `structured_output/__init__.py:101` | 跳过 `fill_bitmask`，不再约束 |
| bitmask index 与 batch 顺序不一致 | `gpu_model_runner.py:968-975` | 逐行重排 `sorted_bitmask` 对齐 batch |
| 本步无结构化请求 | `structured_output/__init__.py:86-87` | `grammar_bitmask` 返回 `None`，`apply_grammar_bitmask` 直接 return |
| accept_token 推进失败 | `backend_xgrammar.py:147-151` | 记 error 日志并返回 False（理论上不应发生，因 bitmask 已约束） |
| xgrammar 不支持的 JSON 特性 | `backend_xgrammar.py:166-213` | `validate_xgrammar_grammar` 抛 `ValueError`；`auto` 模式据此回退 guidance |
| 结构化输出 + 投机解码 | `gpu_model_runner.py:979` | TODO：当前不兼容（`apply_token_bitmask_inplace` 未处理 spec） |

---

## 6. 一图速查：两条调用链

```
链 (a) 采样 ──────────────────────────────────────────────────────────────
 InputBatch.add_request :269   参数写入 _cpu 数组；RANDOM_SEED → torch.Generator
       │                       (gpu_model_runner :331)
 _make_sampling_metadata :537  copy_slice → SamplingMetadata（[num_reqs] 张量 + generators）
       ▼
 gpu_model_runner.execute_model :1075  compute_logits
       │  (若结构化) apply_grammar_bitmask :1078 ──► 见链 (b) B5
       ▼
 Sampler.forward :23
   raw_logprobs = log_softmax(logits)              ← 惩罚/温度前
   apply_allowed_token_ids :249 / apply_bad_words :259 / apply_logits_bias :225 / apply_penalties :182
   sample :85
     greedy = argmax                               （all_greedy → 直接返回 :103）
     /temperature :74 → apply_min_p :203
     topk_topp_sampler :115
        ├ forward_native :86  → apply_top_k_top_p :167 → random_sample :235 (p/Exp(1) argmax, 无同步)
        └ forward_cuda  :102 → flashinfer_sample :259 (拒绝采样, 末尾同步)
     where(temp<eps, greedy, random) :125
   gather_logprobs :136 → SamplerOutput(sampled_token_ids[num_reqs,1])

链 (b) 结构化输出 ────────────────────────────────────────────────────────
 processor._validate_structured_output :144   选后端(xgrammar/guidance/auto)+校验
       ▼
 Request.status = WAITING_FOR_FSM :41
 core.add_request :180 → StructuredOutputManager.grammar_init :39
       └─ executor.submit(_async_create_grammar) → backend.compile_grammar :92  (线程池, GrammarMatcher)
       ▼ 每步 schedule()
 scheduler :284  WAITING_FOR_FSM：grammar Future ready? (result timeout=0.0001) → WAITING
 scheduler :229/:339  structured_output_request_ids[req_id] = 调度 index
 scheduler :399 → StructuredOutputManager.grammar_bitmask :79
       └─ 每请求 grammar.fill_bitmask(bitmask, idx) :155  → np.ndarray 过界
       ▼ GPU runner
 apply_grammar_bitmask :939  按 req 切片 + index 不匹配则重排 → xgr.apply_token_bitmask_inplace :980
       └─ 非法 token logits = -inf
       ▼ 采样 (链 a) 产出 token
 scheduler.update_from_output :655  grammar.accept_tokens(req_id, [token]) :140  推进 FSM
       └─ is_terminated() 后停止后续 fill_bitmask
```
