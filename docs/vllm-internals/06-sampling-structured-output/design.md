# 模块 06 · Sampling + Structured/Guided Output —— 设计文档

> 范围：模型 forward 算出 `hidden_states` 之后、一个 token 真正"诞生"之前的全部逻辑。分两条线：
> 1. **采样（Sampling）**：把 `logits` 变成下一个 token id —— 贪心 vs 随机、温度 / top-k / top-p / min-p / penalty / bad_words / logit_bias，全部在 GPU 上**批量并行**完成。
> 2. **结构化输出（Structured / Guided Output）**：强制模型输出符合 JSON Schema / 正则 / EBNF 文法 / choice 的约束——用 **bitmask 把非法 token 的 logits 置 `-inf`**，再在采样后推进 **FSM/grammar matcher**。
>
> 架构：**V1 为主**（`vllm/v1/sample/`、`vllm/v1/structured_output/`）。本模块是 [模块 00](../00-request-lifecycle/design.md) §F"② 执行 —— GPU forward + 采样"那一步的放大镜，承接模块 00 已提到的 `apply_grammar_bitmask`、`grammar_bitmask`、`StructuredOutputManager.grammar_init`、`WAITING_FOR_FSM`。

---

## 1. 这个模块解决什么问题

模型 forward 的产物是一个 `[num_tokens, vocab_size]` 的 `logits` 张量——它只是"每个词的打分"，并不是 token。从打分到 token，要同时满足几组互相拉扯的诉求：

1. **多样性 vs 确定性可控**：用户既要"贪心/确定复现"（temperature=0、seed 固定），也要"有创意的随机采样"（temperature>0、top-p/top-k 截断长尾）。同一个 batch 里不同请求可以有**完全不同**的采样配置。
2. **逐请求可复现**：带 `seed` 的请求必须**与 batch 里其他请求无关**地复现——不能因为同 batch 来了别的请求就改变随机序列。
3. **吞吐**：采样发生在每一步、每条请求上，是热路径。temperature / top-k / top-p / penalty 必须**向量化批处理**，避免 Python 逐请求循环和不必要的 CPU↔GPU 同步。
4. **结构合法性**：当用户要求输出严格的 JSON / 正则 / 文法时，必须**保证**每一步只能采样到"当前文法状态下合法"的 token，而不是事后校验、失败重试。
5. **结构化约束不能拖慢 GPU**：把文法（可能是复杂 JSON Schema）编译成可在每步快速求值的形式，且编译这种 CPU 重活**不能阻塞** GPU 关键路径。

V1 的方案：**采样是一个纯 GPU 算子链（`Sampler.forward`）**；**结构化输出是"在采样之前往 logits 上叠一层 bitmask 掩码 + 采样之后推进文法状态机"** 的旁路，二者在 `GPUModelRunner.execute_model` 里串接。

---

## 2. 设计目标与约束

- **批量优先**：所有采样参数（temperature、top_p、top_k、min_p、各种 penalty）都以 `[num_reqs]` 的张量形式存在，算子对整个 batch 一次性作用（`SamplingMetadata`，`vllm/v1/sample/metadata.py`）。
- **少同步**：尽量避免 CPU-GPU 同步点。随机采样刻意**不用 `torch.multinomial`**（它会触发同步），改用 "指数分布除法 + argmax" 的 Gumbel-max 技巧（`random_sample`，`topk_topp_sampler.py:235`）。
- **per-request generator**：带 seed 的请求各自持有一个 `torch.Generator`，**只覆盖自己那一行**的随机数，从而 batch 无关可复现（`gpu_model_runner.py:331-333`、`topk_topp_sampler.py:249-255`）。
- **全 batch 走同一条算子链，但用掩码/张量旁路差异**：贪心和随机混在一个 batch，用 `torch.where(temperature < eps, greedy, random)` 选择（`sampler.py:125-130`），而不是把 batch 拆成两组分别跑。
- **结构化输出后端可插拔但引擎级唯一**：支持 `xgrammar` 与 `guidance`（llguidance）两个后端，外加 `auto`（按请求内容自动择一），但**一个引擎实例只用一个后端**，不支持 per-request 选后端（`StructuredOutputManager.grammar_init`，`structured_output/__init__.py:46-58`）。
- **文法编译异步化**：grammar 编译丢进 `ThreadPoolExecutor`，请求先进入 `WAITING_FOR_FSM` 状态，编译完才允许被调度——保证 GPU 不为"等文法"空转（见 §3.4）。
- **bitmask 与采样解耦但同张量**：bitmask 在 scheduler 进程（CPU）生成，序列化为 `np.ndarray` 过界，在 GPU runner 上 in-place 改 logits（`apply_grammar_bitmask`，`gpu_model_runner.py:939`）。

---

## 3. 核心设计思想

### 3.1 采样是一条"原地修改 logits"的算子流水线

`Sampler.forward`（`sampler.py:23`）把"从 logits 到 token"拆成一串**可独立开关**的步骤，每步都倾向于 **in-place** 修改同一个 logits 张量以省显存：

```
logits (float32)
  → apply_allowed_token_ids   (masked_fill_ 非允许 token = -inf)
  → apply_bad_words           (匹配到禁用词前缀，则把末 token = -inf)
  → apply_logits_bias         (logit_bias[token] += bias)
  → apply_penalties           (min_tokens / presence / frequency / repetition)
  → sample:
        greedy_sampled = argmax(logits)            # 贪心分支
        logits = apply_temperature(logits, temp)   # /temp
        logits = apply_min_p(logits, min_p)
        random_sampled = topk_topp_sampler(...)     # 随机分支
        sampled = where(temp < eps, greedy, random) # 逐请求二选一
```

关键设计点：

- **logprobs 用"原始 logits"算**（`sampler.py:28-36`）：在做任何 penalty / 温度缩放**之前**就先 `compute_logprobs(logits)` 存下 `raw_logprobs`。这是 V1 与 V0 的刻意区别——返回给用户的 logprobs 反映"模型真实的概率分布"，而不是被温度/惩罚扭曲后的分布。
- **贪心与随机混批**：`all_greedy` / `all_random` 是 batch 级 fast-path（`sampler.py:96-103`）。若 batch 全贪心，直接 `argmax` 返回，不进随机分支；全随机则跳过 `argmax`；混合时两条分支都算，最后用 `torch.where` 按每行 temperature 选（`sampler.py:125-130`，`out=greedy_sampled` 复用张量）。
- **温度即"贪心开关"**：`temperature < _SAMPLING_EPS`（1e-5）被定义为贪心（`sampling_params.py:510-515` 的 `sampling_type`）。GREEDY 请求在 InputBatch 里 temperature 被置 `-1.0` 以避免后续除零（`gpu_input_batch.py:270-272`）。

### 3.2 随机采样的两个 GPU 技巧：Gumbel-max 与 FlashInfer

朴素的随机采样 `torch.multinomial(probs)` 会触发 **CPU-GPU 同步**（要把累积分布读回 host）。V1 用两条路径避免：

**(a) PyTorch-native `random_sample`（`topk_topp_sampler.py:235-256`）—— Gumbel-max / 指数采样技巧**

```python
q = torch.empty_like(probs)
q.exponential_()                       # 每个元素 ~ Exp(1)
return probs.div_(q).argmax(dim=-1)    # argmax(p / e) 等价于按 p 加权采样
```

`argmax(probs / Exp(1))` 在数学上等价于按 `probs` 做多项式采样（这是 Gumbel-max trick 的指数分布变体），且**全程在 GPU 上、无同步**。

**(b) FlashInfer 采样（`forward_cuda` → `flashinfer_sample`，`topk_topp_sampler.py:259`）**

当需要 top-k/top-p 时，FlashInfer 用**拒绝采样**直接在概率上采样，**避免对 vocab 排序**（排序对大 vocab 很慢）。代价是它**包含一次 CPU-GPU 同步**（检查 `success`），所以放在 forward 末尾、并在失败时回退到 renorm + 重采（`:305-311`）。当 `k is None and p is None` 时刻意退回 `random_sample`，因为后者无同步更快（`:111-115`）。

> 注意一个版本兼容陷阱：FlashInfer ≥ v0.2.3 移除了 sampling API 的 `success` 返回值，与 V1 设计不兼容，故被**默认禁用**、回退到 PyTorch-native（`topk_topp_sampler.py:34-49`）。

### 3.3 top-k / top-p / min-p 的批量实现

- **top-p（核采样）**：对 logits **降序排序** → softmax → 累积和 → 把"累积概率 ≤ 1-p"的低尾置 `-inf` → scatter 回原顺序（`apply_top_k_top_p`，`topk_topp_sampler.py:196-206`）。`top_p_mask[:, -1] = False` 保证**至少保留一个** token。
- **top-k**：若只有 top-k 而无 top-p，走 `apply_top_k_only`（`:210`）—— 用 `logits.topk(max_top_k)` 取每行第 k 大的值作阈值，**不对整个 vocab 排序**，比 top-p 路径快得多。
- **min-p（自适应阈值）**：`apply_min_p`（`sampler.py:203-223`）—— 阈值 = `min_p × max_prob`，把概率低于"最高概率某比例"的 token 砍掉。比固定 top-p 更能适应"分布很尖"vs"分布很平"两种情形。
- 这些参数全是 `[num_reqs]` 张量，通过 `unsqueeze(1)` 广播到 `[num_reqs, vocab]`，对整个 batch 一次算完——**没有逐请求 Python 循环**。

### 3.4 结构化输出：bitmask 约束 logits + FSM 推进

核心思想：一个**编译好的文法**（grammar）本质是个 FSM/下推自动机，它在任意状态下都能回答"下一个 token 哪些合法"。把"合法 token 集合"编码成一个 **bitmask**（vocab 大小的位向量，每 32 个 token 压进一个 int32），采样前用它把**非法 token 的 logits 置 `-inf`**，模型就**只可能**采样出合法 token。采样后再把这个 token 喂回文法，推进 FSM 状态。

```
每一步 decode：
  scheduler 进程（CPU）:
     grammar.fill_bitmask(bitmask, batch_index)   # 当前状态下"合法 token"的位图
     → 打包进 SchedulerOutput.grammar_bitmask (np.ndarray, 过界)
  GPU runner:
     apply_grammar_bitmask: 按 running index 切片 + 排序对齐
       → xgr.apply_token_bitmask_inplace(logits, bitmask)  # 非法 token logits=-inf
     sample(logits) → token
  scheduler 进程（update_from_output）:
     grammar.accept_tokens(req_id, [token])        # 推进 FSM；非法则报错
```

四个设计要点：

1. **约束发生在采样之前**：bitmask 在 `compute_logits` 之后、`sample` 之前应用（`gpu_model_runner.py:1075-1086`），所以约束对所有后续采样算子（温度/top-k/...）都生效——被掩成 `-inf` 的 token 永远不会被采样。
2. **异步编译 + `WAITING_FOR_FSM`**：JSON Schema/正则编译成 grammar 是 CPU 重活（xgrammar 要为整个 vocab 构建 token mask 缓存）。所以 `grammar_init` 把编译丢进 `ThreadPoolExecutor`（`structured_output/__init__.py:60`），请求初始状态是 `WAITING_FOR_FSM`（`request.py:41-43`）。调度器每步检查 `grammar` 是否 ready（非阻塞 `result(timeout=0.0001)`，`structured_output/request.py:31`），ready 才转 `WAITING` 进入正常调度（`scheduler.py:284-291`）。**这就是为什么需要一个独立状态**——否则要么阻塞 GPU 等编译，要么编译没好就调度导致 fill_bitmask 崩溃。
3. **bitmask 按 running index 切片对齐**：scheduler 按"调度顺序"给每个结构化请求分配一个 index 并填 bitmask（`structured_output_request_ids`），但 GPU runner 的 `InputBatch` 排列顺序**可能不同**（持久化 batch、swap 等）。所以 `apply_grammar_bitmask` 要检查 index 是否一致，不一致就**重排 bitmask 行**对齐到 batch 顺序（`gpu_model_runner.py:955-975`）。
4. **bitmask 在 CPU 生成、`np.ndarray` 过界**：xgrammar 的 fill 操作在 CPU 上跑，结果转成 `np.ndarray`（比 tensor 序列化高效）随 `SchedulerOutput` 广播给 worker（`structured_output/__init__.py:106-109`），在 GPU 上才 `torch.from_numpy(...).to(device, non_blocking=True)`。

### 3.5 后端选择：xgrammar / guidance / auto

`Processor._validate_structured_output`（`processor.py:144`）在请求入口决定后端：

- **xgrammar**：默认，基于 [xgrammar](https://github.com/mlc-ai/xgrammar)。把 JSON Schema/正则/EBNF 编译成带缓存的 `GrammarMatcher`，`fill_next_token_bitmask` / `accept_token` 是其核心 API。不支持某些 JSON Schema 特性（pattern、数值范围、`minItems` 等，见 `has_xgrammar_unsupported_json_features`，`backend_xgrammar.py:166`）。
- **guidance**：基于 [llguidance](https://github.com/guidance-ai/llguidance)，特性覆盖更全（支持 xgrammar 不支持的 JSON Schema 特性）。
- **auto**：先试 xgrammar 校验，失败（命中不支持特性）则回退 guidance（`processor.py:173-184`）。这是"opt-in 的有主见行为"，不作默认因为可预测性较差。

校验在**请求入口**就做（编译前先 parse 一遍），避免把非法文法送进异步编译后才发现。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `SamplingParams` | `vllm/sampling_params.py:108` | 用户原始采样配置：temperature、top_p/k、min_p、各 penalty、seed、`guided_decoding` 等 |
| `SamplingType` | `vllm/sampling_params.py:23` | `GREEDY=0` / `RANDOM=1` / `RANDOM_SEED=2`，由 temperature 与 seed 推导（`:510`） |
| **`SamplingMetadata`** | `vllm/v1/sample/metadata.py:9` | 采样的"batch 级输入包"：所有参数已转成 `[num_reqs]` 张量 + `generators` dict + 输出 token 列表 + bad_words/logit_bias/allowed_ids |
| `SamplerOutput` | `vllm/v1/outputs.py` | `Sampler.forward` 产物：`sampled_token_ids [num_reqs,1]` + 可选 `logprobs_tensors`（GPU 张量） |
| `GuidedDecodingParams` | `vllm/sampling_params.py:31` | 结构化输出请求：`json`/`regex`/`choice`/`grammar`/`json_object` 五选一 + `backend` |
| `StructuredOutputRequest` | `vllm/v1/structured_output/request.py:17` | 挂在 `Request` 上的结构化状态：持有 `Future[Grammar]` 或编译好的 `Grammar`；`structured_output_key` 作缓存键 |
| `StructuredOutputManager` | `vllm/v1/structured_output/__init__.py:24` | 引擎级管理器：选后端、异步编译（`grammar_init`）、每步生成 `grammar_bitmask`、持有共享 bitmask 张量 |
| `StructuredOutputGrammar` | `vllm/v1/structured_output/backend_types.py:20` | 请求级文法接口：`accept_tokens` / `fill_bitmask` / `is_terminated` / `reset` |
| `StructuredOutputBackend` | `backend_types.py:63` | 引擎级后端接口：`compile_grammar` / `allocate_token_bitmask` |
| `StructuredOutputOptions` | `backend_types.py:9` | `JSON / JSON_OBJECT / REGEX / GRAMMAR / CHOICE` |

`SamplingMetadata` 的几个"是否为空"标志（`all_greedy`、`all_random`、`no_penalties`、`no_top_p`…）是关键的 fast-path 开关：只要 batch 里没有任何请求用某特性，对应张量就是 `None`，整段算子被跳过（构造见 `gpu_input_batch.py:_make_sampling_metadata` `:537`）。

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 随机采样用指数除法+argmax，不用 `multinomial` | 无 CPU-GPU 同步，全 GPU 批处理 | 数值上等价但不是 bit-exact 的标准多项式采样；理解门槛高 |
| FlashInfer 拒绝采样做 top-k/top-p | 避免对大 vocab 排序，快 | 引入一次同步 + 失败回退逻辑；版本兼容脆弱（≥0.2.3 默认禁用） |
| logprobs 用"惩罚/温度前"的原始 logits | 返回真实概率，语义干净 | 与 V0 行为不同；需多存一份 raw_logprobs |
| 贪心/随机混批 + `torch.where` 选择 | 同 batch 异构采样，GPU 利用率高 | 混合 batch 两条分支都要算（贪心 argmax + 随机），略有冗余 |
| per-request `torch.Generator` 只覆盖自己那行 | seed 复现与 batch 内其他请求无关 | 带 seed 的请求要逐个 `q[i].exponential_(generator=...)`，无法完全向量化（`:254` 注释承认慢） |
| 结构化输出用 bitmask 约束 logits（硬约束） | 保证 100% 合法，无需重试 | bitmask 占 `vocab/32` int32/请求；fill 是 CPU 活；与 spec decode 暂不兼容（`gpu_model_runner.py:979` TODO） |
| grammar 异步编译 + `WAITING_FOR_FSM` | 编译不阻塞 GPU | 多一个请求状态与每步 ready 轮询；首 token 延迟受编译时间影响 |
| 引擎级单后端（不支持 per-request 选后端） | 实现简单、tokenizer info 复用 | 灵活性低；`auto` 用启发式回退弥补 |
| bitmask 在 scheduler(CPU) 生成、np 过界 | 序列化高效；GPU 只做 in-place 掩码 | 需在 GPU runner 处理 index 不匹配的重排（`:968-975`） |

---

## 6. 设计动机出处

- **xgrammar**（2024）—— Dong et al., *"XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models"*（arXiv:2411.15100，mlc-ai/xgrammar）。核心贡献是把文法约束的 token mask 计算做到**接近零开销**：预计算 context-independent token 的 mask、运行时只算 context-dependent 部分，并用位图（bitmask）压缩。vLLM 的 `fill_next_token_bitmask` / `apply_token_bitmask_inplace` 直接来自该库。
- **guidance / llguidance** —— guidance-ai 的 [llguidance](https://github.com/guidance-ai/llguidance)，Rust 实现的约束解码引擎，JSON Schema/Lark 文法特性覆盖比 xgrammar 更全，vLLM 用其 `LLMatcher` + `fill_next_token_bitmask`（`backend_guidance.py`）。
- **outlines（FSM 引导）** —— Willard & Louf, *"Efficient Guided Generation for Large Language Models"*（arXiv:2307.09702）。最早系统化"把正则/文法编译成 FSM，用每个状态的合法 token 集合 mask logits"思路，是结构化输出"bitmask 约束 logits"范式的奠基工作；vLLM V1 的设计与其同源。
- **GPU 采样优化** —— "argmax(p / Exp(1)) 等价于多项式采样"是 Gumbel-max trick（Gumbel 1954；Maddison et al. 2014 *"A* Sampling"*）的指数分布变体；vLLM 用它规避 `torch.multinomial` 的同步开销。top-k/top-p 的拒绝采样实现来自 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)（Ye et al.）。
- **投机解码与采样的关系** —— `RejectionSampler` 严格遵循 Leviathan et al. *"Fast Inference from Transformers via Speculative Decoding"*（arXiv:2211.17192，见 `rejection_sampler.py:26`）；与之配套的 Chen et al. *"Accelerating Large Language Model Decoding with Speculative Sampling"* 为 arXiv:2302.01318。细节属 [模块 05](../05-speculative-decoding/design.md)。

> 实现层面的逐行调用链、`file:line` 对照、精妙之处与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 7. 设计背后的考量与历史教训

### 7.1 设计背后的考量（深化"为什么"）

1. **"无同步采样"是被 decode 热路径倒逼出来的，而非美学选择**：朴素方案是 `torch.multinomial`，正确、可读、bit-exact。但它要把累积分布读回 host，引入 CPU-GPU 同步——在每步每请求的 decode 热路径上，这个同步会让 GPU 干等。于是否决了直观实现，改用 `argmax(probs / Exp(1))` 的 Gumbel-max 指数变体（§3.2）。代价是数值不再是教科书式多项式采样、理解门槛高，但换来全程 GPU、零同步。**约束（热路径不能有同步）直接决定了算法形态**。

2. **per-request `torch.Generator` 是"可复现性"对"向量化"的妥协**：诉求 2（带 seed 的请求必须 batch 无关复现）与诉求 3（向量化吞吐）天然冲突——真正向量化要求所有行共用一个随机流，而 batch 无关复现要求每行有独立流。最终选择"无 seed 走全批向量化、有 seed 的请求逐行 `q[i].exponential_(generator=...)`"的混合路线（`topk_topp_sampler.py:254` 注释直白承认这条路慢）。这是有意识地为正确性牺牲了局部性能，而不是没想到。

3. **logprobs 用"惩罚/温度前的原始 logits"是对 V0 语义的刻意纠偏**：V1 在任何 penalty/温度缩放之前就 `compute_logprobs` 存 `raw_logprobs`（§3.1）。否决了"返回被温度扭曲后的 logprobs"这个更省事的方案，因为那会让用户拿到的概率不反映模型真实分布。代价是多存一份张量、行为与 V0 不一致，但语义更干净——这是把"对用户正确"置于"实现省事"之上。

4. **结构化输出用硬约束（bitmask 置 -inf）而非"采样后校验重试"**：可选方案是先自由采样、再检查是否合法、不合法就重采。否决理由是重试不保证收敛、尾延迟不可控。bitmask 把非法 token 在采样前就掐死，**保证 100% 合法且单步完成**（§3.4）。代价是每请求要占 `vocab/32` 的 int32 位图、fill 是 CPU 重活、且与 spec decode 暂不兼容（`gpu_model_runner.py` TODO）。

5. **结构化输出后端"引擎级单一 + auto 启发式回退"，而非 per-request 任选**：早期方案曾把 backend 当作 per-request 接口参数。后来一系列重构（`#14694` Refactor Structured Output for multiple backends、`#15317` Remove unnecessary backend parameter in structured output interface、`#16578` 把 xgrammar utils 拆到 `backend_xgrammar.py`）把后端选择上提到引擎级，因为 tokenizer info / 编译缓存要在引擎内复用，per-request 选后端会让缓存与生命周期管理复杂化。`auto` 模式（先试 xgrammar、命中不支持特性再退 guidance）用启发式弥补灵活性，但不设默认——因为可预测性较差。

6. **vocab_size 的获取经历了从"昂贵"到"eagerly 缓存"的演进**：结构化输出要按 vocab_size 构造 matcher 和分配 bitmask。早期实现反复推算 vocab_size，催生了一串提交（`#14826` Fix vocab size calculation、`#14851` calculate vocab_size eagerly、`#14630` using cached vocab_size、`#14868` using vocab_size to construct matcher）。教训是：**在热路径/初始化路径上反复计算的"看似廉价的元信息"，要尽早算一次并缓存**——否则既慢又容易因多处推算不一致而出 bug。

### 7.2 重要 bug 修复（真实、精选）

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| `#2937` | temperature 极小（如 1e-4）时除法把 logits 放大到溢出，softmax 出 `nan`/`inf`，输出为空 | 引入 `_MAX_TEMP=1e-2` 下限并 warning。教训：温度作为"除数"在数值上是危险参数，极端值要 clamp，不能信任用户输入正好落在安全区间 |
| `#15049` / `#15907` | top-k 采样里索引张量 dtype 不对（`#15049` 修 long dtype；`#15907` 确保 `apply_top_k_only` 的 `k_index` 是 int64） | GPU 索引算子对 dtype 极敏感，int32/int64 混用会静默出错或崩。教训：跨 GPU 算子传递的索引张量 dtype 必须显式钉死 |
| `#14815` | FlashInfer 采样路径有 bug；后续 `#14788` 干脆临时禁用 FlashInfer Rejection Sampler | FlashInfer 采样含 CPU-GPU 同步 + 失败回退，且版本兼容脆弱（≥0.2.3 移除 `success` 返回值）。教训：引入"含同步+需回退"的外部高性能 kernel，要配好版本门控与 fallback，否则牵一发动全身 |
| `#16529` | 用户传的 `logit_bias` 含越界 vocab id，直接让引擎崩溃 | 在 `processor.py` 与 `sampler.py` 双层校验越界 id。教训：所有"用户可控且会被当索引用"的参数（logit_bias、allowed_token_ids）必须在入口校验，否则是引擎级 DoS |
| `#16403` | 给 repetition penalty 设了上界，限制了合法用法 | 移除人为上界。教训：采样参数的"合理范围"不该被框架武断收窄——penalty 没有理论上界，硬设上界是过度防御 |
| `#14894` | 即使没有任何请求用 bad_words，相关采样逻辑仍有开销 | 用 `no_bad_words` 之类的 batch 级 fast-path 跳过整段。教训：采样是热路径，"特性未启用时必须零开销"，fast-path 标志（`all_greedy`/`no_penalties`/...）是这条设计哲学的体现 |
| `#15407` | 多个 EOS token 时 `min_tokens` 行为不对（未屏蔽全部 EOS） | 修正多 EOS 下的 min_tokens 掩码。教训：凡涉及"EOS"的逻辑都要假设 EOS 可能是一个集合而非单个 id（同类历史教训亦见 V0 的 `#5849`） |

---

## 8. ASCII 全景图

```
                    模型 forward 产出 hidden_states
                              │
                   compute_logits → logits [num_tokens, vocab]
                              │
        ┌─────────────────────┴──────────────────────────────┐
        │   结构化输出旁路（仅 use_structured_output 的请求）  │
        │                                                     │
        │   scheduler(CPU)：grammar.fill_bitmask → bitmask    │
        │   （np.ndarray，随 SchedulerOutput 过界到 worker）   │
        │                     │                               │
        │   apply_grammar_bitmask：                            │
        │     · 按 running index 切片                          │
        │     · index 不一致则重排 bitmask 行对齐 batch        │
        │     · xgr.apply_token_bitmask_inplace               │
        │        → 非法 token 的 logits = -inf                 │
        └─────────────────────┬───────────────────────────────┘
                              ▼
        ┌──────────────  Sampler.forward  ──────────────────┐
        │  raw_logprobs = log_softmax(logits)  ← 惩罚/温度前  │
        │  logits = logits.float()                           │
        │  apply_allowed_token_ids   (mask = -inf)           │
        │  apply_bad_words           (匹配前缀→末token=-inf)  │
        │  apply_logits_bias         (+= bias)               │
        │  apply_penalties           (min/presence/freq/rep) │
        │  ── sample ──                                       │
        │    greedy  = argmax(logits)            （贪心分支） │
        │    logits /= temperature                           │
        │    apply_min_p                                     │
        │    random  = topk_topp_sampler(logits, generators) │
        │              │                                     │
        │       ┌──────┴───────┐                             │
        │   FlashInfer       PyTorch-native                  │
        │  (拒绝采样,        (Gumbel-max:                     │
        │   含同步)          probs/Exp(1) argmax, 无同步)     │
        │              │                                     │
        │    token = where(temp<eps, greedy, random)         │
        └──────────────────────┬─────────────────────────────┘
                               ▼
                       sampled_token_ids  [num_reqs, 1]
                               │
        ┌──────────────────────┴──────────────────────────────┐
        │  scheduler.update_from_output（CPU）：                │
        │    每条结构化请求 grammar.accept_tokens(req_id,[tok]) │
        │      → 推进 FSM；is_terminated() 后停止 fill          │
        └──────────────────────────────────────────────────────┘

  采样参数全部以 [num_reqs] 张量批处理；带 seed 的请求用 per-request
  torch.Generator 只覆盖自己那一行，保证 batch 无关可复现。
```
