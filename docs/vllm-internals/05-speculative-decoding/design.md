# 模块 05 · Speculative Decoding 投机解码 —— 设计文档

> 范围：在「调度（[模块 01](../01-scheduler-batching/design.md)）+ KV/Paged Attention（[模块 02](../02-paged-attention-kvcache/design.md)）+ 采样（[模块 06](../06-sampling-structured-output/design.md)）」之上叠加的一层加速机制 —— 用一个**便宜的 draft** 一次提议 `k` 个候选 token，再用**贵的 target 模型一次性验证**，靠**拒绝采样（rejection sampling）**保证输出分布与"逐 token 跑 target"严格一致。
> 架构：**V1 为主**（`vllm/v1/spec_decode/` + `vllm/v1/sample/rejection_sampler.py`），关键处对比 **V0**（`vllm/spec_decode/`）。V1 把投机解码**重写**进了 `GPUModelRunner`，摒弃了 V0 的 batch expansion。
> 配套实现文档见同目录 [`impl.md`](impl.md)。

---

## 1. 背景与定位

> 如果你还不清楚 **decode 为什么是访存密集（memory-bound）、一次 forward 多算几个 token 几乎不增加耗时** 这些前置直觉，请先读 [PRIMER 第一部分](../PRIMER.md#第一部分llm-推理服务-101先建立心智模型)（尤其 §1.4），这里默认你已经有了那套心智模型。

### 1.1 这个模块在系统里的位置

投机解码是叠加在「调度 + KV + 采样」三层之上的一层**纯加速机制** —— 它不改变最终输出（数学上保证与不用投机时逐 token 采样 target 完全一致），只让 decode 阶段一步多吐几个 token。

- **上游**：调度器（[模块 01](../01-scheduler-batching/design.md)）把上一步 proposer 提议的 `k` 个 draft token 当作"已追加未确认"的 token 一起推进，并经 KV 管理器（[模块 02](../02-paged-attention-kvcache/design.md)）为它们预留 KV slot。
- **本模块**：在 `GPUModelRunner` 的一次 forward 里，**target 模型一次性验证**这 `k` 个 draft（外加一个 bonus 位置），用**拒绝采样**决定接受几个；同时 proposer 为下一步提议新的 `k` 个 draft。
- **下游**：采样器（[模块 06](../06-sampling-structured-output/design.md)）提供 bonus token 的完整采样和拒绝采样的纠正分布；调度器在 `update_from_output` 里按"实际接受了几个"回退 `num_computed_tokens` 的记账。

一句话：**它是 decode 热路径上的一个可选加速层，靠"猜 + 一次验证"摊薄访存成本，且严格无损。**

### 1.2 关键文件与类各干什么（先认人）

| 文件 / 类 | 主要作用（一句话） |
|---|---|
| `config.py` 的 `SpeculativeConfig` | 投机解码总配置：选哪种 proposer（`method`）、猜几个（`num_speculative_tokens` 即 `k`）、接受方法等 |
| `v1/spec_decode/ngram_proposer.py` 的 `NgramProposer` | 无模型 proposer：在已有上下文里用 KMP 找重复 n-gram、复制其后 `k` 个 token 当提议 |
| `v1/spec_decode/eagle.py` 的 `EagleProposer` | 轻量 draft head proposer：复用 target 的 embedding/lm_head，自回归跑 `k` 步生成 draft |
| `v1/spec_decode/metadata.py` 的 `SpecDecodeMetadata` | 验证所需的全部索引与张量（哪些位置是 draft、哪些是 bonus） |
| `v1/sample/rejection_sampler.py` 的 `RejectionSampler` | 验证核心：4 个 Triton 内核实现"逐位置 accept/reject + 修正分布重采" |
| `v1/worker/gpu_model_runner.py` 的 `GPUModelRunner` | 投机解码的整合处：选 proposer、算验证元数据、一次 forward 验证 + 为下一步提议 |
| `v1/core/sched/scheduler.py` 的 `Scheduler` | 把 draft 当待验证 token 调度、预留 KV slot、被拒时回退记账 |
| `v1/spec_decode/metrics.py` 的 `SpecDecodingMetrics` | 统计并打印接受率（投机解码是否划算的第一指标） |

### 1.3 和相似模块 / 概念的区别与联系（专治"听过但分不清"）

- **proposer(draft) vs target**：**proposer（也叫 draft）** 是"便宜的猜测者" —— 它快但不准，提议未来 `k` 个 token；**target** 是用户真正要的那个"贵模型" —— 它准但慢，一次 forward 验证这 `k` 个猜测。关键：proposer 只影响**接受率（速度）**，不影响**最终分布（正确性）**，所以 draft 可以随便简化（只用 temperature），正确性全压在 target 验证 + 拒绝采样这一关上。
- **ngram vs eagle vs draft-model（三类 proposer，最容易混）**：**ngram** 不用任何模型，纯在上下文里找字面重复（零成本，但只在高重复文本里有效）；**eagle** 是一个轻量 draft head，复用 target 的特征与 lm_head（低成本、接受率稳定高，通用首选）；**draft-model** 是一个完整的独立小 LLM（成本可观，需有同族小模型）。取舍主线：proposer 越重→提议越准→但每步开销越大（详见 §3.2）。
- **bonus token**：target 验证 `k` 个 draft 时会**顺手多算一个位置**的 logits（假设 `k` 个全被接受后再往后一格）。只有当 `k` 个 draft **全部 accept** 时，这个白送的 token 才被追加 —— 所以一步最多前进 `k+1` 个 token。它单独用完整采样器采（可享 top_p/top_k），不走拒绝采样内核（详见 §3.4）。
- **拒绝采样（rejection sampling） vs 普通采样**：**普通采样**（[模块 06](../06-sampling-structured-output/design.md)）是"从一个分布里采一个 token"；**拒绝采样**是投机解码特有的"验证"步 —— 逐位置比较 draft 分布 `q` 与 target 分布 `p`，按 `min(1, p/q)` 概率接受 draft，拒绝处从修正分布 `max(0, p−q)` 重采。它的数学保证是"无论 draft 多差，输出都服从 target 分布"（详见 §3.3），这正是"无损加速"的来源。

> decode 为何 memory-bound、continuous batching 等更基础的概念见 [PRIMER 第一部分](../PRIMER.md#第一部分llm-推理服务-101先建立心智模型) 与 [§2.4 易混对照](../PRIMER.md#24-易混概念对照区别与联系专治听过但分不清)。

---

### 1.4 这个模块要解决的问题

LLM 自回归 decode 有两个根本痛点：

1. **逐 token 串行**：要生成第 `t+1` 个 token，必须先有第 `t` 个。一次 forward 只产出一个 token，序列长度决定了 forward 的次数，无法并行。
2. **decode 阶段严重访存受限（memory-bound）**：decode 时 batch 内每条请求只算 1 个 query token，但每一层都要把**整个模型权重**和**全部 KV cache** 从显存搬到 SM。算术强度极低 —— GPU 的算力大量闲置，瓶颈是 HBM 带宽而非 FLOPs。换句话说，"一次 forward 算 1 个 token"和"一次 forward 算 5 个 token"，**耗时几乎一样**，因为权重搬运的固定开销占了大头。

投机解码（Speculative Decoding）正是利用第 2 点：既然一次 forward 多算几个 token 几乎不增加耗时，那就让一个**便宜的 draft 模型**先猜出未来 `k` 个 token，再让**昂贵的 target 模型在一次 forward 里并行验证这 `k` 个猜测**。猜对的部分直接采纳，相当于"一步走了多格"。只要 draft 猜得够准、够便宜，整体 wall-clock 就能成倍下降，而且 —— 这是关键 —— **输出分布与不用投机解码时逐 token 采样 target 完全一致**（数学保证，见 §3.3）。

> 设计动机出处：投机解码由两篇 2023 年同期工作奠基 ——
> - **Leviathan et al., 2023**, *"Fast Inference from Transformers via Speculative Decoding"*（arXiv:2211.17192，Google）。vLLM 的 `RejectionSampler` 注释明确写着 "strictly follows the algorithm described in https://arxiv.org/abs/2211.17192"（`vllm/v1/sample/rejection_sampler.py:25-26`）。
> - **Chen et al., 2023**, *"Accelerating Large Language Model Decoding with Speculative Sampling"*（arXiv:2302.01318，DeepMind）。独立提出等价算法（speculative sampling）。
> 二者的核心贡献都是那条 **modified rejection sampling** 规则：它让"draft 提议 + target 验证"在期望意义上**输出与 target 同分布**，从而是"无损加速"（lossless），不牺牲生成质量。

---

## 2. 设计目标与约束

- **无损（distribution-preserving）**：默认的 `rejection_sampler` 路径必须保证最终采样分布与单独跑 target 一致 —— 这是投机解码的"底线"，不能为了加速牺牲质量。
- **draft 要足够便宜**：draft 的开销必须远小于它省下的 target forward。否则猜得再准也是亏本。这决定了 proposer 的选型谱系（ngram 零模型 → eagle 轻量头 → 独立 draft model）。
- **验证要"一次 forward 搞定"**：target 必须在**一次 forward** 里同时给出 `k` 个 draft 位置的 logits **以及** 第 `k+1` 个 "bonus" 位置的 logits（用于全accept 时白送一个 token）。不能为每个 draft 位置单独跑一次 target —— 那就退化成普通 decode 了。
- **与统一调度抽象兼容**：V1 调度器不区分 prefill/decode，只推进 `num_computed_tokens` 追 `num_tokens_with_spec`（`scheduler.py:122-132`）。投机解码必须套进这个抽象：draft 提议的 `k` 个 token 被当作"已追加但未确认"的 token 参与下一步调度，被拒绝时再回退 `num_computed_tokens`。
- **接受率优先于 draft 质量的完整性**：draft 采样时可以**忽略大部分采样参数**（只用 temperature），因为拒绝采样这一关会把分布纠正回来 —— draft 只影响**接受率（速度）**，不影响**最终分布（正确性）**（`eagle.py:230-234` 注释）。
- **批内异构**：同一个 batch 里，有的请求开了投机解码、有的没开；有的请求 draft 提议了 3 个、有的 1 个、有的 0 个。验证内核必须按 per-request 的 `num_draft_tokens` 处理变长。
- **某些采样特性暂不支持**：投机解码路径不支持 `min_p`、各种 penalty、per-token logprobs（`vllm/v1/spec_decode/utils.py:5-18`）—— 命中这些的请求会被 fallback 回普通 decode。

---

## 3. 核心设计思想

### 3.1 三件套：Proposer（提议）→ Target 验证 → Rejection Sampler（采纳）

投机解码的一步可以拆成三个角色：

```
①  Proposer (draft):   给定上下文，提议未来 k 个 token  q(x) = draft 分布
②  Target verify:       一次 forward，算出这 k 个 draft 位置 + 1 个 bonus 位置的 target logits  p(x)
③  Rejection Sampler:   逐位置比较 p/q，决定 accept / reject，reject 处用"修正分布"重采样，全 accept 则白送 bonus token
```

一步的产出是 **1 到 k+1 个** target 分布下的合法 token：
- 若第 `i` 个 draft 被拒，则前 `i-1` 个 accept + 第 `i` 个 reject 位置的 recovered token，共 `i` 个 token，**丢弃** draft 的第 `i+1..k` 个（它们建立在被推翻的前缀上）。
- 若 `k` 个全 accept，则 `k` 个 accept + 1 个 **bonus token**（target 在第 `k+1` 位置本来就算了 logits，白送），共 `k+1` 个 token。

所以每步**至少前进 1 个 token**（最坏情况第一个 draft 就被拒，等价于普通 decode 一步），**最多前进 k+1 个**。这就是加速的来源。

> 🔗 **回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)**：示例的阶段⑤（decode 第 1 步）原本是"只算 1 个 token id=9"。若对它开投机解码、设 `k=3`：proposer 先猜出 3 个 draft token，target 把 `[当前token, draft1, draft2, draft3]` 拼进序列**一次 forward** 算出 4 个位置的 logits，拒绝采样逐位置验证 —— 若 3 个全对就一步前进 4 个 token（3 accept + 1 bonus），而不是慢悠悠一步一个。这就是把示例里"挤牙膏"的 decode 变成"一次走几格"。

### 3.2 Proposer 谱系：ngram / eagle / draft model 的取舍

V1 当前内置两类 proposer（`gpu_model_runner.py:164-172` 按 `method` 选择），配置见 `SpeculativeConfig.method`（`config.py:2091-2107`，可选 `ngram / eagle / medusa / mlp_speculator / draft_model`）：

| Proposer | 额外模型？ | 提议机制 | 开销 | 接受率 | 适用场景 |
|---|---|---|---|---|---|
| **ngram**（prompt lookup） | **无**（纯 CPU/numba） | 在已有上下文里用 KMP 找最后 `n` 个 token 的历史出现，复制其后 `k` 个 token 作为提议（`ngram_proposer.py:25-58`） | 极低（几乎零成本） | 在高重复场景（代码、RAG 引用、长文摘要照抄）很高，否则常 0 | 文本中存在大量字面重复时近乎白嫖加速 |
| **eagle / eagle-2** | 一个**轻量 draft head**（复用 target 的 embedding 与 lm_head，`eagle.py:209`） | 把 target 最后一层 hidden state 喂进一个小 transformer 头，自回归跑 `k` 步生成 draft（`eagle.py:111-136`） | 低（一个小头跑 k 步） | 高（用了 target 的特征，比独立小模型准） | 通用，draft 质量与开销平衡得最好 |
| **draft model**（独立小模型） | 一个**完整的小 LLM** | 小模型自回归跑 `k` 步 | 中（一个真模型的 k 次 forward） | 取决于 draft 与 target 的"对齐度" | 有现成同族小模型时（如 Llama-70B + Llama-7B draft） |
| **medusa / mlp_speculator** | 多个轻量预测头 | 一次性并行预测多个未来位置 | 低 | 中 | 训练好的 medusa 头 |

> 设计动机出处：
> - **ngram / prompt lookup decoding**：源于 Apoorv Saxena 的 *Prompt Lookup Decoding*（2023，github.com/apoorvumang/prompt-lookup-decoding）。洞见是"输入里出现过的 n-gram 极可能再次出现"，无需任何模型，零成本提议。
> - **Medusa**：Cai et al., 2024, *"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"*（arXiv:2401.10774）。在 target 上加多个并行预测头，避免单独的 draft 模型。
> - **EAGLE / EAGLE-2**：Li et al., 2024, *"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"*（arXiv:2401.15077）及 EAGLE-2（arXiv:2406.16858）。核心洞见：在 **feature（hidden state）层** 而非 token 层做自回归预测，并复用 target 的 embedding/lm_head，使 draft 头极轻却很准。vLLM V1 的 `EagleProposer` 复用 target `lm_head`（`eagle.py:209`）正是这一思路。

**取舍主线**：proposer 越重 → 提议越准（接受率越高）→ 但每步 draft 开销越大。最优点取决于"省下的 target forward × 接受率"是否盖过"draft 的固定开销"。ngram 是"零成本但接受率方差大"，eagle 是"低成本且接受率稳定高"，draft model 是"成本可观、需要好的同族小模型"。

### 3.3 数学核心：拒绝采样为什么保证与 target 同分布

这是整个模块的灵魂。设某个位置 draft 分布为 `q(x)`，target 分布为 `p(x)`，draft 采样出了 token `x̃ ~ q`。**modified rejection sampling** 规则：

```
采样 r ~ Uniform(0,1)
若  r < min(1, p(x̃)/q(x̃))   →  接受 x̃
否则                          →  拒绝，从修正分布 p'(x) = norm(max(0, p(x) - q(x))) 重采样一个 token
```

可以证明，这样产出的 token **恰好服从 target 分布 `p`**。直觉：
- draft 把 `x̃` 采出来的概率是 `q(x̃)`；我们以 `min(1, p/q)` 的概率接受它，于是"经由接受路径得到 `x̃`"的总概率是 `q(x̃)·min(1, p(x̃)/q(x̃)) = min(q(x̃), p(x̃))`。
- 当 `q` 在某些 token 上"过度自信"（`q > p`）时，接受概率 `<1`，多出来的概率质量 `q-p` 被拒绝；这些被拒的情况里，用 `p'(x) ∝ max(0, p(x)-q(x))` 重采样，正好补上那些 `q` 给得不够（`p > q`）的 token。
- 两条路径加起来，每个 token `x` 的总产出概率恰好是 `p(x)`。**无论 `q` 多差，输出分布都是 `p`** —— draft 只影响"接受概率"（即速度），不影响分布（正确性）。这就是为什么 §2 里说 draft 可以随便忽略采样参数。

vLLM 的 Triton 内核精确实现了这条规则：

- **random（采样）路径**（`rejection_sampler.py:481-539` `rejection_random_sample_kernel`）：逐位置 `if draft_prob > 0 and target_prob / draft_prob >= uniform_prob: accept`（`:524`），否则取 recovered token 并把后续全部丢弃（`rejected=True`）。
- **recovered token** 来自修正分布 `max(0, p - q)`（`sample_recovered_tokens_kernel`，`:617` `prob = tl.maximum(target_prob - draft_prob, 0)`），用 **Gumbel-max trick**（`prob/q` 的 argmax，`q ~ Exponential`）从中采样。
- **greedy 路径**（`rejection_greedy_sample_kernel`，`:433-476`）退化为：draft token 必须等于 target 的 argmax 才 accept（`:467`），否则取 target argmax 作为 recovered —— 这就是贪心解码下的"无损"。
- **ngram 特例**：ngram 没有 draft 概率分布，代码把 draft prob 视为 1（`IS_NGRAM` 分支，`:512-513`），修正分布退化为"把 draft token 的概率置 0 再 argmax"（`:594-607`）。

### 3.4 bonus token：k+1 个位置一次验证

target 验证时不是只算 `k` 个 draft 位置的 logits，而是算 **`k+1`** 个 logits：
- 前 `k` 个对应 `k` 个 draft 位置，用于"接受/拒绝/重采样"（`target_logits_indices`）。
- 第 `k+1` 个是 **bonus 位置**：它是"假设 `k` 个 draft 全被接受后，序列再往后一格"的位置。target 在这一格本来就顺手算了 logits（因为 draft token 已经拼进输入序列了），**白送一个 token**（`bonus_logits_indices`）。

关键设计：bonus token **不在拒绝采样内核里采**，而是**单独用完整采样器采**（`gpu_model_runner.py:1093-1098`，对 `bonus_logits` 调 `self.model.sample`），再作为 `bonus_token_ids` 传进 rejection sampler（`:1104-1109`）。原因（`rejection_sampler.py:33-40` 注释）：bonus token 只从 target 分布采，**可以享受 top_p/top_k 等高级采样策略**，而拒绝采样路径不支持这些。把 bonus 拆出来单独采，给了采样更多灵活性。只有当 `k` 个 draft **全部 accept** 时，bonus token 才被追加（内核 `:471-476` / `:534-539` 的 `if not rejected`）。

数据结构上，per-request 的索引由 `_calc_spec_decode_metadata`（`gpu_model_runner.py:753-826`）用纯 numpy 向量化算出，打包成 `SpecDecodeMetadata`（`metadata.py:8-25`）。

### 3.5 V0 batch expansion vs V1 原位 MQA 验证（最大的重写）

这是 V0→V1 在投机解码上**最根本**的差异。

**V0 的 batch expansion**（`vllm/spec_decode/batch_expansion.py`，`BatchExpansionTop1Scorer`）：
要让 target 在一次 forward 里验证 `k` 个 draft，V0 把**每个 (序列, draft 前缀)** 展开成 batch 里**一条独立的单 query 序列**。一条 draft `[t0,t1,t2,t3]` 被拆成 `k+1` 个前缀 `[], [t0], [t0,t1], ..., [t0,t1,t2,t3]`，每个前缀作为一条新 `SequenceGroupMetadata`（`_create_single_target_seq_group_metadata`）。于是 `B` 条请求 × `k` 个 draft 会**膨胀成 `B·(k+1)` 条序列**塞进一次 forward。代码自己注释为 "strictly less efficient than MQA scoring"（`batch_expansion.py:34`）。代价：batch 膨胀 `~(k+1)` 倍带来的 padding、attention 冗余、以及每步在 Python 里重建大量 per-prefix 元数据。

**V0 后期的 MQAScorer**（`vllm/spec_decode/mqa_scorer.py`）：
作为优化，V0 引入 `MQAScorer`：把 `k` 个 draft token 直接拼到**同一条序列**后面，用 **multi-query attention**（一条序列多个 query 位置）在一次 forward 里算完，避免 batch 膨胀。但它有限制（需 FLASH_ATTN 后端 + enforce_eager）。

**V1 的原位验证**（`vllm/v1/`）：
V1 **彻底摒弃 batch expansion**，直接把 MQA 思路做成默认且唯一路径，并**整合进 `GPUModelRunner`**（不再有独立的 `SpecDecodeWorker` 包两个 worker）：
- draft 的 `k` 个 token 在调度时就被当作"待验证 token"追加进请求的 token 序列（`scheduler.py:243-251`），随普通 batch 一起进 forward。
- 一次 target forward 后，用 `logits_indices` / `target_logits_indices` / `bonus_logits_indices` 这三组**索引**从扁平 logits 里"原地"切出每个位置该用的 logits（`gpu_model_runner.py:1093-1103`），**没有任何 batch 复制**。
- 验证用 **Triton rejection sampler**（GPU 内核），而非 V0 的 PyTorch module。

净收益：无 batch 膨胀、无独立 worker 进程、无 Python 端 per-prefix 元数据重建、验证全在 GPU 内核里完成。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `SpeculativeConfig` | `config.py:2081` | 投机解码总配置：`method`（ngram/eagle/...）、`num_speculative_tokens`（即 `k`）、`prompt_lookup_min/max`、`acceptance_method`、draft 模型相关 |
| `Request.spec_token_ids` | `v1/request.py` | 一条请求当前**待验证**的 draft token 列表（上一步 proposer 产出，本步被调度验证） |
| `num_tokens_with_spec` | `v1/request.py`（属性） | `len(prompt) + len(output) + len(spec_token_ids)`，调度器据此算 `num_new_tokens`（`scheduler.py:126-127, 168-169`） |
| `scheduled_spec_decode_tokens` | `SchedulerOutput`（`v1/core/sched/output.py`） | 本步每条请求**实际被调度验证**的 draft token（dict: req_id → token list），`scheduler.py:154,433` |
| **`SpecDecodeMetadata`** | `v1/spec_decode/metadata.py:8` | 验证所需的全部索引与张量：`draft_token_ids`、`num_draft_tokens`、`cu_num_draft_tokens`、`target_logits_indices`、`bonus_logits_indices`、`logits_indices`、`max_spec_len` |
| `draft_probs` / `target_probs` | `rejection_sampler.py` 内部 | draft / target 在各 draft 位置的概率分布（ngram 时 `draft_probs=None`） |
| `bonus_token_ids` | `gpu_model_runner.py:1098` | 单独用完整采样器采出的 bonus token（每请求 1 个），传入 rejection sampler |
| `output_token_ids` | `rejection_sampler.py:167` | 拒绝采样产出，shape `[batch, max_spec_len+1]`，reject 处填 `PLACEHOLDER_TOKEN_ID(-1)`，由 `parse_output` 过滤 |
| `ModelRunnerOutput.spec_token_ids` | `v1/outputs.py` | 本步 proposer 为**下一步**产出的新 draft 提议 |
| `SpecDecodingStats` | `v1/spec_decode/metrics.py:12` | `num_draft_tokens` / `num_accepted_tokens`，用于算接受率 |
| `num_lookahead_tokens` | `scheduler.py:117-120` | eagle 时 = `k`，传给 `allocate_slots` 预留 KV slot（draft token 也要占 KV） |

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 投机解码本身（draft 提议 k + target 验证） | decode 受访存限制时，wall-clock 成倍下降 | draft 开销 + 验证开销；接受率低时反而**变慢**（白跑 draft + 验证了更宽的 logits） |
| draft 忽略大部分采样参数（只用 temperature） | draft 更快、实现更简单（`eagle.py:230-234`） | 接受率略降，但**分布不受影响**（拒绝采样纠正）—— 纯速度取舍 |
| bonus token 单独用完整采样器采 | bonus 可享 top_p/top_k 等策略（`rejection_sampler.py:33-40`） | 多一次 `model.sample` 调用 |
| 拒绝采样（rejection）vs typical acceptance | rejection 严格无损（分布一致） | typical acceptance（`acceptance_method`，`config.py:2108-2125`）接受率更高更快，但**有损**（偏离 target 分布） |
| ngram proposer | 零模型、零显存、CPU 即可（`ngram_proposer.py`） | 接受率高度依赖文本重复度，非重复场景近乎无效 |
| eagle proposer | 接受率高且开销低（复用 target 特征） | 需训练/加载 eagle 头；draft head 不走 CUDA graph（`gpu_model_runner.py:1190-1192`） |
| `k`（num_speculative_tokens）调大 | 接受率高时一步走更多格 | 接受率低时 `k` 越大浪费越多（验证更宽 + draft 跑更多步）；`k` 上限 `MAX_SPEC_LEN=32`（`rejection_sampler.py:20`） |
| V1 原位 MQA 验证（弃 batch expansion） | 无 batch 膨胀、整合进 ModelRunner、GPU 内核验证 | 实现复杂度集中在索引计算（`_calc_spec_decode_metadata` 的向量化）；不支持 min_p/penalty/logprobs |
| 不支持 min_p/penalty/per-token logprobs | 简化验证内核 | 这些请求 fallback 回普通 decode（`utils.py:5-18`） |

### 接受率对加速比的影响（为什么"接受率"是第一指标）

设每步提议 `k` 个 draft、平均接受 `α·k` 个（`α` = 接受率），则每步前进约 `α·k + 1` 个 token（含 bonus 的期望），而每步的成本约等于"1 次 target forward + 1 次 draft 提议"。理想加速比 ≈ `(期望前进 token 数) / (1 + draft 相对成本)`。结论：
- **接受率 `α` 越高，加速越大**；`α→1` 时趋近 `k+1` 倍（被 draft 成本和访存上限拉低）。
- **`α` 太低会负优化**：白跑了 draft，还让 target 算了 `k+1` 宽的 logits、KV 多占了 slot，却几乎没多前进。所以 vLLM 持续统计接受率（`SpecDecodingMetrics`，`metrics.py:46-62` 打印 "Draft acceptance rate"），V0 甚至有 `disable_by_batch_size`：高负载（batch 已经够大、GPU 不再访存受限）时**自动关闭**投机解码，因为此时多算 token 不再"白嫖"。

---

## 6. 一图速查：投机解码一步的数据流

```
        ┌─────────────────────────────────────────────────────────────────────┐
        │  上一步 proposer 已为每条请求产出 draft：req.spec_token_ids = [d1,d2,d3]│
        └─────────────────────────────────────────────────────────────────────┘
                                        │
   ① 调度  scheduler.schedule()         ▼
        num_new_tokens = num_tokens_with_spec - num_computed_tokens   (= 1 个真 token + k 个 draft)
        scheduled_spec_decode_tokens[req] = [d1,d2,d3]      (scheduler.py:243-251)
        allocate_slots(num_lookahead_tokens=k)              (draft token 也要 KV slot)
                                        │
   ② Target 一次 forward                ▼
        把 [last_real, d1, d2, d3] 拼进 batch，一次 forward 得到扁平 logits
        _calc_spec_decode_metadata: 算出 target_logits_indices / bonus_logits_indices
                                        │
                ┌───────────────────────┴────────────────────────┐
                ▼                                                  ▼
   ③a bonus 位置                                       ③b k 个 draft 位置
      model.sample(bonus_logits)  ──► bonus_token_id      target_logits = logits[target_logits_indices]
      (可用 top_p/top_k)                                            │
                │                                                   ▼
                └────────────►  RejectionSampler (Triton)  ◄────────┘
                                 逐位置  accept if p/q ≥ r
                                 reject → recovered ~ max(0,p−q), 丢弃后续
                                 全 accept → 追加 bonus_token
                                        │
                                        ▼
                  output_token_ids  [accept... , recovered | bonus]   (1 ~ k+1 个)
                                        │
   ④ 回填  update_from_output           ▼
        num_rejected = (k+1) - len(generated)            (scheduler.py:605-607)
        request.num_computed_tokens -= num_rejected      ← 回退被拒 draft 占的记账
        observe(num_draft=k, num_accepted=len(generated)-1)  ← 接受率统计
                                        │
   ⑤ 为下一步提议新 draft               ▼
        ngram: KMP 找重复  |  eagle: draft head 自回归跑 k 步
        ModelRunnerOutput.spec_token_ids  ──►  写回 request.spec_token_ids  (回到 ①)
```

> 投机解码是叠加在「调度 + KV + 采样」之上的一层：调度负责把 draft 当待验证 token 推进并在被拒时回退（[模块 01](../01-scheduler-batching/design.md)），KV 负责为 draft 预留 slot（[模块 02](../02-paged-attention-kvcache/design.md)），采样器负责 bonus token 与拒绝采样的纠正分布（[模块 06](../06-sampling-structured-output/design.md)）。逐行 `file:line` 调用链见 [`impl.md`](impl.md)。

---

## 7. 设计背后的考量与历史教训

> 本节深化前文（尤其 §3.5、§5）背后的"为什么"，并从真实 bugfix 里提炼教训。投机解码涉及"draft 提议 / target 验证 / 拒绝采样 / bonus token / KV 记账"多个易错耦合点，历史上 bug 不少且很有启发性。所有 PR 号均来自本仓库 `git log` 真实记录。

### 7.1 设计背后的考量（深化"为什么"）

1. **为什么 V1 要把整套机制重写、彻底弃用 V0 的 batch expansion**：V0 为了"一次 forward 验证 k 个 draft"，把每条 (序列, draft 前缀) 展开成 batch 里一条独立单 query 序列，`B` 条请求 × `k` draft **膨胀成 `B·(k+1)` 条**（§3.5）。代码自己都注释"strictly less efficient than MQA scoring"。这个设计被否决的根因有三：batch 膨胀带来的 padding/attention 冗余、每步在 Python 端重建大量 per-prefix 元数据、以及独立 `SpecDecodeWorker` 包两个 worker 的进程复杂度。V1 改成**原位 MQA 验证 + 三组索引从扁平 logits 切片**（§3.4、§3.5），把 draft token 直接当"待验证 token"追加进同一条序列随普通 batch 进 forward——零 batch 复制、验证全在 Triton 内核里。这是本模块最大的一次重构，也是"为什么 V1 没有 batch_expansion.py"的答案。

2. **为什么 draft 可以随便忽略采样参数，但 target 验证一刻都不能错**：§3.3 的 modified rejection sampling 数学保证"无论 draft 分布 `q` 多差，输出都是 target 分布 `p`"。这条性质把正确性的全部重量压在**拒绝采样这一关**上——draft 只决定接受率（速度），不决定分布（正确性）。所以设计上 draft 采样大胆只用 temperature（§2、`eagle.py:230-234`），但 target 侧的概率必须严格按用户的采样参数算，否则纠正分布 `max(0,p-q)` 里的 `p` 就不是用户要的分布。7.2 的 #10198 正是踩了这条线：target 概率没应用采样参数，导致"无损"不再无损。

3. **为什么 bonus token 要单独用完整采样器采、而不在拒绝采样内核里采**：bonus token（k 个 draft 全 accept 时白送的第 k+1 个）只从 target 分布采，没有 draft 分布要对比。把它留在拒绝采样 Triton 内核里采会被迫继承内核"不支持 top_p/top_k"的限制。于是设计上把 bonus **拆出来单独 `model.sample`**（§3.4），让它能享受完整采样策略，再作为 `bonus_token_ids` 传进 rejection sampler。这是"让简单的快路径保持简单、把复杂采样需求引流到通用采样器"的分工。

4. **为什么 proposer 做成 ngram/eagle/draft-model 谱系而非押注一种**：§3.2 的取舍主线是"proposer 越重→提议越准→但每步开销越大"，最优点取决于"省下的 target forward × 接受率"能否盖过"draft 固定开销"，而这随**文本特征**剧烈变化。高重复文本（代码、RAG 照抄）下 ngram 零成本却接受率极高，通用场景下 eagle（复用 target 的 hidden 特征与 lm_head）平衡最好，有同族小模型时 draft model 才划算。没有一种在所有场景占优，所以做成可配置谱系（`SpeculativeConfig.method`）。

5. **为什么投机解码必须能被"调度抽象"和"接受率"反向约束**：V1 调度器不区分 prefill/decode，draft 的 k 个 token 被当作"已追加未确认"参与调度、被拒时回退 `num_computed_tokens`（§2、§3.5、一图速查的 ④）。这套记账一旦和"KV slot 预留"（`num_lookahead_tokens`，draft 也占 KV）配合错位，就会 KV 串味。更上层，接受率太低时投机解码是**负优化**（白跑 draft + target 算了更宽 logits + 多占 KV），所以 vLLM 持续统计接受率、V0 甚至有 `disable_by_batch_size` 在高负载下自动关闭（§5 末）。即"是否值得投机"本身是个动态决策，不是常开开关。

### 7.2 重要 bug 修复（真实、精选）

> 下列 PR 均可在本仓库 `git log` 查到。投机解码的 bug 大多落在"正确性边界"（拒绝采样是否真无损、bonus/KV 记账是否对齐）和"V0 batch expansion 的元数据重建"上，很能反映 §3 的设计脆弱点。

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| **#9730** `kv corruption with bonus tokens in spec decode` | draft model 多步前向时，**带 bonus token 的序列** KV 写入位置算错，造成 **KV cache 损坏** | bonus token（§3.4）让"哪条序列前进了几个 token"在 draft 多步里变得不均匀，KV 记账必须按 `indices_of_seq_with_bonus_tokens` 区分。教训：bonus token 这个"白送的 +1"会渗透进 KV 写入逻辑，是投机解码记账最易错的角落 |
| **#10198** `apply sampling parameters to target probabilities for consistency in rejection sampling` | V0 batch expansion 给 target 验证用"简化采样参数"，**target 概率没应用用户的 temperature/top_p/top_k**，导致拒绝采样里的 `p` 不是用户分布，**破坏无损性** | 直击 §3.3 的灵魂：拒绝采样无损的前提是 `p` 就是用户要的 target 分布。target 侧任何对采样参数的"偷懒简化"都会让"无损"变"有损"。教训：draft 可简化（§7.1#2），target 一刻不可 |
| **#10350** `Qwen-vl output is inconsistent in speculative decoding` | batch expansion 重建 per-prefix 序列元数据时**漏拷 `mrope_position_delta`**，多模态 RoPE 位置错，Qwen-VL 投机解码输出与不投机时不一致 | 暴露 V0 batch expansion 的根本脆弱点：每步在 Python 端**重建**大量 per-prefix 元数据，漏一个字段就静默出错（§3.5 列为 V1 弃用它的理由之一）。教训：手工重建元数据是 bug 温床，V1 的"原位索引切片"正是为消除这类重建 |
| **#14237** `Fix DeepSeek MTP crash when using TP1ModelRunner with CUDA graph due to shape mismatch` | draft model runner 在 CUDA graph 路径下返回的 `hidden_states` **未按 `selected_token_indices` 裁剪**，形状与图捕获时不符，DeepSeek MTP 崩溃 | draft 头/MTP 与 CUDA graph 的交互很脆（§5 也提到"eagle draft head 不走 CUDA graph"）。修复按 cuda_graph 分支裁剪 hidden。教训：投机解码的 hidden_states 在"全 accept/部分 accept"下长度可变，与定长 CUDA graph 天然冲突 |
| **#15909** `Fix input triton kernel for eagle` | eagle 的 `prepare_input_kernel` 里 `indices = index_start + tl.arange(...)` 在分块循环中**没随块偏移递增**，多块时写入索引错位 | V1 把验证/准备输入做成 Triton 内核（§3.5 的净收益之一），但 GPU 内核里的索引算术（分块 offset）极易错且难发现。教训：从 Python 元数据重建搬到 GPU 内核，省了重建开销，却把 bug 转移成"内核索引算术"这类更隐蔽的形态 |
| **#13362** `Clean up rejection sampler & Fix warning msg` | V1 `RejectionSampler` 的后端选择（FlashInfer vs PyTorch-native）逻辑与告警混乱，`VLLM_USE_FLASHINFER_SAMPLER` 在 V0/V1 语义不一致 | 拒绝采样器有多套后端实现，"默认用哪个、何时告警"必须清晰，否则用户在不知情下走了慢路径。暴露 §3.3 内核有多后端时的配置面复杂度 |
