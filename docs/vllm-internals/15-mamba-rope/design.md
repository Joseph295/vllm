# 模块 15 · Mamba/SSM 与混合架构 + 位置编码（RoPE/M-RoPE/YaRN）—— 设计文档

> 范围：本模块讲两个看似无关、实则都关乎"序列建模如何省内存/如何编码位置"的主题。
> **A 部分 —— Mamba/SSM 与混合架构**：用**常量大小的递归状态**（state space model）替代随序列线性增长的 KV cache，以及 attention 层与 mamba 层如何在同一个模型里共存。
> **B 部分 —— 位置编码**：RoPE（旋转位置编码）如何把相对位置注入注意力，YaRN/NTK 如何在不微调下把上下文外推到训练长度之外，M-RoPE 如何为图像/视频给出三维位置。
> 架构说明：**位置编码（B）在 V1/V0 共用同一套实现**（`vllm/model_executor/layers/rotary_embedding.py` + `csrc/`），M-RoPE 的 position 计算在 V1 走 `gpu_model_runner.py`。**Mamba/SSM（A）在本仓库快照里仍是 V0-only**：所有 mamba 系模型都标了 `SupportsV0Only`（`mamba2.py:172-173`、`jamba.py:362-363`），状态缓存走独立的 `MambaCacheManager`（`models/mamba_cache.py`）而非 V1 的 `KVCacheManager`。这一"为什么还没进 V1"本身就是本模块要讲透的设计张力——见 §A.3 与 §N。

---

## 1. 背景与定位

> 如果你还不清楚 **KV cache 为何随序列线性增长、attention 为何置换不变** 这些前置概念，请先读 [PRIMER 第一部分](../PRIMER.md#第一部分llm-推理服务-101先建立心智模型)（尤其 §1.3 KV cache），这里默认你已经有了那套心智模型。本模块是 A（Mamba/SSM）/ B（位置编码）两部分并列，下面的 1.1~1.3 分别交代两者，1.4 收纳上面的范围说明。

### 1.1 位置

两部分都属于 [PRIMER §2.1](../PRIMER.md#21-两条-track先分清骨架和血肉) 里"模型架构技术"那条 track（随模型而异），但各动一处：

- **A 部分（Mamba/SSM）**：替换 Transformer 的整个 attention 层——用一个**常量大小的递归状态**代替随序列线性增长的 KV cache。混合架构（Jamba 等）里 attention 层和 mamba 层在同一个模型里共存，各取各的缓存。**本快照里 mamba 仍是 V0-only**（见上方架构说明与 §A.3）。
- **B 部分（位置编码）**：插在 attention 的 Q/K 投影**之后、算 attention 之前**，把位置信息旋转进 Q/K。它不碰 KV cache 结构，只改 Q/K 的值；V1/V0 共用同一套实现。

一句话：**A 换的是"历史怎么存"（状态 vs KV），B 改的是"位置怎么进 attention"（旋转 Q/K）——两者都关乎序列建模，但作用在 forward 的不同位置。**

> 🔗 回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)：**B 部分能对上示例**——示例阶段②算 q/k/v 时，真实模型会先对 q、k 按各自位置做 RoPE 旋转（pos0/1/2 各转不同角度）再算 attention 分数；示例为手算简洁省略了这一步。**A 部分（Mamba）与示例的 attention 路径不同**：示例走的是标准 attention（每个 token 存 K/V 进 block_table），而 mamba 层不存 K/V、只维护一个固定大小的状态 `h`，没有 block_table 那套分页寻址，因此这里跳过回扣。

### 1.2 关键文件与类各干什么（先认人）

| 文件 / 类 | 主要作用（一句话） |
|---|---|
| **A** `mamba/mamba_mixer.py` / `mamba_mixer2.py` 的 `MambaMixer`/`MambaMixer2` | Mamba-1/2 的层实现：因果卷积 + selective scan，含 input-dependent 的 Δ/B/C 投影 |
| **A** `mamba/mamba2_metadata.py` 的 `prepare_mamba2_metadata` | top-level 算一次的元数据（seq_idx/chunk_indices…），所有 mamba2 层共享 |
| **A** `models/mamba_cache.py` 的 `MambaCacheManager` | 跨步持有 conv/ssm 状态张量，按请求分配"状态槽"（独立于 V1 的 KVCacheManager） |
| **A** `models/constant_size_cache.py` 的 `ConstantSizeCache` | 状态缓存基类：定长槽的分配 / 释放 / 拷贝，无 block / 无前缀复用 |
| **A** `models/jamba.py` / `mamba2.py` | 混合架构 / 纯 SSM 模型；按层类型分派 KV cache vs 状态缓存（均 `SupportsV0Only`） |
| **B** `layers/rotary_embedding.py` 的 `RotaryEmbedding` 及子类 | 所有 RoPE 变体：持 `cos_sin_cache`，forward 原地旋转 Q/K；各外推法只重写 `_compute_inv_freq` |
| **B** `layers/rotary_embedding.py` 的 `get_rope` | 工厂：按 `rope_scaling` 配置分派到正确子类 + `_ROPE_DICT` 全局缓存实例 |
| **B** `layers/rotary_embedding.py` 的 `MRotaryEmbedding` | M-RoPE：三维 position（时/高/宽）+ 按 `mrope_section` 分段旋转 |
| **B** `csrc/pos_encoding_kernels.cu` | RoPE 的 CUDA kernel：按 position 查 `cos_sin_cache`、每 block 一个 token 旋转 |

### 1.3 和相似模块 / 概念的区别与联系（专治"听过但分不清"）

**A 部分（Mamba/SSM）：**

- **SSM 常量状态 vs KV cache 线性增长**：attention 必须保留每个历史 token 的 K/V 才能"回看"任意位置，缓存随序列 O(L) 增长（[PRIMER §1.3](../PRIMER.md#13-kv-cache它为什么存在为什么是瓶颈)）；SSM 把"回看历史"改成"维护一个固定大小的递归状态 `h`"，缓存恒为 O(1)、decode 一步也是 O(1)。代价是有损压缩（精确检索弱于 attention）。
- **Mamba vs attention**：attention 是"精确但贵"（每个历史 token 可被精确寻址，但显存/算力随长度涨）；Mamba 是"便宜但有损"（状态恒定，但历史被压缩进 `h`）。二者不是替代而是互补。
- **混合架构（hybrid）**：正因为上一条，工程上催生了 Jamba/Bamba/Zamba2 等——**大部分层用便宜的 mamba、少数层穿插 full attention** 兜住精确检索能力。一个模型里两套缓存（状态槽 + KV block）按层类型并存。

**B 部分（位置编码）：**

- **RoPE vs 绝对位置编码**：绝对位置编码（原始 Transformer 的 sinusoidal、可学习 embedding）给 token **加**一个位置向量，注入的是绝对位置，相对位置要模型自己学；RoPE 不加而是**旋转** Q/K，使点积 `q_m·k_n` 在数学上只依赖相对位移 `m−n`——相对位置无参数、无额外计算地自然涌现。
- **M-RoPE vs RoPE**：普通 RoPE 的 position 是一维 token 下标；M-RoPE（Qwen2-VL）把 position 升到三维 `(t,h,w)`，给图像/视频 token 编码空间结构。文本 token 三维取同值时 M-RoPE 退化成普通 RoPE——是 RoPE 的真·超集。
- **YaRN 外推 vs 直接外推**：模型在 `L_train` 上训练，推理想喂更长上下文。直接外推会让高频维产生没见过的相位、注意力崩溃；YaRN（及 NTK/Linear PI）**不微调**地重新分配旋转频率（高频外推、低频插值、中频平滑过渡），让长位置的相位落回训练见过的范围。

---

### 1.4 范围与架构（原开头）

本模块讲两个看似无关、实则都关乎"序列建模如何省内存 / 如何编码位置"的主题：**A 部分** 用常量大小的递归状态替代 KV cache，以及 attention 层与 mamba 层如何共存；**B 部分** RoPE 如何把相对位置注入注意力、YaRN/NTK 如何不微调外推、M-RoPE 如何为图像/视频给出三维位置。架构归属与 V0/V1 现状见本文件开头的范围/架构说明 blockquote；下面 A.1 / B.1 起分别展开两部分各自要解决的具体问题。

---

# A 部分：Mamba/SSM 与混合架构

## A.1 这个模块解决什么问题

Transformer 自回归推理的命门是 KV cache：每生成一个 token，都要保留它之前**所有** token 的 Key/Value 向量，于是显存占用 **随序列长度线性增长**（`O(L)`），而每步 attention 的计算也随上下文长度线性增长（`O(L)` per token，整段 `O(L²)`）。模块 02 用 PagedAttention 把这块显存管得很好，但**无法改变"KV 总量随 L 线性增长"这一根本事实**——上下文越长，KV cache 越吃显存，长上下文服务的瓶颈终究在这里。

**State Space Model（SSM）/ Mamba 换了一条赛道**：它不保留历史每个 token 的 K/V，而是把"到目前为止看过的全部历史"压缩进一个**固定大小的递归状态 `h`**（一个 `(num_heads, head_dim, ssm_state_size)` 的张量）。每来一个新 token：

```
h_t = A_bar · h_{t-1} + B_bar · x_t        （状态递推，h 大小恒定）
y_t = C · h_t + D · x_t                      （输出）
```

无论序列已经多长，状态 `h` 的形状**永远不变**。于是：

1. **显存随序列长度恒定**：`O(1)` per layer，而非 KV 的 `O(L)`。长上下文不再让显存爆炸。
2. **单 token 推理是 `O(1)`**：decode 一步只更新一次固定大小的状态，不必扫历史全部 KV。整段生成是 `O(L)` 而非 `O(L²)`。

代价是：SSM 是一个**有损压缩**——把无限历史塞进固定状态，理论表达力不如能精确回看每个 token 的 attention（"大海捞针"类精确检索任务上 attention 仍占优）。于是工程上催生了**混合架构（hybrid）**：大部分层用便宜的 mamba，少数层穿插 full attention 兜住精确检索能力（Jamba、Bamba、Zamba2、PLaMo2 等）。

vLLM 在 A 部分要解决的工程问题就是：**(1)** 把 Mamba 的"两段式序列变换"（因果卷积 + selective scan）落成能做 continuous batching / chunked prefill 的高效 kernel；**(2)** 把"常量状态"作为一种**新的缓存类型**纳入推理引擎的缓存管理；**(3)** 让一个模型里 attention 层和 mamba 层各取所需的缓存、在同一次 forward 里共存。

> 设计动机出处：Gu & Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*（arXiv:2312.00752, 2023）提出 **selective SSM** 与常量状态推理；Dao & Gu, *"Transformers are SSMs: ... (Mamba-2)"*（arXiv:2405.21060, 2024）给出 SSD（state space duality）与 chunk scan；Lieber et al., *"Jamba: A Hybrid Transformer-Mamba Language Model"*（arXiv:2403.19887, 2024）提出 attention+mamba 混合。

## A.2 设计目标与约束

- **常量状态替代 KV**：mamba 层不分配 KV block，而是为每个在飞行的序列分配**一个固定大小的状态槽**（conv_state + ssm_state），状态大小与序列长度无关。
- **支持 continuous batching**：同一 batch 里不同序列处于不同生成阶段（有的在 prefill、有的在 decode），kernel 必须能用 `query_start_loc`（变长 cu_seqlens）+ `state_indices_tensor`（每序列状态槽下标）一次处理整个 ragged batch。
- **支持 chunked prefill**：长 prompt 分多步 prefill，第二段起要带上前一段算出的状态作为 `initial_states`（`has_initial_states`），mamba2 的 chunk scan 还要算 `chunk_indices`/`chunk_offsets` 处理"chunk 不整除序列边界"。
- **混合架构里两类层各管各的缓存**：attention 层用 KV cache（PagedAttention），mamba 层用 `MambaCacheManager` 的状态槽，二者在模型 forward 里按层类型分派（`jamba.py:338-352`）。
- **CUDA graph 友好**：状态缓存是"seqlen-agnostic"的固定张量，decode 时能被 CUDA graph 捕获/replay（`mamba_cache.py:68`、`constant_size_cache.py:80`）。
- **约束（关键）**：
  - **不支持 prefix caching**：mamba 的状态是历史的有损压缩、不可按 token 内容寻址，没有"共享前缀块"的概念（`mamba2.py:180-181` 直接 `assert not enable_prefix_caching`）。
  - **量化下不支持 TP>1**：mamba2 在 `tp_size>1` 且有量化时直接 assert 拒绝（`mamba_mixer2.py:250-251`，来自 `#14617`）。
  - **V0-only**：状态缓存尚未接入 V1 的统一缓存抽象（详见 §A.3 与 §N）。

## A.3 核心设计思想

### A.3.1 Mamba mixer：两段式序列变换

一个 mamba 层（`MambaMixer` / `MambaMixer2`）的 forward 是一条固定流水线（以 `mamba_mixer.py:137-244` 为准）：

```
hidden ──in_proj──► [x | gate]          ① 门控 MLP 投影，拆出主路 x 和门控 gate
   x ──causal_conv1d──► x'               ② 因果深度卷积（短窗口局部混合），状态=conv_state
   x' ──x_proj──► [Δ | B | C]            ③ selective 参数：Δ(时间步)、B、C 都"输入相关"
   (x', Δ, B, C, A, D) ──selective_scan──► y   ④ SSM 递归扫描，状态=ssm_state
   y ──out_proj──► output                ⑤ 输出投影
```

两个"有状态"的地方：

- **因果卷积（conv_state）**：一个 `conv_kernel_size-1` 宽的滑动窗口，缓存最近几个 token 的卷积输入。prefill 用 `causal_conv1d_fn`（变长、带 cu_seqlens），decode 用 `causal_conv1d_update`（单 token 更新滑窗）。
- **selective scan（ssm_state）**：核心递归 `h_t = A_bar·h_{t-1} + B_bar·x_t`。**"selective" 的含义**：`Δ, B, C` 不是固定参数，而是**从输入 `x` 算出来的**（`x_proj`，`mamba_mixer.py:73-77`）——这正是 Mamba 区别于 LTI（线性时不变）S4 的关键：模型能根据输入内容**选择性地**记住或遗忘（`A` 仍输入无关，注释在 `mamba_mixer.py:29-34` 引了 Mamba 论文 §3.5.2）。prefill 用 `selective_scan_fn`（整段扫描并吐出最终状态），decode 用 `selective_state_update`（拿状态 + 单 token，更新状态并出 1 个输出）。

**为什么常量状态能替代随序列增长的 KV**：attention 必须保留每个历史 token 的 K/V 才能让当前 query"回看"任意位置；SSM 把"回看历史"改成"维护一个递归状态"——历史的信息被**增量地累加进固定大小的 `h`** 里。新 token 只和 `h`（一个固定张量）作用，不和历史逐 token 作用。于是缓存从"`L` 份 K/V"变成"1 份状态"，这就是从 `O(L)` 到 `O(1)` 的来源。

### A.3.2 Mamba-2 与 chunk scan：把递归改写成可并行的矩阵乘

朴素 selective scan 是严格串行的递归（`h_t` 依赖 `h_{t-1}`），prefill 长序列时 GPU 利用率低。Mamba-2 的 SSD（state space duality）洞察：SSM 递归在**块内**可以写成一个（结构化掩码的）矩阵乘，**块间**只需传递块边界的状态。于是 `mamba_chunk_scan_combined`（`ops/ssd_combined.py`）把序列切成 `chunk_size` 的块：

```
块内：用矩阵乘并行算（ssd_bmm / ssd_chunk_scan / ssd_chunk_state）   ← 喂饱 GPU
块间：只传块边界状态（ssd_state_passing）                          ← 串行但量极小
```

`mamba2_metadata.py` 在 **top-level 模型 forward** 一次性算好 `seq_idx`（每个 token 属于 batch 里第几条序列）、`chunk_indices`/`chunk_offsets`（处理 chunk 跨序列边界），所有 mamba2 层复用，避免每层重算（`mamba2_metadata.py:92-101` 注释明示这个"算一次、各层共享"的设计）。

### A.3.3 混合架构：attention 层与 mamba 层如何共存

以 Jamba 为例（`jamba.py`）：模型的层是 `JambaAttentionDecoderLayer` 和 `JambaMambaDecoderLayer` 交替/按配方堆叠（`jamba.py:265-268` 的 `ALL_DECODER_LAYER_TYPES` 字典）。forward 时按层类型分派缓存（`jamba.py:336-352`）：

```python
for layer in self.layers:
    if isinstance(layer, JambaAttentionDecoderLayer):
        kv_cache_index += 1                      # attention 层走 KV cache（PagedAttention）
    if isinstance(layer, JambaMambaDecoderLayer):
        layer_mamba_cache_params = mamba_cache_params.at_layer_idx(mamba_cache_index)
        mamba_cache_index += 1                    # mamba 层走状态缓存
    hidden_states, residual = layer(..., mamba_cache_params=layer_mamba_cache_params)
```

- **attention 层**：和普通 Transformer 一样，KV 写进 PagedAttention 的 block（见 [模块 02](../02-paged-attention-kvcache/design.md)）。
- **mamba 层**：从 `MambaCacheManager` 拿到本层的 `(conv_state[layer], ssm_state[layer], state_indices_tensor)`。

**两种缓存的本质区别**（链回 [模块 02](../02-paged-attention-kvcache/design.md) 的 KV 管理对比）：

| 维度 | KV cache（attention 层） | 状态缓存（mamba 层） |
|---|---|---|
| 大小随序列 | `O(L)`，按 block 增长 | `O(1)`，固定 `(conv+ssm)` 张量 |
| 管理器 | `KVCacheManager`（V1）/ `BlockSpaceManager`（V0） | `MambaCacheManager`（`ConstantSizeCache` 子类） |
| 寻址 | block table：逻辑块→物理块 | `state_indices_tensor`：req→状态槽下标 |
| 前缀复用 | prefix caching（hash 链） | **无**（状态不可按内容寻址） |
| 抢占恢复 | recompute / 重新 prefill | 释放状态槽，重新 prefill |
| 容量上界 | `num_gpu_blocks` 个 block | `max_batch_size` 个状态槽（`mamba_cache.py:32`）|

**V1 缓存抽象如何"容纳"状态缓存？——目前还没有**。这是本模块最重要的事实更正：在本仓库快照里，V1 的 `KVCacheManager` 仍 assert 单一 KV cache group（见 [模块 02](../02-paged-attention-kvcache/design.md) §2 的约束），`kv_cache_interface.py` 里**没有 `MambaSpec`**，`gpu_model_runner.py` 里**没有 mamba 分支**。所有 mamba/hybrid 模型靠 `SupportsV0Only` 标记被引擎路由到 V0 路径，状态缓存由模型自带的 `MambaCacheManager` 独立管理（`current_run_tensors` 在每次 forward 被调）。"把常量状态做成 V1 的一类 KVCacheSpec、让混合架构在 V1 统一调度下跑"是正在演进的方向（[模块 02](../02-paged-attention-kvcache/design.md) §2 的"混合架构尚在演进"注释正是为此埋的伏笔）。见 §N 的历史脉络。

### A.3.4 状态缓存的"槽位"管理：ConstantSizeCache

`MambaCacheManager`（`mamba_cache.py:25`）继承 `ConstantSizeCache`（`constant_size_cache.py:10`），核心是一张 `cache_indices_mapping: {req_id: {seq_id: slot_index}}` 和一个 `free_cache_indices` 空闲槽列表：

- **分配**：新请求来时 `free_cache_indices.pop()` 拿一个槽（`constant_size_cache.py:101`）。
- **n>1 并行采样**：子序列共享父序列的 prefill 状态——`_copy_cache` 把父槽复制到新槽（`constant_size_cache.py:111-113`、`mamba_cache.py:54`）。这是状态缓存版的"copy-on-write"，但因为状态不可共享读，必须**真拷贝**。
- **释放**：请求结束时把槽还回 `free_cache_indices`（`constant_size_cache.py:129-136`）。
- **decode + CUDA graph**：`get_seqlen_agnostic_capture_inputs` 给 graph 一个固定 buffer，`copy_inputs_before_cuda_graphs` 在 replay 前把 `state_indices` 写进 graph 的输入 buffer（`constant_size_cache.py:57-78`）。

## A.4 关键数据结构（A 部分）

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `MambaMixer` | `mamba/mamba_mixer.py:26` | Mamba-1 mixer：conv + selective_scan，含 Δ/B/C 的 input-dependent 投影 |
| `MambaMixer2` | `mamba/mamba_mixer2.py:198` | Mamba-2 mixer：用 chunk scan、支持 n_groups、TP 分片 |
| `Mamba2Metadata` | `mamba/mamba2_metadata.py:15` | top-level 算一次的元数据：`has_prefill`/`has_initial_states`/`chunk_size`/`seq_idx`/`chunk_indices`/`chunk_offsets` |
| `MambaCacheParams` | `models/mamba_cache.py:14` | 一层的状态视图：`conv_state`、`ssm_state`、`state_indices_tensor`；`at_layer_idx()` 取某层 |
| `MambaCacheManager` | `models/mamba_cache.py:25` | 跨步持有 conv/ssm 状态张量 `[num_layers, max_batch, ...]`，按 req 分配槽 |
| `ConstantSizeCache` | `models/constant_size_cache.py:10` | 状态缓存基类：`cache_indices_mapping` + `free_cache_indices`，槽位分配/释放/拷贝 |
| `state_indices_tensor` | （运行时）| `[num_seqs]`，每条序列在状态张量里的槽下标；喂给 selective_scan kernel 选状态 |
| `HasInnerState` / `IsHybrid` / `IsAttentionFree` / `SupportsV0Only` | `models/interfaces.py:317,390,353,547` | 模型能力标记：有内部状态 / 混合 / 纯 SSM 无 attention / 仅 V0 |

## A.5 权衡取舍（A 部分）

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 常量状态替代 KV | 显存 `O(1)` per layer；decode `O(1)`；长上下文不爆显存 | 有损压缩，精确检索（needle-in-haystack）弱于 attention |
| 混合架构（少数 attention + 多数 mamba） | 兼顾长上下文省显存与精确检索能力 | 两套缓存并存，调度/管理复杂度上升 |
| mamba2 chunk scan | prefill 用矩阵乘喂饱 GPU，比串行递归快 | 需算 chunk_indices/offsets 处理边界，元数据逻辑复杂 |
| 独立 `MambaCacheManager`（非 KVCacheManager） | 状态缓存语义简单（定长槽、无 block） | 与 V1 统一缓存抽象割裂 → mamba 模型 `SupportsV0Only`，吃不到 V1 调度红利 |
| 不支持 prefix caching | 实现简单、契合"状态不可按内容寻址" | 共享前缀的工作负载无法复用（system prompt 等每次重算） |
| 量化下禁 TP>1 | 避免量化×分片的正确性雷区 | 大 mamba 模型量化部署受限（`#14617`）|
| `max_batch_size` 个固定状态槽 | 槽位 O(1) 分配，CUDA graph 友好 | 并发上界被 `max_num_seqs` 硬卡死，超出即排队 |

## A.6 设计动机出处（A 部分）

- **Mamba**：Gu & Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*（arXiv:2312.00752）。selective SSM、`Δ/B/C` input-dependent、常量状态推理、硬件感知扫描。`mamba_mixer.py:29-34` 注释直接引该论文 §3.5.2 解释"为何 A 非 selective"。
- **Mamba-2 / SSD**：Dao & Gu, *"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"*（arXiv:2405.21060）。chunk scan、`ops/ssd_*.py` 的实现改编自官方 `state-spaces/mamba`（`ops/mamba_ssm.py:3-4`、`ops/ssd_combined.py:3-4` 的 copyright 注释）。
- **Jamba**：Lieber et al., *"Jamba: A Hybrid Transformer-Mamba Language Model"*（arXiv:2403.19887）。attention+mamba 混合的开创性开源模型，对应 `models/jamba.py`。

---

# B 部分：位置编码（RoPE / M-RoPE / YaRN）

## B.1 这个模块解决什么问题

注意力机制本身是**置换不变**的：把 token 顺序打乱，attention 算出来一样。所以必须**显式注入位置信息**。两个子问题：

1. **如何把位置注入注意力**，且最好注入的是**相对位置**（token i 看 token j 时，关键的是 `i−j` 的距离，而非绝对下标）。
2. **如何外推到训练长度之外**：模型在 `max_position_embeddings=L_train` 上训练，推理却想喂 `4×L_train` 的上下文。直接外推会让注意力崩溃（没见过的高频相位）。能否**不重新微调**就把上下文拉长？

**RoPE（Rotary Position Embedding）** 解决问题 1：不给 token 加一个位置向量，而是**按位置把 Q/K 向量在每个二维子空间里旋转一个角度** `θ_i = position · freq_i`。两个 token 的 attention 分数 `q_m·k_n` 旋转后只依赖 `m−n`（相对位置），位置信息被优雅地编进点积。

**YaRN / NTK-aware scaling** 解决问题 2：通过**修改旋转频率**（拉伸/插值），让模型在长上下文下的相位落回它训练时见过的范围，从而不微调就外推。

**M-RoPE（Multimodal RoPE）** 把问题 1 推广到多模态：图像/视频不是一维 token 流，而有"时间 × 高 × 宽"三个维度。M-RoPE 给每个 token 一个**三维位置** `(t, h, w)`，让模型理解 patch 的空间排布（Qwen2-VL）。

## B.2 设计目标与约束

- **相对位置经点积自然涌现**：RoPE 旋转后 `q_m·k_n` 只依赖 `m−n`，无需显式相对位置偏置矩阵。
- **cos/sin 预计算 + 缓存**：旋转角对所有层、所有请求相同，按 `max_position` 预算一张 `cos_sin_cache` 缓存（`rotary_embedding.py:99-102`），forward 时按 position 查表，避免每步重算三角函数。
- **统一接口 `get_rope`**：模型只调 `get_rope(...)`，由它按 `rope_scaling` 配置分派到正确的子类（`rotary_embedding.py:1151`），并用 `_ROPE_DICT` 全局缓存同参数实例（`rotary_embedding.py:1148,1176-1177`）。
- **多后端**：`RotaryEmbedding` 是 `CustomOp`，有 `forward_native`（PyTorch）/`forward_cuda`（调 `ops.rotary_embedding` 原地 kernel）/`forward_xpu`/`forward_hpu`/`forward_neuron` 多实现。
- **外推不微调**：YaRN/NTK/Llama3 scaling 只改 `_compute_inv_freq` 里的频率，不动权重。
- **约束**：M-RoPE 的 position 是 3×num_tokens 的张量，`cos_sin_cache` 要按 `mrope_section` 分段取（`rotary_embedding.py:962-974`）；chunked prefill 下 mrope position 的记账容易错（见 §N 的 bug）。

## B.3 核心设计思想

### B.3.1 RoPE：用旋转把相对位置编进点积

把 head 维每两维配成一个二维平面，第 `i` 对平面的旋转频率 `freq_i = base^(−2i/d)`（`rotary_embedding.py:104-112`，低维高频、高维低频，几何级数）。位置 `p` 处的旋转角 `θ = p · freq_i`。对 Q/K 的每个二维子向量旋转该角：

```
[q'_2i ]   [cos θ  −sin θ] [q_2i ]
[q'_2i+1] = [sin θ   cos θ] [q_2i+1]
```

关键性质：旋转后两 token 的点积 `q'_m · k'_n = f(q_m, k_n, m−n)` **只依赖相对位移 `m−n`**。这就是 RoPE 的精髓——绝对位置的旋转，在点积里**自动退化成相对位置**，无需任何额外参数。

vLLM 把 `cos`、`sin` 按位置预计算成 `cos_sin_cache`（`[max_pos, rotary_dim]`，前半 cos 后半 sin，`rotary_embedding.py:114-123`），forward 时按 `positions` 查表，CUDA kernel 原地旋转 Q/K（`csrc/pos_encoding_kernels.cu`）。**Neox vs GPT-J 风格**：两者把 head 维配对的方式不同（Neox 是 `[0..d/2)` 配 `[d/2..d)`，GPT-J 是相邻 `2i,2i+1` 配对），kernel 用 `IS_NEOX` 模板参数区分（`pos_encoding_kernels.cu:16-28`）。

### B.3.2 长上下文外推：从 NTK 到 YaRN

直接把 RoPE 用到训练长度之外会失败：高频维度在长位置处产生模型没见过的相位。几种**不微调**的外推法（都只改 `_compute_inv_freq`）：

- **Linear（位置插值 PI）**：把位置 `p` 缩成 `p/scaling_factor`，相当于把所有频率等比压缩（`LinearScalingRotaryEmbedding`，`rotary_embedding.py:404-405`）。简单但损伤高频局部分辨率。
- **Dynamic NTK**：不均匀地改 `base`——按 `scaling_factor` 把 base 放大（`rotary_embedding.py:457-461`），让高频维少插值、低频维多插值，缓解 PI 对局部信息的损伤。
- **YaRN**：更精细。按"波长"把维度分成三段（`_yarn_find_correction_range`，`rotary_embedding.py:482-493`）：**高频维直接外推**（波长 < 上下文，不动）、**低频维插值**（波长 > 上下文）、**中间维用线性 ramp 平滑过渡**（`_yarn_linear_ramp_mask`）。同时引入 `mscale`（`_yarn_get_mscale`，`rotary_embedding.py:506-509`）对 attention logits 做温度补偿。YaRN 是当前长上下文外推的主流（`YaRNScalingRotaryEmbedding._compute_inv_freq`，`rotary_embedding.py:544-560`）。
- **Llama3 scaling**：Llama-3.1 的变体，按波长阈值 `low/high_freq_factor` 分段缩放频率并平滑（`Llama3RotaryEmbedding._compute_inv_freq`，`rotary_embedding.py:830-851`）。
- **LongRoPE（Phi-3）**：用搜索得到的 `short_factor`/`long_factor` 两套频率，按序列长度切换短/长缓存（`Phi3LongRoPEScaledRotaryEmbedding`，`rotary_embedding.py:573`）。

**核心思想统一表述**：长上下文外推 = **重新分配旋转频率**，让"模型在长位置看到的相位分布"尽量贴近"训练时见过的相位分布"。高频管局部、低频管全局，YaRN 的高明在于对二者**区别对待**而非一刀切。

### B.3.3 M-RoPE：给图像/视频三维位置

文本是 1D 序列，position 就是 token 下标。但图像/视频 token 有空间结构。Qwen2-VL 的 **M-RoPE** 把 rotary 维度切成三段 `mrope_section = [t_dim, h_dim, w_dim]`（`rotary_embedding.py:938-940`，断言 `sum == rotary_dim//2`），每段用 position 的一个分量（时间/高/宽）做旋转：

- **文本 token**：三个分量取相同值（退化成普通 RoPE，`gpu_model_runner.py:219-223` 注释）。
- **图像 token**：`t` 恒定、`(h,w)` 按 patch 的二维坐标展开（`get_input_positions_tensor`，`rotary_embedding.py:1098-1107`）。
- **视频 token**：`t` 按帧（乘 `second_per_grid_t · tokens_per_second`）、`(h,w)` 按空间（`rotary_embedding.py:1098-1100`）。

position 张量形状从 `[num_tokens]` 变成 `[3, num_tokens]`，forward 里按 `mrope_section` 把三段 cos/sin 拼起来（`rotary_embedding.py:962-974`）。**`mrope_position_delta`** 记录"多模态 token 把后续文本的 position 撑大了多少"，供 decode 阶段续接位置（`rotary_embedding.py:1118`、`get_next_input_positions_tensor`）。

V1 里这套 position 在 `gpu_model_runner` 管理：prompt 部分预计算（`_update_states`，`gpu_model_runner.py:369-371`），每步把 prompt/completion 的 mrope position 拼进 `mrope_positions` buffer（`_calc_mrope_positions`，`gpu_model_runner.py:701-751`）。详见 [模块 10](../10-multimodal/design.md) 的多模态预处理。

## B.4 关键数据结构（B 部分）

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `RotaryEmbedding` | `rotary_embedding.py:79` | RoPE 基类，持 `cos_sin_cache`，forward 原地旋转 Q/K |
| `cos_sin_cache` | `rotary_embedding.py:99-102` | `[max_pos, rotary_dim]`，前半 cos 后半 sin，按 position 查表 |
| `LinearScalingRotaryEmbedding` | `rotary_embedding.py:345` | 位置插值（PI）；支持多 scaling factor（LoRA 长上下文）|
| `DynamicNTKScalingRotaryEmbedding` | `rotary_embedding.py:431` | Dynamic NTK，改 base 做不均匀插值 |
| `YaRNScalingRotaryEmbedding` | `rotary_embedding.py:512` | YaRN：分段频率 + mscale logit 补偿 |
| `Llama3RotaryEmbedding` | `rotary_embedding.py:808` | Llama-3.1 的波长分段缩放 |
| `Phi3LongRoPEScaledRotaryEmbedding` | `rotary_embedding.py:573` | LongRoPE：短/长两套频率切换 |
| `DeepseekScalingRotaryEmbedding` | `rotary_embedding.py:698` | YaRN 变体，含 `mscale_all_dim` |
| `MRotaryEmbedding` | `rotary_embedding.py:918` | M-RoPE：三维 position + `mrope_section` 分段旋转 |
| `Llama4VisionRotaryEmbedding` | `rotary_embedding.py:854` | Llama4 vision 的复数式 2D rope |
| `get_rope` | `rotary_embedding.py:1151` | 工厂：按 `rope_scaling["rope_type"]` 分派 + `_ROPE_DICT` 缓存 |
| `mrope_positions` (buffer) | `gpu_model_runner.py:224` | V1 的 `[3, max_tokens+1]` 三维 position GPU buffer |

## B.5 权衡取舍（B 部分）

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| RoPE（旋转而非加性位置嵌入）| 相对位置经点积自然涌现；无额外参数；外推友好 | 只作用于 attention 的 Q/K，不进 V/MLP；外推需额外 scaling |
| 预计算 `cos_sin_cache` 查表 | 避免每步算三角函数；CUDA kernel 原地旋转 | 长上下文 cache 占显存（M-RoPE 还 ×4 预留，`rotary_embedding.py:934`）|
| `get_rope` + `_ROPE_DICT` 全局缓存 | 同参数 RoPE 实例复用，省构造开销 | 缓存 key 必须涵盖所有影响参数，否则错误命中（历史上有 `#1867` cache key bug）|
| YaRN 分段频率 | 长上下文外推质量高、不微调 | 实现复杂（correction range/ramp/mscale），超参敏感 |
| Linear PI | 实现极简 | 损伤高频局部分辨率 |
| M-RoPE 三维 position | 多模态空间结构被编码 | position 记账复杂，chunked prefill 易错（`#10388`/`#10403`）|
| native vs cuda 双实现 | 多硬件可移植 | 两套实现需保持数值一致（历史上 forward_native 多次重构）|

## B.6 设计动机出处（B 部分）

- **RoPE**：Su et al., *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*（arXiv:2104.09864, 2021）。旋转注入相对位置的原始论文。
- **YaRN**：Peng et al., *"YaRN: Efficient Context Window Extension of Large Language Models"*（arXiv:2309.00071, 2023）。`rotary_embedding.py:515` 注释直接 credit `github.com/jquesnelle/yarn`。
- **NTK-aware / Dynamic NTK / 位置插值（PI）**：源自 Reddit 社区（bloc97、emozilla、kaiokendev），`rotary_embedding.py:371,434` 注释明确 credit 这几位；PI 学术对应 Chen et al. *"Extending Context Window ... via Positional Interpolation"*（arXiv:2306.15595）。
- **M-RoPE / Qwen2-VL**：Wang et al., *"Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"*（arXiv:2409.12191, 2024）。三维（时/高/宽）位置编码，对应 `MRotaryEmbedding`。
- **LongRoPE**：Ding et al.（arXiv:2402.13753），对应 `Phi3LongRoPEScaledRotaryEmbedding`。

---

## C 部分：设计背后的考量与历史教训

### N.1 设计背后的考量（动机与历史）

1. **【A】Mamba 之所以"还没进 V1"，是因为常量状态和 KV block 是两套语义不兼容的缓存模型**。V1 的 `KVCacheManager` 整套抽象（block table、prefix hash 链、ref_cnt 软驱逐）都是为"随序列增长、可按内容寻址、可共享前缀"的 KV 设计的；而 mamba 状态是**定长、不可寻址、不可共享**的。强行套进 KVCacheSpec 会污染那套优雅的抽象，于是工程上选择先用独立的 `MambaCacheManager` + `SupportsV0Only` 隔离，等 V1 的 `KVCacheSpec` 泛化出"非 block 类缓存"的子类型后再接入。[模块 02](../02-paged-attention-kvcache/design.md) §2 那条"单一 KV cache group""混合架构尚在演进"的约束，正是这件事在 KV 侧投下的影子。

2. **【A】"selective" 是 Mamba 的命门，也是 vLLM 必须做 input-dependent 投影的原因**。S4 等早期 SSM 是 LTI（A/B/C 全固定），可以用卷积一次算完，但表达力弱（不能按内容选择记忆）。Mamba 让 `Δ/B/C` 随输入变（`x_proj`），换来"选择性记忆"，代价是不再能用单个全局卷积、必须做带状态的 scan。vLLM 的 `selective_scan_fn` / `selective_state_update` 两条路径（prefill 整段扫 vs decode 单步更新）正是为这个"有状态递归"服务的。

3. **【A】"算一次元数据、各 mamba 层共享"是 Mamba-2 chunked prefill 的关键省法**。`prepare_mamba2_metadata` 在 top-level forward 算一次 `seq_idx`/`chunk_indices`/`chunk_offsets`，所有层复用（`mamba2_metadata.py:92-101` 注释坦言"本可只在有 initial states 时算，但那需要 attention 元数据，干脆都在顶层算一次"）。这是"宁可偶尔多算一点、也要避免每层重复 device sync"的取舍。

4. **【B】RoPE 选择"旋转"而非"加性位置嵌入"，是为了让相对位置免费涌现**。加性位置嵌入（如原始 Transformer 的 sinusoidal、或可学习 embedding）注入的是绝对位置，相对位置要靠模型自己学。RoPE 的旋转让 `q_m·k_n` 在数学上直接只依赖 `m−n`，相对位置**无参数、无额外计算**地出现在点积里——这是它能横扫现代 LLM 的根本原因，也是为什么外推问题能归结为"改频率"这一件干净的事。

5. **【B】长上下文外推全部收敛到"只改 `_compute_inv_freq`"，是 vLLM RoPE 体系最漂亮的抽象**。Linear/NTK/YaRN/Llama3/LongRoPE 五花八门，但在代码里都只是 `RotaryEmbedding` 的子类重写 `_compute_inv_freq`（或 `_compute_cos_sin_cache`），forward 旋转逻辑完全复用。`get_rope` 工厂 + `_ROPE_DICT` 缓存把"选哪种外推"压成一个配置分派。这让新外推法的接入成本极低，也让 kernel 永远只需懂"按 cos_sin_cache 旋转"这一件事。

6. **【B】M-RoPE 把 position 从 `[L]` 升到 `[3,L]`，逼着整条 position 记账重写**。一旦 position 有三维，chunked prefill 下"这一步该取 position 的哪一段"、decode 续接位置时"delta 是多少"都成了易错点（见 N.2 的 `#10388`/`#10403`）。vLLM 把 prompt 段 position 预计算、completion 段用 `mrope_position_delta` 续算（`gpu_model_runner.py:_calc_mrope_positions`），是为了在不重算整段的前提下正确续接——这也是多模态长生成的隐蔽复杂度来源。

### N.2 重要 bug 修复（真实、精选）

> PR 号来自 `git log -- vllm/model_executor/layers/mamba/ vllm/model_executor/layers/rotary_embedding.py`，均已 `git show` 核实。

| PR | 主题 | 暴露的设计点 / 教训 |
|---|---|---|
| `#14617`（A）| Mamba2 在量化时强制禁止 TP>1 | 量化 × 张量并行 × mamba 的特殊分片（n_groups/head 复制）三者叠加是正确性雷区；与其修复全部组合，不如显式 `assert` 把不支持的组合挡在门外（`mamba_mixer2.py:250-251`）。教训：有些组合的代价是"先禁掉比先支持更负责任" |
| `#15423`（A）| Bamba-9B 的 mamba2 块里冗余计算 | mamba2 prefill 在 batch 全是 decode 或无 initial states 时仍做了多余的状态准备/拷贝；说明"continuous batching 下 prefill/decode 混合"的分支极易留下只在某些 batch 形态才触发的冗余 |
| `#14778` / `#14848`(revert) / `#14857`（A）| Mamba2 prefill "一连串不必要的内存拷贝" | 同一性能优化被合入(`#14778`)、回退(`#14848`)、再正确合入(`#14857`)。状态张量的 in-place 更新 vs 拷贝边界极微妙，改错就是静默数值错误——和 [模块 02](../02-paged-attention-kvcache/design.md) §8 里"同一 kernel 边界被改两次"如出一辙：**有状态更新路径的拷贝/原地之争，必须有测试守住** |
| `#10388`（B）| chunked prefill 开启时 M-RoPE position 计算错 | 多模态三维 position 在"长 prompt 分块 prefill"时，每块该取 position 的哪一段算错。坐实 M-RoPE 把 position 记账复杂度抬了一个量级 |
| `#10403`（B）| 非最后一个 prefill chunk 的 `mrope_position_delta` 错 | `delta`（多模态撑大后续文本 position 的量）只在最后一块才该 finalize，中间块算 delta 会污染续接。教训：跨 chunk 的"位置增量"是有状态量，必须分清"何时才能定稿" |
| `#14548`（B）| Qwen2-VL/2.5-VL 的 `second_per_grid_ts`（视频帧时间间隔）处理错 | 视频的时间维 position 依赖每帧时间间隔；处理错会让视频 token 的时间位置全偏。多模态 position 的每个维度都有自己的物理含义，少乘一个因子就是一类静默错误 |
| `#2983` / `#2984`（B）| 初始化 YaRN RoPE 的 bug / YaRN model len 的断言 | YaRN 引入时 `_compute_inv_freq` 的分段频率/`max_position × scaling` 的长度计算易错；两个紧邻 PR 修初始化 + 修长度断言，说明外推法的"频率重分配"数学一旦写错，表现为长上下文质量悄悄崩坏 |
| `#1867`（B）| RoPE cache key 错误 | `_ROPE_DICT` 的缓存 key 没涵盖全部影响参数 → 不同配置错误命中同一个 RoPE 实例。和 [模块 02](../02-paged-attention-kvcache/design.md) §8 的 prefix-cache "键完备性即正确性"是同一类教训：**内容寻址/参数寻址的缓存，键设计就是正确性本身** |
| `#11604`（B）| `DeepseekScalingRotaryEmbedding` 里漏了一句 `print` | 小但真实：调试语句进了热路径。提醒位置编码在每层每步都跑，任何残留 I/O 都是性能毒药 |

---

## 一图速览

```
A 部分：KV cache（O(L)）  vs  SSM 常量状态（O(1)）
─────────────────────────────────────────────────────────────
Attention 层：每个历史 token 存一份 K/V，随序列线性增长
   t=1: [k1,v1]
   t=2: [k1,v1][k2,v2]
   t=3: [k1,v1][k2,v2][k3,v3]   ← 缓存随 L 增长（PagedAttention 管它，见模块02）

Mamba 层：把全部历史压进一个固定大小的状态 h
   t=1: h ← A·h + B·x1                  conv_state(滑窗) + ssm_state(递归)
   t=2: h ← A·h + B·x2     ┐ 状态形状永不变 → 显存 O(1)
   t=3: h ← A·h + B·x3     ┘ decode 一步 O(1)，不扫历史

混合架构（Jamba/Bamba/Zamba2）：按层分派两套缓存
   [mamba][mamba][ATTN][mamba][mamba][ATTN]...
     │              │
   状态槽         KV block
   (MambaCacheManager)  (KVCacheManager, 见模块02)
   ※ 本快照：mamba 走 V0（SupportsV0Only），状态缓存尚未并入 V1 抽象

─────────────────────────────────────────────────────────────
B 部分：RoPE 旋转 → 相对位置；YaRN 改频率 → 外推；M-RoPE → 三维
─────────────────────────────────────────────────────────────
RoPE：按 position p 旋转 Q/K 的每个二维子空间 θ = p·freq_i
   q'_m · k'_n  =  f(q_m, k_n, m−n)     ← 点积只依赖相对位置 m−n

频率分布（低维高频管局部，高维低频管全局）：
   freq_i = base^(−2i/d)
   外推 = 重新分配 freq，让长位置的相位落回训练见过的范围
     Linear:  p → p/s        （全频等比压缩，简单但损局部）
     NTK:     改 base         （不均匀插值）
     YaRN:    高频外推 | 中频 ramp | 低频插值 + mscale logit 补偿  ← 主流

M-RoPE（Qwen2-VL）：position 从 [L] → [3, L]
   文本 token: (p, p, p)            ← 退化成普通 RoPE
   图像 token: (t常量, h坐标, w坐标)
   视频 token: (t帧×Δt, h, w)
   按 mrope_section=[t,h,w] 分段取 cos/sin 旋转

get_rope(rope_scaling) ──分派──► RotaryEmbedding 子类（只重写 _compute_inv_freq）
   ──► forward_cuda ──► ops.rotary_embedding（按 cos_sin_cache 原地旋转 Q/K）
```

> 实现层面的逐行调用链、`file:line` 对照与边界情况，见同目录 [`impl.md`](impl.md)。
