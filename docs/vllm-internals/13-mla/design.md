# 模块 13 · MLA（Multi-head Latent Attention，DeepSeek）—— 设计文档

> 范围：DeepSeek 系列（V2 / V3 / R1）使用的 **Multi-head Latent Attention (MLA)** 在 vLLM 中的设计。MLA 是 **PagedAttention（[模块 02](../02-paged-attention-kvcache/design.md)）的一个注意力变体**——它仍然把 KV cache 切成 block、用 block table 做逻辑→物理寻址，但**存进 cache 的不是完整的多头 K/V，而是一条低秩 latent 向量**，从而把 KV cache 显存压缩数倍。本模块只讲 MLA 特有的设计；分页、block table、prefix caching 等通用机制请回看 [模块 02](../02-paged-attention-kvcache/design.md)。
> 架构：**V1（`vllm/v1/attention/backends/mla/`）为主**。MLA 在 V0 也有实现，但本文聚焦 V1。

---

## 1. 这个模块解决什么问题

标准 Multi-Head Attention (MHA) 的 KV cache 是性能与显存的命门（见 [模块 02](../02-paged-attention-kvcache/design.md) §1）。它的显存占用正比于：

```
KV cache 每 token 字节数 = 2(K和V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

对 DeepSeek-V3 这种**深、宽**的模型，`num_heads=128`、`head_dim=128`，每 token 的完整 K/V 是 `2 × 128 × 128 = 32768` 个元素/层。长上下文（128K）下，单是 KV cache 就能吃掉几十 GB 显存，**直接限制了 batch size 和可服务的并发数**。

GQA/MQA（Grouped/Multi-Query Attention）通过**减少 KV head 数**来压缩 KV cache，但会损失模型质量。DeepSeek 的 MLA 走了另一条路：

> **不缓存完整的 K/V，而是把它们压缩成一条低秩的 latent 向量 `kv_c`（维度 `kv_lora_rank`，DSV3 中为 512），需要时再用一个上投影矩阵 `W_UK/W_UV` 还原出每个 head 的 K/V。**

于是 MLA 缓存的每 token 只有 `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` 个元素（**单 head**，不是 `num_heads × head_dim`），相比 MHA 的 `2 × 128 × 128 = 32768` 元素压缩了约 **57 倍**（同时只存一份、不分 K/V 两份）。这就是 DeepSeek 能在 128 头、128K 上下文下仍保持可观吞吐的关键。

**核心矛盾**：把 K/V 压成 latent 省了显存，但 attention 计算需要完整的多头 K/V。MLA 的全部精妙之处，就在于**如何在不把 latent 重新展开成完整 KV cache 的前提下，正确而高效地完成 attention**——尤其是 decode 阶段。

---

## 2. 设计目标与约束

- **KV cache 只存 latent**：物理 cache 里每 token 只放一条 `[kv_lora_rank + qk_rope_head_dim]` 的向量，复用模块 02 的 block/block table/slot 机制，但 KV cache 张量形状退化为单 head（`common.py:239` `get_kv_cache_shape` 返回 `(num_blocks, block_size, head_size)`，`num_kv_heads` 恒为 1）。
- **沿用 PagedAttention 的分页**：latent 向量按 block 存、用 block table 寻址、slot_mapping 写入——这些与 [模块 02](../02-paged-attention-kvcache/design.md) 完全一致，MLA 只换了"每个 slot 里存什么"。
- **两条算法路径**：prefill（compute-bound）用"上投影还原完整 K/V 再做 MHA"的**计算友好**路径；decode（memory-bound）用"吸收上投影、对 latent 直接做 MQA"的**数据搬运友好**路径（`common.py:9-27` 文件头注释是设计纲领）。
- **decoupled RoPE**：RoPE 不能直接作用在压缩后的 latent 上（压缩破坏了位置结构），所以 MLA 把 query/key 拆成 **nope 部分（不加 RoPE，`qk_nope_head_dim=128`）** 和 **rope 部分（单独走 RoPE，`qk_rope_head_dim=64`）**，二者拼接后做 attention。这是 MLA 与压缩能共存的前提。
- **吸收（absorb）矩阵**：decode 时不物化完整 K/V，而是把上投影矩阵 `W_UK`（K 上投影）合并进 Q、把 `W_UV`（V 上投影）合并进 O，使 attention 直接在 latent 维度上以 MQA 形式运行（`common.py:648-658` `_q_proj_and_k_up_proj`、`:638-645` `_v_up_proj_and_o_proj`）。
- **prefill/decode 分批（reorder_batch）**：因为两条路径算法不同、对 batch 的张量布局要求不同，MLA backend 必须把同一 step 里的 decode 请求和 prefill 请求**在 batch 内分到两段**（decode 在前、prefill 在后），否则无法分别走两套 kernel。这是 MLA 对 [模块 01](../01-scheduler-batching/design.md) 调度产物的一个额外约束。
- **约束**：
  - MLA 的 head_size 固定为 576（`common.py:248` `get_supported_head_sizes`），即 `kv_lora_rank(512) + qk_rope_head_dim(64)`。
  - **不支持 cascade attention**（`common.py:252` `use_cascade_attention` 恒返回 False）——公共前缀复用见 [模块 02](../02-paged-attention-kvcache/design.md) §3，MLA 暂未接入。
  - FlashMLA / TritonMLA 均**不支持 FP8 KV cache、alibi、sliding window、logits_soft_cap**（`flashmla.py:104-121`、`triton_mla.py:50-67`）。

---

## 3. 核心设计思想

### 3.1 latent KV 压缩：用一条低秩向量取代完整 KV

MLA 的输入投影链（模型侧 `deepseek_v2.py:466-482` `DeepseekV2MLAAttention.forward`）：

```
hidden_states ─ kv_a_proj_with_mqa ─► [kv_c | k_pe]   (一次投影同时产出 latent 和 decoupled rope key)
                                          │      └─ k_pe: [Skv, qk_rope_head_dim=64]   走 RoPE
                                          └─ kv_c: [Skv, kv_lora_rank=512]            经 kv_a_layernorm 归一化
```

**只有 `kv_c`（latent）和 `k_pe`（rope key）被写进 KV cache**（`common.py:917-926` `concat_and_cache_mla`），合计 `512 + 64 = 576` 维。完整的多头 K/V 从不进 cache——它们在需要时由 `kv_b_proj`（= `[W_UK; W_UV]` 拼接，`deepseek_v2.py:409-414`）从 `kv_c` 上投影还原。

各维度（`common.py:30-60` 文件头定义，DSV3 实际值）：

| 符号 | 含义 | DSV3 值 |
|---|---|---|
| `Lkv` = `kv_lora_rank` | KV latent 维度 | 512 |
| `Lq` = `q_lora_rank` | Q latent 维度（可选，省 Q 投影参数） | 1536 |
| `P` = `qk_nope_head_dim` | 不加 RoPE 的 QK head 维 | 128 |
| `R` = `qk_rope_head_dim` | 走 RoPE 的 QK head 维（decoupled） | 64 |
| `V` = `v_head_dim` | V head 维 | 128 |
| `N` = `num_heads` | 注意力头数 | 128 |

> **压缩比**：MHA 缓存 `2 × N × head_dim = 2 × 128 × 128 = 32768` 元素/token/层；MLA 缓存 `Lkv + R = 576` 元素/token/层。压缩约 **57×**。这就是 DeepSeek 长上下文高吞吐的物理基础。

### 3.2 decoupled RoPE：为什么 rope 维要单独拎出来

RoPE 把位置信息以**旋转**的方式编码进 Q/K。问题是：如果先把 K 压缩成低秩 `kv_c` 再缓存，**RoPE 无法在解压后再正确施加**——因为 `kv_b_proj` 的上投影是线性的，而 RoPE 旋转依赖绝对位置，两者不可交换。换句话说，"压缩"和"位置旋转"在同一份向量上无法共存。

DeepSeek 的解法是 **decoupled（解耦）RoPE**：把 QK 拆成两段，**只对一小段加 RoPE**：

```
q_head = [ q_nope (P=128, 不旋转) | q_pe (R=64, 旋转) ]   QK head 维 = P + R = 192
k_head = [ k_nope (P=128, 由latent上投影) | k_pe (R=64, 旋转, 独立缓存) ]
```

- `q_nope` / `k_nope`：走低秩压缩通道（k_nope 由 `kv_c @ W_UK` 还原），**不加 RoPE**。
- `q_pe` / `k_pe`：单独一小段（64 维），**只有它走 RoPE**。`k_pe` 对所有 head 共享（MQA 式），直接和 latent 一起缓存。

attention 分数 = `q_nope·k_nope + q_pe·k_pe`，两段拼接后一次算完。这样压缩通道完全不碰 RoPE，位置信息全部承载在那 64 维 `*_pe` 上。`common.py` 里 RoPE 被**搬进 attention backend 内部施加**（`common.py:903-905` decode、`:913-915` prefill），因为只有 backend 才知道 latent 已解压成哪些 head——这是 MLA 与普通 attention 的一个结构性差异（`MLACommonPrefillMetadata.input_positions` 因此要带进 metadata，`common.py:270-272`）。

### 3.3 "吸收"（absorb）：decode 为何能把上投影并进 Q/O 省算

这是 MLA 最反直觉、也最关键的一步。先看朴素做法（prefill 用的"计算友好"路径，`common.py:63-83`）：

```
k_nope = kv_c @ W_UK   →  [Skv, N, P]     ← 把 latent 上投影成完整多头 K
v      = kv_c @ W_UV   →  [Skv, N, V]     ← 上投影成完整多头 V
然后做标准 MHA(q, [k_nope|k_pe], v)
```

这要为 **每个缓存 token** 物化出 `[Skv, N, P]` 的完整 K——decode 时 `Skv` 是整个上下文长度，这等于把压缩的 KV cache 又**展开回 MHA 的规模**，省下的显存带宽全吐回去了。

**吸收（absorb）的洞察**：矩阵乘法可结合。decode 的注意力分数中，`q_nope` 和 `k_nope` 的点积可以改写：

```
q_nope · k_nope = q_nope · (kv_c @ W_UK) = (q_nope @ W_UK^T) · kv_c
                                            └────── ql_nope ──────┘
```

把 `W_UK` **吸收进 query**（`q_nope @ W_UK^T → ql_nope`，`common.py:648-658` `_q_proj_and_k_up_proj`，预先转置好的 `W_UK_T` 在 `:709` 构造），就不必为每个缓存 token 还原 `k_nope`——直接拿 `ql_nope`（已在 latent 维度）和缓存里的 `kv_c` 做点积。同理输出侧：

```
o = (attn_weights · kv_c) @ W_UV   →   把 W_UV 吸收进输出投影
```

`_v_up_proj_and_o_proj`（`common.py:638-645`）先 `bmm(attn_out, W_UV)` 再 `o_proj`。于是 decode 的 attention **整个跑在 latent 维度 `Lkv=512` 上**，且因为缓存里每 token 只有一条共享向量（不分 head），它退化成 **MQA（Multi-Query Attention）**——一份 KV 被所有 query head 共享。

> **代价**：吸收后 QK 的 head 维从 `P+R=192` 变成 `Lkv+R=512+64=576`，**算术量反而增大**（512 > 128）。但 decode 是 **memory-bound**（瓶颈在读 KV cache 的显存带宽，不在算力），MQA 形式让每 token 只读 576 维而非 `N×head_dim`，**省下的带宽远超多花的算力**。这正是文件头注释（`common.py:91-115`）说 decode 走"data-movement friendly"路径的含义。

### 3.4 prefill vs decode：两套算法、一套 cache

| | **prefill 路径**（compute-bound） | **decode 路径**（memory-bound） |
|---|---|---|
| 代码 | `_forward_prefill`（`common.py:788`） | `_forward_decode`（抽象 `:845`，子类实现） |
| 还原 KV？ | **是**：`kv_c @ kv_b_proj` 物化完整 `k_nope/v`（`:799-802`） | **否**：吸收 `W_UK/W_UV`，attention 直接吃 latent |
| attention 形式 | **MHA**，QK 头维 = `P+R=192`，V 头维 = `V=128` | **MQA**，QK 头维 = `Lkv+R=576`，V 头维 = `Lkv=512` |
| kernel | `flash_attn_varlen_func`（FA2/FA3） | FlashMLA `flash_mla_with_kvcache` 或 Triton `decode_attention_fwd` |
| 为何这样选 | `Sq/Skv ≈ 1`，算力足够，物化划算且能用现成 FA kernel | `Sq/Skv` 很大（query 1 个、KV 几千个），瓶颈在带宽，MQA 省读 |

**判据**：当前 vLLM 简单地用"调度器标的 prefill/decode"来分（`common.py:16-18` 注释承认这是粗略启发式），具体地：`reorder_batch` 里**只有 `num_scheduled_tokens == 1` 的请求算 decode**（`common.py:399`），其余都走 prefill（因为 `_forward_decode` 当前只支持 query 长度为 1）。chunked prefill 是 prefill 路径的一个变体（见 §3.5）。

### 3.5 chunked prefill：用固定 workspace 装下长上下文

prefill 路径要物化 `k_nope = kv_c @ W_UK`，形状 `[Skv, N, P]`。当 prefill 请求带很长的已缓存 context（`Skv` 大）时，这个物化会 OOM。MLA 的解法（`common.py:118-184` 注释 + `_compute_prefill_context` `:711`）：**把 context 切成固定大小的 chunk**，每个 chunk 单独 `gather_cache` 进一块固定 workspace、算一段 attention，再用 `merge_attn_states`（online-softmax 合并，见 [模块 11](../11-gpu-kernels-memory/impl.md)）把多段结果拼起来。workspace 大小被限制在 64K token（`common.py:354-371`），把"context 越长内存越大"变成"内存恒定、迭代次数变多"。这是 MLA 版的 FlashAttention 分块思想，但分的是 **context 维**而非 query 维。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `MLACommonBackend` | `common.py:222` | MLA backend 基类：`get_kv_cache_shape` 返回单 head latent cache 形状、`head_size=576`、禁用 cascade |
| `MLACommonMetadata` | `common.py:291` | 一步 attention 的元数据：`num_decodes/num_decode_tokens/num_prefills`（prefill/decode 分界）、`slot_mapping`、`decode`/`prefill` 子元数据 |
| `MLACommonDecodeMetadata` | `common.py:280` | decode 段元数据：`input_positions`（给 backend 内部 RoPE）、`block_table`、`seq_lens` |
| `MLACommonPrefillMetadata` | `common.py:256` | prefill 段元数据：`block_table`、`query_start_loc`、可选 `ChunkedContextMetadata` |
| `ChunkedContextMetadata` | `common.py:260` | chunked prefill：`cu_seq_lens`、`starts`、`workspace`（固定大小） |
| `MLACommonMetadataBuilder` | `common.py:337` | 构建上述元数据；持有 `reorder_batch`（prefill/decode 分批）、chunked prefill workspace 分配 |
| `MLACommonImpl` | `common.py:570` | attention 实现基类：`forward` 分流 decode/prefill、吸收矩阵 `W_UK_T`/`W_UV`、`_forward_prefill`、抽象 `_forward_decode` |
| `FlashMLAImpl` / `TritonMLAImpl` | `flashmla.py:80` / `triton_mla.py:29` | 两种 decode kernel 后端，仅 `_forward_decode` 不同 |
| `W_UK_T`, `W_UV` | `common.py:709`, `:707` | 从 `kv_b_proj` 拆出并转置好的上投影矩阵；decode 时被"吸收"进 Q/O |
| KV cache 张量 | `common.py:239` | `(num_blocks, block_size, 576)`——**单 head latent**，对比 MHA 的 `[2, num_blocks, block_size, num_kv_heads, head_size]`（[模块 02](../02-paged-attention-kvcache/design.md) §3.1） |

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 缓存 latent 而非完整 KV | KV cache 显存压缩约 57×，batch / 上下文大幅放大 | 每次 attention 要上投影还原（prefill）或吸收改写（decode），逻辑复杂度高 |
| decode 吸收上投影、走 MQA | 省 KV cache 读带宽（memory-bound 主因），不物化完整 K/V | QK 头维涨到 576，算力反增；只在 memory-bound 时划算（故 decode 才用） |
| prefill 物化完整 KV、走 MHA | 能直接复用现成 FlashAttention kernel，compute-bound 下划算 | 物化 `[Skv,N,P]` 占显存 → 需 chunked prefill 兜底长 context |
| decoupled RoPE（拆 nope/rope 维） | 让压缩与位置编码共存 | QK head 维变成两段拼接，RoPE 要搬进 backend 内施加，metadata 多带 positions |
| prefill/decode 分批（reorder_batch） | 两套 kernel 各自处理规整张量 | 给调度产物加约束、每 step 要 swap batch 状态（见 [模块 01](../01-scheduler-batching/design.md)） |
| W_UV/W_UK_T 存 fp16/bf16 副本做 bmm | 无需量化 bmm kernel，实现简单 | 多一份 16-bit 权重显存（`common.py:685-687` 注释承认，但占用小） |
| FlashMLA 强制 block_size=64 | FlashMLA kernel 要求 | 与默认 block_size 不同，平台层要强制改（`cuda.py:157-161`） |
| 不支持 FP8 KV / cascade / sliding window | 缩小实现面、先把主路径做对 | 功能受限，长前缀复用、FP8 省显存暂缺 |

**与 GQA/MQA 的路线对比**：GQA/MQA 靠**减少 KV head 数**压 cache（structural，损质量）；MLA 靠**低秩压缩 + 运行时还原**压 cache（保留了等效的多头表达力，质量损失小）。代价是 MLA 的 attention 实现远比 GQA 复杂——需要吸收矩阵、两套路径、decoupled RoPE。这是"用实现复杂度换质量与压缩比"的取舍。

---

## 6. 设计动机出处

- **MLA 原始设计**：DeepSeek-AI, *"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"*（arXiv:2405.04434, 2024）。提出 low-rank KV joint compression、decoupled RoPE、以及 decode 时的矩阵吸收。vLLM 文件头注释（`common.py:20-21`）明确标注本实现以此论文 + FlashInfer 实现为参考。
- **DeepSeek-V3**：DeepSeek-AI, *"DeepSeek-V3 Technical Report"*（arXiv:2412.19437, 2024）。给出本模块引用的实际维度（`Lkv=512, Lq=1536, P=128, R=64, V=128, N=128`）。
- **FlashMLA**：DeepSeek 开源的 MLA decode kernel（github.com/deepseek-ai/FlashMLA）。vLLM 的 `FlashMLABackend`（`flashmla.py`）封装其 `flash_mla_with_kvcache`，是 V1 默认 MLA decode 后端。
- **FlashInfer**：vLLM 注释引用其 MLA 实现思路（github.com/flashinfer-ai/flashinfer PR #551）作为 compute/data-movement 双路径划分的参考。
- **PagedAttention**：Kwon et al., SOSP 2023（arXiv:2309.06180）。MLA 复用其分页 KV 管理（见 [模块 02](../02-paged-attention-kvcache/design.md) §6），只替换 slot 内容。

> 以上为**设计动机**来源；所有 `file:line` 代码引用均以仓库实际代码为准（见 [`impl.md`](impl.md)）。

---

## 7. 一图速览：MLA 的两条路径

```
                 hidden_states [Sq, H]
                        │
          ┌─────────────┴──────────────┐
   q_a_proj/q_b_proj            kv_a_proj_with_mqa
   (Q 低秩→上投影)                     │
          │                  ┌─────────┴──────────┐
        q [Sq,N,P+R]      kv_c [Sq,Lkv=512]   k_pe [Sq,R=64]
          │               (latent, layernorm)  (decoupled rope key)
   split nope/pe                  │                  │
          │                       └──── concat_and_cache_mla ───► KV cache
          │                            存 [block,block_size,576]  ← 只存 latent+rope，单 head
          │                                       │
   ┌──────┴───────────────────────┐              ▼
   │  reorder_batch: decode在前/prefill在后        从 cache 取 latent
   ▼                              ▼
┌─ DECODE (memory-bound) ─┐   ┌─ PREFILL (compute-bound) ───────┐
│ 吸收 W_UK → ql_nope     │   │ kv_b_proj 上投影还原完整 k_nope/v │
│ attention 跑在 latent   │   │ MHA(q, [k_nope|k_pe], v)         │
│ 维 (Lkv+R=576), MQA形式 │   │ 长context → chunked + merge_lse  │
│ 吸收 W_UV → o_proj      │   │ o_proj                            │
└──────────┬──────────────┘   └──────────┬───────────────────────┘
           └────────────┬─────────────────┘
                        ▼
                  output [Sq, H]

对比 MHA KV cache：[2, blk, block_size, N=128, head_dim=128]  ← 每token 32768 元素
MLA  KV cache    ：[blk, block_size, 576]                     ← 每token 576 元素 (≈57× 压缩)
```

> 实现层面的逐行调用链、`file:line` 对照与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 8. 设计背后的考量与历史教训

### 8.1 设计背后的考量（动机与历史）

1. **"两套路径"否决了"统一一条 attention"的省事方案**。最自然的实现是不管 prefill / decode 都把 latent 上投影回完整 KV、走一条标准 MHA。但那会让 decode 把压缩省下的显存带宽**全吐回去**——decode 时 `Skv` 是整段上下文，物化 `[Skv,N,P]` 等于临时重建了 MHA 规模的 K/V。vLLM 因此**按计算特性分流**：compute-bound 的 prefill 物化（还能白嫖现成 FlashAttention kernel），memory-bound 的 decode 吸收上投影、对 latent 跑 MQA。文件头注释（`common.py:9-27`）把这条"data-movement vs compute friendly"的划分立为整个 backend 的设计纲领。代价是两套代码路径、两套 metadata、两套 kernel，复杂度陡增。

2. **"吸收矩阵"是用代数恒等式换显存带宽，但只在 memory-bound 才成立**。`q_nope·(kv_c@W_UK) = (q_nope@W_UK^T)·kv_c` 这个结合律改写，把"为每个缓存 token 还原 k_nope"消掉了，代价是 QK 头维从 192 涨到 576、算力反增。这个取舍**只在 decode（瓶颈是读 KV 带宽、不是算力）才划算**——所以吸收路径被严格限定在 decode。理解 MLA 的关键，就是认清"省的是带宽、花的是算力，而 decode 恰好缺带宽不缺算力"。

3. **decoupled RoPE 是"压缩"与"位置编码"不可调和时被逼出来的妥协**。RoPE 的旋转依赖绝对位置、和低秩线性上投影不可交换，无法在解压后补加。于是 DeepSeek 把 QK 切成"压缩通道（nope，不旋转）+ 一小段旋转通道（pe，64 维）",让位置信息全部承载在那 64 维上、彻底避开压缩通道。这也解释了为什么 vLLM 要**把 RoPE 搬进 attention backend 内部**施加（`common.py:903/913`）——只有 backend 知道 latent 当前解压成哪些 head、positions 该如何对齐，所以 `input_positions` 被特意塞进 metadata（`common.py:270-272`）。

4. **`reorder_batch` 把 prefill/decode 分批，是"两套 kernel 各要规整张量"的直接后果**。decode kernel（MQA、query 长 1）和 prefill kernel（MHA、变长 query）对 batch 的张量布局要求完全不同，无法在一个交错的 batch 上同时跑。MLA 因此要求把 decode 请求挪到 batch 前段、prefill 挪到后段（`common.py:380-440`），用尽量少的 `swap_states` 完成（注释解释 decode 请求通常"久驻"在 batch 前部、新请求 append 在后部，所以交换很少）。这是 MLA 对统一调度抽象（[模块 01](../01-scheduler-batching/design.md)）施加的一个额外约束——`gpu_model_runner.py:474-478` 专门留了 `reorder_batch` 钩子给"想区分 compute/memory-bound 的 backend"。

5. **演进脉络：从"materialize"到"absorb"是一次关键性能重构**。最初 V1 MLA（`#13789`）的 decode 也会物化中间结果；`#14770`（*MLA get rid of materialization*）才把上投影彻底吸收掉、不再产出完整 K/V 中间张量，配合 `#14540`（*Improve MLA on V1*）把 CPU 开销也压下来。这条线说明 MLA 的工程价值不止"压 cache"，还在于"压 cache 之后如何不把省下的带宽在 decode 重新花掉"——吸收是这一目标的最后一块拼图。

6. **`merge_attn_states` 把 chunked prefill 和未来的 cascade 复用统一到 online-softmax 合并上**。长 context prefill 必须分块（否则物化 OOM），分块就要把多段 attention 结果按 log-sum-exp 正确合并。`#16173` 把这个合并落成 3× 加速的 CUDA kernel（见 [模块 11](../11-gpu-kernels-memory/impl.md)），同时也是 cascade attention（[模块 02](../02-paged-attention-kvcache/design.md) §3）的同一原语——一个 kernel 服务两处"分段算再合并"的需求。

### 8.2 重要 bug 修复（真实，PR 来自 `git log -- vllm/v1/attention/backends/mla/`）

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| `#13789` | V1 MLA 初始支持落地 | 把 MLA 接进 V1 统一调度 + paged KV 框架的奠基提交；确立 `MLACommonMetadata` 的 prefill/decode 分段结构 |
| `#13867` | V1 FlashMLA backend 落地 | 引入 `FlashMLABackend` + `flash_mla_with_kvcache`，需要 `get_mla_metadata` 预算 tile 调度（`flashmla.py:61-77`）；带出 block_size=64 的强制约束 |
| `#14253` | **MLA + V1 非法内存访问 + 精度错误** | 重写 metadata：把 `num_decodes/num_decode_tokens/num_prefills` 从可选改为**必填字段**、`reorder_batch` 返回 `modified_batch` 触发 `refresh_sampling_metadata`。教训：prefill/decode 分批一旦与采样元数据不同步，就是越界 + 静默精度错——分批是正确性的一部分，不只是性能优化 |
| `#14476` | **DeepSeek 精度错误** | 把 `rotary_emb` 改为直接取其 `forward_native`/`forward_cuda`（`common.py:620-622`），绕开 torch library 包装层在 attention custom op 内部的开销/语义问题；牵出"RoPE 必须在 backend 内正确施加"这条 decoupled RoPE 的落地细节 |
| `#13897` | MLA prefill **context 性能**差 | 修 chunked prefill 的 context 计算路径性能；坐实"长 context prefill 必须分块 + 固定 workspace"否则要么慢要么 OOM（`common.py:118-184` 注释） |
| `#14384` / `#14471` | *Reduce MLA CPU overheads* 被**回退** | `#14384` 想削 CPU 开销，但引入问题被 `#14471` revert，随后由 `#14540` 重做。教训：MLA 的 metadata 构建在 CPU 热路径上（每 step 都要算 chunk/cu_seq_lens），优化它容易碰正确性，需谨慎分步 |
| `#14540` | *Improve MLA on V1*（性能） | 重做被回退的 CPU 开销优化，把 metadata 构建中的 GPU↔CPU 同步压到最小（`common.py:454-456` 注释专门告诫避免 GPU→CPU sync） |
| `#14770` | *MLA get rid of materialization* | 把 decode 路径残余的中间张量物化消除，让吸收路径真正只在 latent 维运行；这是"省下的带宽不在 decode 重新花掉"的关键提交 |
| `#16173` | `merge_attn_states` CUDA kernel（3× 加速） | 把 chunked prefill / cascade 共用的 online-softmax 合并落成专用 kernel；同时修复其与 CUDA graph 捕获的兼容性（见 [模块 02](../02-paged-attention-kvcache/design.md) §8.2 `#16693`） |
