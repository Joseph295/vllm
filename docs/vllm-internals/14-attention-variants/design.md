# 模块 14 · 注意力变体：SWA / Cascade / GQA·MQA —— 设计文档

> 范围：vLLM 对三种"非标准 full attention"变体的支持。它们都不是新的 attention 算法，而是围绕 **KV cache 的存与算**做文章——本质都是 [模块 02](../02-paged-attention-kvcache/design.md) PagedAttention/Prefix Caching 抽象的**特化或复用**：
> - **SWA（Sliding Window Attention，滑动窗口注意力）**：每个 query 只 attend 最近 `W` 个 token，于是窗口外的 KV block **可以提前释放**，把 KV cache 占用从 O(L) 压到 O(W)。
> - **Cascade Attention（级联注意力）**：当一批请求**共享一段公共前缀**（如同一 system prompt），把"对公共前缀的 attention"**只算一次**，再与各请求独有后缀的 attention 用 log-sum-exp 合并，省 KV 显存带宽。
> - **GQA / MQA（Grouped-Query / Multi-Query Attention，分组查询/多查询注意力）**：让多个 query 头**共享同一组 KV 头**，KV 头数从 `num_attention_heads` 降到 `num_kv_heads`，直接把 KV cache 显存按 `num_queries_per_kv` 倍数缩小。
>
> 架构：**V1（`vllm/v1/`）为主**。三种变体在 V1 里分别落在 KV cache 管理（SWA）、attention backend（Cascade）、模型层 + cache 格式（GQA/MQA）三处。CUDA/Triton kernel 的线程级细节见 [模块 11](../11-gpu-kernels-memory/impl.md)。

---

## 1. 这个模块解决什么问题

标准的 full causal attention 有两条"成本曲线"随上下文增长而恶化：

1. **KV cache 显存随序列长度线性增长**：每个 token 的 K/V 都要永久缓存（见 [模块 02](../02-paged-attention-kvcache/design.md)），128K 上下文的单条请求 KV 可达数 GB。
2. **KV cache 显存随 KV 头数线性增长**：标准 MHA 每个 attention 头都有独立 K/V，`num_kv_heads == num_attention_heads`，几十个头乘以序列长度，是显存大头。
3. **公共前缀被重复计算 / 重复读取**：批内多条请求共享同一段长前缀时，每条都对整段前缀重新算一遍 attention，浪费 GPU 显存带宽（前缀的 KV 在 cache 里只存一份，但 attention 要把它读 N 遍）。

三种变体各打一个痛点，且**互相正交**，可叠加：

| 变体 | 攻击的成本 | 核心手段 | 显存/带宽收益 |
|---|---|---|---|
| **SWA** | KV 随**序列长度**增长 | 限制 attention 跨度为窗口 `W`，窗口外 KV block 提前释放 | KV 占用 O(L) → O(W) |
| **GQA/MQA** | KV 随**头数**增长 | 多 query 头共享 KV 头，`num_kv_heads ≪ num_attention_heads` | KV 占用 ÷ `num_queries_per_kv`（MQA 极端到 ÷ 头数） |
| **Cascade** | **公共前缀**被重复读 | 公共前缀 attention 算一次 + 各自后缀 attention，LSE 合并 | 前缀 KV 读 N 次 → 读 1 次 |

> **关键认知**：这三者没有引入新的 KV cache 结构。SWA 复用 paged block 的"提前 free"；GQA/MQA 只是把 cache 张量的 `num_kv_heads` 维度变小；Cascade 完全不碰 cache 布局，只在 kernel 调度上把同一份共享前缀 block table 喂两次。它们都是 [模块 02](../02-paged-attention-kvcache/design.md) 抽象的**自然延伸**。

---

## 2. 设计目标与约束

- **SWA 与 paged KV / prefix cache 必须协同**：滑窗外 block 的释放要走 [模块 02](../02-paged-attention-kvcache/design.md) 同一套 `free_blocks`，且释放后逻辑下标必须保持连续（用 `null_block` 占位），否则 block table、slot_mapping 全部错位。
- **SWA 仍要支持 prefix caching**：滑窗下"命中"不再是"最长公共前缀"，而是"窗口长度的连续命中段"——必须重新定义命中判据（`find_longest_cache_hit` 在 SWA 下从右往左扫）。
- **Cascade 必须 numerically 等价于普通 attention**：拆成"前缀 attention + 后缀 attention"再合并，结果必须**逐 bit 等价**于一次性算（靠 log-sum-exp 的可结合性，见 §3.2），不能有精度退化。
- **Cascade 只在"划得来"时启用**：用启发式 + 性能模型决定（公共前缀够长、请求够多、与 FlashDecoding 对比），未命中场景几乎零开销（早返回）。
- **GQA/MQA 对 cache 管理透明**：KV 头分组只改变"一个 token 的 KV 有多大"（`page_size_bytes`），不改变 block/block table/prefix hash 任何逻辑——KV manager 完全不知道有没有 GQA。
- **正确性约束**：
  - SWA：`num_computed_tokens` 仍是 `block_size` 整数倍（继承 [模块 02](../02-paged-attention-kvcache/design.md) 不变量）；窗口的"连续块数" `cdiv(W-1, block_size)`。
  - Cascade：公共前缀长度必须是 `block_size` 整数倍，且不超过批内**最小** `num_computed_tokens`（否则前缀 attention 的"无 mask 双向"会算错，见 §3.2）。
  - Cascade 与 ALiBi / sliding window **互斥**（当前 kernel 不支持，见 §5）。
  - GQA/MQA：`num_attention_heads % num_kv_heads == 0`（`flash_attn.py:392` 的 assert）。
- **V1 当前限制**：`KVCacheManager` 仅支持单一 KV cache group，即整模型 attention 类型一致（纯 full 或纯 SWA）。**交错**（interleaved）的 full+SWA 混合架构（如 Gemma2、Cohere2、Llama4）尚未在 KV manager 层高效共存，退化为全 full 处理（`kv_cache_interface.py:146-165` 注释；检测见 §6 `#14896`）。

---

## 3. 核心设计思想

### 3.1 SWA：限制窗口 → 窗口外 block 提前释放

**动机出处**：滑动窗口 attention 由 **Longformer**（Beltagy et al. 2020，arXiv:2004.05150）系统化用于长文档，**Mistral 7B**（Jiang et al. 2023，arXiv:2310.06825）把固定窗口 `W=4096` 用进主流 LLM——核心论点是"信息可以跨层逐跳传播，每层只看局部窗口即可覆盖很长的有效感受野"。

SWA 的 attention 规则：位置 `i` 的 query 只 attend `[i-W+1, i]` 这 `W` 个 key。在 flash-attention 里这就是一个参数 `window_size=(W-1, 0)`（`flash_attn.py:382-385`）——左看 `W-1`，右看 0（causal）。kernel 层面几乎免费。

真正的设计在 **KV cache 管理**：既然位置 `i` 永远不会再被 `> i+W` 的任何 query attend，那么**当生成推进到 `i+W` 之后，位置 `i` 所在的 block 就再无用处，可以释放**。这把 KV 占用从"全序列 O(L)"压到"窗口 O(W)"。

vLLM 在 V1 里把它实现成 `SlidingWindowManager`（`specialized_manager.py:90`），与 paged KV / prefix cache 协同的三个要点：

1. **窗口外 block 走同一套 `free_blocks` 提前释放**（`specialized_manager.py:132` `remove_skipped_blocks` → `kv_cache_manager.py:210-212`）：每次 `allocate_slots` 前，先算出"最后一个仍有用的块" `last_useful_block = (num_computed_tokens - W + 1) // block_size`，把它之前的块全部摘掉、加入空闲链表，让它们能立刻被别的请求复用。释放发生在分配新块**之前**，以减少驱逐。
2. **被释放的逻辑槽用 `null_block` 占位**（`specialized_manager.py:147`）：block table 的逻辑下标必须连续（worker 侧 slot_mapping、attention kernel 都按 `position // block_size` 索引），不能稀疏。于是被释放的位置填 `null_block`（恒为 block 0，`block_pool.py:54-57`），让 block table 始终是规整数组，且 attention 永不会读到这些 null 槽（因为它们在窗口外，kernel 的 `window_size` mask 直接屏蔽）。
3. **prefix cache 命中判据从右往左扫**（`specialized_manager.py:102-130`）：full attention 命中是"从头连续命中的最长前缀"；SWA 下，因为只需要窗口内的 KV，命中变成"找到 `cdiv(W-1, block_size)` 个**连续**命中块即可"——从右往左扫描 block_hashes，凑够连续窗口块就早停。窗口外的块即便 hash 命中也无意义（不会被 attend）。

> SWA 与 [模块 02](../02-paged-attention-kvcache/design.md) 的关系：SWA 就是 `SpecializedManager` 的另一个子类（与 `FullAttentionManager` 并列），只重写 `find_longest_cache_hit` 和 `remove_skipped_blocks` 两个方法，**复用** BlockPool、空闲链表、ref_cnt、`null_block` 等全部基础设施。

### 3.2 Cascade：公共前缀只算一次，再与后缀 LSE 合并

**动机出处**：这是 vLLM 自己的优化（`#11635`，*[V1] Implement Cascade Attention*），思路与 **FlashInfer 的 Cascade Inference**（多级 KV 共享）一脉相承，目标和 SGLang 的 **RadixAttention**（见 [模块 02](../02-paged-attention-kvcache/design.md) §5）一致——都是"避免对共享前缀重复算"。区别：RadixAttention 用基数树在**存**这一层共享 KV，Cascade 在 prefix caching 已经让前缀 KV **只存一份**的基础上，进一步让前缀 attention **只算一份**。

**问题**：prefix caching（[模块 02](../02-paged-attention-kvcache/design.md)）已经让 N 条共享前缀的请求把前缀 KV 存成一份（block table 都指向同一批物理块）。但**计算**上，每条请求的 decode 仍要把这段共享前缀的 KV **完整读一遍**算 attention——N 条请求就读 N 遍同一份 KV，浪费显存带宽（decode 是带宽瓶颈）。

**洞察**：把每条请求 `i` 的 attention 拆成两段：
- **公共前缀段**：所有请求的 query 对**同一段共享前缀 KV** 的 attention。这段对所有请求是"同一份 KV"，可以把所有请求的 query 拼在一起、对前缀 KV **只读一次**算出来。
- **后缀段**：每条请求 query 对自己**独有的后缀 KV** 的 attention。

两段算完后用 **log-sum-exp 合并**还原成完整 attention。数学基础：attention 输出是对 value 的加权和，权重是 softmax；把 key 集合切成两个不相交子集 P（前缀）和 S（后缀）分别算 partial attention `(O_P, lse_P)` 和 `(O_S, lse_S)`，则完整结果
```
O = (O_P · e^{lse_P} + O_S · e^{lse_S}) / (e^{lse_P} + e^{lse_S})
```
这正是 flash-attention 的 online-softmax 在"分块"维度上的可结合性——`merge_attn_states`（`flash_attn.py:706`）就在做这件事。**结果逐 bit 等价**于一次性算，没有精度损失。

vLLM 的实现（`cascade_attention`, `flash_attn.py:621`）：
- **前缀 kernel**：把所有请求的 query 拼成一个"超 batch"（`cu_prefix_query_lens=[0, num_tokens]`），用**任一请求的前缀 block table**（`block_table[:1]`，因为大家前缀块完全相同）作 KV，`causal=False`（前缀对所有 query 无 mask），输出 `(prefix_output, prefix_lse)`（`flash_attn.py:656-677`）。
- **后缀 kernel**：每条请求各自的 query 对 `block_table[:, num_common_kv_blocks:]`（跳过公共块后的后缀块）算 `causal=True`，输出 `(suffix_output, suffix_lse)`（`flash_attn.py:682-703`）。
- **合并**：`merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)`（`flash_attn.py:706`）按上式 LSE 合并。

**公共前缀的确定**链回调度层：
- 调度器每步算 `num_common_prefix_blocks`（`scheduler.py:392-397`），判据极简——**某 block 的 `ref_cnt == len(running)`**（所有 running 请求都指向它）即公共（`kv_cache_manager.py:331-377`，`get_num_common_prefix_blocks`）。这复用了 prefix caching 的 ref_cnt（命中时每个共享者 `touch` +1）。链回 [模块 01](../01-scheduler-batching/design.md) 的调度产物 `SchedulerOutput`。
- model runner 再把"块数"换算成"token 长度"并做一系列截断（`gpu_model_runner.py:617-699` `_compute_cascade_attn_prefix_len`）：①截到 `block_size` 整数倍；②**截到批内最小 `num_computed_tokens`**——否则前缀 kernel 的"无 mask 双向" attention 会让某请求 attend 到本属于它 query 的 token（注释 `gpu_model_runner.py:652-682` 用具体例子讲透了这个 off-by-one）；③再过 `use_cascade_attention` 启发式（`flash_attn.py:553`）拍板。

### 3.3 GQA / MQA：多 query 头共享 KV 头

**动机出处**：**MQA**（Multi-Query Attention）由 **Shazeer 2019**（arXiv:1911.02150，*Fast Transformer Decoding: One Write-Head is All You Need*）提出——所有 query 头共享**一组** K/V，把 decode 时的 KV 读带宽和显存降到 1/头数，代价是质量略降。**GQA**（Grouped-Query Attention）由 **Ainslie et al. 2023**（arXiv:2305.13245，*GQA: Training Generalized Multi-Query Transformer Models...*）提出折中——把 query 头分成 `G` 组，每组共享一组 K/V，`G = num_kv_heads` 介于 MHA（`G = num_heads`）和 MQA（`G = 1`）之间，质量几乎无损而省显存。Llama-2 70B、Mistral、Qwen 等主流模型全部用 GQA。

**vLLM 视角下，GQA/MQA 几乎是"免费"的**，因为它只改变一个量：**`num_kv_heads`**。

- **模型层**：每层 attention 用 `num_kv_heads`（可 < `num_attention_heads`）声明 K/V 投影大小（`llama.py:120-129`、`config.py:1021-1033` `get_num_kv_heads`）。`num_queries_per_kv = num_heads // num_kv_heads`（`flash_attn.py:393`、`layer.py:271`）就是"每个 KV 头被几个 query 头共享"。
- **KV cache 格式**：物理 KV cache 张量形状 `[2, num_blocks, block_size, num_kv_heads, head_size]`（`flash_attn.py:56-64`），`num_kv_heads` 维度直接小了 `num_queries_per_kv` 倍——这就是显存收益的全部来源。`page_size_bytes = coef * block_size * num_kv_heads * head_size * dtype_size`（`kv_cache_interface.py:64-69`），KV manager 拿到的"每页字节数"自然变小，**它根本不知道有没有 GQA**。
- **kernel 层**：FlashAttention / PagedAttention kernel **原生支持** GQA——query 张量 `[*, num_heads, head_size]`、KV 张量 `[*, num_kv_heads, head_size]`，kernel 内部让 `num_queries_per_kv` 个连续 query 头映射到同一个 KV 头去读 K/V。不需要把 KV 物理复制 `num_queries_per_kv` 份（那会抵消显存收益）。只有少数走 PyTorch SDPA 的 fallback 路径（`layer.py:302-305`）才用 `repeat_interleave` 显式扩展 KV——主流 paged 路径不复制。

> GQA/MQA 与 [模块 02](../02-paged-attention-kvcache/design.md) 的关系：**零侵入**。block、block table、prefix hash、ref_cnt 全不变；唯一变化是 cache 张量第 4 维（`num_kv_heads`）变小、kernel 多一层 `head // num_queries_per_kv` 的索引。这也是为什么 GQA 能和 SWA、Cascade、prefix caching 任意叠加。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `SlidingWindowSpec` | `kv_cache_interface.py:84` | SWA 层的 KV cache 格式：在 `AttentionSpec` 基础上加 `sliding_window`；`type_id` 含窗口值，`max_memory_usage_bytes` 只按 `W-1 + batched_tokens` 估（不按 max_model_len） |
| `SlidingWindowManager` | `specialized_manager.py:90` | SWA 的 `SpecializedManager` 子类：重写 `find_longest_cache_hit`（从右往左凑连续窗口块）+ `remove_skipped_blocks`（释放窗口外块，null_block 占位） |
| `sliding_window_contiguous_blocks` | `specialized_manager.py:98` | `cdiv(W-1, block_size)`，prefix cache 命中所需的"连续命中块数" |
| `FullAttentionSpec` / `FullAttentionManager` | `kv_cache_interface.py:72` / `specialized_manager.py:69` | 对照：标准 full attention，`find_longest_cache_hit` 从左连续命中、`remove_skipped_blocks` 永远返回空 |
| `FlashAttentionMetadata` | `flash_attn.py:71` | attention 一步元数据；Cascade 相关字段：`use_cascade`、`common_prefix_len`、`cu_prefix_query_lens`、`prefix_kv_lens`、`suffix_kv_lens`（`:89-94`） |
| `num_common_prefix_blocks` | `scheduler.py:392` / `output.py` SchedulerOutput | 调度器算出的"所有 running 请求共享的前缀块数"，过界到 model runner |
| `merge_attn_states` | `attention/ops/merge_attn_states.py:9` | LSE 合并前缀/后缀 partial attention，CUDA kernel（`csrc/attention/merge_attn_states.cu`）+ Triton fallback |
| `num_kv_heads` | `config.py:1021` `get_num_kv_heads` | GQA/MQA 的核心量：每 GPU 的 KV 头数（按 TP 切分，min 为 1） |
| `num_queries_per_kv` | `flash_attn.py:393`、`layer.py:271` | `num_heads // num_kv_heads`，每个 KV 头被几个 query 头共享 |
| `page_size_bytes` | `kv_cache_interface.py:64` | `coef * block_size * num_kv_heads * head_size * dtype_size`；GQA 通过 `num_kv_heads` 缩小它 |
| `window_size`（runner） | `gpu_model_runner.py`（`#14896` 引入） | `sliding_window or interleaved_sliding_window`，用于检测交错 SWA、关 cascade |

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| SWA 窗口外 block 提前释放 | KV 占用 O(L)→O(W)，长上下文显存大降 | 窗口外信息彻底丢失（模型靠跨层传播弥补）；释放逻辑需 null_block 占位、命中判据要改 |
| SWA 命中从右往左扫 | 正确反映"只需窗口内 KV"的命中语义 | 当前是 O(len(block_hashes)) 朴素扫描，低命中率场景未优化（`specialized_manager.py:104-108` TODO） |
| Cascade 公共前缀算一次 | 前缀 KV 读 N→1 次，省 decode 带宽 | 多两个 kernel + 一次 merge；前缀长度受最小 computed_tokens 截断；与 ALiBi/SWA 互斥 |
| Cascade 用启发式 + 性能模型 | 只在划得来时启用，未命中早返回零开销 | 阈值（256 token、8 请求）是经验值（`flash_attn.py:569,578` 标注 TODO 调优）；性能模型"很粗糙"（`:599` 注释自承） |
| Cascade 默认开、可一键关 | 大多数共享前缀场景自动受益 | 边界 bug 时需逃生阀 → 加 `disable_cascade_attn` 开关（`#15243`） |
| GQA/MQA 只缩 `num_kv_heads` | KV 显存 ÷ `num_queries_per_kv`，对 cache 管理零侵入 | 质量略降（MQA 明显、GQA 几乎无感）；属模型架构选择，vLLM 仅承接 |
| GQA paged kernel 不复制 KV | 不抵消显存收益 | kernel 需支持 query/KV 头数不等的映射（FA/PA 原生支持；少数 fallback 用 repeat_interleave） |
| 单 KV cache group（暂不支持真交错混合） | KV manager 简单，一张 block table 走天下 | Gemma2/Cohere2/Llama4 等 full+SWA 交错架构退化为全 full（`kv_cache_interface.py:146-165`） |

**Cascade vs RadixAttention / FlashInfer Cascade**：三者目标一致（共享前缀不重复算）。RadixAttention（SGLang）用基数树在"存"层做任意粒度共享 + 树上 LRU；vLLM 的 prefix caching 已在 block 粒度做"存一份"，Cascade 在此之上加"算一次"，实现是两个 flash-attn kernel + LSE 合并，与现有 paged 路径无缝；FlashInfer 的 Cascade Inference 把这个思路推广到多级 KV。vLLM 选了"最小改动复用 flash-attn"的路线，代价是只在两级（前缀/后缀）、启发式触发。

---

## 6. 设计动机出处

- **SWA / Longformer**：Beltagy, Peters, Cohan, *"Longformer: The Long-Document Transformer"*（arXiv:2004.05150, 2020）。提出滑动窗口 + 全局注意力混合，论证局部窗口经多层堆叠可覆盖长程依赖。
- **SWA / Mistral**：Jiang et al., *"Mistral 7B"*（arXiv:2310.06825, 2023）。把固定 `W=4096` 滑窗用进主流 7B LLM，并明确指出"滚动缓冲 KV cache"——窗口外 KV 可丢弃，正是 vLLM SWA block 提前释放的动机。
- **Cascade Attention**：vLLM `#11635`（*[V1] Implement Cascade Attention*，Woosuk Kwon）。LSE 合并多段 partial attention 的思路与 **FlashInfer Cascade Inference**（flashinfer.ai 博客）、flash-attention 的 online-softmax 可结合性同源；目标对标 SGLang **RadixAttention**（Zheng et al., arXiv:2312.07104，见 [模块 02](../02-paged-attention-kvcache/design.md) §5）。
- **MQA**：Shazeer, *"Fast Transformer Decoding: One Write-Head is All You Need"*（arXiv:1911.02150, 2019）。所有 query 头共享单组 K/V，decode 带宽/显存降到 1/头数。
- **GQA**：Ainslie et al., *"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"*（arXiv:2305.13245, 2023）。query 头分组共享 K/V，质量与 MHA 几乎持平而省显存，成为 Llama-2 70B 等的默认。

> 以上为**设计动机**来源；所有 `file:line` 代码引用以仓库实际代码为准（见 [`impl.md`](impl.md)）。

---

## 7. 一图速览：三种变体如何改写 full attention

```
                        full attention（基线）
  KV: [k0 k1 k2 ... kL]（每个 token 全缓存，num_kv_heads = num_heads）
  query_i attend 全部 [k0..ki]，每条请求各读各的（哪怕前缀相同）

──────────────────────────────────────────────────────────────────────
SWA：窗口 W，窗口外 block 提前释放（KV O(L)→O(W)）
  position:  0  1  2  3  4  5  6  7  8  9  (W=4)
  block_table(逻辑): [null][null][ B5 ][ B6 ]   ← 前两块已 free，null_block 占位
                              └─ 窗口内才保留 ─┘
  query_9 只 attend [k6 k7 k8 k9]，kernel window_size=(W-1,0) 屏蔽窗外
  释放: remove_skipped_blocks → free_blocks（复用模块02空闲链表）

──────────────────────────────────────────────────────────────────────
Cascade：批内共享前缀 P，独有后缀 S（前缀算一次）
  reqA: [== P ==][ Sa ]      ref_cnt(P块)=len(running) → 判定为公共
  reqB: [== P ==][ Sb ]
  reqC: [== P ==][ Sc ]
        └block_table[:1]┘ └ block_table[:, num_common:] ┘
   ① 前缀 kernel: 所有 query 拼一起 × P 的KV(读一次,causal=False) → (O_P, lse_P)
   ② 后缀 kernel: 各 query × 各自 S(causal=True)            → (O_S, lse_S)
   ③ merge_attn_states: O = (O_P·e^lse_P + O_S·e^lse_S)/(e^lse_P + e^lse_S)
                                                 ↑ LSE 合并，逐bit等价

──────────────────────────────────────────────────────────────────────
GQA/MQA：多 query 头共享 KV 头（KV ÷ num_queries_per_kv）
  num_heads=8, num_kv_heads=2  → num_queries_per_kv=4
  query 头:  q0 q1 q2 q3 | q4 q5 q6 q7
                \_______/      \_______/
  KV  头:       kv0            kv1     ← cache 第4维只存2个头
  cache: [2, num_blocks, block_size, num_kv_heads=2, head_size]
  kernel: head_kv = head_q // num_queries_per_kv（不物理复制 KV）

  三者正交，可叠加：GQA + SWA + prefix-cache + Cascade 同时生效
```

> 实现层面的逐行调用链、`file:line` 对照、kernel 内边界处理，见同目录 [`impl.md`](impl.md)。

---

## 8. 设计背后的考量与历史教训

### 8.1 设计背后的考量（动机与历史）

1. **三种变体被刻意做成"对 [模块 02](../02-paged-attention-kvcache/design.md) 的特化/复用"而非另起炉灶**。SWA 没有发明新的 cache 结构，而是当 `SpecializedManager` 的一个子类，只重写命中查找与块释放两个方法，BlockPool / 空闲链表 / ref_cnt / null_block 全部复用。GQA/MQA 干脆连 manager 都不碰，只把 cache 张量的 `num_kv_heads` 维缩小。Cascade 连 cache 布局都不动，纯在 kernel 调度上把同一份 block table 喂两遍。这是"用同一套分页抽象覆盖所有变体"的工程选择——代价是真正的混合架构（交错 full+SWA）暂时受限于"单 KV cache group"而退化（见 §8.2 `#14896`）。

2. **SWA 命中判据"从右往左扫"是被"窗口语义"逼出来的，且明知不优仍先求正确**。full attention 命中是"从头最长连续前缀"；SWA 只需窗口内 KV，所以命中变成"凑够 `cdiv(W-1,block_size)` 个连续命中块"。当前实现是从右往左 O(len) 朴素扫（`specialized_manager.py:102-130`），代码里直接挂了 TODO（`:104-108`）说低命中率场景可优化到亚线性。这是"先正确、再优化"的典型——SWA + prefix caching 能共存本身就是 `#13069` 才解锁的。

3. **Cascade 的公共前缀长度被"双重截断"，每一刀都对应一个正确性陷阱**。①截到 `block_size` 整数倍——因为后缀 block table 要从块边界切（`block_table[:, num_common_kv_blocks:]`）。②截到批内**最小** `num_computed_tokens`——因为前缀 kernel 用 `causal=False` 的无 mask 双向 attention，若公共前缀越过某请求的 computed 边界，该请求的 query 会 attend 到本该被 causal mask 屏蔽的 token。`gpu_model_runner.py:652-682` 用 reqA/reqB/reqC 三个具体例子推导出"必须 capped by min computed_tokens"，并解释为何实现里连那个 +1 都不加（为保证恒用两个 kernel、后缀不空）。这段注释是理解 Cascade 正确性的钥匙。

4. **Cascade 是"默认开 + 启发式 + 一键关"三件套，因为它是性能优化而非正确性需求**。启用要过三关：公共前缀 ≥256 token、请求数 ≥8、再走性能模型对比 FlashDecoding（`flash_attn.py:553-618`）。阈值都标了 TODO 待调，性能模型自承"很粗糙"。又因为它下放到 kernel、边界多，专门加了 `disable_cascade_attn` 开关（`#15243`）当逃生阀——优化类特性必须能一键退回安全路径。

5. **GQA/MQA 的"零侵入"是 KV manager 抽象设计对的回报**。因为 KV manager 只认 `page_size_bytes`，不认头数，GQA 把 `num_kv_heads` 缩小后整条分配/命中/驱逐链路无感。这让 GQA 能和 SWA、Cascade、prefix caching 任意叠加——正交性不是巧合，是"manager 不关心 cache 内部布局、只关心页大小"这一抽象边界的直接产物。代价：真正需要"按层不同头数/不同 attention 类型"的混合架构，反而因为这层抽象太平而暂不支持（单 group 限制）。

6. **`merge_attn_states` 从 Triton 起步、后补 CUDA kernel，体现"先能用再加速"**。Cascade 落地时（`#11635`）合并用 Triton；后续 `#16173` 加了 3x 加速的 CUDA kernel，并保留 Triton 作 FP8/非对齐 headdim 的 fallback（`merge_attn_states.py:17-41`）。这与下放到 kernel 的优化都要守 CUDA graph 兼容（`#16693`）是同一类教训：KV 复用类优化的最后一公里在 kernel，且要兼顾多 dtype/多硬件/graph capture。

### 8.2 重要 bug 修复与演进（真实，PR 来自 `git log -- vllm/v1/core/specialized_manager.py vllm/v1/attention/backends/` 及关联提交）

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| `#9679` | *[V1] Support sliding window attention* | V1 SWA 的起点：在 flash_attn backend 支持 `window_size` 参数。SWA 先从 kernel 参数做起，KV 管理层的提前释放是后来才补的（`#14097`） |
| `#14097` | *[V1] Implement sliding window attention in kv_cache_manager* | 把 SWA 从"kernel 参数"提升为"KV 管理一等公民"：`SlidingWindowManager` 落地，窗口外 block 经 `remove_skipped_blocks` 提前释放 + null_block 占位。这是本模块 SWA 与 paged KV 协同的核心提交（`specialized_manager.py` 唯一一条 git log） |
| `#13069` | *[V1] Allow sliding window + prefix caching* | SWA 与 prefix caching 此前不能共存；本 PR 让 SWA 下的命中判据（连续窗口块）与 prefix cache 兼容。教训：变体叠加不是自动的，命中语义要为 SWA 重新定义 |
| `#11635` | *[V1] Implement Cascade Attention* | Cascade 的起点：`cascade_attention` + `use_cascade_attention` 启发式 + LSE 合并。确立"前缀 attention 算一次 + 后缀 + merge"范式 |
| `#14896` | *[V1][BugFix] Detect interleaved sliding window attention* | 交错 SWA 模型（Gemma2/Cohere2/Llama4）的 `sliding_window` 为 None、真正窗口在 `interleaved_sliding_window`；漏检会让 cascade 误启用、SWA 失效。修复引入 `window_size = sliding_window or interleaved_sliding_window`，并据此关 cascade。教训：模型架构的"非标准命名"会静默绕过变体检测 |
| `#15211` | *Fix use_cascade_attention handling for Alibi-based models* | ALiBi 模型误启用 cascade（cascade kernel 不支持 ALiBi，`flash_attn.py:643` assert）。修复在 `use_cascade_attention` 早返回 False。教训：cascade 的"支持性检查"与"收益启发式"必须前者优先，否则触发 assert/错误结果 |
| `#15243` | *[V1] Add flag to disable cascade attention* | 加 `disable_cascade_attn` 配置开关。优化类特性需要一键逃生阀，应对未覆盖的边界 |
| `#16209` | *Fix Llama4 - Index Error When Single Request Near Max Context* | 本质是 local/chunked attention（iRoPE，Llama4）构造 virtual batch 的 block_indices 在接近 max context 时越界。修复 `block_indices.flatten().clip(max=block_table.shape[1]-1)`（`flash_attn.py:267`）。教训：窗口/分块类变体在"恰好顶到边界"时的 off-by-one 极易漏，需边界测试守 |
| `#16693` | *fix potential cuda graph broken for merge_attn_states kernel* | cascade 合并 kernel 可能破坏 CUDA graph 捕获（与 [模块 02](../02-paged-attention-kvcache/design.md) §8.2 同一条目）。教训：KV 复用优化下放 kernel 后要额外守 CUDA graph 约束，见 [模块 07](../07-cuda-graph-compile/design.md) |
| `#16173` | *support merge_attn_states CUDA kernel, 3x speedup* | 把 cascade 的 LSE 合并从 Triton 升级为 CUDA kernel，保留 Triton 作 FP8/非对齐 fallback。演进而非 bug，体现"先能用再加速" |

> 说明：`vllm/v1/core/specialized_manager.py` 的 `git log` 仅有 `#14097` 一条（该文件较新）；SWA/Cascade 的其余演进散落在 `vllm/v1/attention/backends/` 与 `gpu_model_runner.py`/`scheduler.py`，上表已据 `git log` 与 `git show` 逐条核实，未作编造。GQA/MQA 在 vLLM 侧无专属 bug 流（属模型架构承接，无独立修复历史）。

```
设计演进时间线（据 git log，时间自上而下）
  #9679   V1 SWA：kernel 支持 window_size
    │
  #11635  Cascade Attention 落地（前缀算一次 + LSE 合并）
    │
  #13069  SWA + prefix caching 共存
    │
  #14097  SlidingWindowManager：窗口外 block 提前释放（KV 管理一等公民）
    │
  #14896  检测交错 SWA（interleaved_sliding_window）→ 正确关 cascade
    │
  #15211  ALiBi 模型不误用 cascade   #15243  disable_cascade_attn 开关
    │
  #16173  merge_attn_states CUDA kernel 3x   #16209 Llama4 边界越界
    │
  #16693  守住 cascade merge 的 CUDA graph 兼容
```
