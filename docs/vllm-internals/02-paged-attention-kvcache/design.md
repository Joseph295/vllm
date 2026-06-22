# 模块 02 · PagedAttention + KV Cache 管理 + Prefix Caching —— 设计文档

> 范围：vLLM 如何**管理 KV cache 显存**。包含三层紧密耦合的机制：(1) **PagedAttention** —— 把每条序列的 KV cache 切成固定大小的 **block（页）**，用 **block table** 把逻辑块映射到物理块，从而消灭显存碎片；(2) **KV Cache 管理** —— block 的分配、引用计数、驱逐、抢占；(3) **Prefix Caching** —— 用 token 内容 hash 让不同请求**复用**已算好的前缀 block。
> 架构：**V1（`vllm/v1/core/`）为主**，关键处对比 V0（`vllm/core/`）。CUDA kernel 部分 V0/V1 共用（`csrc/`）。

---

## 1. 背景与定位

> 如果你还不清楚 **KV cache 为什么存在、为什么是显存瓶颈**，请先读 [PRIMER §1.3](../PRIMER.md#13-kv-cache它为什么存在为什么是瓶颈)，这里默认你已经有了那套心智模型。

### 1.1 这个模块在系统里的位置

本模块是 KV cache 的**显存账本与底层 kernel**——别的模块"决定要算什么"，本模块负责"这些 token 的 KV 放哪、怎么读、怎么省"：

- **上游**：调度器（[模块 01](../01-scheduler-batching/design.md)）每要调度一个请求，都先问本模块"**放不放得下**"、"前缀有没有命中"，拿到 block 分配结果。
- **本模块**：把 KV cache 显存切成定长 **block（页）**，用 **block table** 把逻辑块映射到物理块（消灭碎片）；管 block 的分配/引用计数/驱逐/抢占；用 hash 让不同请求**复用**相同前缀的 block（prefix caching）。
- **下游**：把 `block_table` / `slot_mapping` 喂给 attention kernel（[模块 11](../11-gpu-kernels-memory/design.md)），kernel 据此跨非连续物理块读写 KV、完成 attention。

一句话：**它是 vLLM 高吞吐的物理基础——在有限显存里尽量多塞并发请求、并让相同前缀只算一次。**

### 1.2 关键文件与类各干什么（先认人）

| 文件 / 类 | 主要作用（一句话） |
|---|---|
| `v1/core/kv_cache_manager.py` 的 `KVCacheManager` | 对调度器的**门面**：`allocate_slots`（分配）/ `get_computed_blocks`（查前缀命中）/ `free`（释放） |
| `v1/core/block_pool.py` 的 `BlockPool` | 物理块**总池**：持有所有 block、空闲链表、命中字典、负责缓存满块与驱逐 |
| `v1/core/kv_cache_utils.py` 的 `KVCacheBlock` / `FreeKVCacheBlockQueue` / `BlockHashType` | 一个物理块的元数据、空闲块双向链表、prefix cache 的 hash 键 |
| `v1/core/specialized_manager.py` 的 `FullAttentionManager` 等 | 按 attention 类型分派的**命中查找 & 驱逐**逻辑（full / sliding window） |
| `v1/worker/block_table.py` 的 `BlockTable` | GPU 上的 block table 张量 `[max_num_reqs, max_num_blocks_per_req]`，喂给 kernel |
| `v1/attention/backends/flash_attn.py` | 构建 `FlashAttentionMetadata`（含 `block_table`/`slot_mapping`），调 attention kernel |
| `csrc/attention/`、`csrc/cache_kernels.cu` | CUDA kernel（V0/V1 共用）：`reshape_and_cache`（写 KV）、paged attention（读 KV） |

### 1.3 和相似模块 / 概念的区别与联系（专治"听过但分不清"）

- **PagedAttention vs 朴素 attention**：唯一本质区别是**多一次 `block_table[逻辑块] → 物理块` 的间接寻址**（见 §3.1）。朴素 attention 假设一条序列的 KV 连续存放；PagedAttention 让 KV 切成定长块、物理上可乱放、可共享，代价是 kernel 每次读 KV 多查一次表。
- **prefix caching vs chunked prefill**：二者都让 prefill 变便宜，但**一个蹭现成、一个切自己**。prefix caching（本模块）=「复用别人/自己已算过的相同前缀 KV」；chunked prefill（[模块 01](../01-scheduler-batching/design.md)）=「把自己的长 prompt 拆成几步算」。详见 [PRIMER §2.4](../PRIMER.md#24-易混概念对照区别与联系专治听过但分不清)。
- **抢占（preempt）vs 交换（swap）**：显存不足时 V1 只做"**重计算式抢占**"——`free` 掉低优先级请求的 block、打回 `waiting` 从头重算（§3.5）；V0 还支持 **swap-to-CPU**（把 KV 换到内存再换回），V1 配合默认 prefix caching 后重算更划算，故砍掉 swap。
- **逻辑块 vs 物理块**：**逻辑块**是"这条序列的第 i 段 token"（按序列连续编号）；**物理块**是显存里真实的那块 page（`block_id`，可乱序、可被多条序列共享）。`block_table` 就是这两者之间的页表。

🔗 **回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)**：例子里 `[11,7,4]` 共 3 个 token、`block_size=2`，需要 2 个逻辑块，分到物理块 `block_table=[7,3]`；每个 token 的全局位置 `slot_mapping=[14,15,6]`（slot = 物理块号×2 + 块内偏移）。本模块要解决的就是"怎么把这 2 个逻辑块映射到物理块 7、3，且若别的请求前缀也是 `[11,7]` 就让它直接共享物理块 7"。

---

### 1.4 这个模块要解决的问题

LLM 自回归推理时，每生成一个 token 都要读取它之前所有 token 的 Key/Value 向量（attention）。把这些 K/V 缓存下来（KV cache）避免重算，是推理性能的命门。但 KV cache 的**显存管理**有一组很难调和的矛盾：

1. **序列长度事先不知道**：一条请求会生成多少 token（直到遇到 EOS）无法预知。传统做法（如 早期 HF/FasterTransformer）给每条序列**预留一段连续显存**，按 `max_seq_len` 申请。
2. **连续分配 → 严重碎片化**：
   - **内部碎片（internal fragmentation）**：按 `max_seq_len` 预留，但请求实际只生成几十个 token，预留的剩余空间全部浪费。
   - **外部碎片（external fragmentation）**：不同请求预留的连续段长短不一，请求结束后留下大小不一的"空洞"，新请求要连续段却拼不出来。
   - PagedAttention 论文实测：朴素连续分配下，**实际有效利用的 KV 显存只有 20.4%~38.2%**，其余全是碎片浪费（来源见 §6）。
3. **前缀无法共享**：system prompt、few-shot 示例、同一文档的多轮问答——大量请求共享相同前缀，连续分配方案下每条请求各存一份，重复计算 + 重复占显存。
4. **kernel 要能跨非连续块算 attention**：一旦 KV 不再连续存放，attention kernel 必须能"按一张映射表跳着读"才能正确工作。

vLLM 的核心洞察：**把操作系统的虚拟内存分页（paging）搬到 GPU 上管理 KV cache**。显存被切成固定大小的 **block**（页），序列的逻辑连续 KV 被映射到一组**物理上可以不连续**的 block，映射关系记在 **block table**（页表）里。于是：

- 显存按 block 粒度分配 → **外部碎片彻底消失**（所有 block 等大，任意空闲 block 都能用）。
- 只为已经产生的 token 分配 block，按需增长 → **内部碎片降到至多一个 block**（最后一个未满的 block）。
- 相同内容的 block 可以被多条序列的 block table 同时指向 → **前缀天然可共享**（prefix caching / copy-on-write 的物理基础）。

> 这就是论文标题 *"Efficient Memory Management for LLM Serving with **PagedAttention**"* 的全部含义：用分页做高效显存管理。

---

## 2. 设计目标与约束

- **零外部碎片、近零内部碎片**：block 等大，浪费上界 = `block_size - 1` 个 token 的 KV（每条序列最后一个不满的 block）。
- **按需增长**：序列每多算 `block_size` 个 token 才申请一个新 block，绝不预留 `max_seq_len`。
- **物理块可任意复用**：block table 是一层间接（indirection），逻辑块 → 物理块的映射可随意，使得"抢占后重新分配""前缀共享""copy-on-write"都只是改 block table，不搬数据。
- **attention kernel 必须接受 block table**：kernel 入参里带 `block_tables` 和 `slot_mapping`，运行时把逻辑位置翻译成物理地址（这是 PagedAttention 与普通 flash-attention 的唯一本质区别）。
- **前缀复用对用户透明**：开 `enable_prefix_caching` 后自动命中，无需用户标注哪段是公共前缀（区别于需要显式 prefix 的早期方案）。这就是 **Automatic Prefix Caching (APC)**。
- **正确性约束**：
  - 只有**写满**（full）的 block 才能被 cache/共享——半满 block 内容还会变，不能复用。
  - 被多条序列共享的 block **不可原地写**，必须 **copy-on-write**。
  - `num_computed_tokens` 始终是 `block_size` 的整数倍（V1 KV manager 的核心不变量，简化记账）。
- **V1 额外约束**：当前 `KVCacheManager` 仅支持**单一 KV cache group**（`kv_cache_manager.py:31` 的 assert）——即整模型 KV cache 类型一致（纯 full attention，或纯 sliding window）。混合架构（部分层 full + 部分层 sliding window）尚在演进（`kv_cache_interface.py:146-165` 注释）。

---

## 3. 核心设计思想

### 3.1 PagedAttention：block、block table、slot

三个概念是理解一切的基础：

- **block（物理块 / 页）**：KV cache 显存被切成 `num_gpu_blocks` 个等大 block，每个 block 容纳 `block_size` 个 token 的 K 和 V（典型 `block_size=16`）。物理 block 由整数 `block_id`（0 .. num_gpu_blocks-1）标识。物理 KV cache 张量形状（FlashAttention backend，`flash_attn.py:56-64`）：

  ```
  kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            └K/V┘ └─物理块维─┘ └─块内偏移─┘
  ```

- **block table（页表）**：每条请求一行，`block_table[req][logical_block_idx] = physical_block_id`。逻辑块 i 装的是该序列第 `i*block_size .. (i+1)*block_size` 个 token。逻辑块在序列里**连续编号**，但映射到的物理块**可以乱序、不连续、甚至与别的序列共享**。

- **slot（槽位）**：一个 token 在 KV cache 里的**全局扁平位置** = `physical_block_id * block_size + block_offset`。写入 KV 时用 `slot_mapping[token] = slot`；读取时 kernel 用 block table 反查。

**逻辑→物理映射的两条路径**（贯穿全模块）：

```
写 KV（reshape_and_cache）：           读 KV（attention kernel）：
  token 在序列的 position                kernel 要算 query 对第 j 个 KV 的注意力
    → logical_block = position//block_size   → logical_block = j // block_size
    → 查 block_table 得 physical_block       → physical_block = block_table[logical_block]
    → slot = physical_block*block_size       → 地址 = physical_block*block_stride
              + position%block_size                    + (j%block_size)*...
    → 写入 cache[slot]                       → 读 cache[地址]
```

这就是 PagedAttention kernel 与朴素 attention 的唯一区别：**多一次 `block_table[logical] → physical` 的间接寻址**。物理实现见 [`impl.md`](impl.md) §2；attention kernel 如何用 block table 取 KV、reshape_and_cache 写入的**线程/warp 级 CUDA 实现**见 [模块 11](../11-gpu-kernels-memory/impl.md)。

### 3.2 V0 → V1 的 KV 管理对比（为什么 V1 重写了这一层）

V0 和 V1 都实现了 PagedAttention（CUDA kernel 共用），但**显存管理的软件架构**差别巨大：

| 维度 | V0（`vllm/core/`） | V1（`vllm/v1/core/`） |
|---|---|---|
| 入口管理器 | `BlockSpaceManager`（`block_manager.py`） | `KVCacheManager`（`kv_cache_manager.py`） |
| block 抽象 | `Block` 对象层级（immutable/mutable block，`block/` 目录一整套类） | 扁平的 `KVCacheBlock` dataclass（`kv_cache_utils.py:112`），只是元数据 |
| 空闲块管理 | `NaiveBlockAllocator` 的空闲集合 + **独立 `LRUEvictor`**（`evictor.py`，最小堆按 last_accessed） | **单一双向链表 `FreeKVCacheBlockQueue`**（`kv_cache_utils.py:161`），空闲 + 驱逐候选合一，O(1) 中间删除 |
| 驱逐策略 | LRUEvictor 用堆，`evict()` 弹最久未用 | 链表头 = LRU 驱逐目标，**不需要单独 evictor**；free 时反序入队让尾部块先被驱逐（`kv_cache_manager.py:308-314`） |
| prefix cache hash | `PrefixCachingBlock.content_hash`，Python `hash((is_first, prev_hash, *tokens, extra))`（`prefix_caching_block.py:927`） | `BlockHashType` 命名元组 + hash 链（`kv_cache_utils.py:21,392`），默认可选 **SHA256** 抗碰撞 |
| 命中查找 | 经 `PrefixCachingBlockAllocator._cached_blocks` dict（hash→block_id） | `BlockPool.cached_block_hash_to_block`（hash→{block_id:block}），`find_longest_cache_hit` 沿 hash 链走（`specialized_manager.py:71`） |
| copy-on-write | 显式 `CopyOnWriteTracker`（`block/common.py`）+ `RefCounter`，`append_slots` 返回 CoW pair，再调 `copy_blocks` kernel | **V1 不在 prefill/decode 主路径做 CoW**：不去重 block（`block_pool.py:46-50` 注释），block table append-only；共享只发生在 prefix cache 命中（只读前缀） |
| 调度耦合 | block 管理与 V0 scheduler 紧耦合，prefill/decode 分离 | KV manager 纯函数式接口（`allocate_slots`/`get_computed_blocks`/`free`），被统一调度抽象调用（见 [模块 01](../01-scheduler-batching/design.md)） |

**V1 重写的核心动机**：V0 的 `block/` 目录是一套层层包装的对象（block allocator → block → block table → evictor），分配一个 block 要穿过多层 Python 调用，且 prefix caching、CoW、swap 各有独立子系统，难维护、有开销。V1 把它**压扁**成"一个 `BlockPool` + 一条双向链表 + 一张 hash 字典"，并把驱逐折叠进链表本身。代价是放弃了 V0 的一些灵活性（如运行中 block 去重、CPU swap），换来更简单、更快的热路径。

> V1 设计动机出处：vLLM V1 博客 *"vLLM V1: A Major Upgrade to vLLM's Core Architecture"*（blog.vllm.ai, 2025-01）把 "**zero-overhead prefix caching**" 列为 V1 目标之一——prefix caching 默认开启且对未命中场景几乎零开销，正是靠这套扁平结构 + hash 链实现。

### 3.3 Prefix Caching：用 hash 链把"前缀复用"变成"查字典"

朴素想法"两条请求前缀相同就共享"难在：怎么**快速判断**两段 token 前缀相同？vLLM 的答案是 **block 级 hash 链（hash chain）**：

```
block 0 hash = H(NONE_HASH, tokens[0:16])
block 1 hash = H(block0_hash, tokens[16:32])      ← 依赖前一块的 hash
block 2 hash = H(block1_hash, tokens[32:48])
...
```

每个 block 的 hash **包含了它前面所有 block 的 hash**（通过 `parent_block_hash`，`kv_cache_utils.py:392-420`）。于是：

- 两条请求的第 i 个 block hash 相同 **当且仅当**它们前 i 个 block 的 token 完全一致（前缀相同）。这正是 token 前缀匹配的等价条件。
- 命中查找退化成：沿 hash 链查字典 `cached_block_hash_to_block`，**第一个查不到的 block 就是前缀分叉点**（`specialized_manager.py:71-82`，`FullAttentionManager.find_longest_cache_hit`）。命中前缀的物理块直接被新请求的 block table 指向，对应 token 不再计算（计入 `num_computed_tokens`）。
- 多模态/LoRA 请求把 `mm_hash`、`lora_id` 作为 **extra_keys** 混进 hash（`kv_cache_utils.py:364-389`），避免"token 相同但图片/适配器不同"的错误命中。

这与 SGLang 的 **RadixAttention** 是两种实现同一目标的路线（对比见 §5）。

### 3.4 引用计数与驱逐：block 的生命周期

每个 `KVCacheBlock` 有 `ref_cnt`（`kv_cache_utils.py:117`）。一个 block 的状态由 `ref_cnt` 和"是否在空闲链表里"共同决定：

```
ref_cnt > 0                    →  正被某些请求使用，不可分配给别人
ref_cnt == 0 且已 cache        →  "驱逐候选"：仍躺在 cached_block_hash_to_block 里
                                   （内容还在、还能被 prefix cache 命中），
                                   同时挂在 free_block_queue 里等待被复用/驱逐
ref_cnt == 0 且未 cache        →  纯空闲块
```

**关键设计**：ref_cnt 归零的 cached block **不立即销毁**，而是进空闲链表当"驱逐候选"。只要还没被真正分配出去覆盖，它就还能被后来的相同前缀命中（"软驱逐"）。只有当空闲块不够、它被 `get_new_blocks` 真正取走时，才 `_maybe_evict_cached_block` 抹掉它的 hash 元数据（`block_pool.py:158-210`）。**驱逐顺序 = 链表顺序 = LRU**（链表头先被取）。

### 3.5 抢占（preemption）：显存不足时的退路

当所有 block 都在用、新请求又放不下时，V1 调度器（见 [模块 01](../01-scheduler-batching/design.md)）**抢占**最低优先级的 running 请求：`free` 掉它的全部 block（`scheduler.py:204-205`），把它打回 `WAITING` 且 `num_computed_tokens=0`，下次从头 re-prefill。这是 PagedAttention 论文里 "**recomputation**" 恢复策略的 V1 实现（V0 抢占还支持 "swap to CPU"，V1 抢占只用 recomputation、不再有 swap，更简单）。抢占循环有个精妙的自环退出条件（`scheduler.py:214`），详见 [`impl.md`](impl.md) §4。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `KVCacheBlock` | `kv_cache_utils.py:112` | 一个物理 block 的元数据：`block_id`、`ref_cnt`、`_block_hash`、双向链表指针 `prev/next_free_block` |
| `BlockHashType` | `kv_cache_utils.py:21` | `(hash_value, token_ids, extra_keys)` 命名元组，作为 cache 字典的 key；带 token_ids 抗碰撞 |
| `FreeKVCacheBlockQueue` | `kv_cache_utils.py:161` | 空闲块**双向链表**（自己手搓而非 deque，为 O(1) 中间删除）。头=最早释放=LRU 驱逐目标 |
| `BlockPool` | `block_pool.py:16` | 物理块总池：持有所有 `KVCacheBlock`、空闲链表、`cached_block_hash_to_block` 命中字典、`null_block` |
| `cached_block_hash_to_block` | `block_pool.py:51` | `{block_hash: {block_id: block}}`，prefix cache 命中查找的核心字典 |
| `KVCacheManager` | `kv_cache_manager.py:20` | 对调度器的门面：`allocate_slots` / `get_computed_blocks` / `free`；持有 `req_to_blocks`、`req_to_block_hashes` |
| `SpecializedManager` | `specialized_manager.py:11` | 按 attention 类型分派的命中/驱逐逻辑：`FullAttentionManager` vs `SlidingWindowManager`（SWA/Cascade 详见 [模块 14](../14-attention-variants/design.md)；MLA 详见 [模块 13](../13-mla/design.md)） |
| `KVCacheSpec` / `FullAttentionSpec` / `SlidingWindowSpec` | `kv_cache_interface.py:15,73,85` | 描述一层 KV cache 的格式（block_size、num_kv_heads、head_size、dtype、page_size_bytes） |
| `KVCacheConfig` / `KVCacheGroupSpec` | `kv_cache_interface.py:136,124` | 显存切分结果：`num_blocks`、每层张量大小、layer 分组 |
| `BlockTable`（worker 侧） | `worker/block_table.py:11` | GPU 上的 block table 张量 `[max_num_reqs, max_num_blocks_per_req]`，喂给 attention kernel |
| `FlashAttentionMetadata` | `attention/backends/flash_attn.py:71` | 一步 forward 的 attention 元数据：`block_table`、`slot_mapping`、`seq_lens` 等 |
| `null_block` | `block_pool.py:57` | `block_id=0` 的占位块，给 sliding window 等"被跳过的逻辑块"用，ref_cnt 不维护 |

**block 在 `allocate_slots` 中的版图**（`kv_cache_manager.py:182-196` 的 ASCII 注释，是理解分配逻辑的关键）：

```
| < computed > | < new computed > |  < new >  | < pre-allocated > |
└─前次已算的块─┘ └─本次prefix命中─┘ └─新申请─┘ └─预留减少抖动────┘
|                  < required >                |
|                    < full >               |   ← 这些会被 cache（写满的块）
```

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 分页（固定 block_size） | 消灭外部碎片；内部碎片上界 = `block_size-1` token | 多一层 block table 间接寻址，kernel 略复杂；`block_size` 太大则内部碎片回升，太小则 block table 变长 |
| block table 间接寻址 | 抢占/共享/CoW 只改表不搬数据 | attention kernel 必须定制（不能直接用现成连续-KV kernel）；每个 token 多一次访存查表 |
| prefix caching 默认开启 | 命中时跳过整段 prefill，显存复用 | 未命中也要算 hash（开销）；故 V1 用扁平结构把未命中开销压到接近零（"zero-overhead"） |
| hash 链 + 内容字典 | 前缀匹配 = O(前缀块数) 查字典，无需逐 token 比 | hash 碰撞风险（用 SHA256 + 存 token_ids 双保险，`kv_cache_utils.py:21-26`）；hash 计算非零成本 |
| ref_cnt=0 的 cached block 软驱逐 | 释放后仍可被命中，最大化复用 | 占着 hash 字典；真正分配时才驱逐，逻辑稍绕 |
| 抢占用 recomputation（非 swap） | 实现简单，无 CPU↔GPU 搬运带宽开销 | 抢占的请求要从头重算（浪费已算 token）；长 prompt 被抢代价大 |
| 单 KV cache group 限制 | 简化 manager（一张 block table 走天下） | 暂不支持 full+sliding 混合架构高效共存（`unify_hybrid_kv_cache_specs` 退化为全 full） |
| append-only block table（不运行时去重） | block_id 永不变，block table 可只增不改，逻辑简单 | 内容相同但 hash 未命中的两个块不会合并，略损复用率（`block_pool.py:46-50` 注释明示） |

**与 SGLang RadixAttention 的对比**：两者都做自动前缀复用。RadixAttention 用一棵 **radix tree（基数树）** 组织 KV，前缀匹配是树上的最长前缀路径查找，天然支持任意粒度共享、LRU 驱逐叶子。vLLM 用 **block 级 hash 链 + 平坦字典**，匹配粒度是 block（`block_size` 个 token），实现更简单、与现有 paged kernel 无缝对接，但共享只能落在 block 边界。两者是"树结构精细 vs 哈希平坦简单"的工程权衡。

---

## 6. 设计动机出处

- **PagedAttention 论文**：Kwon et al., *"Efficient Memory Management for Large Language Model Serving with PagedAttention"*, **SOSP 2023**（arXiv:2309.06180）。提出把 OS 虚拟内存分页思想用于 KV cache，给出碎片化实测（有效利用率 20%~38%）、block table、copy-on-write、共享前缀等核心设计。本模块 §1 的碎片化数据、§3.1 的分页思想、§3.5 的 recomputation/swap 恢复策略均出自此。
- **Automatic Prefix Caching（APC）**：vLLM 官方文档 *"Automatic Prefix Caching"*（docs.vllm.ai）。阐述用 block hash 自动复用前缀、对用户透明、`enable_prefix_caching` 开关。本模块 §3.3 的 hash 链与命中逻辑对应其设计。
- **vLLM V1 博客**：*"vLLM V1: A Major Upgrade to vLLM's Core Architecture"*（blog.vllm.ai, 2025-01）。把 "zero-overhead prefix caching" 与简化的 KV 管理列为 V1 目标。本模块 §3.2 的 V0→V1 重写动机出自此。
- **RadixAttention / SGLang**：Zheng et al., *"SGLang: Efficient Execution of Structured Language Model Programs"*（arXiv:2312.07104）。用 radix tree 做自动 KV 复用，本模块 §5 用作对比路线。

> 以上为**设计动机**来源；所有 `file:line` 代码引用均以仓库实际代码为准（见 [`impl.md`](impl.md)）。

---

## 7. 一图速览：从碎片化到分页

```
朴素连续分配（碎片化）：                    PagedAttention（分页）：
                                          物理 KV cache 显存（block 池，等大块）：
请求A: [████████░░░░░░░░░░░░]  ← 预留max     ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐
       已用    内部碎片(浪费)                 │B0│B1│B2│B3│B4│B5│B6│B7│..│
请求B: [██░░░░░░░░░░]   (结束→留空洞)         └──┴──┴──┴──┴──┴──┴──┴──┴──┘
请求C: [需要连续大段，但空洞拼不出] ← 外部碎片    ▲  ▲     ▲  ▲     ▲
                                            │  │     │  │     └─ 请求C 逻辑块0 → B6
   有效利用率 ~20-38%（论文实测）            │  │     │  └─ 请求B 逻辑块1 → B4
                                            │  │     └─ 请求B 逻辑块0 → B3
                                            │  └─ 请求A 逻辑块1 → B1
                                            └─ 请求A 逻辑块0 → B0

block_table:  A=[B0,B1]  B=[B3,B4]  C=[B6,...]   ← 物理块乱序、不连续、可共享

────────────────────────────────────────────────────────────────────
Prefix Caching（共享前缀）：A、B 前缀相同
  block hash:  H0 = H(∅, tok[0:16])      ← 两请求算出同一个 H0
               H1 = H(H0, tok[16:32])     ← 同一个 H1
  cached_block_hash_to_block: { H0→B0, H1→B1, ... }
  → 请求B 命中 H0,H1：block_table_B = [B0, B1, B5...]  ← 直接指向A的物理块
                                       └共享(ref_cnt=2)┘ └B自己新申请┘
  → 命中部分跳过 prefill，只算 B5 起的新 token
────────────────────────────────────────────────────────────────────
attention kernel 读 KV（逻辑→物理）：
  要算 query 对第 j 个 KV：
     logical = j // block_size
     physical = block_table[logical]          ← 间接寻址（PagedAttention 精髓）
     addr = physical * block_stride + (j % block_size) * head_stride
     读 k_cache[addr] / v_cache[addr]
```

> 实现层面的逐行调用链、`file:line` 对照、kernel 内的指针算术与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 8. 设计背后的考量与历史教训

### 8.1 设计背后的考量（动机与历史）

1. **V1 把 V0 的"对象层级 block allocator"压扁，否决了"继续封装"的路线**。V0 的 `block/` 目录是一套层层包装（block allocator → block → block table → 独立 `LRUEvictor`），分配一个 block 要穿过多层 Python 调用，prefix caching、CoW、swap 各有独立子系统。V1 把它压成"一个 `BlockPool` + 一条双向链表 `FreeKVCacheBlockQueue` + 一张 hash 字典"，**把驱逐折叠进链表本身**（链表头即 LRU 目标，free 时反序入队），不再需要单独 evictor。代价是放弃 V0 的运行中 block 去重与 CPU swap，换来热路径更短、未命中开销趋零——这正是"zero-overhead prefix caching"的实现基础。

2. **`num_computed_tokens` 必须是 block_size 整数倍，是被刻意立成不变量的"简化记账"**。`allocate_slots` 全程假设这一点，因此 prefix 全命中导致"无新 token"时，宁可回退一整个 block 重算（见 §8.2 `#11186`），也不破坏不变量。这是"用一点冗余计算换记账简单性"的典型取舍——把复杂度挡在数据结构之外。

3. **"ref_cnt 归零不立即销毁、而是当驱逐候选"是为了把复用率压到最大**。归零的 cached block 仍留在 `cached_block_hash_to_block` 里、同时挂进空闲链表，只有被 `get_new_blocks` 真正取走时才 `_maybe_evict_cached_block` 抹掉 hash（软驱逐）。否决的简单方案是"归零即删"，但那会让"刚释放就来的相同前缀"白白错过命中。代价是逻辑更绕、字典占用更久。

4. **prefix caching 的 hash 链经历了从"裸 Python hash"到"抗碰撞 + extra_keys"的演进**。最初只对 token 串做 hash；后续意识到①不同请求 token 串相同但上下文不同会错误命中，②Python `hash` 有碰撞风险。于是 `#12603` 把 `mm_hash`/`lora_id` 等作为 **extra_keys** 混进 block hash，`#12621` 引入更强的碰撞规避（并默认可选 SHA256），`BlockHashType` 还把 `token_ids` 一并存进 key 做双保险。教训：**内容寻址的缓存，"键设计"就是正确性本身**，少一个维度就是一类静默错误。

5. **CUDA kernel 这一层"以重构/演进为主，但有几个高启发的早期正确性 bug"**。PagedAttention kernel 与朴素 attention 的唯一本质区别是多一次 `block_table[logical]→physical` 间接寻址，而恰恰是这层间接寻址 + 块尾不满，制造了最经典的几类 GPU bug：整数溢出、块边界越界、块尾 NaN（见 §8.2）。这些不是设计缺陷，而是"分页"这一抽象在 kernel 落地时必然要补的边界处理。

### 8.2 重要 bug 修复（真实、精选）

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| `#11186` | prompt 长度恰为 block_size 整数倍且 prefix 全命中时无新 token 可算 | 坐实 **`num_computed_tokens` 必须是 block_size 整数倍**这条不变量：修复回退**整个 block**（`-= block_size`）重算，注释明言"因为 `allocate_slots()` 假设它总是 block_size 的倍数" |
| `#11565` | `allocate_slots` 里 `_touch(computed_blocks)` 调用位置不当：在确认能分配新块**之前**就 touch 了命中块 | 若随后因空闲块不足分配失败，已被 touch 的 computed block 状态被污染。修复把 `_touch` **移到分配检查之后**——"先确认成功再改共享状态"，否则失败路径会留下脏引用计数 |
| `#14073` | 重复缓存"已在 prefix cache 里的 block"，靠脆弱的"找第一个未缓存块"循环兜底（代码里原本带 `FIXME: 这本不该发生`） | 缓存写入路径应**只缓存不在 prefix cache 中的 block**，由 `num_cached_blocks` 精确界定，而非靠遍历探测。教训：用 `FIXME` 标注的"理论上不会发生"往往真的会发生，应从源头修而非加兜底 |
| `#12603` | 两个 token 串相同但 `extra_keys`（如多模态/LoRA）不同的 block 算出同一 hash → 错误命中 | prefix cache 的命中键必须涵盖一切影响 KV 内容的上下文；把 `extra_keys` 混入 block hash。内容寻址缓存的"键完备性"即正确性 |
| `#1514` | attention/cache kernel 中 `physical_block_number` 用 int32，乘以大 `kv_block_stride` 时**整数溢出** | 分页的间接寻址在大模型/长上下文下，物理块号 × stride 会超出 int32；修复统一 cast 到 int64。教训：地址算术必须按最坏规模选位宽 |
| `#936` / `#1241` | 块尾不满时 V 向量含越界 token → **NaN**；`#936` 修复时写出 `j <= V_VEC_SIZE` 的 off-by-one，`#1241` 再改回 `j < V_VEC_SIZE` | "块尾不满"是分页抽象的固有边界：最后一个 block 几乎总是半满，越界元素必须显式清零。同一处边界被改了两次，正说明 kernel 边界条件极易写错、需测试守住 |
| `#16693` | `merge_attn_states` kernel 可能破坏 CUDA graph 捕获 | cascade/分块 attention 合并多段结果时的 kernel 与 CUDA graph 的兼容性边界；说明 KV 复用类优化下放到 kernel 后，要额外守住 graph capture 的约束 |
