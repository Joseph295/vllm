# 模块 02 · PagedAttention + KV Cache 管理 + Prefix Caching —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的实现细节：从调度器请求一块 block，到 block table 进入 attention kernel 完成逻辑→物理寻址的完整链路。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
> Python 侧以 **V1（`vllm/v1/core/`、`vllm/v1/worker/`）** 为主；CUDA kernel（`csrc/`）V0/V1 共用。

---

## 1. 代码地图

```
CUDA kernel 层 csrc/                          （V0/V1 共用）
  ├─ attention/paged_attention_v1.cu          # 单 kernel 版 paged attention 启动器
  ├─ attention/paged_attention_v2.cu          # 分块 reduce 版（长序列）
  ├─ attention/attention_kernels.cuh          # 核心 kernel 模板：block_table 逻辑→物理寻址
  ├─ cache_kernels.cu                         # reshape_and_cache（写KV）、copy_blocks（CoW）
  └─ (Python 绑定) vllm/_custom_ops.py        # paged_attention_v1/v2、reshape_and_cache*、copy_blocks

V1 KV 管理层 vllm/v1/core/
  ├─ kv_cache_manager.py        # KVCacheManager：对调度器的门面（allocate_slots / get_computed_blocks / free）
  ├─ block_pool.py              # BlockPool：物理块总池、空闲链表、命中字典、cache_full_blocks、驱逐
  ├─ kv_cache_utils.py          # KVCacheBlock、BlockHashType、FreeKVCacheBlockQueue、hash_block_tokens
  ├─ specialized_manager.py     # FullAttentionManager / SlidingWindowManager：命中查找 & 驱逐策略
  └─ kv_cache_interface.py      # KVCacheSpec / FullAttentionSpec / KVCacheConfig（显存切分）

V1 调度耦合 vllm/v1/core/sched/
  └─ scheduler.py               # 调用 KV manager 分配/释放/抢占；打包 block_ids 进 SchedulerOutput

V1 Worker / kernel 对接 vllm/v1/worker/
  ├─ gpu_model_runner.py        # 计算 slot_mapping、维护 block_table 张量、调 attention backend
  └─ block_table.py             # BlockTable：GPU 上 [max_num_reqs, max_num_blocks_per_req] 张量

V1 attention backend vllm/v1/attention/backends/
  └─ flash_attn.py              # 构建 FlashAttentionMetadata（block_table/slot_mapping），调 kernel

V0 对比 vllm/core/
  ├─ block_manager.py           # BlockSpaceManager（allocate/append_slots/fork/free）
  ├─ block/prefix_caching_block.py、block/common.py  # PrefixCachingBlockAllocator、CoWTracker、RefCounter
  └─ evictor.py                 # LRUEvictor（V1 已用 free_block_queue 替代）
```

---

## 2. 端到端调用链：从 block 分配到 kernel 读 block table

下面追踪**一条请求的一个调度步**里，KV/block 相关的完整链路。分两条主线：**写 KV** 和 **读 KV（attention）**，中间夹着 block 的分配与 prefix cache 命中。

### 阶段 A：调度器请求 block —— prefix 命中 + 分配

> **这一步在干嘛**：调度器要给一个请求安排 KV 空间。先问 prefix cache"前面这段 token 有没有别人/自己已经算过"——命中的部分直接白嫖（不再计算），只为剩下的 token 真正申请新物理块。这是 prefix caching 省算力和 PagedAttention 分配显存的入口。
>
> 🔗 **回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)**：例子里 `[11,7,4]` 若是全新请求，prefix 不命中，就为 2 个逻辑块申请物理块 `[7,3]`；若另一条请求前缀也是 `[11,7]`，它会命中物理块 7、直接共享，省去重算。

**A1.** `vllm/v1/core/sched/scheduler.py:305` 调度 WAITING 请求时，先查 prefix cache：
```python
computed_blocks, num_computed_tokens = \
    self.kv_cache_manager.get_computed_blocks(request)
num_new_tokens = request.num_tokens - num_computed_tokens   # 命中的前缀不再算
```
> 命中的前缀 token 数直接从待算 token 里扣掉——这就是 prefix caching 省算力的入口。

**A2.** `vllm/v1/core/kv_cache_manager.py:103` `get_computed_blocks()` —— **prefix 命中查找**
```python
block_hashes = self.req_to_block_hashes[request.request_id]
if not block_hashes:
    block_hashes = hash_request_tokens(self.caching_hash_fn, self.block_size, request)
    self.req_to_block_hashes[request.request_id] = block_hashes
...
computed_blocks = self.specialized_manager.find_longest_cache_hit(block_hashes)
...
num_computed_tokens = len(computed_blocks) * self.block_size   # 永远是 block_size 整数倍
```
- `:124` `hash_request_tokens` 把整条 prompt 切块、算 **hash 链**（见 A3）。结果缓存在 `req_to_block_hashes`，避免反复算。
- `:133-142` **边界**：若 prompt 长度恰好是 `block_size` 整数倍且全部命中，弹掉最后一个 block hash——因为 attention 至少要算最后一个 token，不能全跳过（否则没有 query）。
- `:147` 委托 `find_longest_cache_hit` 沿 hash 链查字典。

**A3.** `vllm/v1/core/kv_cache_utils.py:423` `hash_request_tokens()` —— **hash 链构造**
```python
parent_block_hash_value = None
for start in range(0, len(token_ids), block_size):
    block_token_ids = token_ids[start:start + block_size]
    if len(block_token_ids) < block_size:
        break                                   # 不满的块不参与 hash（不能共享）
    block_hash = hash_block_tokens(hash_function, parent_block_hash_value,
                                   block_token_ids, req_extra_keys)
    ret.append(block_hash)
    parent_block_hash_value = block_hash.hash_value   # 链：当前块 hash 喂给下一块
```
`hash_block_tokens`（`kv_cache_utils.py:392`）把 `(parent_hash, token_tuple, extra_keys)` 一起 hash：
```python
return BlockHashType(
    hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
    curr_block_token_ids_tuple, extra_keys)
```
> `extra_keys` 来自 mm_hash / lora_id（`generate_block_hash_extra_keys`, `:364`），保证多模态/LoRA 不误命中。`BlockHashType` 额外存 `token_ids` 是 hash 碰撞的第二道保险（`:21-26`）。

**A4.** `vllm/v1/core/specialized_manager.py:71` `FullAttentionManager.find_longest_cache_hit()`
```python
computed_blocks: list[KVCacheBlock] = []
for block_hash in block_hashes:
    if cached_block := self.block_pool.get_cached_block(block_hash):
        computed_blocks.append(cached_block)
    else:
        break    # 链上第一个 miss → 前缀分叉点，后面必然全 miss
return computed_blocks
```
> 这是 hash 链的精髓：**一旦某块 miss 就能立刻停**，因为后续块的 hash 依赖了这个 miss 块，不可能命中。`get_cached_block`（`block_pool.py:59`）查 `cached_block_hash_to_block` 字典。

**A5.** `vllm/v1/core/sched/scheduler.py:332` 分配新 block：
```python
new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, computed_blocks)
if new_blocks is None:
    break    # 显存不足，本步不调度该请求
```

**A6.** `vllm/v1/core/kv_cache_manager.py:163` `allocate_slots()` —— **分配 + 触碰命中块 + 缓存满块**
```python
num_computed_tokens = (request.num_computed_tokens +
                       len(new_computed_blocks) * self.block_size)
num_required_blocks = cdiv(num_computed_tokens + num_tokens + num_lookahead_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks) - len(new_computed_blocks)
...
# 命中块若是驱逐候选(ref_cnt==0)，不能算作可用空闲块
num_evictable_computed_blocks = sum(1 for blk in new_computed_blocks if blk.ref_cnt == 0)
if (num_new_blocks > self.block_pool.get_num_free_blocks() - num_evictable_computed_blocks):
    return None                                   # :229-232 显存不足
if self.enable_caching:
    self.block_pool.touch(new_computed_blocks)    # :236 命中块 ref_cnt++，从空闲链表摘除
req_blocks.extend(new_computed_blocks)            # :244 先挂命中块
...
new_blocks = self.block_pool.get_new_blocks(num_new_blocks)   # :268 取新物理块
req_blocks.extend(new_blocks)
...
self.block_pool.cache_full_blocks(...)            # :284 把本步写满的块登记进 cache
```
- `:227-232` **关键边界**：prefix 命中的块如果 ref_cnt==0（曾是别人释放的驱逐候选），它名义上还躺在空闲链表里，必须从"可用空闲数"里减掉，否则会重复计数导致超分配。
- `:236` `touch()` 把命中块 ref_cnt++ 并从空闲链表移除（`block_pool.py:212`）——**这就是 prefix 共享的引用计数动作**：A、B 两请求命中同一块，该块 ref_cnt 变成 2。
- `:254-264` **预分配**：多申请 `num_preallocate_blocks` 个块，减少每步都更新链表/ref_cnt 的开销（`kv_cache_manager.py:44-56` 注释）。
- `:281-292` `cache_full_blocks` 只缓存**写满且非投机**的块（投机 token 可能被 reject，不能 cache）。

**A7.** `vllm/v1/core/block_pool.py:158` `get_new_blocks()` —— **从链表头取块 + 必要时驱逐**
```python
curr_block = self.free_block_queue.popleft()      # 链表头 = LRU
assert curr_block.ref_cnt == 0
if self.enable_caching:
    self._maybe_evict_cached_block(curr_block)     # :182 该块若还带旧 hash，抹掉它（真正驱逐）
curr_block.incr_ref()
```
`_maybe_evict_cached_block`（`block_pool.py:190`）：
```python
block_hash = block.block_hash
if block_hash and block_hash in self.cached_block_hash_to_block:
    block.reset_hash()                                          # 清元数据
    del self.cached_block_hash_to_block[block_hash][block.block_id]
```
> **软驱逐的兑现点**：一个块 ref_cnt 归零后只是进了链表（仍可被命中），直到此刻被真正取走分配给新请求，才抹掉 hash、从命中字典删除。在此之前它都还能被 prefix cache 救回。

**A8.** `vllm/v1/core/block_pool.py:76` `cache_full_blocks()` —— **把写满的块登记进命中字典**
```python
for i, blk in enumerate(new_full_blocks):
    if i < len(new_block_hashes):
        block_hash = new_block_hashes[i]          # prompt 块：hash 在 A3 已算好，直接复用
    else:
        ...                                       # 生成的块：现算 hash（:132-151）
        block_hash = hash_block_tokens(hash_fn, prev_block_hash_value, block_tokens, extra_keys)
        block_hashes.append(block_hash)
    blk.block_hash = block_hash
    self.cached_block_hash_to_block[block_hash][blk.block_id] = blk    # :155 登记，可被后续命中
    prev_block_hash_value = block_hash.hash_value
```

调度器把分配到的 block_id 打包进 `SchedulerOutput`（`scheduler.py:358`）：`req_to_new_block_ids[req_id] = [b.block_id for b in computed_blocks + new_blocks]`，随调度结果过界到 worker。

### 阶段 B：worker 侧 —— 维护 block table 张量、算 slot_mapping

> **这一步在干嘛**：调度器分到的物理块号过界到 worker 后，把它们写进 GPU 上的 block table 张量（这条请求的"逻辑块→物理块"页表）；再为本步要算的每个 token 算出它的全局 **slot**（= 物理块号×block_size + 块内偏移），告诉 kernel"这个 token 的 KV 该写到显存哪个槽位"。
>
> 🔗 **回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)**：这一步就是算出例子里的 `slot_mapping = [14, 15, 6]`（token0→7×2+0=14、token1→7×2+1=15、token2→3×2+0=6）。

**B1.** `vllm/v1/worker/gpu_model_runner.py:420` `_update_states()`：把新分到的 block_id 追加进 GPU block table 行
```python
self.input_batch.block_table.append_row(req_data.new_block_ids, req_index)
```
`BlockTable.append_row`（`worker/block_table.py:39`）只往 numpy 视图里写增量：
```python
start = self.num_blocks_per_row[row_idx]
self.block_table_np[row_idx, start:start + num_blocks] = block_ids   # append-only
```
> block table 是 `[max_num_reqs, max_num_blocks_per_req]` 的 int32 张量（`block_table.py:25`），CPU(pinned)+GPU 双份，`commit()`（`:69`）异步 H2D。**append-only** 正是 design §5 "block_id 永不变" 的体现。

**B2.** `vllm/v1/worker/gpu_model_runner.py:541` `_prepare_inputs()`：**计算 slot_mapping（逻辑位置→物理 slot）**
```python
# block_table_indices = req * max_num_blocks_per_req + position // block_size
block_table_indices = (req_indices * self.max_num_blocks_per_req +
                       positions_np // self.block_size)
block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()   # 查表得物理块号
block_offsets = positions_np % self.block_size
np.add(block_numbers * self.block_size, block_offsets,
       out=self.slot_mapping_np[:total_num_scheduled_tokens])           # slot = phys*bs + offset
```
> 这是 design §3.1 "写 KV 路径" 的代码实现：`position → logical_block → (查表) physical_block → slot`。`:539` 注释专门说明**不能**用 `token_indices // block_size`，因为 `max_model_len` 未必整除 `block_size`。纯 numpy 向量化，无 Python 逐 token 循环。

### 阶段 C：attention backend —— 把 block_table / slot_mapping 装进 metadata

> **这一步在干嘛**：把上一步算好的 `block_table`、`slot_mapping`、各请求序列长度等，打包成 attention kernel 一步所需的元数据 `FlashAttentionMetadata`——这就是交给 kernel 的"寻址说明书"。

**C1.** `vllm/v1/attention/backends/flash_attn.py:286` `FlashAttentionMetadataBuilder.build()`
```python
block_table = self.runner.input_batch.block_table.get_device_tensor()[:num_reqs]
slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(device, non_blocking=True).long()
...
attn_metadata = FlashAttentionMetadata(... block_table=block_table, slot_mapping=slot_mapping,
                                       seq_lens=seq_lens, ...)
```
> `FlashAttentionMetadata`（`:71`）就是 attention 一步所需的全部 paged 元数据。`use_cascade`（`:321`）用于公共前缀的 cascade attention（见 §3）。

### 阶段 D：写 KV —— reshape_and_cache_flash

> **这一步在干嘛**：把本步算出的新 token 的 K、V 写进分页 KV cache。kernel 拿 `slot_mapping` 里每个 token 的 slot，反解出"物理块号 + 块内偏移"，把 K/V 放到显存对应地址——这就是"逻辑位置→物理槽位"的写入落地。
>
> 🔗 **回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)**：例子里 token2 的 slot=6 在这里被分解回 `block_idx = 6/2 = 3`、`block_offset = 6%2 = 0`，于是 k2/v2 被写进物理块 3 的偏移 0。

**D1.** `vllm/v1/attention/backends/flash_attn.py:460` `forward()` 先把本步算出的 K/V 写进 paged cache：
```python
key_cache, value_cache = kv_cache.unbind(0)
torch.ops._C_cache_ops.reshape_and_cache_flash(
    key, value, key_cache, value_cache,
    attn_metadata.slot_mapping,             # ← 每个 token 写到哪个 slot
    self.kv_cache_dtype, layer._k_scale, layer._v_scale)
```
Python 绑定 `vllm/_custom_ops.py:1328` `reshape_and_cache_flash`。

**D2.** `csrc/cache_kernels.cu:222` `reshape_and_cache_kernel`（标准版；flash 版 `:282` 同理）—— **slot 分解为块号+块内偏移**
```cuda
const int64_t slot_idx = slot_mapping[token_idx];
if (slot_idx < 0) return;                          // padding token 跳过
const int64_t block_idx = slot_idx / block_size;   // ← 物理块号
const int64_t block_offset = slot_idx % block_size;// ← 块内偏移
...
const int64_t tgt_key_idx =
    block_idx * num_heads * (head_size / x) * block_size * x + ...
    + block_offset * x + x_offset;                 // 写入 paged cache 对应地址
```
> design §3.1 的 `slot = physical_block*block_size + offset` 在这里被**逆向**分解回 `(block_idx, block_offset)` 来定位 cache 张量地址。`slot_idx < 0` 是 padding 的哨兵值。

### 阶段 E：读 KV —— attention kernel 用 block table 逻辑→物理寻址

> **这一步在干嘛**：算 attention 时，当前 token 的 query 要去读前面所有 token 的 K/V。但 KV 散落在不连续的物理块里，kernel 必须用 `block_table[逻辑块] → 物理块` 跳着读——这一次间接寻址就是 PagedAttention 与朴素 attention 的全部区别。
>
> 🔗 **回扣[贯穿示例](../PRIMER.md#第三部分一个贯穿全文的玩具示例强烈建议先看懂这个)**：decode 步算 `q3` 对前 4 个位置的 attention 时，kernel 用 `block_table=[7,3]` 把逻辑位置翻成物理 slot——pos0/1 在物理块 7、pos2/3 在物理块 3——再去取对应 KV。

**E1.** Python 侧（FlashAttention 路径）把 `block_table` 传给 `flash_attn_varlen_func`（`flash_attn.py:503-522`），kernel 内部按 block table 跳读 paged KV。

**E2.** 经典 PagedAttention kernel（`csrc/attention/`，paged_attention_v1/v2 与多数自定义 backend 共用此模式）—— **逻辑块→物理块的间接寻址**

`csrc/attention/attention_kernels.cuh:207` 先定位本序列的 block table 行：
```cuda
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
```
`csrc/attention/attention_kernels.cuh:257` **核心一行**：逻辑块号 → 物理块号
```cuda
const int64_t physical_block_number =
    static_cast<int64_t>(block_table[block_idx]);   // ← PagedAttention 的精髓
```
随后用 `physical_block_number` 算出 K/V cache 的真实地址（`:273` 起）：
```cuda
const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                       + kv_head_idx * kv_head_stride + physical_block_offset * x;
```
> 这就是 design §1 "kernel 要能跨非连续块算 attention" 的全部实现：把朴素 attention 里"连续 KV 数组的下标"换成"`block_table[逻辑块]` 查出的物理块号"。多了一次间接寻址，就让物理块可以任意乱序、共享、被抢占重分配。Python 绑定见 `_custom_ops.py:40`（v1）/`:69`（v2）。这段 kernel 内部的**线程/warp 映射、每线程负责哪些 head/token、reshape_and_cache 写入 kernel 的 GPU 视角细节**见 [模块 11](../11-gpu-kernels-memory/impl.md)。

至此一个调度步的 KV 读写闭环完成；下一步 `update_from_output` 里若请求结束就 `free`（见 §4）。

---

## 3. cascade attention：公共前缀只算一次

当多条 running 请求共享长公共前缀（如同一 system prompt），重复对前缀算 attention 是浪费。V1 用 **cascade attention** 把"公共前缀 attention"和"各自后缀 attention"拆开算再合并。

- **公共前缀块数**由 `vllm/v1/core/kv_cache_manager.py:331` `get_num_common_prefix_blocks()` 计算：
  ```python
  for block in blocks:
      if block.ref_cnt == num_running_requests:   # 所有 running 请求都指向它 → 公共
          num_common_blocks += 1
      else:
          break
  ```
  > 判据极简：**block 的 ref_cnt 等于 running 请求总数**，说明每条 running 请求都共享它。一旦遇到第一个非公共块就停。
- 是否启用由启发式 `use_cascade_attention`（`flash_attn.py:553`）决定（公共前缀 ≥256 token、请求数 ≥8、无 alibi/sliding window，再走性能模型对比 FlashDecoding）。
- 实际计算 `cascade_attention`（`flash_attn.py:621`）：前缀用 `block_table[:1]`（任一请求的前缀块即可，因共享）算一次（`:667`），后缀用 `block_table[:, num_common_kv_blocks:]` 各算（`:693`），最后 `merge_attn_states`（`:706`）合并。

---

## 4. 释放与抢占

### 4.1 free —— 反序释放让尾块先驱逐

`vllm/v1/core/kv_cache_manager.py:298` `free()`
```python
blocks = self.req_to_blocks.pop(request.request_id, [])
ordered_blocks = blocks
if self.enable_caching:
    ordered_blocks = reversed(blocks)     # 反序！尾块先入空闲链表 = 先被驱逐
self.block_pool.free_blocks(ordered_blocks)
```
`block_pool.free_blocks`（`:227`）逐块 `decr_ref()`，ref_cnt 归零才 `append` 进空闲链表尾：
```python
block.decr_ref()
if block.ref_cnt == 0 and block != self.null_block:
    self.free_block_queue.append(block)
```
> **为什么反序**：一条序列的块按 token 顺序排列，**头部块（前缀）更可能被别的请求复用**，尾部块（这条请求独有的生成内容）几乎不会被复用。反序入队 → 尾块排在空闲链表更靠前 → 优先被驱逐，**保护可复用的前缀块**（`kv_cache_utils.py:170-177` 注释解释了这个 LRU 排序）。共享块（ref_cnt 仍 >0）不入链表，自然不会被误驱逐。

### 4.2 preempt —— 显存不足时的自环退出

`vllm/v1/core/sched/scheduler.py:196` running 请求分配失败时抢占：
```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is None:
        preempted_req = self.running.pop()               # 抢最低优先级
        self.kv_cache_manager.free(preempted_req)        # 释放其全部 block
        preempted_req.status = RequestStatus.PREEMPTED
        preempted_req.num_computed_tokens = 0            # 打回从头 re-prefill
        self.waiting.appendleft(preempted_req)
        if preempted_req == request:                     # :214 抢到自己 → 没有更低优先级可抢
            can_schedule = False
            break
    else:
        break
```
> **`preempted_req == request` 是关键边界**：当被抢的恰好是当前正想调度的请求自己，说明已无更低优先级可抢，放弃调度它，否则会死循环。这是 design §3.5 recomputation 恢复策略的实现（`num_computed_tokens=0` 即丢弃已算 KV，下次重算）。

---

## 5. 实现中的精妙之处（Tricks 与非显然设计）

> 下面每条都是读 KV 管理代码时容易一眼滑过、理解了才算真懂的点。格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · 手搓双向链表替代 deque，为的是 O(1) 中间删除
- **代码**：`kv_cache_utils.py:161` `FreeKVCacheBlockQueue`，`remove()`（`:208`）直接改 `prev/next_free_block` 指针。
- **精妙之处**：prefix cache 命中一个块时，要把它从"空闲链表中间"摘出来（`touch` → `remove`）。Python `deque` 中间删除是 O(n)。这里把链表指针直接挂在 `KVCacheBlock` 上（`kv_cache_utils.py:124-125`），删除/插入全 O(1)，且**不分配任何 Python 对象**（`:166-168` 注释）——纯指针操作，逼近 C 的 deque 性能。这是 V1 把 V0 的独立 evictor 折叠掉的关键支撑。

### T2 · 空闲链表同时是"空闲池"和"驱逐候选池"，省掉独立 evictor
- **代码**：`block_pool.py:158` `get_new_blocks` 从链表头取，取到带 hash 的块就 `_maybe_evict_cached_block`（`:182`）。
- **精妙之处**：V0 有独立 `LRUEvictor`（`evictor.py`，最小堆）。V1 发现：**驱逐候选块本质就是 ref_cnt==0 的 cached 块**，它们本来就该躺在空闲链表里等复用。于是把两者合一——链表头即 LRU 驱逐目标，"驱逐"只是分配时顺手抹掉旧 hash。没有单独的堆、没有 `last_accessed` 时间戳比较，少一整个子系统。

### T3 · ref_cnt==0 的 cached 块"软驱逐"，释放后还能被命中救回
- **代码**：`block_pool.py:227-239` free 只 append 进链表、不抹 hash；`kv_cache_manager.py:227-230` 分配时把这种块算作 `num_evictable_computed_blocks`。
- **精妙之处**：一个块释放后 hash 元数据**保留**，仍挂在 `cached_block_hash_to_block` 里。在它被真正分配出去覆盖之前，后来的相同前缀请求**还能命中它**（`get_computed_blocks` 照样查到），`touch` 把它从链表里救回、ref_cnt 复活。这把"释放"和"销毁"解耦，最大化前缀复用窗口——但代价是分配时要小心：这种块名义在空闲链表但即将被命中占用，所以 `:229` 要从可用空闲数里减掉，否则超分配。

### T4 · 反序释放保护前缀块（LRU 的方向性）
- **代码**：`kv_cache_manager.py:308-314` `reversed(blocks)`。
- **精妙之处**：见 §4.1。直觉是"先进先出"，但这里**故意反序**——序列尾部块是该请求独有的生成内容（复用价值低），头部块是前缀（复用价值高）。反序入空闲链表让尾块排前、优先被驱逐，前缀块尽量久留。一行 `reversed` 编码了对 KV 复用模式的深刻理解。

### T5 · hash 链让前缀匹配能"一 miss 即停"
- **代码**：`specialized_manager.py:74-82` 遇到第一个 miss 就 `break`。
- **精妙之处**：因为每个 block hash 包含父块 hash（`kv_cache_utils.py:455-458`），第 i 块命中**蕴含**前 i-1 块都命中。所以查找是沿链单向走、第一个 miss 处即前缀分叉点，无需回溯、无需查后续块。把"两段 token 前缀是否相同"这个 O(L) 的比较，变成 O(命中块数) 的查字典。

### T6 · block hash 把 token_ids 一起存进 key，双保险抗碰撞
- **代码**：`kv_cache_utils.py:21-32` `BlockHashType(hash_value, token_ids, extra_keys)`；`:417-420` 整个元组作字典 key。
- **精妙之处**：字典 key 不只是 hash 整数，还带上原始 `token_ids` 和 `extra_keys`。即使两段不同 token 算出相同 `hash_value`（builtin `hash` 有碰撞可能），元组级别的 `__eq__` 也会因 token_ids 不同而判不等，**杜绝错误命中**。用户可选 SHA256（`kv_cache_manager.py:41`）让碰撞概率降到可忽略；token_ids 是第二道防线。

### T7 · 写满才缓存、投机 token 不缓存
- **代码**：`kv_cache_manager.py:281-282` `num_full_blocks_after_append = (num_computed_tokens + num_tokens - len(spec_token_ids)) // block_size`；`block_pool.py:105-107` 只处理 `[num_cached:num_full]`。
- **精妙之处**：两条正确性约束合一。(a) **半满块不缓存**——块内容还会变，缓存了就是错的；只 cache `// block_size` 取整后的满块数。(b) **投机解码的 token 不缓存**——它们可能被 verify 阶段 reject 回退，缓存了会让别的请求命中到"将被撤销"的 KV。所以算满块数时先 `- len(spec_token_ids)`。

### T8 · 全命中时弹掉最后一个 block hash，保证有 query 可算
- **代码**：`kv_cache_manager.py:133-144`。
- **精妙之处**：若 prompt 长度恰是 `block_size` 整数倍、且每个块都命中，那么 `num_computed_tokens == num_tokens`，意味着"没有一个 token 需要计算"——但 attention 至少要为最后一个位置产生 query/输出。于是故意弹掉最后一个 block hash，强制最后一个块 recompute。注释（`:137-141`）说明这是因为 `allocate_slots` 假设 `num_computed_tokens` 是 `block_size` 整数倍，无法只回退一个 token，只能整块回退。

### T9 · null_block：给"被跳过的逻辑块"一个统一占位
- **代码**：`block_pool.py:54-57` `null_block = free_block_queue.popleft()`（block_id=0，ref_cnt 不维护）；`specialized_manager.py:109` sliding window 用它填充被淘汰位置。
- **精妙之处**：sliding window attention 下，窗口外的旧块会被释放，但 block table 的逻辑下标必须连续。用 `null_block`（恒为 block 0）填这些"逻辑存在、物理已弃"的槽（`remove_skipped_blocks`, `:132`），让 block table 始终是规整数组、无需稀疏表示。它的 ref_cnt 故意不维护，且各处 `block != self.null_block` 判断防止误把它 free（`block_pool.py:223,238`）。

### T10 · slot_mapping 不能用 token_indices // block_size
- **代码**：`gpu_model_runner.py:539-548`，特意分两步：先查 block table 得物理块号，再 `physical*block_size + offset`。
- **精妙之处**：`token_indices` 是 `position + req * max_model_len`（`:524`），而 `max_model_len` 未必整除 `block_size`，直接整除会算错块边界。正确做法是用 `position // block_size` 得逻辑块号、查 block table 得物理块号、再乘 `block_size` 加偏移。注释（`:539-541`）专门警示这个陷阱。

### T11 · 预分配块摊薄链表/ref_cnt 更新开销
- **代码**：`kv_cache_manager.py:254-264` 多取 `num_preallocate_blocks` 个块。
- **精妙之处**：decode 阶段每步只多算 1 个 token，若每跨一个 block 边界才申请会频繁动链表。预分配一批块（默认 64 token，`:54`），让请求在用完预分配前**完全不碰 KV manager**。注释（`:44-53`）强调这区别于 lookahead——不保证总有 N 个空块，用完才再申请，以少量显存浪费换调度热路径的开销下降。

### T12 · block table append-only：不去重、不改 block_id
- **代码**：`block_pool.py:46-50` 注释明示不去重；`worker/block_table.py:39` `append_row` 只增。
- **精妙之处**：两个内容相同但 hash 未及时命中的块**不会被合并**，看似损失复用率。但换来 block_id 一旦分配永不变、block table 只增不改——worker 侧只需 `append_row` 写增量、attention kernel 读到的物理块号稳定（对 CUDA graph replay 友好，见 [模块 07](../07-cuda-graph-compile/design.md)）。这是"略损复用率换整条写路径简单"的工程取舍。

### T13 · cascade attention 用 ref_cnt 判公共前缀
- **代码**：`kv_cache_manager.py:372-376`。
- **精妙之处**：判断"所有 running 请求是否共享某前缀块"不需要逐请求比对 token——直接看块的 `ref_cnt == num_running_requests`。因为 prefix 命中时每个共享者都 `touch` 过它一次（T3），ref_cnt 精确等于共享者数量。引用计数在这里被复用成"公共性检测器"。边界（`:343-357` 注释）：running 数 ≥ 调度数，可能有未调度的 running 请求不共享前缀，导致保守返回 0。

### T14 · CoW 退化：V1 主路径不做 copy-on-write
- **代码**：V1 无 `CopyOnWriteTracker`；对比 V0 `block/common.py:98` `CopyOnWriteTracker` + `block_manager.py:250` `clear_copy_on_writes()` + `_custom_ops.py:1357` `copy_blocks`（kernel `cache_kernels.cu:83`）。
- **精妙之处**：V0 用 `fork()` + CoW 支持 beam search / parallel sampling 的序列分叉（共享块、写时复制）。V1 因为取消了运行时 block 去重、且 prefix 共享的块都是**只读前缀**（命中块不会被新请求写），主 prefill/decode 路径**不需要 CoW**——共享块永远不被原地改写，新 token 一律写进各自新申请的块。`copy_blocks` kernel 仍保留（给跨设备/PD 分离搬块用），但不在 V1 attention 热路径。这是 design §3.2 表格"V1 不做 CoW"的代码层证据。

### T15 · 显存切分用二分搜索给出可行 max_model_len
- **代码**：`kv_cache_utils.py:462` `estimate_max_model_len` 二分；`:510` `check_enough_kv_cache_memory`；`:601` `_get_kv_cache_config_uniform_type` 把显存均分各层。
- **精妙之处**：`num_blocks = available_memory // page_size // num_layers`（`:621`）。若连一条 `max_model_len` 的请求都放不下，不是直接报错了事，而是**二分搜索**出"当前显存能支持的最大 model_len"，在报错信息里告诉用户该调到多少（`:538-543`），把不可行配置变成可操作的提示。

---

## 6. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| prefix caching 关闭 | `kv_cache_manager.py:116-118` | `get_computed_blocks` 直接返回 `([], 0)`，不算 hash |
| prompt_logprobs 请求 | `kv_cache_manager.py:130-131` | 跳过 prefix caching（需重算 logprobs，不能复用 KV） |
| prompt 长度整除 block_size 且全命中 | `kv_cache_manager.py:133-144` | 弹掉最后一个 block hash，强制最后块 recompute（保证有 query，见 T8） |
| 命中块是驱逐候选(ref_cnt=0) | `kv_cache_manager.py:227-230` | 从可用空闲数里扣除，避免超分配 |
| 显存不足（running） | `scheduler.py:196-221` | 抢占最低优先级请求；抢到自己则放弃调度（自环退出，见 §4.2） |
| 显存不足（waiting） | `scheduler.py:332-336` | `allocate_slots` 返回 None → `break`，本步不调度 |
| 超过单请求最大块数 | `kv_cache_manager.py:257-264` | `min(..., max_num_blocks_per_req - len(req_blocks))` 截断，防 block table 越界 |
| padding token（slot<0） | `cache_kernels.cu:224-227` | kernel 直接 return，不写 cache |
| sliding window 窗口外块 | `specialized_manager.py:132-148` | `remove_skipped_blocks` 用 null_block 替换、返回待释放块 |
| reset_prefix_cache 时仍有块在用 | `block_pool.py:250-255` | 除 null_block 外还有占用块则拒绝 reset，打 warning |
| 请求 abort（分配前就结束） | `kv_cache_manager.py:307` | `req_to_blocks.pop(req_id, [])` 默认空列表，安全 |
| 混合 KV cache 类型（full+sliding） | `kv_cache_utils.py:657-681` | `unify_hybrid_kv_cache_specs` 把 sliding 全转 full（暂不支持真混合） |
| 多于 1 个 KV cache group | `kv_cache_manager.py:31-33` | assert 报错（当前不支持） |

---

## 7. 一图速查：block 分配 → kernel 寻址

```
[调度器] scheduler.py:305  get_computed_blocks ──► kv_cache_manager.py:103
            │  ① hash_request_tokens :423  (token切块→hash链, parent链式)
            │  ② specialized_manager.py:71  find_longest_cache_hit (沿链查字典,一miss即停)
            │  └─► 命中前缀块 computed_blocks (这些token不再算)
            ▼
        scheduler.py:332  allocate_slots ──► kv_cache_manager.py:163
            │  touch(命中块) :236  (ref_cnt++, 从空闲链表摘除 ← prefix共享)
            │  get_new_blocks :268 ──► block_pool.py:158 (链表头popleft, 必要时驱逐旧hash)
            │  cache_full_blocks :284 ──► block_pool.py:76 (写满块登记进 cached_block_hash_to_block)
            └─► SchedulerOutput.req_to_new_block_ids :358  [block_id...]  ──► 过界到 worker
            ▼ ──────────────────────────── worker 边界 ────────────────────────────
[worker] gpu_model_runner.py:420  block_table.append_row (block_id 写进 GPU block table 张量)
         gpu_model_runner.py:541  算 slot_mapping:
              block_table_indices = req*max_blocks + position//block_size
              physical = block_table[idx];  slot = physical*block_size + position%block_size
            ▼
[backend] flash_attn.py:286  build → FlashAttentionMetadata{block_table, slot_mapping, seq_lens}
            ├─ 写KV: flash_attn.py:460 reshape_and_cache_flash ──► cache_kernels.cu:222
            │         slot → block_idx=slot/bs, block_offset=slot%bs → 写 paged cache
            └─ 读KV: flash_attn.py:503 flash_attn_varlen_func(block_table=...)
                      └─► attention_kernels.cuh:257
                            physical_block = block_table[logical_block]   ← PagedAttention 精髓
                            addr = physical_block*stride + offset → 读 k_cache/v_cache
            ▼
[释放] scheduler.py:204 preempt / :733 finish ──► kv_cache_manager.py:298 free
         reversed(blocks) (尾块先驱逐,护前缀) ──► block_pool.py:227 decr_ref, ref_cnt==0 入空闲链表尾
```
