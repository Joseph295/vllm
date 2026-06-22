# 模块 14 · 注意力变体：SWA / Cascade / GQA·MQA —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的实现细节：SWA 的 KV block 提前释放、Cascade 的公共前缀检测与两段 attention、GQA/MQA 的 KV 头映射。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
> Python 侧以 **V1（`vllm/v1/`）** 为主；attention kernel（FlashAttention / `csrc/`）部分细节见 [模块 11](../11-gpu-kernels-memory/impl.md)。三种变体都构建在 [模块 02](../02-paged-attention-kvcache/impl.md) 的 PagedAttention / BlockPool 之上。

---

## 1. 代码地图

```
SWA（滑动窗口） —— KV cache 管理层 vllm/v1/core/
  ├─ specialized_manager.py     # SlidingWindowManager: find_longest_cache_hit(从右扫) / remove_skipped_blocks(释放窗外块)
  ├─ kv_cache_interface.py      # SlidingWindowSpec: sliding_window 字段、page_size、max_memory_usage 估算
  ├─ kv_cache_manager.py        # allocate_slots :210 调 remove_skipped_blocks → free_blocks
  └─ block_pool.py              # null_block(:57)、free_blocks(:227)、touch(:212)（与模块02共用）

SWA —— kernel 参数 vllm/v1/attention/backends/
  └─ flash_attn.py:382-385      # window_size=(sliding_window-1, 0) 传入 flash_attn_varlen_func

Cascade（级联） —— 调度 + backend + kernel
  ├─ vllm/v1/core/sched/scheduler.py:392      # num_common_prefix_blocks (ref_cnt==len(running))
  ├─ vllm/v1/core/kv_cache_manager.py:331     # get_num_common_prefix_blocks
  ├─ vllm/v1/worker/gpu_model_runner.py:617   # _compute_cascade_attn_prefix_len (块→token, 双重截断)
  └─ vllm/v1/attention/backends/flash_attn.py
        ├─ :286 build           # 构 cascade metadata (cu_prefix_query_lens / prefix_kv_lens / suffix_kv_lens)
        ├─ :553 use_cascade_attention  # 启发式 + 性能模型
        ├─ :621 cascade_attention      # 前缀kernel + 后缀kernel + merge
        └─ :706 merge_attn_states      # LSE 合并 → vllm/attention/ops/merge_attn_states.py

GQA/MQA —— 模型层 + cache 格式 + kernel
  ├─ vllm/config.py:1021        # get_num_kv_heads (TP 切分, min 1)
  ├─ vllm/model_executor/models/llama.py:120-129  # num_kv_heads 按 TP 复制/切分
  ├─ vllm/attention/layer.py:268-302  # num_queries_per_kv; SDPA fallback 的 repeat_interleave
  └─ vllm/v1/attention/backends/flash_attn.py
        ├─ :56-64  get_kv_cache_shape  # [2, num_blocks, block_size, num_kv_heads, head_size]
        └─ :392-393 num_queries_per_kv = num_heads // num_kv_heads
```

---

## 2. 调用链一：SWA 的 KV block 管理与释放

> **这条链在干嘛**：SWA 的精髓不在 kernel（kernel 只是加个 `window_size` 参数屏蔽窗外），而在 KV 管理——既然位置 i 的 token 永远不会再被 i+W 之后的 query 看到，那它所在的 block 就可以提前释放、还给别的请求用。这条链追的就是"窗口外 block 怎么被摘掉、被 `null_block` 占位，以及 SWA 下 prefix cache 命中怎么重新定义"。

追踪一条 SWA 请求的一个调度步里，"窗口外 block 如何被提前释放、命中如何重定义"。

### A. SWA 的 cache 格式声明 —— `SlidingWindowSpec`

`vllm/v1/kv_cache_interface.py:84` 定义 SWA 层的 KV cache 格式：
```python
@dataclass
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int

    def __post_init__(self):
        assert not self.use_mla, "MLA is not supported for sliding window"
```
- `:91-93` `type_id` 含 `sliding_window`，使 SWA 层与 full attention 层被分到不同 KV cache group。
- `:95-111` `max_memory_usage_bytes` 是 SWA 省显存的账面体现：
  ```python
  num_tokens = min(self.sliding_window - 1 + max_num_batched_tokens, max_model_len)
  return (cdiv(num_tokens, self.block_size) + 1) * self.page_size_bytes
  ```
  > 不按 `max_model_len` 估，而是按 `W-1 + 一批 token`——这就是 SWA 把 KV 占用从 O(L) 压到 O(W) 在显存预算上的兑现。`+1` 是因为窗口可能不从块边界开始（`:107-110` 注释举例：block_size=4、6 token 的窗口需 2 块）。

### B. SWA 的 prefix cache 命中 —— 从右往左凑连续窗口块

`vllm/v1/core/specialized_manager.py:90` `SlidingWindowManager.__init__`：
```python
self.sliding_window = kv_cache_spec.sliding_window
# -1 since the input token itself is also included in the window
self.sliding_window_contiguous_blocks = cdiv(
    (kv_cache_spec.sliding_window - 1), self.block_size)   # 命中所需连续块数
self._null_block = block_pool.null_block
```

`specialized_manager.py:102` `find_longest_cache_hit` —— **SWA 命中判据**（对照 `FullAttentionManager.find_longest_cache_hit:71` 的从左连续命中）：
```python
computed_blocks = [self._null_block] * len(block_hashes)   # 默认全 null
num_contiguous_blocks = 0
# Search from right to left and early stop when a match is found.
for i in range(len(block_hashes) - 1, -1, -1):
    if cached_block := self.block_pool.get_cached_block(block_hashes[i]):
        computed_blocks[i] = cached_block
        num_contiguous_blocks += 1
        if num_contiguous_blocks >= self.sliding_window_contiguous_blocks:
            del computed_blocks[i + num_contiguous_blocks:]   # 裁掉尾部多余
            return computed_blocks
    else:
        num_contiguous_blocks = 0   # 断了就重新计数
del computed_blocks[num_contiguous_blocks:]
return computed_blocks
```
> **解读**：full attention 只需从头连续命中；SWA 只需窗口内的 KV，所以命中变成"找到 `sliding_window_contiguous_blocks` 个**连续**命中块"。从右往左扫、断了就把计数清零（`:126`）。例子（`:121` 注释）：window 块数=2 时 `[NULL,NULL,8,3,NULL,9]` 命中段是 `[NULL,NULL,8,3]`。窗口外的命中块即便 hash 存在也用 `null_block` 占位（永不会被 attend）。`get_cached_block` 查的是 [模块 02](../02-paged-attention-kvcache/impl.md) 的 `cached_block_hash_to_block` 字典。

### C. 窗口外 block 的提前释放 —— `remove_skipped_blocks`

`vllm/v1/core/kv_cache_manager.py:204` `allocate_slots` 在分配新块**之前**先释放窗外块：
```python
# Free the blocks that are skipped during the attention computation
# (e.g., tokens outside the sliding window).
# We can do this even if we cannot schedule this request ...
# Should call this function before allocating new blocks to reduce evictions.
removed_blocks = self.specialized_manager.remove_skipped_blocks(
    req_blocks, request.num_computed_tokens)
self.block_pool.free_blocks(removed_blocks)
```
> **解读**：释放放在分配前，目的是"先腾出窗外块、再算新块需求"，减少触发驱逐（`:206-209` 注释）。`free_blocks` 是 [模块 02](../02-paged-attention-kvcache/impl.md) 的同一个释放入口——SWA **完全复用**空闲链表/ref_cnt，不另起炉灶。

`vllm/v1/core/specialized_manager.py:132` `SlidingWindowManager.remove_skipped_blocks`：
```python
last_useful_token = num_computed_tokens - self.sliding_window + 1
last_useful_block = last_useful_token // self.block_size

removed_blocks: list[KVCacheBlock] = []
for i in range(last_useful_block - 1, -1, -1):
    if blocks[i] == self._null_block:
        # already null → 它之前的也都被前几次调用置 null 了
        break
    removed_blocks.append(blocks[i])
    blocks[i] = self._null_block      # ← 用 null_block 顶替逻辑槽
return removed_blocks
```
> **解读**：算出"最后一个仍在窗口内的块" `last_useful_block`，把它之前的块从后往前逐个：①加入待释放列表、②在 block table 里**置为 `null_block`**。置 null 是关键——block table 逻辑下标必须连续（worker 算 slot_mapping、kernel 按 `position//block_size` 索引都依赖连续性），不能稀疏。遇到已是 null 的块就 break（`:141-145`：之前的调用已把更早的块置 null，无需重复）。返回的块由上层 `free_blocks` 真正释放，**本函数不释放**（接口注释 `:54-58` 明确）。`FullAttentionManager.remove_skipped_blocks:84` 永远返回 `[]`（full 无窗外块）。

### D. kernel 侧的窗口屏蔽

`vllm/v1/attention/backends/flash_attn.py:382` `FlashAttentionImpl.__init__`：
```python
if sliding_window is None:
    self.sliding_window = (-1, -1)        # 无窗口
else:
    self.sliding_window = (sliding_window - 1, 0)   # 左看 W-1，右看 0(causal)
```
`:515` 传入 kernel：`window_size=self.sliding_window`。
> **解读**：被 `null_block` 占位的窗外槽即便物理上 block 0 有内容，也因 `window_size` mask 而**永不被 attend**——SWA 的正确性由"管理层释放 + kernel 屏蔽"双重保证。`null_block`（`block_pool.py:57`）恒为 block 0、ref_cnt 不维护，各处 `block != self.null_block` 判断（`block_pool.py:223,238`）防止它被误 free。

---

## 3. 调用链二：Cascade 的公共前缀检测与两段 attention

> **这条链在干嘛**：当批内多条请求共享一段公共前缀（如同一 system prompt），不想让每条都把前缀 KV 重读一遍。这条链追的是：①调度器怎么靠 ref_cnt 检测出"所有 running 请求都共享的前缀块数"；②model runner 怎么把"块数"换成"安全的 token 长度"并判断划不划算；③backend 怎么用两个 kernel（前缀算一次 + 各自后缀）+ LSE 合并把结果还原成等价于一次算。

追踪"公共前缀如何被检测、如何只算一次、如何与后缀合并"。

### A. 调度层检测公共前缀块数

`vllm/v1/core/sched/scheduler.py:392` 每个调度步算公共前缀块数：
```python
num_common_prefix_blocks = 0
if self.running:
    any_request = self.running[0]
    num_common_prefix_blocks = (
        self.kv_cache_manager.get_num_common_prefix_blocks(
            any_request, len(self.running)))
```
随后打包进 `SchedulerOutput`（`scheduler.py:435` `num_common_prefix_blocks=...`），过界到 worker。链回 [模块 01](../01-scheduler-batching/design.md) 的调度产物。

`vllm/v1/core/kv_cache_manager.py:331` `get_num_common_prefix_blocks` —— **判据：ref_cnt == running 数**：
```python
assert request.status == RequestStatus.RUNNING
blocks = self.req_to_blocks[request.request_id]
num_common_blocks = 0
for block in blocks:
    if block.ref_cnt == num_running_requests:   # 所有 running 请求都指向它
        num_common_blocks += 1
    else:
        break                                   # 第一个非公共块即停
return num_common_blocks
```
> **解读**：不需要逐请求比对 token——直接看块的 `ref_cnt`。因为 prefix caching 命中时每个共享者都 `touch` 过该块一次（[模块 02](../02-paged-attention-kvcache/impl.md) T3/T13），ref_cnt 精确等于共享者数。引用计数被复用成"公共性检测器"。**边界**（`:343-357` 注释）：`len(running) ≥ 本步调度数`，可能有未调度的 running 请求不共享前缀 → 保守返回 0。

### B. model runner：块数 → token 长度，双重截断

`vllm/v1/worker/gpu_model_runner.py:572` 调用点：
```python
common_prefix_len = 0
if self.cascade_attn_enabled:       # not disable_cascade_attn (:140)
    common_prefix_len = self._compute_cascade_attn_prefix_len(
        num_scheduled_tokens, scheduler_output.num_common_prefix_blocks)
```

`vllm/v1/worker/gpu_model_runner.py:617` `_compute_cascade_attn_prefix_len`：
```python
common_prefix_len = num_common_prefix_blocks * self.block_size
if common_prefix_len == 0:
    return 0                                    # 常见情形，早返回
...
num_reqs = len(num_scheduled_tokens)
common_prefix_len = min(                        # ① 截到批内最小 computed_tokens
    common_prefix_len,
    self.input_batch.num_computed_tokens_cpu[:num_reqs].min())
common_prefix_len = (common_prefix_len // self.block_size *
                     self.block_size)           # ② 截到 block_size 整数倍
use_cascade = self.attn_backend.use_cascade_attention(
    common_prefix_len=common_prefix_len,
    query_lens=num_scheduled_tokens,
    num_query_heads=self.num_query_heads,
    num_kv_heads=self.num_kv_heads,
    use_alibi=self.use_alibi,
    use_sliding_window=self.window_size is not None,
    num_sms=self.num_sms)
return common_prefix_len if use_cascade else 0
```
> **解读**：`:652-682` 的长注释用 reqA/reqB/reqC 三例推导出**为何必须截到 min computed_tokens**——前缀 kernel 用 `causal=False` 无 mask 双向 attention，若公共前缀越过某请求的 computed 边界，该请求 query 会 attend 到本该被 causal mask 屏蔽的 token，结果错误。实现里甚至不加注释提到的 +1（`:670-682`），为的是保证后缀 kernel 永不收到空输入（恒用两个 kernel）。`use_sliding_window=self.window_size is not None` 用的是 `#14896` 引入的 `window_size`（= `sliding_window or interleaved_sliding_window`），确保交错 SWA 模型不误用 cascade。

`vllm/v1/attention/backends/flash_attn.py:553` `use_cascade_attention` —— **支持性检查 + 收益启发式**：
```python
if common_prefix_len < 256:           # 前缀太短不划算
    return False
if use_alibi or use_sliding_window:   # cascade kernel 不支持这俩
    return False
num_reqs = len(query_lens)
if num_reqs < 8:                      # 请求太少不划算
    return False
num_queries_per_kv = num_query_heads // num_kv_heads
use_flash_decoding = (num_queries_per_kv > 1 and not use_sliding_window
                      and not use_alibi and np.all(query_lens == 1))
if not use_flash_decoding:
    return True                       # 普通 attention 不用 FlashDecoding → cascade 更省带宽
# 否则用粗糙性能模型对比 cascade vs FlashDecoding 的 CTA 波数（:601-618）
return cascade_time < flash_decoding_time
```
> **解读**：先做"支持性检查"（ALiBi/SWA 直接 False，否则会触发 `cascade_attention:643-646` 的 assert），再做"收益启发式"。阈值 256/8 标了 TODO 待调（`:569,578`），性能模型自承"很粗糙"（`:599`）。注意 `num_queries_per_kv`（GQA 量）在这里决定是否走 FlashDecoding——三种变体在此交汇。

### C. 构建 cascade metadata

`vllm/v1/attention/backends/flash_attn.py:321` `build`：
```python
use_cascade = common_prefix_len > 0
if use_cascade:
    cu_prefix_query_lens = torch.tensor([0, num_actual_tokens], ...)  # 所有query拼成一个超batch
    prefix_kv_lens = torch.tensor([common_prefix_len], ...)           # 前缀KV长度(单一)
    suffix_kv_lens = (self.runner.seq_lens_np[:num_reqs] - common_prefix_len)  # 各请求后缀KV长度
    suffix_kv_lens = torch.from_numpy(suffix_kv_lens).to(...)
```
> **解读**：cascade 的精髓在这三个长度张量。`cu_prefix_query_lens=[0, num_actual_tokens]` 把**所有请求的 query 当成一条序列**喂给前缀 kernel（前缀对谁都一样，无需分 batch）；`prefix_kv_lens` 只有一个值（共享前缀长度）；`suffix_kv_lens` 每请求一个（各自后缀长度 = 总 seq_len − 公共前缀）。

### D. 两段 attention + LSE 合并

`vllm/v1/attention/backends/flash_attn.py:485` `forward` 分流：
```python
if not attn_metadata.use_cascade or use_local_attn:
    flash_attn_varlen_func(...)      # 普通路径
    return output
# Cascade attention (rare case).
cascade_attention(output[:num_actual_tokens], query[:num_actual_tokens], ...)
```

`vllm/v1/attention/backends/flash_attn.py:621` `cascade_attention`：
```python
assert alibi_slopes is None, "Cascade attention does not support ALiBi."
assert sliding_window == (-1, -1), "Cascade attention does not support sliding window."
assert common_prefix_len % block_size == 0
num_common_kv_blocks = common_prefix_len // block_size
assert num_common_kv_blocks > 0

# ① 前缀 kernel：所有 query × 共享前缀 KV，读一次，无 mask
prefix_output, prefix_lse = flash_attn_varlen_func(
    q=query, k=key_cache, v=value_cache,
    cu_seqlens_q=cu_prefix_query_lens,      # [0, num_tokens] 拼成一条
    seqused_k=prefix_kv_lens,
    max_seqlen_k=common_prefix_len,
    causal=False,                           # ← 前缀对所有 query 无 causal mask
    block_table=block_table[:1],            # ← 任一请求的前缀块即可(都相同)
    return_softmax_lse=True, ...)

# ② 后缀 kernel：各请求 query × 各自后缀 KV
suffix_output, suffix_lse = flash_attn_varlen_func(
    q=query, k=key_cache, v=value_cache,
    cu_seqlens_q=cu_query_lens,             # 各请求分开
    seqused_k=suffix_kv_lens,
    max_seqlen_k=max_kv_len - common_prefix_len,
    causal=True,                            # ← 后缀仍 causal
    block_table=block_table[:, num_common_kv_blocks:],   # ← 跳过公共块
    return_softmax_lse=True, ...)

# ③ LSE 合并
merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
```
> **解读**：这就是 design §3.2 的代码兑现。前缀 kernel 用 `block_table[:1]`（公共块对所有请求物理相同，取第 0 行即可）+ `causal=False`（前缀全可见）把前缀 KV **只读一次**；后缀 kernel 用 `block_table[:, num_common_kv_blocks:]` 切掉公共块、`causal=True` 各算各的。两段都 `return_softmax_lse=True` 拿回 log-sum-exp。

`vllm/attention/ops/merge_attn_states.py:9` `merge_attn_states` —— **LSE 合并**：
```python
if (current_platform.is_cuda() and supported_dtypes(output) and supported_headdim(output)):
    from vllm._custom_ops import merge_attn_states         # CUDA kernel (#16173, 3x)
    return merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse)
else:
    from vllm.attention.ops.triton_merge_attn_states import merge_attn_states  # Triton fallback
    return merge_attn_states(...)
```
> **解读**：按 `O = (O_P·e^{lse_P} + O_S·e^{lse_S}) / (e^{lse_P}+e^{lse_S})` 合并，结果逐 bit 等价于一次性算（online-softmax 可结合性）。CUDA kernel 走 128b 向量化（`:23-30` 注释，要求 headdim 对齐），FP8/非对齐退回 Triton。kernel 线程级细节见 [模块 11](../11-gpu-kernels-memory/impl.md)。

---

## 4. 调用链三：GQA/MQA 的 KV 头映射

> **这条链在干嘛**：GQA/MQA 在 vLLM 里几乎"免费"，因为它只改一个量——`num_kv_heads`。这条链追的是这个量怎么从 HF config 读出来、按 TP 切分/复制，一路把 cache 张量第 4 维缩小（显存收益的全部来源），最后让 kernel 内部把多个 query 头映射到同一个 KV 头去读（不物理复制 KV，否则就白省了）。

追踪 `num_kv_heads` 如何从 config 一路压缩到 cache 张量与 kernel。

### A. config 算出每 GPU 的 KV 头数

`vllm/config.py:1021` `get_num_kv_heads`：
```python
def get_num_kv_heads(self, parallel_config):
    if self.use_mla:
        return 1                       # MLA decode 退化成 MQA
    total_num_kv_heads = self.get_total_num_kv_heads()
    # TP 切分；KV 头少于 TP 时复制，保证每 GPU 至少 1 个
    return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```
`:967` `get_total_num_kv_heads` 从 HF config 各种字段读（`num_key_value_heads`、`multi_query` 单头、ChatGLM `multi_query_group_num` 等，`:1003-1019`）。非 GQA 模型则 `num_kv_heads == num_attention_heads`（`:1017-1019`）。

`vllm/model_executor/models/llama.py:120` 模型层据此切/复制：
```python
self.total_num_kv_heads = num_kv_heads
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0     # 切分
else:
    assert tp_size % self.total_num_kv_heads == 0     # 复制(KV头少于TP)
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
self.kv_size = self.num_kv_heads * self.head_dim       # K/V 投影输出维度
```
> **解读**：GQA/MQA 的 KV 头数比 query 头数少，TP 切分时若 `num_kv_heads < tp_size` 就**复制** KV 头到各 GPU（每 GPU 至少 1 个），否则正常切。`kv_size` 据此变小，QKV 投影直接产出更窄的 K/V。

### B. cache 格式：`num_kv_heads` 维变小

`vllm/v1/attention/backends/flash_attn.py:56` `get_kv_cache_shape`：
```python
return (2, num_blocks, block_size, num_kv_heads, head_size)
#                                   ^^^^^^^^^^^^ ← GQA 在这里省显存
```
`vllm/v1/kv_cache_interface.py:64` `page_size_bytes`：
```python
coef = 1 if self.use_mla else 2
return coef * self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)
```
> **解读**：GQA 的全部显存收益就在 cache 张量第 4 维 `num_kv_heads`（而非 `num_heads`）。`page_size_bytes` 据此缩小 `num_queries_per_kv` 倍，[模块 02](../02-paged-attention-kvcache/impl.md) 的 KV manager 拿到更小的"每页字节数"后，分配/命中/驱逐链路**完全无感**——manager 不认头数、只认页大小，这是 GQA 零侵入的根因。

### C. kernel：query 头映射到共享 KV 头

`vllm/v1/attention/backends/flash_attn.py:392` `FlashAttentionImpl.__init__`：
```python
assert self.num_heads % self.num_kv_heads == 0
self.num_queries_per_kv = self.num_heads // self.num_kv_heads   # 每个KV头被几个query头共享
```
forward 里 query 形状 `[num_tokens, num_heads, head_size]`、KV cache `[..., num_kv_heads, head_size]`，直接传给 `flash_attn_varlen_func`（`:503`）——**FlashAttention/PagedAttention kernel 原生支持** query 头数 ≠ KV 头数，内部让 `num_queries_per_kv` 个连续 query 头索引同一个 KV 头，**不物理复制 KV**。

唯一显式复制 KV 的是非 paged 的 SDPA fallback 路径，`vllm/attention/layer.py:302`：
```python
if (num_repeat := self.num_queries_per_kv) > 1:
    # Handle MQA and GQA
    key = torch.repeat_interleave(key, num_repeat, dim=2)
    value = torch.repeat_interleave(value, num_repeat, dim=2)
```
> **解读**：主流 paged 路径靠 kernel 索引共享、零复制（否则抵消显存收益）；只有 encoder-only 等走 PyTorch SDPA 的 fallback 才 `repeat_interleave` 把 KV 头扩回 query 头数（因为 SDPA 要求 Q/K/V 头数相等）。这是 GQA 在 vLLM 里**唯一**会物理复制 KV 的地方，且不在 decode 热路径。

---

## 5. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · SWA 命中从右往左扫，full 从左往右扫
- **代码**：`specialized_manager.py:113` SWA 反向 range；`:74` full 正向。
- **精妙之处**：full attention 要"最长前缀"（从头连续）；SWA 只要"窗口内连续 `cdiv(W-1,bs)` 块"。窗口贴在序列**右端**，所以从右扫、凑够即停。一旦断了 `num_contiguous_blocks=0` 重新计数（`:126`），保证返回的是**连续**段而非零散命中。同一抽象方法两种实现，精准编码了两种 attention 的命中语义差异。

### T2 · 窗口外块用 null_block 占位而非删除
- **代码**：`specialized_manager.py:147` `blocks[i] = self._null_block`。
- **精妙之处**：block table 逻辑下标必须连续（worker slot_mapping、kernel `position//block_size` 都依赖），不能稀疏。释放窗外块后留下的"逻辑存在、物理已弃"的槽，统一填 `null_block`（恒为 block 0）。kernel 的 `window_size` mask 保证这些 null 槽永不被 attend——管理层占位 + kernel 屏蔽双保险。

### T3 · 释放放在分配新块之前
- **代码**：`kv_cache_manager.py:204-212`，`remove_skipped_blocks` → `free_blocks` 在 `get_new_blocks` 之前。
- **精妙之处**：先把窗外块还给空闲链表，再算"还差几块"。这样这条 SWA 请求自己释放的块能立刻拿来给自己用，最大限度减少触发对**别人**块的驱逐（`:206-209` 注释明示）。释放与分配的顺序本身是一个优化。

### T4 · SWA 的 max_memory 按窗口估、不按 max_model_len
- **代码**：`kv_cache_interface.py:104` `min(W-1+max_num_batched_tokens, max_model_len)`。
- **精妙之处**：full attention 的 `max_memory_usage_bytes` 按 `max_model_len` 估（`:79-81`）；SWA 只按"窗口 + 一批新 token"估，且 `+1` 容纳窗口不对齐块边界。这让 SWA 模型在相同显存下能放下**多得多**的并发请求——省显存的账面落点。

### T5 · 公共前缀判据复用 ref_cnt，零额外比对
- **代码**：`kv_cache_manager.py:373` `block.ref_cnt == num_running_requests`。
- **精妙之处**：判断"所有 running 请求是否共享某块"不逐 token 比对，直接看 ref_cnt——因为 prefix 命中时每个共享者 `touch` +1（[模块 02](../02-paged-attention-kvcache/impl.md) T3）。引用计数被复用成"公共性检测器"。第一个非公共块即停（`:375-376`），因为后续块不可能比前缀更公共。

### T6 · cascade 前缀长度必须截到批内最小 computed_tokens
- **代码**：`gpu_model_runner.py:684-686` `min(..., num_computed_tokens_cpu[:num_reqs].min())`。
- **精妙之处**：前缀 kernel 用 `causal=False`（无 mask）。若公共前缀越过某请求的 computed 边界，该请求 query 会 attend 到本该被 causal 屏蔽的未来 token → 静默错误。`:652-682` 用三个具体请求推导出这条截断，是 cascade 正确性的命门。还故意不加 +1，为保证后缀 kernel 永不收空输入。

### T7 · cascade 前缀 kernel 把所有 query 拼成一条序列
- **代码**：`flash_attn.py:323` `cu_prefix_query_lens = [0, num_actual_tokens]`。
- **精妙之处**：前缀对所有请求是同一份 KV，所以无需分 batch——把全 batch 的 query 当一条超长序列、对前缀 KV 读**一次**。这正是"前缀只算一次"的实现：N 条请求的前缀 attention 合并成一个 kernel call，前缀 KV 带宽从 N× 降到 1×。

### T8 · cascade 前缀用 block_table[:1]，后缀用 block_table[:, num_common:]
- **代码**：`flash_attn.py:667` `block_table[:1]`；`:693` `block_table[:, num_common_kv_blocks:]`。
- **精妙之处**：公共块对所有请求物理相同，取**任一行**（第 0 行）即可作前缀 KV 索引；后缀则把每行**列方向**切掉前 `num_common_kv_blocks` 块。一份 block table 两种切法，无需重建——cascade 不碰 cache 布局的体现。

### T9 · LSE 合并保证逐 bit 等价
- **代码**：`flash_attn.py:706` `merge_attn_states`；公式 `O=(O_P·e^lse_P + O_S·e^lse_S)/(e^lse_P+e^lse_S)`。
- **精妙之处**：把 KV 切成前缀/后缀两个不相交子集分别算 partial attention，再用各自的 log-sum-exp 合并，结果与一次性算**数学上恒等**（flash-attn online-softmax 的分块可结合性）。所以 cascade 是纯性能优化、零精度损失——这是它敢默认开启的底气。

### T10 · cascade 是"两 kernel + merge"，但仍恒用两个 kernel
- **代码**：`gpu_model_runner.py:670-682` 注释；`cascade_attention:652` `assert num_common_kv_blocks > 0`。
- **精妙之处**：实现刻意保证后缀 kernel 永不为空（前缀截断不加 +1，留至少一块给后缀）。否则"某请求全部 token 都在公共前缀里"会让后缀 kernel 收到空输入——当前实现不支持。用"恒两个 kernel"的约束换实现简单。

### T11 · GQA 对 KV manager 零侵入，只缩 page_size_bytes
- **代码**：`kv_cache_interface.py:64-69`；manager 全程只用 `page_size_bytes`。
- **精妙之处**：GQA 把 cache 张量 `num_kv_heads` 维缩小，page 变小，KV manager（block/命中/驱逐）**完全不知道有没有 GQA**。这层"manager 只认页大小、不认头数"的抽象边界，让 GQA 能和 SWA/Cascade/prefix-cache 任意叠加——正交性是抽象设计对的回报。

### T12 · GQA paged 路径零复制，仅 SDPA fallback repeat_interleave
- **代码**：`layer.py:302` `repeat_interleave`（仅 fallback）；FA/PA kernel 原生支持。
- **精妙之处**：若把 KV 物理复制 `num_queries_per_kv` 份喂 kernel，会抵消 GQA 的显存收益。所以主流 paged 路径靠 kernel 内 `head_kv = head_q // num_queries_per_kv` 索引共享、不复制；只有要求 Q/K/V 头数相等的 PyTorch SDPA fallback 才显式扩展，且不在 decode 热路径。

### T13 · MLA decode 退化成 MQA（num_kv_heads=1）
- **代码**：`config.py:1023-1025` `if self.use_mla: return 1`。
- **精妙之处**：MLA（Multi-head Latent Attention，DeepSeek）decode 时只存一个 latent 向量，等价于 `num_kv_heads=1` 的 MQA。vLLM 直接让 `get_num_kv_heads` 对 MLA 返回 1，复用 GQA/MQA 的整套 KV 头映射机制，且 `page_size_bytes` 的 `coef=1`（`kv_cache_interface.py:67`，MLA 只存单 latent 而非 K+V 两份）。

### T14 · 交错 SWA 用 window_size 兜住非标准命名
- **代码**：`gpu_model_runner.py`（`#14896`）`window_size = sliding_window or interleaved_sliding_window`。
- **精妙之处**：Gemma2/Cohere2/Llama4 把窗口放在 `interleaved_sliding_window` 而非 `sliding_window`（后者为 None）。漏读会让 cascade 误启用、SWA 检测失效。`window_size` 兜底合并两个字段，并据此 `use_sliding_window`（`:696`）正确关 cascade。教训：模型的非标准 config 命名会静默绕过变体检测。

### T15 · merge_attn_states 双后端：CUDA 优先、Triton 兜底
- **代码**：`merge_attn_states.py:32-41`。
- **精妙之处**：CUDA kernel（`#16173`，3x）走 128b 向量化，要求 headdim 按 dtype 对齐（fp32 %4、半精 %8，`:26-30`）且非 FP8；不满足则退回 Triton。一行 dtype/headdim 检查决定走哪条，保证正确性优先于性能。还要守 CUDA graph 兼容（`#16693`，见 [模块 07](../07-cuda-graph-compile/design.md)）。

---

## 6. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| SWA 命中不足窗口块数 | `specialized_manager.py:127-130` | 即便 `< sliding_window_contiguous_blocks` 也保留已命中的连续段，不强求凑满 |
| SWA 窗外块已是 null | `specialized_manager.py:141-145` | break（更早的块已被前次调用置 null，无需重复释放） |
| SWA + MLA | `kv_cache_interface.py:88-89` | `assert not use_mla`（SWA 不支持 MLA） |
| cascade 公共前缀 = 0 | `gpu_model_runner.py:640-642` | 早返回 0（常见情形，避免无谓计算） |
| cascade 前缀长度非块整数倍 | `flash_attn.py:650` `assert common_prefix_len % block_size == 0` | 已在 runner `:688` 截到整数倍，此处守不变量 |
| cascade + ALiBi | `flash_attn.py:575,643` | `use_cascade_attention` 早返回 False；kernel assert 兜底（`#15211`） |
| cascade + sliding window | `flash_attn.py:575,645-646` | 同上，不支持 → 不启用；交错 SWA 经 `window_size` 检测（`#14896`） |
| cascade 请求数 < 8 或前缀 < 256 | `flash_attn.py:572,580` | 不划算，返回 False，走普通 attention |
| cascade 被全局禁用 | `gpu_model_runner.py:140` `cascade_attn_enabled` | `disable_cascade_attn` → `common_prefix_len=0`（`#15243`） |
| 公共前缀 > 部分 running 未调度 | `kv_cache_manager.py:353-357` | 保守返回 0（无法检测未调度 running 是否共享） |
| local/chunked attn 接近 max context | `flash_attn.py:267` | `block_indices.clip(max=block_table.shape[1]-1)` 防越界（`#16209`） |
| GQA KV 头少于 TP size | `llama.py:125-128` | 复制 KV 头到各 GPU（`tp_size % num_kv_heads == 0`），每 GPU ≥1 头 |
| MLA（退化 MQA） | `config.py:1023-1025` | `num_kv_heads` 强制为 1，`page_size` coef=1 |

---

## 7. 一图速查：三条调用链

```
【SWA：窗外 block 提前释放】
 allocate_slots  kv_cache_manager.py:204
   └─ remove_skipped_blocks  specialized_manager.py:132
        last_useful_block = (num_computed_tokens - W + 1)//block_size
        for i in [last_useful_block-1 .. 0]:  blocks[i] = null_block; 收集
   └─ free_blocks  block_pool.py:227 (复用模块02空闲链表, decr_ref)
 命中: find_longest_cache_hit  specialized_manager.py:102 (从右扫,凑 cdiv(W-1,bs) 连续块)
 kernel: flash_attn.py:515  window_size=(W-1, 0)  屏蔽窗外

【Cascade：公共前缀算一次 + LSE 合并】
 scheduler.py:392  num_common_prefix_blocks
   └─ kv_cache_manager.py:331  ref_cnt == len(running) → 公共块数
 SchedulerOutput → gpu_model_runner.py:617  _compute_cascade_attn_prefix_len
        ① min(批内最小 computed_tokens)   ② 截 block_size 整数倍
        ③ use_cascade_attention :553 (前缀≥256 & 请求≥8 & 非ALiBi/SWA & 性能模型)
   └─ flash_attn.py:286 build → metadata{cu_prefix_query_lens, prefix_kv_lens, suffix_kv_lens}
   └─ flash_attn.py:621 cascade_attention
        ① 前缀: 所有query拼一起 × block_table[:1] (causal=False) → (O_P, lse_P)
        ② 后缀: 各query × block_table[:, num_common:] (causal=True) → (O_S, lse_S)
        ③ merge_attn_states :706  O=(O_P·e^lse_P+O_S·e^lse_S)/(e^lse_P+e^lse_S)

【GQA/MQA：多 query 头共享 KV 头】
 config.py:1021  get_num_kv_heads (TP 切分/复制, min 1; MLA→1)
   └─ llama.py:120-129  num_kv_heads, kv_size 变窄
   └─ kv_cache_interface.py:64  page_size_bytes ∝ num_kv_heads (KV manager 无感)
   └─ flash_attn.py:56  cache=[2,num_blocks,block_size,num_kv_heads,head_size]
   └─ flash_attn.py:393  num_queries_per_kv = num_heads//num_kv_heads
        kernel 内 head_kv = head_q // num_queries_per_kv (零复制)
        仅 SDPA fallback layer.py:302 repeat_interleave 显式扩展
```
