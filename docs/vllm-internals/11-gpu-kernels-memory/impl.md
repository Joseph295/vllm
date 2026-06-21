# 模块 11 · GPU 执行底层：Kernel 与显存 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照 CUDA/C++ 源码**的实现细节：从 Python `_custom_ops.py` 绑定 → C++ launcher → CUDA kernel 的完整链路，深入 `csrc/**` 的 `.cu`/`.cuh`。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
> CUDA kernel（`csrc/`）**V0/V1 共用**；显存 profiling / KV 初始化以 **V1**（`vllm/v1/worker/`）为主。本模块是 [模块 02](../02-paged-attention-kvcache/impl.md)（KV 软件管理）/ [模块 03](../03-distributed-parallel/impl.md)（并行）/ [模块 07](../07-cuda-graph-compile/impl.md)（graph）的 GPU 底层延伸，概念处指回不重复。

---

## 1. 代码地图

```
PagedAttention kernel   csrc/attention/                       （V0/V1 共用）
  ├─ attention_kernels.cuh      # ★核心 paged_attention_kernel（device 函数）+ v1/v2 global 包装 + v2 reduce kernel
  ├─ paged_attention_v1.cu      # v1 launcher：单 kernel 版（grid.z=1）
  ├─ paged_attention_v2.cu      # v2 launcher：split-KV（grid.z=分区数）+ reduce kernel
  ├─ attention_utils.cuh        # Qk_dot：Q·K 点积 + 线程组内 shuffle reduce
  ├─ attention_generic.cuh      # Vec / FloatVec 模板、zero()
  ├─ dtype_float16/bfloat16/float32/fp8.cuh   # 各 dtype 的 Vec 特化、mul/fma/sum/向量转换
  └─ merge_attn_states.cu       # cascade / split-KV 局部 attention 合并（128bit pack）

KV cache 写入 / 拷贝   csrc/cache_kernels.cu                  （V0/V1 共用）
  ├─ reshape_and_cache_kernel         # 经典 paged 布局写入
  ├─ reshape_and_cache_flash_kernel   # flash 规整布局写入（V1 默认）
  ├─ concat_and_cache_mla_kernel      # MLA 联合 KV 写入
  └─ copy_blocks_kernel / swap_blocks # block 拷贝（CoW/PD）、设备间换页

custom all-reduce   csrc/custom_all_reduce.cu / .cuh          （TP 用，见 模块 03）
  ├─ CustomAllreduce 类           # IPC handle 管理、1/2-stage 分派、graph buffer 注册
  ├─ cross_device_reduce_1stage   # one-shot：直接 P2P 读所有 rank 求和
  ├─ cross_device_reduce_2stage   # two-shot：reduce-scatter + all-gather
  └─ barrier_at_start / barrier_at_end   # 自旋 flag 跨卡同步

量化 GEMM（择要）   csrc/quantization/                        （见 模块 08）
  ├─ permute_cols.cu              # 权重列离线重排（dequant-fuse 前置）
  └─ marlin / gptq_marlin/        # int4×fp16 GEMM kernel

Python 绑定          vllm/_custom_ops.py                       注册见 csrc/torch_bindings.cpp
  ├─ paged_attention_v1/v2  (:40 / :69)      → torch.ops._C.*
  ├─ reshape_and_cache(_flash) (:1313/:1328) → torch.ops._C_cache_ops.*
  ├─ copy_blocks (:1357) / merge_attn_states (:142)
  └─ paged_attn.py:128  use_v1/v2 启发式选择

显存 profiling / KV 初始化   vllm/v1/worker/
  ├─ gpu_worker.py:139  determine_available_memory  # 跑 dummy forward 量峰值
  ├─ gpu_model_runner.py:1628  initialize_kv_cache  # 切 block、torch.zeros 分配 KV 张量
  ├─ gpu_model_runner.py:1600  capture_model        # CUDA graph 捕获
  └─ gpu_model_runner.py:464   _prepare_inputs       # pinned + non_blocking H2D
```

---

## 2. 端到端调用链：从 Python 绑定到 CUDA kernel（PagedAttention 主线）

追踪一次 decode attention：Python 决定 v1/v2 → C++ launcher 配 grid/shared mem → CUDA kernel 算。

### 阶段 A：Python 侧选 v1 还是 v2

**A1.** `vllm/attention/ops/paged_attn.py:128` —— **启发式选择**
```python
max_num_partitions = (max_seq_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE  # _PARTITION_SIZE=512
use_v1 = (max_seq_len <= 8192
          and (max_num_partitions == 1 or num_seqs * num_heads > 512))
```
> 解读：分区数=1（短序列）或 `num_seqs*num_heads` 已经够大（并行度够）就用 V1，省掉 reduce 开销；长序列（>8192，单 block 装不下 shared memory）或并行度不足时用 V2 沿 KV 维切分。`_PARTITION_SIZE=512` 必须和 C++ 侧 `paged_attention_v2.cu:51` 的默认值一致（`:14` 注释强调）。

**A2.** V2 路径要预分配三个临时张量（`paged_attn.py:157-167`）：`tmp_output[num_seqs,num_heads,max_num_partitions,head_size]`、`exp_sums` / `max_logits`[num_seqs,num_heads,max_num_partitions]`。这些就是各 partition 局部 softmax 结果的落点。

**A3.** Python 绑定 `vllm/_custom_ops.py:40`（v1）/ `:69`（v2）只是转发到 `torch.ops._C.paged_attention_v1/v2`（注册见 `csrc/torch_bindings.cpp:51` / `:65`）。

### 阶段 B：C++ launcher 配 grid 与 shared memory

**B1.** `csrc/attention/paged_attention_v1.cu:52` `paged_attention_v1_launcher` —— **grid 与 shared mem 计算**
```cpp
dim3 grid(num_heads, num_seqs, 1);          // V1: grid.z = 1
dim3 block(NUM_THREADS);                     // 默认 128 线程 = 4 warps
int padded_max_seq_len = DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
int logits_size  = padded_max_seq_len * sizeof(float);     // shared mem 放每个 KV 的 logit
int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);  // 跨 warp reduce 的暂存
int shared_mem_size = std::max(logits_size, outputs_size);  // :93 两者复用同一块 shared mem
```
> 解读：shared memory **复用**——先放 logits（每个 KV token 一个 float），算完 softmax 后同一块内存改放跨 warp reduce 的输出暂存（`outputs_size`），取两者 max。这是省 shared memory 的关键 trick（见 T6）。`q_vecs` 和 `red_smem` 是另外的静态 `__shared__`。

**B2.** v2 launcher（`paged_attention_v2.cu:52`）的 grid 多一维：
```cpp
int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);   // :91
dim3 grid(num_heads, num_seqs, max_num_partitions);   // :96  ← grid.z = 分区数
dim3 reduce_grid(num_heads, num_seqs);                 // :99  reduce kernel 的 grid
int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);  // :100 放各分区 max+exp_sum
```
`LAUNCH_PAGED_ATTENTION_V2`（`:32`）先 launch 主 kernel 写各分区局部结果，紧接着 launch `paged_attention_v2_reduce_kernel`（`:43`）合并。两个 kernel 在同一 stream 串行，第二个自然等第一个完成。

**B3.** head_size 通过 `switch` 在编译期定死（`paged_attention_v1.cu:99-133`）：
```cpp
switch (head_size) {
  case 64:  LAUNCH_PAGED_ATTENTION_V1(64);  break;
  case 128: LAUNCH_PAGED_ATTENTION_V1(128); break;
  ...   default: TORCH_CHECK(false, "Unsupported head size");
}
```
> 解读：head_size、block_size、dtype 全是模板参数（`paged_attention_v1.cu:49`），每个组合编译出独立 kernel。换来编译期常量、循环全展开，代价是编译慢、二进制大（`:100` 注释：只编译模型实际用到的 head_size 以省编译时间）。dtype 分派由 `DISPATCH_BY_KV_CACHE_DTYPE`（`quant_utils.cuh:532`）展开，block_size 由 `CALL_V1_LAUNCHER_BLOCK_SIZE`（`:153`）switch。

**B4.** V1 还要给 kernel 设置动态 shared memory 上限（`paged_attention_v1.cu:33` `LAUNCH_PAGED_ATTENTION_V1` 宏内）：
```cpp
VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
    (void*)vllm::paged_attention_v1_kernel<...>, shared_mem_size);
```
> 因为 logits 可能很大（长序列），超过默认 48KB 动态 shared mem，必须显式 opt-in 更大的 shared memory。

### 阶段 C：CUDA kernel 主体（`attention_kernels.cuh:90` `paged_attention_kernel`）

逐段对照（行号即 `attention_kernels.cuh`）：

**C1. 解析 grid 坐标、算分区范围（`:111-136`）**
```cuda
const int seq_idx = blockIdx.y;
const int partition_idx = blockIdx.z;
const int seq_len = seq_lens[seq_idx];
if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) return;  // :116 本分区无 KV→退出
const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
const int end_block_idx   = MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
```
> V1 时 `USE_PARTITIONING=false`，一个 block 扫整条序列 `[0, num_seq_blocks)`；V2 时只扫本分区 `[start, end)`。

**C2. 三级线程层级常量（`:138-171`）**
```cuda
constexpr int THREAD_GROUP_SIZE   = MAX(WARP_SIZE / BLOCK_SIZE, 1);     // block_size=16 → 2
constexpr int NUM_THREAD_GROUPS   = NUM_THREADS / THREAD_GROUP_SIZE;
constexpr int NUM_WARPS           = NUM_THREADS / WARP_SIZE;            // 128/32 = 4
constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);  // :162 向量化粒度
constexpr int NUM_VECS_PER_THREAD = (HEAD_SIZE / THREAD_GROUP_SIZE) / VEC_SIZE;
const int thread_group_idx    = thread_idx / THREAD_GROUP_SIZE;   // 这是第几个线程组
const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;   // 组内第几个线程
```
> 解读：`VEC_SIZE` 让每个线程一次取/算 **16 字节**（128bit）。fp16 + `THREAD_GROUP_SIZE=2` → `VEC_SIZE = 16/(2*2) = 4`，即 `K_vec = uint2`（4 个 fp16）。`Vec` 的 dtype 特化见 `dtype_float16.cuh:35-50`。

**C3. Q 载入 shared memory（`:179-189`）**
```cuda
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
}
__syncthreads();
```
> 解读：Q 被所有 KV token 复用，故一次性放进 shared memory，按 `[thread_group_offset][vec]` 切片——组内每个线程负责 Q 的不同段，正好对应后面 Q·K 的分工。`q` 可能不连续（从 qkv 融合张量切出），用 `q_stride`（`:179` 注释）。

**C4. shared memory 规划（`:192-201`）**
```cuda
extern __shared__ char shared_mem[];
float* logits = reinterpret_cast<float*>(shared_mem);   // 每个 KV token 一个 fp32 logit
__shared__ float red_smem[2 * NUM_WARPS];                // 跨 warp reduce 暂存
constexpr int x = 16 / sizeof(cache_t);                  // K cache 重排的 x（fp16→8）
```

**C5. 取本序列的 block table 行（`:207`）—— PagedAttention 间接寻址的起点**
```cuda
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
```

**C6. 主循环：warp 分 KV block，取 K 算 Q·K（`:227-308`）**
```cuda
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
     block_idx += NUM_WARPS) {                            // ★每个 warp 隔 NUM_WARPS 取一个 block
  const int64_t physical_block_number =
      static_cast<int64_t>(block_table[block_idx]);       // :257 ★逻辑块→物理块（int64 防溢出）
  for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    K_vec k_vecs[NUM_VECS_PER_THREAD];
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
      const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride  // :273 物理块基址
                             + kv_head_idx * kv_head_stride + physical_block_offset * x;
      const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
      const int offset1 = (vec_idx * VEC_SIZE) / x;   // :277 K cache 重排寻址
      const int offset2 = (vec_idx * VEC_SIZE) % x;
      k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
    }
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
    qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;  // :297 ALiBi 偏置
    if (thread_group_offset == 0) {                       // 组长写结果
      const bool mask = token_idx >= seq_len;             // :302 超出序列的 padding
      logits[token_idx - start_token_idx] = mask ? 0.f : qk;
      qk_max = mask ? qk_max : fmaxf(qk_max, qk);
    }
  }
}
```
> 解读：
> - `:227` warp 把 KV block **交错**分给各 warp（stride = NUM_WARPS），负载均衡。
> - `:257` 就是 PagedAttention 的精髓一行——逻辑块号 `block_idx` 经 block table 翻译成物理块号。强转 int64 防 `physical_block_number * kv_block_stride` 溢出（`:229` 注释）。
> - `:277` K cache 的 `head_size/x, block_size, x` 重排在这里被"反算"出真实地址：`offset1 * BLOCK_SIZE * x` 跨过 head_size 维、`offset2` 是 x 维内偏移。这层重排让一个 warp 取一个 block 落连续 128bit（见 T3）。
> - `Qk_dot::dot`（`attention_utils.cuh:31`）内部 `fma` 累加后做**线程组内 shfl_xor reduce**（`:42`）得到完整点积。
> - `:302` padding token（`token_idx >= seq_len`）的 logit 强制 0、不更新 max——正确性关键。

**C7. softmax 的两级 reduce（`:310-346`）**
```cuda
// (1) warp 内：先把各线程组的 max 归约成整个 warp 的 max
for (int mask = WARP_SIZE/2; mask >= THREAD_GROUP_SIZE; mask /= 2)
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));   // :314
if (lane == 0) red_smem[warp_idx] = qk_max;  __syncthreads();   // 各 warp max 进 shared mem
// (2) 跨 warp：从 shared mem 读各 warp max，再 warp 内归约成 block 全局 max
qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
for (int mask = NUM_WARPS/2; mask >= 1; mask /= 2)
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));    // :326
qk_max = VLLM_SHFL_SYNC(qk_max, 0);                              // 广播给所有线程
// exp 与求和
float exp_sum = 0.f;
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);  logits[i] = val;  exp_sum += val;
}
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);   // :339 又一次两级 reduce
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) logits[i] *= inv_sum;
```
> 解读：softmax 的 max 和 sum 都是"**warp 内 shfl + 跨 warp shared memory**"两级归约。`block_sum`（`:50`）是这套模式的封装。fp32 累加 logits 保精度（`:193` 注释）。

**C8. V2 落盘局部结果（`:348-357`）**
```cuda
if (USE_PARTITIONING && thread_idx == 0) {
    max_logits[... + partition_idx] = qk_max;     // 本分区局部 max
    exp_sums[...  + partition_idx]  = exp_sum;     // 本分区局部 exp_sum
}
```
> 解读：V2 每个分区把局部 max 和 exp_sum 落盘，供 reduce kernel rescale 合并。V1 时 `USE_PARTITIONING=false`，跳过。

**C9. 加权 V 累加（`:359-434`）**
```cuda
constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
float accs[NUM_ROWS_PER_THREAD];                          // 每线程负责若干 head_size 行
for (int block_idx = start + warp_idx; block_idx < end; block_idx += NUM_WARPS) {
  const int64_t physical_block_number = block_table[block_idx];  // :394 再次间接寻址取 V
  L_vec logits_vec; from_float(logits_vec, ...);          // 取本 block 的 softmax 权重
  const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
    if (row_idx < HEAD_SIZE) {
      V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + row_idx*BLOCK_SIZE + physical_block_offset);
      if (block_idx == num_seq_blocks - 1) {              // :420 最后一个 block：清 padding 的 V
        for (int j = 0; j < V_VEC_SIZE; j++)
          v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;  // 防 NaN
      }
      accs[i] += dot(logits_vec, v_vec);
    }
  }
}
```
> 解读：V cache 布局 `[..., head_size, block_size]`，所以 `row_idx*BLOCK_SIZE` 跨 head_size 行。`:420` 是著名的 **NaN 防御**（issue #641）——最后一个 block 里超出 seq_len 的 V 槽可能是未初始化的 NaN，参与 `dot` 会污染结果，故显式清零。

**C10. 输出的跨 warp reduce（`:436-495`）**
```cuda
// warp 内：把同一 head_size 行在 warp 内归约
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++)
  for (int mask = NUM_V_VECS_PER_ROW/2; mask >= 1; mask /= 2)
    acc += VLLM_SHFL_XOR_SYNC(acc, mask);                 // :441
__syncthreads();
float* out_smem = reinterpret_cast<float*>(shared_mem);  // :452 ★复用 logits 的 shared mem 当输出暂存
for (int i = NUM_WARPS; i > 1; i /= 2) {                  // 跨 warp 树形归约
    int mid = i / 2;
    if (warp_idx >= mid && warp_idx < i) { /* 上半 warp 写 shared mem */ }
    __syncthreads();
    if (warp_idx < mid) { /* 下半 warp 累加 */ }
    __syncthreads();
}
if (warp_idx == 0) { /* warp0 写最终 out[seq,head,:] */ }  // :484
```
> 解读：`:449` 的 `__syncthreads()` + `:452` 复用 shared memory——logits 用完了，同一块 shared mem 改当跨 warp reduce 的暂存（对应 B1 的 `std::max(logits_size, outputs_size)`）。跨 warp 用树形归约（`NUM_WARPS → NUM_WARPS/2 → ... → 1`）。

### 阶段 D：V2 reduce kernel 合并各分区（`attention_kernels.cuh:567`）

```cuda
__global__ void paged_attention_v2_reduce_kernel(...) {
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if (num_partitions == 1) { /* :582 只一个分区→直接 copy tmp_out 到 out，省 reduce */ return; }
  // (1) 各分区 max → 全局 max（两级 reduce）
  for (...) max_logit = fmaxf(max_logit, max_logits_ptr[i]);    // :611
  // (2) rescale 各分区 exp_sum 再求全局 exp_sum
  float rescaled = exp_sums_ptr[i] * expf(shared_max_logits[i] - max_logit);  // :646
  global_exp_sum += rescaled;  shared_exp_sums[i] = rescaled;
  global_exp_sum = block_sum<NUM_WARPS>(...);                    // :651
  const float inv = __fdividef(1.f, global_exp_sum + 1e-6f);
  // (3) 加权合并各分区输出
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0;
    for (int j = 0; j < num_partitions; ++j)
      acc += to_float(tmp_out_ptr[j*HEAD_SIZE + i]) * shared_exp_sums[j] * inv;  // :664
    from_float(out_ptr[i], acc);
  }
}
```
> 解读：这是 **online-softmax / log-sum-exp 合并**公式——各分区算的是局部 softmax，要按 `exp(local_max - global_max)` 重新加权才能拼成全局 softmax。和 cascade attention 的 `merge_attn_states`（[模块 02](../02-paged-attention-kvcache/impl.md) §3）、FlashDecoding 是同一套数学。`:582` 单分区快捷路径避免无谓 reduce。

---

## 3. KV 写入 kernel 调用链（`reshape_and_cache`）

**E1.** attention backend 在 forward 开头写 KV：`flash_attn.py` 调 `_custom_ops.py:1328 reshape_and_cache_flash` → `torch.ops._C_cache_ops.reshape_and_cache_flash`（注册 `torch_bindings.cpp:595`）。

**E2.** C++ launcher `cache_kernels.cu:411 reshape_and_cache_flash`：
```cpp
int num_tokens = slot_mapping.size(0);   // :430 ★用 slot_mapping 而非 key 的 token 数
dim3 grid(num_tokens);                     // 每个 token 一个 thread block
dim3 block(std::min(num_heads * head_size, 512));  // 块内线程并行写 num_heads*head_size 个元素
```
> 解读：`:420-429` 注释解释为何用 `slot_mapping.size(0)`：V1 下 `key` 含 CUDA graph padding（更长），`slot_mapping` 才是真实 token 数。多出的 padding key 不该写 cache。

**E3.** kernel `cache_kernels.cu:265 reshape_and_cache_flash_kernel`：
```cuda
const int64_t token_idx = blockIdx.x;
const int64_t slot_idx = slot_mapping[token_idx];
if (slot_idx < 0) return;                          // :279 padding token 哨兵，跳过
const int64_t block_idx    = slot_idx / block_size;  // 物理块号
const int64_t block_offset = slot_idx % block_size;  // 块内偏移
for (int i = threadIdx.x; i < num_heads*head_size; i += blockDim.x) {
  const int head_idx = i / head_size, head_offset = i % head_size;
  const int64_t tgt = block_idx * block_stride + block_offset * num_heads*head_size
                      + head_idx * head_size + head_offset;   // :290 规整布局地址
  if constexpr (kv_dt == kAuto) key_cache[tgt] = key[token_idx*key_stride + i];
  else key_cache[tgt] = fp8::scaled_convert<cache_t,scalar_t,kv_dt>(tgt_key, *k_scale);  // :299 fp8
}
```
> 解读：flash 版地址简单（规整布局）。经典版 `reshape_and_cache_kernel`（`:211`）地址复杂得多（`:242` 的 `head_size/x, block_size, x` 重排），对应 §2 C6 的访存优化。fp8 路径用 `scaled_convert` 量化写入（KV 显存砍半）。`slot = block*block_size + offset` 在此被**逆向**分解回 `(block_idx, block_offset)`——和 [模块 02](../02-paged-attention-kvcache/impl.md) §2 阶段 D 的写 KV 是同一处，本文从 kernel 内部看。

---

## 4. 显存 profiling 与 KV 初始化（Python 侧）

**F1.** `gpu_worker.py:139 determine_available_memory` —— **跑 dummy forward 量峰值**
```python
torch.cuda.reset_peak_memory_stats()                       # :152
_, total_gpu_memory = torch.cuda.mem_get_info()
self.model_runner.profile_run()                            # :157 真跑一次最大 batch dummy forward
peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # :169 torch 峰值
torch.cuda.empty_cache()
torch_allocated  = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # :175
total_allocated  = mem_get_info()[1] - mem_get_info()[0]   # :177 实际占用（含非 torch）
non_torch = total_allocated - torch_allocated              # :179 NCCL / IPC buffer 等
if non_torch > 0: peak_memory += non_torch                 # :181 补进峰值
available = total_gpu_memory * gpu_memory_utilization - peak_memory  # :182
```
> 解读：核心是 `:179` 的 non-torch 修正——NCCL 通信 buffer、custom all-reduce 的 IPC buffer 不走 torch allocator，必须从 `mem_get_info` 与 torch 计数的差值补回，否则高估可用显存而 OOM（`:171-173` 注释）。`:162` 断言 profiling 期间别的进程没动显存。

**F2.** `gpu_model_runner.py:1628 initialize_kv_cache` —— **切 block、分配 KV 张量**
```python
if len(kv_cache_config.kv_cache_groups) > 1:
    raise NotImplementedError("Hybrid models ... not supported yet.")   # :1635 单 group 约束
for kv_cache_group in kv_cache_config.kv_cache_groups:
  for layer_name in kv_cache_group.layer_names:
    num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes    # :1647 切块
    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size)                # flash: [2,nb,bs,kvh,hs]
    kv_caches[layer_name] = torch.zeros(kv_cache_shape, dtype, device)  # :1661 一次性分配 KV 张量
```
> 解读：`page_size_bytes` 是一个 block 一层的字节数；`available / page_size / num_layers` 即 block 数。FlashAttention 的 `get_kv_cache_shape`（`flash_attn.py:64`）返回 `[2, num_blocks, block_size, num_kv_heads, head_size]`（K/V 合一维）。`torch.zeros` 一次性把整个 KV 池分配出来——之后所有请求都在这块固定显存里按 block 周转，地址固定（对 CUDA graph 友好）。单 KV cache group 限制见 [模块 02](../02-paged-attention-kvcache/design.md) §2。

**F3.** `gpu_model_runner.py:1600 capture_model` —— **CUDA graph 捕获**
```python
with graph_capture(device=self.device):
    for num_tokens in reversed(self.cudagraph_batch_sizes):   # :1614 ★从大到小捕获
        for _ in range(cudagraph_num_of_warmups): self._dummy_run(num_tokens)
        self._dummy_run(num_tokens)
```
> 解读：`reversed(...)` 让大 shape 先捕获、小 shape 复用其 memory pool（`:1611` 注释），省显存。详见 [模块 07](../07-cuda-graph-compile/impl.md)。

**F4.** `gpu_model_runner.py:559 _prepare_inputs` —— **pinned + non_blocking H2D**
```python
self.input_ids[:n].copy_(self.input_ids_cpu[:n], non_blocking=True)   # :559
self.positions[:n].copy_(self.positions_cpu[:n], non_blocking=True)    # :568
```
> 解读：`*_cpu` 是 `pin_memory=True` 的 staging buffer（`:249-272`），配 `non_blocking=True` 让 H2D 异步、与上一步 GPU compute 重叠。target（`input_ids`/`positions`）是固定持久 buffer（CUDA graph 要求）。slot_mapping 在 `:546` 用纯 numpy 算好（见 [模块 02](../02-paged-attention-kvcache/impl.md) §2 阶段 B），再异步拷上 GPU。

---

## 5. 实现中的精妙之处（Tricks 与非显然设计）

> 每条都是读 CUDA kernel 时容易一眼滑过、理解了才算真懂的点。格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · THREAD_GROUP：几个线程合算一个 KV token 的点积
- **代码**：`attention_kernels.cuh:138` `THREAD_GROUP_SIZE = MAX(WARP_SIZE/BLOCK_SIZE,1)`；`attention_utils.cuh:42` 组内 `shfl_xor` reduce。
- **精妙之处**：朴素做法"一线程算一个 token 的 head_size 维点积"会让单线程串行 128 次乘加，且取数粒度小。这里让 `THREAD_GROUP_SIZE` 个线程**各算 head_size 的一段**、组内 shuffle 归约出完整点积。一举两得：(a) 点积并行；(b) 每线程一次取 `VEC_SIZE`=16 字节正好一条 `ld.128`。`block_size=16` 时组大小=2，一个 warp（32 线程）刚好同时处理 16 个 token = 一个 KV block，warp 与 block 对齐。

### T2 · 向量化访存：VEC_SIZE 锁定 128bit
- **代码**：`attention_kernels.cuh:162` `VEC_SIZE = MAX(16/(THREAD_GROUP_SIZE*sizeof(scalar_t)),1)`；`dtype_float16.cuh:35-50` 把 `Vec<uint16_t,4>` 映射成 `uint2` 等。
- **精妙之处**：所有取 Q/K/V 都按 `Vec` 类型 `reinterpret_cast` 后整体 load，编译器生成 `ld.128`（一次 128bit）。decode attention 是**带宽瓶颈**，把访存事务数压到最少是头等大事。`Vec`/`FloatVec` 模板（`attention_generic.cuh:26`）为每种 dtype + 向量长度特化出对应的 CUDA 向量类型。

### T3 · K cache 的 `head_size/x, block_size, x` 重排
- **代码**：布局注释 `attention_kernels.cuh:97`；寻址 `:273-282`（`offset1*BLOCK_SIZE*x + offset2`）；写入 `cache_kernels.cu:242`。
- **精妙之处**：K cache 不是直觉的 `[block, head, block_size, head_size]`，而是把 head_size 拆成 `head_size/x` 与 `x`（`x=16/sizeof(cache_t)`，fp16→8），再把 `block_size` 插在中间。目的：一个 warp 取一个 block 的 16 个 token 时，相邻 lane 访问的地址落在**连续的 128bit 段**上，既触发合并访存（coalescing）又规避 shared/global 的访存冲突。这是为 §2 C6 的 warp 访存模式量身定做的布局。代价是写入地址算术复杂（`reshape_and_cache_kernel:242` 那一长串）。

### T4 · 两级 reduce：warp 内 shuffle + 跨 warp shared memory
- **代码**：`block_sum`（`:50`）；qk_max reduce（`:313-330`）；输出 reduce（`:436-481`）。
- **精妙之处**：thread block 内归约不能只靠 `__shfl`（只在 warp 内有效）。统一模式是：先 `shfl_xor` 在 32 线程的 warp 内归约 → 每 warp 写一个值进 `red_smem` → `__syncthreads()` → 再用一个 warp 把这 `NUM_WARPS` 个值归约 → `shfl` 广播。softmax 的 max、sum、最终输出全用这套两级模式。`red_smem[2*NUM_WARPS]`（`:196`）开两份是给 max 和 sum 各留位置。

### T5 · `:257` 逻辑块→物理块，强转 int64 防溢出
- **代码**：`attention_kernels.cuh:257` / `:394` `static_cast<int64_t>(block_table[block_idx])`。
- **精妙之处**：block table 存的是 int32 块号，但 `physical_block_number * kv_block_stride`（stride 可能很大）在 int32 下会溢出。先强转 int64 再乘——`:229` 注释专门警示。这是 PagedAttention 间接寻址唯一的额外开销：一次查表 + 一次 int64 乘法。

### T6 · shared memory 复用：logits 用完当输出暂存
- **代码**：`paged_attention_v1.cu:93` `shared_mem_size = std::max(logits_size, outputs_size)`；`attention_kernels.cuh:449` `__syncthreads()` + `:452` 同一指针 reinterpret 成 `out_smem`。
- **精妙之处**：logits（每 KV 一个 float）和跨 warp reduce 的输出暂存（`(NUM_WARPS/2)*head_size` 个 float）**生命周期不重叠**——logits 在 softmax 后就没用了。于是同一块动态 shared memory 先当 logits、`__syncthreads()` 后改当输出暂存，shared mem 占用取两者 max 而非和。长序列时 logits 大，正好省下可观 shared memory。

### T7 · padding token 的双重防护：logit 清零 + V 清零
- **代码**：logit `:302` `mask ? 0.f : qk`；V `:420-429` 最后一个 block 显式 `zero_value`。
- **精妙之处**：序列长度未必整除 block_size，最后一个 block 有"逻辑越界"的槽。这些槽：(a) 的 logit 必须 0（否则参与 softmax 分母）；(b) 的 V 可能是**未初始化的 NaN**（KV cache `torch.zeros` 过但若 fp8/对齐 padding 可能残留），`dot(logits, NaN)` 会污染输出。`:421` 注释直接引用 issue #641。两处都只在"最后一个 block"判断（`block_idx == num_seq_blocks-1`），不拖慢主循环。

### T8 · V1/V2 共用一个 device 函数，模板参数 PARTITION_SIZE 区分
- **代码**：`paged_attention_kernel`（`:90`）的 `int PARTITION_SIZE=0`；v1 包装传 0（`:519`），v2 传 512（`:556`）；`USE_PARTITIONING = PARTITION_SIZE > 0`（`:114`）。
- **精妙之处**：避免维护两份近乎相同的 kernel。`USE_PARTITIONING` 是编译期常量，`if constexpr` 式的分支在 V1 编译时整段消失——V1 kernel 没有任何分区开销，V2 才有落盘局部结果的代码（`:348`）。一套源码、零运行时分支、两种最优 kernel。

### T9 · V2 reduce 的单分区快捷路径
- **代码**：`attention_kernels.cuh:582` `if (num_partitions == 1) { copy tmp_out→out; return; }`。
- **精妙之处**：V2 grid 按 `max_num_partitions`（按 max_seq_len 算）铺，但某条短序列可能实际只占 1 个分区。这时 reduce 无意义，直接 memcpy tmp_out 到 out 即可，省掉整套 log-sum-exp 归约。同一个 batch 里长短序列混合时，短序列走快捷路径。

### T10 · custom all-reduce 的 36-block 网格与 NVLink 争用
- **代码**：`custom_all_reduce.cuh:31` `kMaxBlocks=36`；`:519` 注释。
- **精妙之处**：直觉是"用越多 block 越快"，但注释（来自实测：A100/A10/A30/T4/V100）说**太多 SM 反而在 NVLink 总线上争用**，36 block 是网格搜索的最优。这也解释了为何 NCCL kernel 同样只用少量 SM——通信 kernel 是带宽/总线瓶颈，不是计算瓶颈。

### T11 · one-shot vs two-shot 的选择阈值
- **代码**：`custom_all_reduce.cuh:561` `REDUCE_CASE`：`world_size==2` 或小消息（`bytes < 512KB/256KB`）走 1-stage，否则 2-stage。
- **精妙之处**：1-stage 每卡读全部对端数据（通信量 = `ngpus × size`），延迟 = 1 次 P2P 读，适合小消息（decode 激活很小，延迟敏感）。2-stage 做 reduce-scatter+all-gather（通信量 ≈ `2×size`），省带宽但多一次同步，适合大消息（prefill）。阈值随 world_size 收紧（卡越多、单卡读越多，越早切 2-stage）。

### T12 · 2-stage 必须 tid 对齐保证跨卡可见性
- **代码**：`custom_all_reduce.cuh:348-363`，注释 `:348-352`。
- **精妙之处**：2-stage 的 reduce-scatter 阶段线程 i 算 `start+i` 的和，all-gather 阶段**同一个线程 i** 必须去 gather `start+i`。因为跨设备的内存可见性**只在相同 tid 的线程间**由 barrier 保证。若换线程去读，可能读到对端尚未刷新的旧值。这是 P2P 弱内存模型下的正确性陷阱。

### T13 · all-reduce 同步用交替计数器避免竞态
- **代码**：`custom_all_reduce.cuh:53` `Signal{start[][], end[][]}` 两套；`:47-52` 注释。
- **精妙之处**：起始和结束各用一套 flag 数组。原因：对端 block 可能在当前 block 还没过第一个同步点时，就到了第二个同步点并写 `counter+1`，若共用一套计数器会读到错误值。两套交替使用杜绝这种 ABA 式竞态。flag 用 `st.release/ld.acquire`（`:157-179`）保证内存序。

### T14 · custom all-reduce 的 CUDA graph 兼容：捕获期占位、replay 前填指针
- **代码**：`custom_all_reduce.cuh:540-544`（捕获期压 `graph_unreg_buffers_`）；`:489 register_graph_buffers`（replay 前填实际 peer 指针）；`:382-397` 注释详述流程。
- **精妙之处**：CUDA graph 要求 kernel 参数在捕获时固定，但 peer 显存指针要等所有 rank 交换 IPC handle 后才知道。解法：捕获时 `cudaStreamIsCapturing` 检测到正在捕获，就**递增 rank_data 指针当占位参数**压进 buffer；捕获后各 rank `get_graph_buffer_ipc_meta` 导出 handle、Python all-gather、再 `register_graph_buffers` 把实际对端指针写进占位处。这是 design §3.6 "固定地址" 约束在分布式下的精确兑现。

### T15 · IPC handle 必须用 allocation 的 base 地址
- **代码**：`custom_all_reduce.cuh:447-455` `cuPointerGetAttribute(rangeStartAddrAttr)` 取 base，存 `offset = ptr - base`。
- **精妙之处**：`cudaIpcGetMemHandle` 只能对**分配的起始地址**生成 handle。但 all-reduce 的 input 往往是某大 allocation 中间的一段。所以要先用 `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR` 找到 base 地址生成 handle，另存偏移；对端打开 handle 后 `handle += offset` 还原真实地址（`:502`）。直接对中间地址生 handle 会得到错地址。

### T16 · Marlin 权重重排：让取数天然对齐 tensor core
- **代码**：`permute_cols.cu:14 permute_cols_kernel`；`:42` `out_half[cur_k] = a_row_half[src_pos]`。
- **精妙之处**：tensor core 的 mma 指令对操作数有固定的 lane↔数据布局要求。直接按原列序取低精度权重，dequant 后喂 mma 要做混乱的重排。Marlin **离线**把权重列 permute 成"GPU 取一摞数正好喂给 mma"的顺序，GEMM 时一次 `ld.128`（`int4` = 128bit，`:84`）取连续权重直接用。permute 是一次性的（加载权重时），换来运行时零重排开销。按 `sms` 个 block、每 block 处理若干行（`:82`）。详见 [模块 08](../08-quantization/design.md)。

### T17 · merge_attn_states 的 128bit pack 与 fp32 FMA
- **代码**：`merge_attn_states.cu:20` `pack_128b_t = uint4`；`:66-76` 逐元素转 fp32 做 FMA 再转回。
- **精妙之处**：cascade / split-KV 合并要对 `[num_tokens, num_heads, head_size]` 逐元素加权和。每线程一次取 128bit（`uint4`，`pack_size` 个元素），但**累加用 fp32**（`:67` 注释"Always use float for FMA to keep high precision"）——half/bf16 直接相加会丢精度。这是"访存按低精度 pack、计算按 fp32"的标准向量化模式，和 attention kernel 的 fp32 accumulator（`:371`）一脉相承。

---

## 6. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| KV 写入遇 padding token（slot<0） | `cache_kernels.cu:279` / `:224` | kernel 直接 return，不写 cache |
| V1 下 key 含 graph padding 比 slot_mapping 长 | `cache_kernels.cu:420-430` | 用 `slot_mapping.size(0)` 当真实 token 数 |
| attention 最后一个 block 越界槽的 logit | `attention_kernels.cuh:302` | `mask ? 0.f : qk`，不计入 softmax |
| attention 最后一个 block 越界槽的 V（可能 NaN） | `attention_kernels.cuh:420-429` | 显式置 `zero_value` 防污染（issue #641） |
| V2 分区超出 seq_len（无 KV 可算） | `attention_kernels.cuh:116` | thread block 直接 return |
| V2 实际只占 1 个分区 | `attention_kernels.cuh:582` | reduce kernel 走快捷 copy，跳过 log-sum-exp |
| 不支持的 head_size | `paged_attention_v1.cu:130` | `TORCH_CHECK(false, "Unsupported head size")` |
| 不支持的 block_size（非 8/16/32） | `paged_attention_v1.cu:164` | `TORCH_CHECK(false, "Unsupported block size")` |
| fp8 KV cache | `cache_kernels.cu:255` / `attention_kernels.cuh:283` | 写入 `scaled_convert` 量化、读取反量化 |
| 显存 profiling 期间别的进程动了显存 | `gpu_worker.py:162` | assert 报错（要求 init>current free） |
| 非 torch 分配（NCCL/IPC buffer） | `gpu_worker.py:179-181` | 计入峰值，防高估可用显存 OOM |
| 混合 KV cache 类型（>1 group） | `gpu_model_runner.py:1635` | `NotImplementedError`（暂不支持） |
| custom all-reduce 卡数非偶数 / >8 | `custom_all_reduce.cu:17-20` | 抛异常（仅支持 2/4/6/8 卡） |
| all-reduce 输入长度非 pack 整数倍 | `custom_all_reduce.cuh:529` | 抛异常（要求是 `16/sizeof(T)` 倍数） |
| all-reduce 正在 CUDA graph 捕获 | `custom_all_reduce.cuh:540-544` | 占位指针压 `graph_unreg_buffers_`，replay 前填实际 peer 指针 |
| permute_cols 非 16bit / 列非 8 倍数 | `permute_cols.cu:72-76` | `TORCH_CHECK` 报错 |

---

## 7. 一图速查：Python 绑定 → C++ launcher → CUDA kernel

```
[Python] paged_attn.py:128  use_v1 = (max_seq_len<=8192 && (parts==1 || seqs*heads>512))
            │
            ├─V1─► _custom_ops.py:40  paged_attention_v1 ─► torch.ops._C.paged_attention_v1
            │         └─► paged_attention_v1.cu:52  launcher
            │               grid(num_heads, num_seqs, 1)   block(128)   shared=max(logits,outputs)
            │               switch(head_size) → LAUNCH_PAGED_ATTENTION_V1(128)
            │                  └─► attention_kernels.cuh:90  paged_attention_kernel<PARTITION_SIZE=0>
            │
            └─V2─► _custom_ops.py:69  paged_attention_v2 ─► torch.ops._C.paged_attention_v2
                      └─► paged_attention_v2.cu:52  launcher
                            grid(num_heads, num_seqs, ceil(seq/512))   PARTITION_SIZE=512
                            ① paged_attention_kernel<512> → 各分区写 tmp_out/exp_sums/max_logits
                            ② paged_attention_v2_reduce_kernel:567 → log-sum-exp 合并 → out

  ── kernel 内部 (attention_kernels.cuh) ──
     :207  block_table = block_tables + seq_idx*max_num_blocks_per_seq
     :179  Q → shared memory (q_vecs, 按 thread_group 切片)
     :227  for block: warp 交错分 KV block
     :257    physical = block_table[block_idx]            ← ★逻辑→物理 间接寻址 (int64)
     :273    取 K (head_size/x,block_size,x 重排, ld.128 向量化)
            Qk_dot::dot → 线程组内 shfl reduce → qk
     :313  softmax: warp内 shfl + 跨warp red_smem 两级 reduce → max → exp → sum
     :380  for block: physical = block_table[...]; accs += dot(logits, V)
     :436  输出: warp内 reduce + 跨warp 树形 reduce (复用 logits 的 shared mem)
     :484  warp0 写 out

  ── KV 写入 (cache_kernels.cu) ──
     _custom_ops.py:1328 reshape_and_cache_flash ─► :411 launcher (grid=num_tokens)
        └─► :265 kernel:  slot=slot_mapping[tok]; if slot<0 return
                          block=slot/bs, offset=slot%bs → 写 paged cache (fp8 可量化)

  ── 显存 (gpu_worker.py / gpu_model_runner.py) ──
     :139 determine_available_memory:  profile_run() → peak + non_torch → total*util - peak
     :1628 initialize_kv_cache:  num_blocks = size/page_size → torch.zeros([2,nb,bs,kvh,hs])
     :1600 capture_model:  reversed(sizes) 大→小 复用 graph memory pool

  ── all-reduce (custom_all_reduce.cuh, TP) ──
     allreduce:528  pack=16/sizeof(T); cudaStreamIsCapturing? → graph 占位 : 查 buffers_
     :561 REDUCE_CASE:  小消息→1stage (P2P 直读求和) / 大消息→2stage (reduce-scatter+allgather)
                        barrier_at_start/end 自旋 flag 跨卡同步 (交替计数器防竞态)
```

> 上层 KV 软件管理（block 分配/prefix cache/抢占）见 [模块 02](../02-paged-attention-kvcache/impl.md)；CUDA graph 捕获机制见 [模块 07](../07-cuda-graph-compile/impl.md)；分布式通信编排见 [模块 03](../03-distributed-parallel/impl.md)；量化 GEMM 全貌见 [模块 08](../08-quantization/impl.md)。本模块聚焦这些机制在 GPU kernel / 显存层面的实现。
