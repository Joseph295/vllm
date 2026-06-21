# 模块 11 · GPU 执行底层：Kernel 与显存 —— 设计文档

> 范围：Python 编排（调度、KV 管理、并行策略）**之下**，GPU 上真正发生的计算与显存操作。本模块深入到 `csrc/**` 的 CUDA/C++ 源码，讲清楚：PagedAttention kernel 里 thread block / warp / thread group 怎么分工算 attention；KV 怎么按 paged 布局写进显存；KV 显存怎么 profiling 出来、怎么切块；CUDA graph 在 GPU 上怎么省开销；custom all-reduce 怎么用 IPC 直接读写对端显存；量化 GEMM 为什么要重排权重。
> 定位：这是 [模块 02](../02-paged-attention-kvcache/design.md)（PagedAttention/KV 管理）、[模块 03](../03-distributed-parallel/design.md)（分布式并行）、[模块 07](../07-cuda-graph-compile/design.md)（CUDA graph/编译）、[模块 08](../08-quantization/design.md)（量化）的 **GPU 底层延伸**。上层模块讲"软件怎么编排"，本模块讲"GPU 上到底怎么算、怎么放"。概念处用相对链接指回，不重复。
> 架构：CUDA kernel（`csrc/`）**V0/V1 共用**；显存 profiling 与 KV 初始化以 **V1**（`vllm/v1/worker/`）为主。

---

## 1. 这个模块解决什么问题

上层模块把请求调度好、把 block 分配好、把 block table 填好之后，剩下的全是 GPU 上的硬活：

1. **decode attention 在 GPU 上是个"反 GEMM"的怪兽**：prefill 阶段 Q 是一长串 token，可以用稠密 GEMM 打满 tensor core；但 decode 阶段每条序列只有**1 个 query token**，要对它之前**成百上千个 KV** 算注意力。这是个又瘦又长（`[1, head_size] × [seq_len, head_size]^T`）的形状，标准 GEMM kernel 在这里效率极低（M=1，tensor core 大量空转）。还要叠加 PagedAttention 的"KV 物理上不连续、要查 block table 跳读"。这正是 vLLM 自己手写 `paged_attention_kernel` 的原因。
2. **KV 写入要落到分页布局**：模型每层算出新的 K/V 后，要按 `slot_mapping` 写进**物理上分块、且为了访存对齐做了维度重排**的 KV cache 张量。这步若用 PyTorch 逐 token 的 `index_put_` 会极慢，必须一个 CUDA kernel 搞定。
3. **KV 显存到底能开多大全靠猜不行**：KV cache 要吃掉权重和激活之外**剩下的全部显存**。但"剩下多少"取决于模型激活峰值，事先算不准。vLLM 的办法是**真跑一次 dummy forward**量峰值，再把剩余显存切成等大 block。
4. **decode 是访存瓶颈、launch 开销敏感**：单步 decode 计算量极小，GPU 上 kernel 真正执行可能只要几十微秒，而 CPU 侧 launch 一串 kernel 的开销可能更大。**CUDA graph** 把整张 kernel 图录下来一次性 replay，省掉逐 kernel 的 launch 开销——但代价是所有 buffer 地址、kernel 参数必须**固定**。
5. **多卡之间 all-reduce 是 TP 的关键路径**：每层 attention/MLP 后要跨卡求和激活。NCCL 对小消息（decode 时激活很小）延迟偏高。vLLM 的 **custom all-reduce** 用 CUDA IPC 让各卡**直接读写对端显存**（NVLink P2P），对小消息显著降延迟。
6. **低精度权重在 GPU 上不能直接喂 tensor core**：int4/int8 权重要先 dequant 成 fp16，但 dequant 和 GEMM 分两步会浪费带宽。Marlin 这类 kernel 把 dequant **融进** GEMM 的取数阶段，且为了让取数访存规整，**离线把权重列重排**过。

这六件事，分别对应本模块的六大主题。核心一句话：**上层把"算什么、放哪里"决定好，本模块解决"在 GPU 的线程/warp/显存层面，到底怎么把它算出来、放下去，且要快"**。

---

## 2. 设计目标与约束

- **decode attention 要把并行度榨干**：grid 维度要覆盖 `(head, seq, partition)`，让每个 SM 都有活干；长序列要能沿 KV 维切 partition（V2），避免单 block 串行扫几千个 KV。
- **访存优先于计算**：decode attention 是带宽瓶颈，所有取 Q/K/V 都要**向量化**（一次取 16 字节 = 128bit，触发 `ld.128`），KV cache 的物理布局要为"warp 连续取一个 block"而设计（K cache 的 `head_size/x, block_size, x` 重排就是为此）。
- **block table 间接寻址必须在 kernel 内完成**：kernel 入参带 `block_tables`，每访问一个逻辑 KV block 先查表得物理块号——这是 PagedAttention 与普通 flash-attention 的唯一本质区别（见 [模块 02](../02-paged-attention-kvcache/design.md) §3.1）。
- **KV 显存分配要 profiling 驱动、留安全裕度**：用 `gpu_memory_utilization`（默认 0.9）作为总显存上限，减去 dummy forward 量到的峰值（含非 torch 分配如 NCCL buffer），剩下的才给 KV。
- **CUDA graph 友好**：所有进 kernel 的 buffer（input_ids、positions、slot_mapping、block_table、KV cache）必须是**持久、地址固定**的张量；block table append-only、block_id 永不变（见 [模块 02](../02-paged-attention-kvcache/design.md) §5 T12）正是为了 replay 时地址稳定。
- **custom all-reduce 要可被 CUDA graph 捕获**：peer 指针在 graph 捕获时未知，必须用"捕获期占位、replay 前填实际指针"的机制（`graph_unreg_buffers_`）。
- **正确性 corner case**：attention 里超出 `seq_len` 的 padding token 的 logits 要 mask 成 0、V 要清零（防 NaN）；KV 写入时 `slot < 0` 的 padding token 要跳过。

---

## 3. 核心设计思想

### 3.1 PagedAttention kernel：线程层级的三级分工（本模块重中之重）

decode attention 的核心 kernel 是 `csrc/attention/attention_kernels.cuh` 里的 `paged_attention_kernel`（一个 `__device__` 函数模板，被 v1/v2 两个 `__global__` 入口包一层）。理解它要先建立**三级线程层级**：

```
Grid:    (num_heads, num_seqs, max_num_partitions)   ← 一个 thread block 算「1个head × 1条seq × 1个KV分区」
  └─ Thread Block (默认 NUM_THREADS=128 线程 = 4 warps)
        └─ Warp (32 线程)           ← 每个 warp 负责若干个 KV block（沿 KV 序列维划分）
             └─ THREAD_GROUP        ← 几个线程合伙处理「一个 KV token」的 Q·K 点积
```

**为什么这样分**：decode 时一个 block 要算"1 个 query 对一串 KV"的注意力。最外层 grid 把工作按 `(head, seq, partition)` 三维铺开——`num_heads × num_seqs` 已经能填满中小 batch；当序列很长、block 数不够填满 GPU 时，再沿 KV 序列维切 `partition`（V2，见 §3.2）。

**THREAD_GROUP（线程组）是这个 kernel 最精巧的设计**：
- 定义在 `attention_kernels.cuh:138`：`THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1)`。当 `block_size=16`、`WARP_SIZE=32` 时，`THREAD_GROUP_SIZE = 2`，即**2 个线程合伙算一个 KV token 的 Q·K**。一个 warp（32 线程）于是同时处理 16 个 token（= 一个 KV block）。
- 为什么要"几个线程合伙算一个 token"而不是"一个线程算一个 token"？因为 `head_size`（如 128）的 Q·K 点积有 128 个乘加。一个线程独算要串行 128 次；让 `THREAD_GROUP_SIZE` 个线程**各算一段、再 warp 内 reduce**，能并行起来。同时让取数向量化：每个线程组每次取 16 字节（`VEC_SIZE = 16 / (THREAD_GROUP_SIZE * sizeof(scalar_t))`，`:162`），正好一条 `ld.128`。

**Q 进 shared memory**：Q（`[head_size]`）被所有 KV token 共用，所以先一次性载入 shared memory（`q_vecs`，`:180`），按 thread group 切片存放，之后每个线程组反复从 shared memory 读自己那段 Q 去和不同 K 点积。

**KV 的物理布局是为"warp 连续取一个 block"设计的**。K cache 形状是 `[num_blocks, num_kv_heads, head_size/x, block_size, x]`（注意 `block_size` 被塞到 `head_size` 维里面去了，`x = 16/sizeof(cache_t)`）。这个看似古怪的重排，是为了让一个 warp 取一个 block 的 16 个 token 时，访存落在连续的 128bit 段上（见 impl.md T3）。

**Q·K 之后的两段 reduce（softmax 的关键）**：
1. **warp 内 reduce 求 qk_max**：每个线程组算出自己 token 的 qk，先 `__shfl_xor` 在线程组内归约（`attention_utils.cuh:42`），再 warp 内归约求该 warp 的局部 max（`:314`）。
2. **跨 warp reduce**：各 warp 的 max 写进 shared memory `red_smem`，再归约成整个 block 的 max（`:320-330`）。然后算 `exp(logit - max)`、求 exp_sum（`block_sum`，`:50`，又是一次"warp 内 + 跨 warp"两级 reduce）。

**Q·K·V 的 V 累加**：softmax 权重（`logits`）算好后，再过一遍 KV block，这次取 V cache（布局 `[num_blocks, num_kv_heads, head_size, block_size]`），每个线程累加 `logits · V` 到寄存器 `accs[]`（`:431`），最后 warp 内 + 跨 warp reduce 出 `[head_size]` 的输出（`:436-481`）。

> 一句话总结 kernel 内幕：**grid 铺 (head,seq,partition)，warp 分 KV block，thread group 合算一个 token 的点积，shared memory 放 Q 和中间 logits，所有归约都是"warp 内 shuffle + 跨 warp shared memory"两级**。详尽逐行见 [`impl.md`](impl.md) §2、§3。

### 3.2 V1 vs V2：为什么长序列要沿 KV 维切 partition

同一个 `paged_attention_kernel`，靠模板参数 `PARTITION_SIZE` 区分两种用法：

| | **V1**（`paged_attention_v1.cu`） | **V2**（`paged_attention_v2.cu`） |
|---|---|---|
| `PARTITION_SIZE` | 0（不分区） | 512 |
| grid.z | 1 | `ceil(max_seq_len / 512)` |
| 每个 block 扫多少 KV | 整条序列 | 一个 512-token 分区 |
| 输出 | 直接写最终 out | 各分区写 `tmp_out` + `exp_sums` + `max_logits`，再由 **reduce kernel** 合并 |
| 何时用 | 短序列、或 `num_seqs*num_heads` 大（并行度已够） | 长序列（>8192）或分区数 >1 且并行度不足 |

**动机**：V1 一个 thread block 串行扫完整条序列的所有 KV block。当 batch 小（`num_seqs*num_heads` 不足以填满 GPU 的 SM）、序列又长时，GPU 大量 SM 闲着，单个 block 却要串行扫几千个 KV——并行度严重不足。

**V2 的解法是"split-KV"**：把一条序列的 KV 沿序列维切成 512-token 的 partition，**每个 partition 一个 thread block 并行算**自己那段的局部 softmax（局部 max、局部 exp_sum、局部加权 V），写进临时张量。然后第二个 `paged_attention_v2_reduce_kernel`（`:567`）把同一 (seq,head) 的所有 partition 用 **online-softmax 的 rescale 公式**合并：先求全局 max，把各 partition 的 exp_sum 按 `exp(local_max - global_max)` rescale 再相加，最后加权合并各 partition 的输出。

V1/V2 的选择在 Python 侧 `vllm/attention/ops/paged_attn.py:128`：

```python
use_v1 = (max_seq_len <= 8192
          and (max_num_partitions == 1 or num_seqs * num_heads > 512))
```

> 这本质是把 **FlashAttention 的 online softmax / split-KV** 思想用在 paged decode 上。V2 的 reduce 和 cascade attention 的 `merge_attn_states`（[模块 02](../02-paged-attention-kvcache/impl.md) §3）是同一类"合并多段 attention 局部结果"的操作，数学公式一致（log-sum-exp 合并）。

### 3.3 GQA/MQA 在 kernel 里只是一行除法

decode attention 要把 `num_heads` 个 query head 映射到 `num_kv_heads` 个 KV head（GQA：多个 Q head 共享一个 KV head；MQA：所有 Q head 共享一个）。kernel 里这只是（`attention_kernels.cuh:152-153`）：

```cuda
const int num_queries_per_kv = num_heads / num_kv_heads;
const int kv_head_idx = head_idx / num_queries_per_kv;
```

grid 仍按 `num_heads` 铺（每个 query head 一个 block 平面），但取 KV 时用 `kv_head_idx` 去算 KV cache 地址——**多个 query head 的 block 自然落到同一份 KV 数据上**，零额外开销实现 KV head 共享。这就是 GQA 省 KV 显存的物理基础。

### 3.4 KV 写入：slot_mapping 驱动的分页落盘

模型每层算出新 K/V 后，`reshape_and_cache` / `reshape_and_cache_flash`（`csrc/cache_kernels.cu`）按 `slot_mapping` 把它们写进 paged cache。核心逻辑：

```
每个 token 一个 thread block：
  slot = slot_mapping[token_idx]        ← 上层算好的全局扁平位置
  if slot < 0: return                   ← padding token 跳过（哨兵）
  block_idx    = slot / block_size      ← 物理块号
  block_offset = slot % block_size      ← 块内偏移
  block 内 num_heads*head_size 个元素由 blockDim.x 个线程并行写
```

两个版本的区别只在**目标 cache 的布局**：
- `reshape_and_cache`（经典 paged attention 用）：K cache 重排成 `[num_blocks, num_heads, head_size/x, block_size, x]`，写入地址算术复杂（`:242`），目的是配合 §3.1 的 warp 连续访存。
- `reshape_and_cache_flash`（FlashAttention backend 用，V1 默认）：K/V cache 都是规整的 `[num_blocks, block_size, num_heads, head_size]`，写入地址简单（`:290`），因为 FlashAttention 自己的 kernel 接受这种布局。

两者都支持 **fp8 量化写入**：若 `kv_cache_dtype != auto`，写入前用 `fp8::scaled_convert` 把 fp16 K/V 乘 scale 转成 fp8 存（`:255`），读取时（attention kernel 内）再转回。这把 KV cache 显存砍半，是长上下文的关键。详见 [模块 02](../02-paged-attention-kvcache/impl.md) §2 阶段 D。

### 3.5 KV 显存：profiling 出来、再切块

KV cache 要吃掉"权重 + 激活峰值"之外剩下的所有显存。但激活峰值算不准，所以 vLLM **真跑一次 dummy forward**（`gpu_worker.py:139` `determine_available_memory`）：

```
1. reset_peak_memory_stats()                          # 清零 torch 峰值计数
2. model_runner.profile_run()                         # 跑一次最大 batch 的 dummy forward
3. peak = memory_stats()["allocated_bytes.all.peak"]  # torch 量到的峰值
4. non_torch = (总显存 - 空闲) - torch当前分配          # NCCL 等非 torch 分配（可达几 GB）
5. peak += non_torch
6. KV 可用 = 总显存 * gpu_memory_utilization - peak
```

关键设计点：
- **必须真跑 forward**：因为激活峰值（尤其是 prefill 大 batch 的中间张量）无法静态推算。
- **要数 non-torch 分配**：NCCL 通信 buffer、custom all-reduce 的 IPC buffer 不走 torch allocator，必须从 `mem_get_info` 和 torch 计数的差值里把它们补回峰值，否则会高估可用显存而 OOM（`gpu_worker.py:171-181`）。
- 量出的字节数交给 KV cache 配置层切成 `num_blocks` 等大 block（`initialize_kv_cache`，`gpu_model_runner.py:1628`）：`num_blocks = tensor_size // page_size_bytes`。KV cache 张量用 `torch.zeros(kv_cache_shape, ...)` 一次性分配（`:1661`），形状由 attention backend 给（FlashAttention：`[2, num_blocks, block_size, num_kv_heads, head_size]`，`flash_attn.py:64`）。

显存切分的二分搜索逻辑（放不下时算出最大可行 `max_model_len`）见 [模块 02](../02-paged-attention-kvcache/impl.md) §5 T15。

### 3.6 CUDA graph：从 GPU 视角看为什么省、为什么要固定地址

decode 单步计算量极小，CPU 侧"launch 几十上百个 kernel"的开销反而成为瓶颈（每个 kernel launch 有几微秒固定开销，几十个叠起来就盖过了实际计算）。**CUDA graph 把一整步的 kernel 调用序列录成一张图，replay 时 GPU 驱动一次性提交，省掉逐 kernel 的 CPU launch 开销**。

从 GPU/显存视角，capture 有两个硬约束（`gpu_model_runner.py:1600` `capture_model`）：
1. **buffer 地址必须固定**：graph 录的是"对某个固定显存地址执行某 kernel"。所以 input_ids、positions、slot_mapping、block_table、KV cache 都必须是**预分配的持久张量**（`gpu_model_runner.py:203-272` 那一批 `torch.zeros` persistent buffer）。每步把新数据 `copy_` 进这些固定 buffer，再 replay。这也是 [模块 02](../02-paged-attention-kvcache/design.md) §5 "block_id 永不变、block table append-only" 的根本原因——block table 张量地址固定，内容原地更新。
2. **memory pool 共享**：capture 多个 batch size 时，**从大到小**捕获（`:1611-1614` `reversed(...)`），让小 shape 复用大 shape 已分配的 graph memory pool，避免每个 size 各占一份显存。

详见 [模块 07](../07-cuda-graph-compile/design.md)；本模块只补"为什么 GPU 上非要固定地址"这个底层动机。

### 3.7 CUDA stream 与 H2D 重叠

`_prepare_inputs`（`gpu_model_runner.py:464`）在 CPU 侧用 numpy 算好 input_ids、positions、slot_mapping 等（pinned staging buffer，`pin_memory=True`），然后 `copy_(..., non_blocking=True)` 异步 H2D（`:559-570`）。pinned 内存 + `non_blocking=True` 让这些 H2D 拷贝**能与上一步的 GPU compute 在不同 stream 上重叠**，不阻塞主计算流。这是把 [模块 00](../00-request-lifecycle/design.md) "CPU 活与 GPU 活重叠" 的思想落到 stream 层面。

### 3.8 custom all-reduce：用 IPC 直接读写对端显存

TP 下每层要 all-reduce 激活。NCCL 对小消息（decode 激活很小）延迟偏高。vLLM 的 custom all-reduce（`csrc/custom_all_reduce.cu` / `.cuh`）核心是：

- **CUDA IPC 让各卡互相能读对端显存**：每卡分配一块 buffer，用 `cudaIpcGetMemHandle` 导出 handle，Python 侧 all-gather 后各卡 `cudaIpcOpenMemHandle` 打开，于是得到指向**对端显存的指针**（`custom_all_reduce.cuh:427`）。在 NVLink 全连接拓扑下，kernel 可以直接 load/store 对端显存（P2P）。
- **one-shot（1-stage）vs two-shot（2-stage）**：
  - **1-stage**（`cross_device_reduce_1stage`，`:296`）：每个线程直接读**所有** rank 同一位置的值求和写回。延迟 = 1 次 P2P 读。适合小消息（带宽不是瓶颈、延迟才是）。
  - **2-stage**（`cross_device_reduce_2stage`，`:319`）：reduce-scatter + all-gather。每卡只 reduce `1/ngpus` 的数据再 gather。通信量更省，适合大消息。
  - 选择在 `:561` `REDUCE_CASE`：world_size=2 或小消息走 1-stage，否则走 2-stage。
- **为什么比 NCCL 低延迟**：自定义 kernel 跳过 NCCL 的通用调度/proto 协商，对小消息直接 P2P 读写 + 自旋 flag 同步（`barrier_at_start/end`，`:197/:218`），路径极短。只用 36 个 block（`:519` 注释：太多 block 反而在 NVLink 上争用）。
- **CUDA graph 兼容性**：peer 指针在 graph 捕获时未知，所以捕获期只把 input 指针压进 `graph_unreg_buffers_`（`:542`），replay 前再通过 IPC handle 填实际对端指针（`register_graph_buffers`，`:489`）。这正是 §3.6 "固定地址" 约束在分布式下的延伸。

详见 [模块 03](../03-distributed-parallel/design.md)；本模块补 GPU IPC 与 one/two-shot 的 kernel 细节。

### 3.9 量化 GEMM：权重重排 + dequant-fuse

低精度（int4/int8）权重 GEMM 的 kernel（`csrc/quantization/marlin`、`gptq_marlin`）要解决"既要省显存带宽（权重低精度）、又要喂饱 tensor core（计算用 fp16）"。两个关键设计：

- **离线权重列重排**（`csrc/permute_cols.cu`）：tensor core 的取数有固定的 lane↔数据映射（mma 指令要求特定布局）。直接按原始列顺序取低精度权重，访存会乱、还要逐元素 dequant。Marlin 把权重的列**离线 permute** 成"GPU 取数时天然对齐 tensor core 布局"的顺序，于是 GEMM 时一次 `ld` 取一摞权重正好喂给 mma。`permute_cols_kernel`（`:14`）就是按 `perm` 索引重排列、且按 16bit 类型向量化（`int4` = 128bit 一次搬）。
- **dequant 融进 GEMM 取数**：不单独跑一个 dequant kernel 把全部权重转 fp16（那要写一整份 fp16 权重回显存，浪费带宽），而是在 GEMM 的内层循环里，**取到低精度权重的当下就地 dequant 成 fp16** 再喂 mma。权重始终以低精度躺在显存里、低精度被读进来，只有寄存器里短暂存在 fp16——带宽省在"读的是 int4 而非 fp16"。

详见 [模块 08](../08-quantization/design.md)；本模块补"为什么要 permute、dequant 怎么 fuse"的 kernel 动机。

---

## 4. 关键数据结构 / 显存布局

| 结构 / 布局 | 定义位置 | 角色 |
|---|---|---|
| `paged_attention_kernel` | `attention_kernels.cuh:90` | 核心 device 函数：paged decode attention，被 v1/v2 包装 |
| K cache 布局（经典） | `attention_kernels.cuh:97` 注释 | `[num_blocks, num_kv_heads, head_size/x, block_size, x]`，`x=16/sizeof(cache_t)`，为 warp 连续 128bit 访存重排 |
| V cache 布局（经典） | `attention_kernels.cuh:99` 注释 | `[num_blocks, num_kv_heads, head_size, block_size]` |
| K/V cache 布局（flash, V1） | `flash_attn.py:64` | `[2, num_blocks, block_size, num_kv_heads, head_size]`，规整、无重排 |
| `THREAD_GROUP_SIZE` | `attention_kernels.cuh:138` | `MAX(WARP_SIZE/BLOCK_SIZE,1)`，几个线程合算一个 KV token 的 Q·K |
| `Vec<T, VEC_SIZE>` / `Q_vec` / `K_vec` / `V_vec` | `attention_generic.cuh:26`、`dtype_*.cuh` | 向量化访存类型（`uint4` 等），一次取/算 16 字节 |
| `Qk_dot` | `attention_utils.cuh:49` | Q·K 点积 + 线程组内 shuffle reduce |
| `exp_sums` / `max_logits` / `tmp_out` | `paged_attention_v2.cu:53` | V2 各 partition 的局部 softmax 结果，供 reduce kernel 合并 |
| `paged_attention_v2_reduce_kernel` | `attention_kernels.cuh:567` | 把各 KV partition 的局部 attention 用 log-sum-exp rescale 合并 |
| `reshape_and_cache(_flash)_kernel` | `cache_kernels.cu:211` / `:265` | slot_mapping 驱动的 KV 分页写入（含 fp8 量化） |
| `copy_blocks_kernel` | `cache_kernels.cu:73` | block 级拷贝（CoW / PD 分离搬块） |
| `merge_attn_states_kernel` | `merge_attn_states.cu:15` | cascade / split-KV 的局部 attention 合并（128bit pack） |
| `Signal` | `custom_all_reduce.cuh:53` | all-reduce 跨卡自旋同步的 flag 数组（per-block, per-rank） |
| `RankData` | `custom_all_reduce.cuh:59` | 8 个 rank 的 peer 指针（IPC 打开后填入） |
| `CustomAllreduce` | `custom_all_reduce.cuh:370` | all-reduce 主类：IPC handle 管理、1/2-stage 分派、graph buffer |
| KV 可用显存 | `gpu_worker.py:182` | `总显存 * util - (torch峰值 + 非torch分配)` |

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 手写 paged decode kernel（不用 cuBLAS GEMM） | decode 的 M=1 瘦长形状专门优化；原生支持 block table 间接寻址 | 维护一大坨模板 CUDA；每个 head_size / block_size 组合单独编译（`paged_attention_v1.cu:99` switch），编译慢、二进制大 |
| THREAD_GROUP 合算一个 token | Q·K 点积并行 + 向量化访存；warp 内可整齐处理一个 block | 引入线程组内 reduce（多一层 shuffle）；`head_size % thread_group_size == 0` 约束 |
| K cache 维度重排（`head_size/x, block_size, x`） | warp 取一个 block 落连续 128bit，访存高效、规避 bank conflict | 写入地址算术复杂（`reshape_and_cache`）；与 flash 布局不兼容，要两套写 kernel |
| V1/V2 双 kernel（split-KV） | 长序列/小 batch 也能填满 GPU | 多一个 reduce kernel + tmp 张量显存；选择靠启发式（`paged_attn.py:128`），非最优 |
| fp8 KV cache | KV 显存砍半，长上下文/大 batch 可行 | 量化误差；读写各多一次 scaled_convert |
| profiling 量显存（跑 dummy forward） | 自动榨干剩余显存，无需手调 | 启动多花几秒；假设 profiling 期间别的进程不动显存（`gpu_worker.py:160`） |
| `gpu_memory_utilization` 默认 0.9 | 留 10% 裕度防 OOM / 碎片 | 太高易 OOM，太低浪费；要数 non-torch 分配否则仍可能 OOM |
| CUDA graph 固定地址 | 省 launch 开销，decode 吞吐显著上升 | 所有 buffer 必须持久预分配；动态 shape 要按 size 分别捕获、占显存 |
| custom all-reduce（IPC P2P） | 小消息低延迟，胜过 NCCL | 仅限 NVLink 全连接 + ≤8 卡 + 偶数卡（`custom_all_reduce.cu:17-20`）；graph 捕获要特殊处理 peer 指针 |
| Marlin 离线权重重排 | dequant-fuse，低精度 GEMM 喂满 tensor core | 权重加载时要 permute（一次性）；kernel 与特定布局强绑定 |

---

## 6. 设计动机出处

- **PagedAttention 论文**：Kwon et al., *"Efficient Memory Management for Large Language Model Serving with PagedAttention"*, **SOSP 2023**（arXiv:2309.06180）。本模块 §3.1 的 paged decode kernel、§3.4 的 block table 间接寻址、KV 分页布局均出自此。kernel 实现改编自 NVIDIA **FasterTransformer** 的 `decoder_masked_multihead_attention`（见各 `.cuh` 文件头注释）。
- **FlashAttention / online softmax**：Dao et al., *"FlashAttention"*（arXiv:2205.14135）及 *"FlashAttention-2"*（arXiv:2307.08691）。本模块 §3.2 V2 的 split-KV + log-sum-exp 合并、§3.1 的 online softmax 思想出自此。`merge_attn_states.cu:12` 注释明确引用 arXiv:2501.01005（split-KV 合并公式）。
- **FlashDecoding**：split-KV decode 的并行化思想（Dao et al.，斯坦福博客 2023-10），对应 V2 沿 KV 维分 partition 提升并行度。
- **vLLM custom all-reduce**：vLLM 文档与源码注释（`custom_all_reduce.cuh:519`）说明 36 block 的网格搜索结论、1-stage/2-stage 选择依据。one-shot/two-shot all-reduce 是经典 NCCL 算法术语（reduce-scatter + all-gather）。
- **Marlin**：Frantar et al., *"MARLIN: Mixed-precision Auto-Regressive Linear kernels"*（GPTQ 作者团队），核心是 int4×fp16 GEMM 的 dequant-fuse 与权重重排。本模块 §3.9 对应其设计。

> 以上为**设计动机**来源；所有 `file:line` 代码引用均以仓库实际代码为准（见 [`impl.md`](impl.md)）。

---

## 7. 一图速览：PagedAttention kernel 的线程/warp 映射

```
Grid (num_heads, num_seqs, max_num_partitions)
  例：head=0, seq=0, partition=0 → 一个 Thread Block (128 线程 = 4 warps)
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  shared memory:  q_vecs[Q 切片]   logits[本分区每个 KV token 的 qk]         │
  │                                                                            │
  │  block_table = block_tables + seq_idx * max_num_blocks_per_seq  ← 本seq页表 │
  │                                                                            │
  │  for block_idx = warp_idx; block_idx < end; block_idx += NUM_WARPS:        │
  │     physical_block = block_table[block_idx]      ← ★逻辑块→物理块 间接寻址   │
  │     ┌─ warp 0 ─────────────────────────────────────────────┐               │
  │     │  THREAD_GROUP_SIZE=2:  线程(0,1) 合算 token0 的 Q·K     │               │
  │     │                        线程(2,3) 合算 token1 的 Q·K     │               │
  │     │                        ...      合算 token15(一个block) │               │
  │     │  组内 shfl_xor reduce → 每个token的 qk                  │               │
  │     └────────────────────────────────────────────────────────┘             │
  │     warp1/2/3 同理处理 block_idx+1, +2, +3 ...                              │
  │                                                                            │
  │  ── softmax ──                                                             │
  │  warp 内 shfl reduce → 各 warp 局部 qk_max                                  │
  │  跨 warp (red_smem) reduce → block 全局 qk_max ─┐                           │
  │  logits[i] = exp(logit - qk_max);  exp_sum = block_sum(...)  ←两级reduce    │
  │  logits[i] *= 1/exp_sum                                                     │
  │                                                                            │
  │  ── 加权 V ──                                                              │
  │  for block: physical = block_table[block_idx]                              │
  │     accs[row] += dot(logits_vec, V_vec)    ← 每线程累加若干 head_size 行     │
  │  warp 内 reduce + 跨 warp reduce → out[head_size]                           │
  └──────────────────────────────────────────────────────────────────────────┘
       │ V1: 直接写 out[seq,head,:]
       │ V2: 写 tmp_out + exp_sums + max_logits，再由 reduce_kernel 合并：
       └─► reduce_kernel: 全局max ← 各partition max; rescale exp_sum;
                          out = Σ_p  tmp_out_p * (exp_sum_p * exp(max_p-globalmax)) / Σ
                          （log-sum-exp 合并，= FlashDecoding / merge_attn_states 同款）
```

> 实现层面的逐行调用链、kernel 内的指针算术、向量化访存与 bank conflict 规避、reduce 细节，见同目录 [`impl.md`](impl.md)。

---

## 8. 设计背后的考量与历史教训

### 8.1 设计背后的考量（深化"为什么"）

1. **为什么手写 paged decode kernel、而非复用 cuBLAS/FlashAttention**。decode 的 query 只有 1 个 token（M=1），标准 GEMM 在 tensor core 上大量空转；再叠加 PagedAttention 的"KV 物理不连续、要查 block table 跳读"——现成 GEMM/flash kernel 都不接受这种间接寻址。所以 vLLM 否决了"复用通用 kernel"的路线，改编自 NVIDIA FasterTransformer 的 `decoder_masked_multihead_attention`（见 `.cuh` 文件头注释）手写 `paged_attention_kernel`。代价是维护一大坨模板 CUDA（每个 head_size/block_size 组合单独编译、编译慢二进制大），但这是"既要 paged、又要 decode 高效"无法回避的成本。

2. **V1→V2 split-KV 是被"小 batch + 长序列"的并行度不足倒逼出来的**。最初只有 V1：一个 thread block 串行扫完整条序列的 KV。当 `num_seqs*num_heads` 填不满 GPU 的 SM、序列又很长时，大量 SM 闲置而单 block 串行扫几千 KV。V2 把 KV 沿序列维切成 512-token partition 并行算、再用 online-softmax 的 reduce kernel 合并——本质是把 FlashDecoding 的 split-KV 思想搬到 paged decode。选择走 V1 还是 V2 至今仍是启发式（`paged_attn.py:128`），承认了"没有静态最优解、只能按 batch/seq 规模猜"。

3. **KV 显存为何要真跑 dummy forward 量、而非静态推算**。激活峰值（尤其 prefill 大 batch 的中间张量）无法静态算准，所以 vLLM 的办法是 `profile_run` 真跑一次最大 batch 的 forward 量峰值（`gpu_worker.py:139`）。一个关键且反直觉的点：NCCL 通信 buffer、custom all-reduce 的 IPC buffer **不走 torch allocator**，必须从 `mem_get_info` 与 torch 计数的差值里把这些 non-torch 分配补回峰值，否则会高估可用显存而 OOM（`gpu_worker.py:171-181`）。这条"要数 non-torch 分配"的教训是踩过 OOM 坑后才补上的。

4. **custom all-reduce 用 IPC P2P，是对"NCCL 小消息延迟偏高"的针对性绕行**。decode 时激活很小，NCCL 的通用调度/proto 协商开销盖过实际传输。custom all-reduce 用 `cudaIpcOpenMemHandle` 让各卡直接读写对端显存、自旋 flag 同步，路径极短。但它有强约束（仅 NVLink 全连接 + ≤8 卡 + 偶数卡）、且只用 36 个 block（`custom_all_reduce.cuh:519` 注释：block 太多反而在 NVLink 上争用）——这是"只在特定拓扑下才划算"的专用优化，不是通用替代。

5. **CUDA graph 的"固定地址"约束如何向上反推整个数据通路的设计**。graph 录的是"对某固定显存地址执行某 kernel"，所以 input_ids/positions/slot_mapping/block_table/KV cache 都必须是预分配持久张量，每步 `copy_` 进去再 replay。这条底层约束直接解释了上层为何"block_id 永不变、block table append-only"（[模块 02](../02-paged-attention-kvcache/design.md) §5），也解释了 custom all-reduce 为何要"捕获期占位、replay 前填实际 peer 指针"（`graph_unreg_buffers_`）。一个看似底层的 GPU 约束，反向决定了上层多个模块的数据结构必须 append-only/地址稳定。

### 8.2 重要 bug 修复（真实、精选）

下列 PR 号 / issue 号均经 `git log`/`git show` 或代码注释在本仓库实际核对（注明出处）。

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| **#16693** [Bugfix][Kernel] fix potential cuda graph broken for merge_attn_states kernel | `merge_attn_states_kernel` 的 launch 宏没显式指定 stream（`<<<grid, block>>>` 用了默认 stream），cascade/split-KV 合并在 CUDA graph 捕获下可能被打断 | 修复改成 `<<<grid, block, 0, stream>>>`。教训：任何要被 CUDA graph 捕获的 kernel launch 都必须显式跑在当前 stream 上，默认 stream 会让 capture 失效——§8.1.5"固定地址/固定 stream"约束在一个不起眼的 launch 宏上被违背 |
| **#5dba2575** Resolve race conditions in Marlin kernel (#11493) | `gptq_marlin.cu` 存在线程间竞态，多线程对共享数据的读写顺序未正确同步，量化 GEMM 偶发结果错误 | 量化 kernel 把 dequant 融进 GEMM、多线程共享 staging 数据，同步稍有不慎即 data race。教训：dequant-fuse 的高性能 kernel 共享状态密集，正确的 barrier/同步是正确性底线，竞态在大 M 维或特定 occupancy 下才偶发、极难复现 |
| **#8558** [Bugfix] Fix potentially unsafe custom allreduce synchronization | custom all-reduce 的跨卡自旋同步（`Signal`/barrier）内存序不够安全，可能读到对端尚未写完的数据 | 重写了 128 行同步逻辑。教训：custom all-reduce 跳过 NCCL 自己实现跨卡同步，acquire/release 内存序必须严格正确——P2P 直接读写对端显存时，"对端是否写完"完全靠 flag 同步保证，弱内存序会让 1-stage 读到脏数据 |
| **#6852** [Bug Fix] Illegal memory access, FP8 Llama 3.1 405b | cutlass w8a8 的 broadcast load epilogue 在 FP8 大模型上越界访存，触发 illegal memory access | 量化 GEMM 的 epilogue（scale 广播）地址算术在特定 shape 下越界。教训：低精度 GEMM 的 scale 广播路径对 shape 边界敏感，405b 这类极端维度最容易把 epilogue 的隐式 shape 假设暴露出来 |
| **issue #641**（代码注释，`attention_kernels.cuh:424`） | 序列末尾、超出 `seq_len` 的 padding token 对应的 V 向量可能含 NaN，直接参与加权和会污染输出 | kernel 在最后一个 block 显式把越界位置的 `v_vec` 清零（`token_idx + j < seq_len ? v : zero`）。教训：paged attention 按 block 对齐扫描，最后一个 block 几乎总有越界 token，其 KV 是未初始化/脏数据，必须在累加前显式 mask 清零防 NaN——正确性 corner case 直接写进了 kernel 内层循环 |

补充：`gpu_model_runner.py:217` 的 M-RoPE `mrope_positions` 故意多留一个 dummy 位置使其非连续以兼容 torch.compile，注释引用了 `pull/12128#discussion_r1926431923`（已核对存在）——这是"为兼容编译器而刻意让张量非连续"的设计取舍，非 bug，但同属本模块"底层约束反推数据结构"的典型。

无"挖不到 bug"的情况：本模块涉及的 `csrc/**` kernel 与 V1 worker 路径 bug 修复历史丰富。
