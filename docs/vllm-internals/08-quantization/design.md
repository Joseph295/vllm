# 模块 08 · Quantization 量化 —— 设计文档

> 范围：vLLM 如何用**低精度数值格式**（INT4/INT8/FP8/FP4 等）压缩权重、激活、KV cache，从而降低显存占用、提升吞吐、降低成本。本模块讲清三个层次：(1) 量化方法的分类与原理（weight-only 与 W8A8 的本质区别），(2) vLLM 用一套 `QuantizationConfig + QuantizeMethodBase` 插件抽象做到**无侵入替换**任意 Linear/MoE/Attention 的计算逻辑，(3) 为什么高性能 kernel（Marlin/Machete/CUTLASS）需要在加载后对权重做重排（`process_weights_after_loading`）。
> 它是一个**横切关注点**：量化不改调度、不改 KV 分页，只改"某一层的 weights 怎么存、forward 怎么算"。代码主干在 `vllm/model_executor/layers/quantization/`，接入点在 `linear.py` 与 `attention/layer.py`。

---

## 1. 背景与定位

> 如果你还不清楚 **weights/KV cache 为何是显存大头、decode 为何 memory-bound** 这些前置概念，建议先读 [PRIMER 第一部分](../PRIMER.md#第一部分llm-推理服务-101先建立心智模型)（尤其 1.3 KV cache、1.4 memory-bound）。一句话先垫底：**量化 = 用更少的比特存权重 / 激活 / KV，省显存、省访存，换一点点精度。**

### 1.1 这个模块在系统里的位置

量化是一个**横切关注点**：它不改调度、不改 KV 分页、不改采样，只改"某一层的 weights 怎么存、forward 那一步怎么算"。

- **上游**：启动期解析 checkpoint 里的 `quantization_config`，决定每层用哪种量化方法（[模块 00](../00-request-lifecycle/design.md) 模型加载阶段）。
- **本模块**：在每个 Linear / FusedMoE / Attention 层的"创建权重 → 加载后重排 → forward"三步里介入，把全精度计算换成低精度计算。
- **下游**：量化层产出的结果对模型其余部分**透明**——上层看到的还是同样 shape 的 FP16 输出，只是中间用低比特算/存。它和编译（[模块 07](../07-cuda-graph-compile/design.md)）、KV 分页（[模块 02](../02-paged-attention-kvcache/design.md)）正交叠加。

一句话：**它是"插"在每层权重创建与 forward 之间的一套可替换策略，模型代码写一遍 fp16 即可同时支持所有量化方法。**

### 1.2 关键文件与类各干什么（先认人）

| 文件 / 类 | 主要作用（一句话） |
|---|---|
| `quantization/base_config.py` 的 `QuantizationConfig` (ABC) | 一种量化"方法"的配置 + 工厂；`get_quant_method(layer, prefix)` 是核心分派入口 |
| `quantization/base_config.py` 的 `QuantizeMethodBase` (ABC) | 单层量化策略三段式：`create_weights` / `process_weights_after_loading` / `apply` |
| `quantization/__init__.py` 的 `QUANTIZATION_METHODS` | 全局方法名注册表（25+ 种），名→类，运行时升级 kernel 靠遍历它 |
| `layers/linear.py` 的 `LinearBase.__init__` | **无侵入替换发生处**：构造每个 Linear 时调 `get_quant_method` 拿到 method |
| `quantization/gptq_marlin.py` / `awq_marlin.py` | weight-only（W4A16）的 GPTQ/AWQ + Marlin kernel 实现 |
| `quantization/fp8.py` | W8A8/FP8 的 Linear / MoE / KV cache 三类方法 |
| `quantization/kv_cache.py` 的 `BaseKVCacheMethod` | KV cache 量化：给 Attention 层加 `k_scale`/`v_scale`/`q_scale` |
| `utils/marlin_utils.py` | 把权重/scale/zero_point 重排成 Tensor-Core MMA 布局（性能关键的一次性重排） |

### 1.3 和相似模块 / 概念的区别与联系（专治"听过但分不清"）

- **weight-only（W4A16）vs W8A8 vs KV cache 量化**：三类正交，解决不同瓶颈。**weight-only**（GPTQ/AWQ）只压**权重**到 4/8-bit、激活保持 FP16，GEMM 时把权重反量化回 FP16 再算——**省访存、加速 memory-bound 的 decode，但算力不省**。**W8A8/FP8** 把**权重和激活都**压到 8-bit，直接走低精度 Tensor Core——**省算力、加速 compute-bound 的 prefill 与大 batch**。**KV cache 量化** 只压**写进 KV cache 的 k/v**、不动权重/GEMM——**扩容上下文与并发**。一个模型可以同时 weight-only + FP8 KV。
- **GPTQ vs AWQ vs FP8**：前两者是 weight-only 的两种**标定算法**——GPTQ 用二阶 Hessian 信息逐列量化、可带 act-order（`desc_act`）重排列；AWQ 按激活幅度保护"显著权重通道"、必带 zero_point、不需要 reorder。FP8 是**数值格式**（E4M3，4 位指数 3 位尾数），既可只量权重也可 W8A8 量激活。一句话：**GPTQ/AWQ 回答"INT4 怎么量得准"，FP8 回答"用哪种 8-bit 浮点"。**
- **scale 与 zero_point**：量化的核心两个旁路参数。**scale** 是缩放系数——把低比特整数还原回浮点的倍率（`w_fp16 = q_int * scale`）；**zero_point** 是零点偏移——非对称量化时整数 0 对应的浮点值（`w_fp16 = (q_int - zero_point) * scale`）。对称量化只需 scale（GPTQ），非对称量化两者都要（AWQ）。group-wise 量化是每 128 个权重共享一组 (scale, zp)，精度远好于整行一个 scale。
- **Marlin 为何要权重重排**：Marlin/Machete 这类高性能 kernel 把量化权重**直接喂进 NVIDIA Tensor Core 的 MMA 指令**，MMA 要求操作数按一种特定的、非行主序的 lane 排布。若 forward 时再重排，每次 GEMM 都要 reshuffle、得不偿失。于是 vLLM 在加载后**一次性**把权重 shuffle 成 MMA 期望的交错布局（`process_weights_after_loading`），换 forward 时"量化权重零额外 reshuffle 直流 MMA"。
- **本模块 vs 编译（[模块 07](../07-cuda-graph-compile/design.md)）**：量化产出的算子也必须**能被 torch.compile 追踪、能进 cudagraph**——量化 buffer 必须是固定地址的普通 `Parameter`、`apply` 路径必须可被 Dynamo 追踪。这条隐形约束催生过专门修复（见 §7.2）。

---

### 1.4 这个模块要解决的问题

LLM 推理的瓶颈分两段，量化对两段都有用：

1. **显存（capacity）**：一个 70B 模型 FP16 权重就要 140GB，单卡放不下。**weights** 是显存大头，其次是 **KV cache**（长上下文、大 batch 时 KV 能轻松超过权重）。把 weights 从 16-bit 降到 4-bit，显存直接降到 ~1/4；把 KV cache 从 FP16 降到 FP8，KV 容量翻倍 → 能跑更长上下文、更大 batch。
2. **带宽 / 吞吐（throughput）**：decode 阶段是**访存受限**（memory-bound）——每生成一个 token 都要把整个权重矩阵从 HBM 读一遍。weights 越小，读得越快，decode 越快。这是 weight-only 量化（GPTQ/AWQ）的核心收益来源：**即使算的时候要反量化回 FP16，省下的访存也远大于反量化开销**。
3. **算力（compute）**：prefill 阶段和大 batch decode 是**算力受限**（compute-bound）。这时要用**低精度 Tensor Core**（FP8/INT8 的 MMA 指令吞吐是 FP16 的 2×）才能真正加速——这就需要**激活也量化**（W8A8/FP8），让 GEMM 的两个操作数都是低精度。

一句话：**weight-only 主要省显存+加速 decode；W8A8/FP8 主要加速 compute-bound 的 prefill 与大 batch；KV cache 量化扩容上下文/并发。** vLLM 把这三类统一在一套插件抽象下。

---

## 2. 设计目标与约束

- **无侵入替换**：模型代码（`LlamaForCausalLM` 等）完全不感知量化。它只管声明 `RowParallelLinear(...)`、`Attention(...)`，由框架根据 `quant_config` 决定这一层用哪种 `QuantizeMethodBase`。模型作者**写一遍 fp16 模型即可同时支持所有量化方法**。
- **方法可插拔、可扩展**：每种量化方法（gptq/awq/fp8/compressed-tensors/...）是一个 `QuantizationConfig` 子类，注册进全局表（`__init__.py` 的 `QUANTIZATION_METHODS`），并提供 `register_quantization_config` 让第三方注册自定义方法。
- **checkpoint 格式自适应**：能读 HF 上各家工具产出的格式（AutoGPTQ、AutoAWQ、llm-compressor 的 compressed-tensors、TensorRT-LLM ModelOpt、bitsandbytes...），并在运行时**自动升级到更快的 kernel**（如把 `gptq` checkpoint 用 `gptq_marlin` kernel 跑）。
- **硬件能力门控**：每种方法声明 `get_min_capability()`（Volta 70 / Turing 75 / Ampere 80 / Hopper 90），框架据此校验并在不支持时优雅降级（如 FP8 在无原生 FP8 硬件的卡上 fallback 到 Marlin weight-only）。
- **三类层统一**：Linear、FusedMoE、Attention(KV cache) 共用 `QuantizationConfig.get_quant_method(layer, prefix)` 这一个分派入口，按 `layer` 类型返回不同的 method。
- **与 torch.compile / CUDA graph 兼容**（见 [模块 07](../07-cuda-graph-compile/design.md)）：量化层的 buffer 必须是固定地址的普通 `Parameter`（不能是带元数据的 Parameter 子类），`apply` 路径必须是可被 Dynamo 追踪的算子调用。

---

## 3. 核心设计思想

### 3.1 量化方法的三大分类

**(A) Weight-only（W4A16 / W8A16，激活保持 FP16/BF16）**
- 代表：**GPTQ**（`gptq.py`/`gptq_marlin.py`）、**AWQ**（`awq.py`/`awq_marlin.py`）、compressed-tensors 的 `wNa16`。
- 只把 weights 量化到 INT4/INT8（按 group 量化，带 scale，AWQ 还带 zero_point）；激活 `x` 保持 FP16。
- GEMM 时**先把权重反量化回 FP16**（或在 Marlin kernel 内部边读边反量化），再做 FP16 矩阵乘。所以**算力不省，省的是访存**——非常适合 memory-bound 的 decode。
- 动机出处：**GPTQ**（Frantar et al., *"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"*, arXiv:2210.17323, 2022）用二阶 Hessian 信息逐列量化以最小化量化误差；**AWQ**（Lin et al., *"AWQ: Activation-aware Weight Quantization"*, arXiv:2306.00978, 2023）发现"少数显著权重通道"对精度至关重要，按激活幅度对权重通道缩放后再量化，因此 AWQ 不需要 GPTQ 的 reorder（`desc_act`）也能保精度。

**(B) W8A8 / FP8（权重和激活都量化，走低精度 Tensor Core）**
- 代表：**FP8**（`fp8.py`，E4M3 格式）、compressed-tensors 的 `w8a8` int8/fp8、ModelOpt（`modelopt.py`）、QQQ（W4A8）。
- weights 和 activations **都**降到 8-bit。GEMM 直接用 INT8/FP8 Tensor Core（`cutlass_scaled_mm` 或 `torch._scaled_mm`），输出再用 `scale_a * scale_b` 反量化回 FP16。**算力真正翻倍**，适合 compute-bound 场景。
- 激活量化的两种 scale 策略：**static**（per-tensor，scale 从 checkpoint 加载，提前标定好）和 **dynamic**（运行时按激活算 scale，可以 per-token 更精确）。FP8 默认在支持 CUTLASS 时用 **per-token dynamic**（`Fp8LinearOp(use_per_token_if_dynamic=...)`，`fp8.py:182`）。
- 动机出处：**SmoothQuant**（Xiao et al., arXiv:2211.10438, 2022）提出把激活的"难量化"通过等价变换迁移到权重上，使 W8A8 可行；**FP8**（Micikevicius et al., *"FP8 Formats for Deep Learning"*, arXiv:2209.05433, 2022）定义了 **E4M3**（4 位指数 3 位尾数，动态范围小、精度高，用于权重/激活）和 **E5M2**（5 位指数 2 位尾数，范围大、精度低，多用于梯度/某些 KV cache）。vLLM 的 FP8 linear 只用 E4M3（`torch._scaled_mm` 限制）。

**(C) KV cache 量化（FP8 KV）**
- 代表：`kv_cache.py` 的 `BaseKVCacheMethod`、`fp8.py` 的 `Fp8KVCacheMethod`。
- 把写入 KV cache 的 k/v 张量量化到 FP8（E4M3），读取时反量化。**不动权重、不动 GEMM**，只动 attention 层往 paged KV cache 写/读的那一步（paged KV / `reshape_and_cache` 见 [模块 02](../02-paged-attention-kvcache/design.md)）。
- 收益：KV cache 容量翻倍 → 同显存下上下文更长、并发更高。代价：attention 数值精度略降，需要标定好的 `k_scale`/`v_scale`。

> 三类是正交的：一个模型可以同时 weight-only(W4A16) + FP8 KV cache。

### 3.2 group-wise 量化的 scale 与 zero_point（weight-only 的关键）

把一行权重整体用一个 scale 量化误差太大。group-wise 量化把每个输出通道的输入维度切成大小为 `group_size`（典型 128）的小组，**每组一个 scale**（对称量化）或 **scale + zero_point**（非对称量化）：

```
反量化:  w_fp16 = (q_int - zero_point) * scale       # 每个 group 一组 (scale, zp)
```

- **GPTQ** 对称量化为主（`uint4b8` = 无符号 4-bit 偏移 8），scale 按 group，`g_idx`（act-order/`desc_act`）记录列的重排顺序。
- **AWQ** 非对称量化，**必带 zero_point**（`qzeros`），scale 按 group（`GroupQuantScaleParameter`，`awq.py:141`）。
- `group_size == -1` 退化为 **per-channel**（每个输出通道一个 scale，`ChannelQuantScaleParameter`）。
- TP（张量并行，见 [模块 03](../03-distributed-parallel/design.md)）下 scale 的切分是个微妙点：当 scale 沿输入维分组且该层做了 row-parallel 切分，scale 也要跟着切（`scales_and_zp_input_dim=0`）；但若用了 act-order 或 group 跨越分片边界，则需在**所有 rank 上复制完整 scale**（`marlin_repeat_scales_on_all_ranks`，`gptq_marlin.py:249`）。

### 3.3 统一插件抽象：`QuantizationConfig` + `QuantizeMethodBase`

整个模块的骨架就两个抽象基类（`base_config.py`）：

```
QuantizationConfig (ABC)                      # 一种量化"方法"的配置 + 工厂
  ├─ from_config(hf_quant_cfg) -> 实例         # 从 HF checkpoint 的 quantization_config 解析
  ├─ get_min_capability() / get_name() ...     # 元信息、硬件门控
  ├─ override_quantization_method(...)         # 运行时升级 kernel（gptq → gptq_marlin）
  └─ get_quant_method(layer, prefix)           # ★ 核心工厂：按 layer 类型返回一个 method
                                               #   LinearBase  -> XxxLinearMethod
                                               #   FusedMoE    -> XxxMoEMethod
                                               #   Attention   -> XxxKVCacheMethod / BaseKVCacheMethod
                                               #   命中 ignore -> UnquantizedLinearMethod

QuantizeMethodBase (ABC)                       # "这一层怎么存权重、怎么算"的策略
  ├─ create_weights(layer, ...)                # ① 注册量化权重/scale/zp 等参数（替代普通 weight）
  ├─ process_weights_after_loading(layer)      # ② 加载完后重排/重量化（Marlin 重排在此发生）
  └─ apply(layer, x, bias)                     # ③ forward：反量化 / 低精度 GEMM
        子类: LinearMethodBase / FusedMoEMethodBase / BaseKVCacheMethod
```

**无侵入替换的关键一行**：`LinearBase.__init__`（`linear.py:227-232`）——模型构造每个 Linear 层时，若有 `quant_config` 就调 `quant_config.get_quant_method(self, prefix)` 拿到 method 并存为 `self.quant_method`；没有就用 `UnquantizedLinearMethod`。之后 `forward` 一律走 `self.quant_method.apply(self, x, bias)`（`linear.py:321/474/1258`）。模型代码对此**毫无感知**，量化方法被"插"进了每层的 weight 创建与 forward。

### 3.4 三阶段生命周期：create → process → apply

量化层的权重和普通层不一样，要走三步（对应 `QuantizeMethodBase` 三个方法）：

1. **create_weights**：不再注册一个 `[out, in]` 的 FP16 `weight`，而是注册量化布局的若干 buffer——如 GPTQ 的 `qweight`(int32 packed)、`scales`、`qzeros`、`g_idx`。这些用 vLLM 的 `Parameter` 子类（`PackedvLLMParameter`/`GroupQuantScaleParameter` 等），携带 `input_dim/output_dim/packed_dim` 元数据，让 **weight_loader 知道 TP 下怎么切分**。
2. **process_weights_after_loading**：权重从 checkpoint 加载完后调用一次（`loader.py:179`）。在这里做**一次性的重排/重量化**——把 checkpoint 的通用布局转成**特定 kernel 要的布局**（Marlin Tensor Core 交错布局、FP8 的转置+per-tensor 重量化等）。这是性能的关键，见 3.5。
3. **apply**：forward 热路径。要么"反量化权重 → FP16 GEMM"（weight-only 大 batch、AWQ 的 `awq_dequantize`+`matmul`），要么"调融合 kernel 边读边反量化"（`gptq_marlin_gemm`），要么"量化激活 → 低精度 GEMM → 反量化输出"（FP8/INT8 W8A8）。

### 3.5 为什么高性能 kernel 需要权重重排（process_weights_after_loading 的精髓）

checkpoint 里的量化权重是**按通用规则 packing** 的（如 GPTQ 把 8 个 int4 沿某维塞进一个 int32）。但 Marlin/Machete 这类 kernel 直接把量化权重喂进 NVIDIA **Tensor Core 的 MMA 指令**，MMA 要求每个 warp 的操作数元素按一种**特定的、非行主序的 lane 排布**。如果 forward 时再去重排，每次 GEMM 都要 reshuffle，得不偿失。

于是 vLLM 在加载后**一次性**把权重 shuffle 成 MMA 期望的交错布局（`ops.gptq_marlin_repack`），把 scale/zero_point 也按对应 permutation 重排（`marlin_permute_scales` / `marlin_zero_points`，`marlin_utils.py:217/248`），并 pack 进 int32 以匹配 kernel 的向量化反量化路径。代价是加载慢一点点、多一份内存峰值；收益是 forward 时**量化权重直接流进 MMA 单元，零额外 reshuffle**。`marlin_zero_points` 的注释（`marlin_utils.py:250-251`）点明："zero-points 在每个 MMA 上都要用，所以也要按 MMA 访问模式重排"。Marlin **GEMM kernel 边读边反量化直流 MMA 的 GPU 实现、以及 permute_cols 权重重排 kernel 的 csrc/ 细节**见 [模块 11](../11-gpu-kernels-memory/design.md)。

GPTQ 的 act-order（`desc_act`）还要在此 `argsort(g_idx)` 把列按量化分组排序（`gptq.py:254` / Marlin kernel 的 `g_idx_sort_indices`），FP8 W8A8 则在此把多个逻辑分片（qkv、gate_up）的不同 per-tensor scale **重量化合并成单一 scale**（`requantize_with_max_scale`，`fp8.py:370`），因为 `torch._scaled_mm` 只接受 per-tensor scale。

### 3.6 运行时 kernel 升级与降级

- **升级**：HF 上一个普通 `gptq` checkpoint，若硬件支持 Marlin，`GPTQMarlinConfig.override_quantization_method`（`gptq_marlin.py:133`）会在 `config.py:749-756` 的方法遍历中"截胡"，把方法名改成 `gptq_marlin`，用快得多的 Marlin kernel 跑同一份权重。awq → awq_marlin 同理。
- **降级**：FP8 在 capability < 89（无原生 FP8 矩阵指令）的卡上，`Fp8LinearMethod.__init__` 设 `use_marlin=True`（`fp8.py:171`），退化为 **FP8 权重 + Marlin weight-only**（激活不量化，`fp8.py:386` `del input_scale`），保证能跑只是不享受 W8A8 的算力加速。
- **kernel 选择器**：weight-only 的 Marlin/Machete/Exllama/AllSpark 由 `choose_mp_linear_kernel`（`kernels/mixed_precision/__init__.py:27`）按 `_POSSIBLE_KERNELS` 优先级 + capability + `can_implement` 逐个试，返回第一个可用的。Machete（capability 90，Hopper）优先级最高，其次 AllSpark、Marlin、Exllama。

---

## 4. 关键数据结构

| 结构 / 抽象 | 定义位置 | 角色 |
|---|---|---|
| `QuantizationConfig` (ABC) | `base_config.py:60` | 一种量化方法的配置+工厂；`get_quant_method` 是核心分派入口 |
| `QuantizeMethodBase` (ABC) | `base_config.py:11` | 单层量化策略：`create_weights`/`process_weights_after_loading`/`apply` |
| `LinearMethodBase` | `linear.py:136` | `QuantizeMethodBase` 的 Linear 特化（`apply(layer,x,bias)`） |
| `UnquantizedLinearMethod` | `linear.py:170` | 不量化的兜底：普通 `weight` + `F.linear` |
| `BaseKVCacheMethod` | `kv_cache.py:13` | KV cache 量化：给 Attention 层加 `k_scale`/`v_scale`/`q_scale` |
| `QUANTIZATION_METHODS` | `quantization/__init__.py:8` | 全局方法名注册表（25+ 种），`get_quantization_config` 名→类 |
| `MPLinearLayerConfig` | `kernels/mixed_precision/MPLinearKernel.py:13` | weight-only kernel 的统一配置（weight_type/group_size/has_g_idx/zero_points） |
| `MPLinearKernel` | 同上 `:24` | weight-only kernel 抽象（Marlin/Machete/Exllama/AllSpark 子类） |
| `CompressedTensorsScheme` | `compressed_tensors/schemes/compressed_tensors_scheme.py:11` | compressed-tensors 的 per-layer scheme 抽象 |
| `scalar_types.uint4b8` 等 | `vllm/scalar_type.py` | 量化数值类型（位宽+偏移+是否有符号），驱动 kernel 支持判定 |
| `PackedvLLMParameter` / `GroupQuantScaleParameter` / `ChannelQuantScaleParameter` / `PerTensorScaleParameter` | `vllm/model_executor/parameter.py` | 携带 TP 切分元数据的量化参数，weight_loader 据此切分 |

**典型量化层的参数布局对照**：

| 方法 | weight buffer | scale | zero_point | 其他 |
|---|---|---|---|---|
| GPTQ (W4A16) | `qweight` int32(packed 8×int4) | `scales` per-group | `qzeros` int32(packed) | `g_idx`(act-order) |
| AWQ (W4A16) | `qweight` int32 | `scales` per-group | `qzeros` int32（必有） | — |
| FP8 (W8A8) | `weight` float8_e4m3fn | `weight_scale` per-tensor/channel | — | `input_scale`(static 时) |
| FP8-block | `weight` float8_e4m3fn | `weight_scale_inv` per-block(128×128) | — | 激活 per-token dynamic |
| INT8 W8A8 | `weight` int8 | `weight_scale` per-channel | — | `input_scale`/dynamic + 可选 azp |
| FP8 KV | （在 Attention 层）`_k_scale`/`_v_scale`/`_q_scale` per-tensor | — | — | cache 存 float8_e4m3fn |

---

## 5. 权衡取舍（精度 vs 速度 vs 显存）

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| weight-only W4A16（GPTQ/AWQ） | 显存 ~1/4，decode（memory-bound）显著加速 | 反量化开销使 compute-bound（大 batch/prefill）几乎不加速；4-bit 有精度损失 |
| W8A8 / FP8（激活也量化） | 用低精度 Tensor Core，prefill/大 batch 算力翻倍 | 激活量化更敏感，需 static 标定或 dynamic 算 scale；E4M3 动态范围小，离群值可能溢出 |
| group-wise scale（group_size=128） | 精度远好于 per-tensor | 多存一份 scale；TP 切分逻辑复杂（跨分片 group 要复制 scale） |
| Marlin 等 kernel 的权重重排 | forward 零 reshuffle，量化权重直流 MMA | 加载时一次性重排成本 + 内存峰值；布局与 kernel 强绑定，换 kernel 要重排 |
| 运行时升级 gptq→gptq_marlin | 同一份 checkpoint 自动跑更快 kernel | 仅特定硬件/配置可转；转换判定（`is_gptq_marlin_compatible`）有约束 |
| FP8 KV cache | KV 容量翻倍，上下文/并发翻倍 | attention 精度略降；需要标定 `k/v_scale`，否则用 1.0 会掉点（`kv_cache.py:93` 警告） |
| per-token dynamic 激活 scale | 比 per-tensor 更精确，抗离群 | 每次 forward 多一次 quant kernel；需 CUTLASS 支持 |
| 无侵入插件抽象 | 模型写一遍支持全部量化方法 | 抽象层多、间接调用多；新 kernel 要塞进 `create/process/apply` 三段式契约 |

---

## 6. 设计动机出处（标注）

- **GPTQ**：Frantar, Ashkboos, Hoefler, Alistarh. *"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."* arXiv:2210.17323, 2022. —— 二阶（Hessian）误差补偿的逐列 4-bit PTQ。代码引用见 `gptq.py:27`。
- **AWQ**：Lin et al. *"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."* arXiv:2306.00978, 2023. —— 按激活幅度保护显著权重通道。代码引用见 `awq.py:19`。
- **SmoothQuant**：Xiao et al. *"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models."* arXiv:2211.10438, 2022. —— W8A8 可行性的奠基（激活难量化 → 迁移到权重）。
- **FP8 (E4M3/E5M2)**：Micikevicius et al. *"FP8 Formats for Deep Learning."* arXiv:2209.05433, 2022. —— 两种 8-bit 浮点格式的定义与权衡；vLLM linear 用 E4M3（`fp8.py` 注释）。
- **Marlin kernel**：Frantar et al. *"MARLIN: Mixed-precision Auto-Regressive Parallel Inference."*（NeurIPS workshop / IST-DASLab）—— 面向 batch 推理、把量化权重直流 Tensor Core 的混合精度 GEMM；vLLM 集成于 `csrc/quantization/gptq_marlin/` 与 `marlin_utils.py`。
- **compressed-tensors**：Neural Magic / Red Hat 的 `llm-compressor` 产出的统一 checkpoint 格式，用 `config_groups`(targets/ignore) 描述每层 scheme；vLLM 适配见 `compressed_tensors/compressed_tensors.py`。
- **Machete**：vLLM 自带的 Hopper（capability 90）mixed-precision GEMM，weight 预排布走 `machete_prepack_B`（`machete.py`），优先级高于 Marlin。

> 实现层面的逐行调用链、`file:line` 对照、kernel 重排细节与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 7. 设计背后的考量与历史教训

### 7.1 设计背后的考量（深化"为什么"）

1. **三段式契约（create → process → apply）是被"高性能 kernel 要特殊布局"倒逼的**：最朴素的设计是只有一个 `apply`，forward 时按需反量化。但 Marlin/Machete 这类 kernel 把量化权重直接喂进 Tensor Core 的 MMA 指令，要求权重按非行主序的 lane 排布（§3.5）。若每次 GEMM 都 reshuffle，得不偿失。于是否决"懒重排"，引入独立的 `process_weights_after_loading` 阶段做**一次性**重排。代价是加载慢一点、多一份内存峰值，换 forward 时零 reshuffle、量化权重直流 MMA。**布局成本被前移到启动期，是因为它在热路径上无法摊销**。

2. **`QuantizationConfig.get_quant_method(layer, prefix)` 单一分派入口，是为了让模型代码"写一遍 fp16 即支持全部量化"**：替代方案是每种量化方法在模型代码里插 if/else。否决理由是那会让 25+ 种方法 × N 个模型组合爆炸，且模型作者要懂量化。最终把分派收敛到一个工厂方法，按 `layer` 类型（Linear/FusedMoE/Attention）返回不同 method，模型对量化**毫无感知**（§3.3）。代价是抽象层多、间接调用多、新 kernel 必须塞进三段式契约。

3. **运行时 kernel 升级（gptq→gptq_marlin）而非要求用户重新量化**：HF 上海量 checkpoint 是用 AutoGPTQ/AutoAWQ 产的通用格式。可选方案是只支持"为 Marlin 专门量化的 checkpoint"。否决理由是那会割裂生态。于是用 `override_quantization_method` 在 config 解析阶段"截胡"，把同一份权重用更快的 Marlin kernel 跑（§3.6）。这是"自适应 checkpoint 格式"约束的直接产物——**适配生态既有格式，比追求理论最优布局更重要**。

4. **硬件能力门控 + 优雅降级，而非"不支持就报错"**：FP8 在 capability < 89 的卡上没有原生 FP8 矩阵指令。可以直接拒绝运行，但 V1 选择降级为"FP8 权重 + Marlin weight-only"（激活不量化，`fp8.py` 删 input_scale）。每种方法声明 `get_min_capability()`，框架据此校验并降级。**让模型"能跑、只是不享受全部加速"，胜过"在这张卡上根本跑不了"**。

5. **per-tensor / per-channel / per-token / per-group 是"精度 vs 开销"光谱上的多个刻度，没有银弹**：FP8 linear 在支持 CUTLASS 时默认 per-token dynamic（更精确、抗离群），但 `torch._scaled_mm` 只接受 per-tensor scale——所以 `process_weights_after_loading` 要把 qkv、gate_up 这些逻辑分片的不同 per-tensor scale 用 `requantize_with_max_scale` 重量化合并成单一 scale（§3.5）。group-wise（128）精度远好于 per-tensor，但 TP 切分时跨分片的 group 要在所有 rank 上复制完整 scale（`marlin_repeat_scales_on_all_ranks`）。**kernel API 的限制（只收 per-tensor）反过来约束了上游的 scale 布局设计**。

6. **量化层必须与 torch.compile / CUDA graph 兼容，是一条贯穿全模块的隐形约束**（见 [模块 07](../07-cuda-graph-compile/design.md)）：量化 buffer 必须是固定地址的普通 `Parameter`，`apply` 路径必须可被 Dynamo 追踪。这条约束催生过专门的修复（`#13918` 让 FP8 Linear 兼容 torch.compile、`#13425` 绕开 V1+CUDA graph+`torch._scaled_mm` fallback）。教训：在 vLLM 里，"一个量化 kernel 算得对"只是及格线，**它还必须能被编译、能进 cudagraph**，否则在 V1 默认路径上根本用不了。

### 7.2 重要 bug 修复（真实、精选）

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| `#13918` | FP8 Linear 与 torch.compile 不兼容 | `apply` 路径里有 Dynamo 无法追踪的操作。教训：量化 kernel 的正确性之外，还有"可被编译追踪"这条 V1 硬约束，二者都要满足 |
| `#13425` | V1 + CUDA graph + `torch._scaled_mm` 的 fallback 路径出问题 | `torch._scaled_mm` 在某些情形会走 fallback，与 cudagraph 固定执行流冲突。教训：依赖 PyTorch 内部算子的"隐式 fallback"在 cudagraph 下是定时炸弹，要显式 work around |
| `#13632` | nvfp4 量化 kernel 用了 `getStreamFromPool` 拿到错误的 stream（而非当前 stream） | 改用 `getCurrentCUDAStream`。教训：自定义 CUDA kernel 必须在**调用方的当前 stream** 上跑，从 pool 另取 stream 会破坏与 cudagraph / 上下游 kernel 的执行序，引发隐蔽的竞态/数据错乱 |
| `#12696` | Hopper `cutlass_scaled_mm` 的 per-token/per-channel 量化算错 | CUTLASS W8A8 GEMM 的 scale 广播维度在 Hopper 上需特殊处理。教训：低精度 GEMM 的 scale 应用是数值正确性的命门，per-token/per-channel 的广播方向极易写反 |
| `#13784` | FP8 + Expert Parallel（MoE）组合下出错 | MoE 的多专家 + 量化 + EP 切分三者交织，`get_quant_method` 在 FusedMoE 路径上的处理有缺陷。教训：量化与 MoE/并行的正交性是设计目标，但正交≠自动正确，组合路径要单独测 |
| `#15319` / `#15398` | 给 Marlin kernel 传了非连续（non-contiguous）输入的修复（`#15319`）随后被回滚（`#15398`） | Marlin kernel 假设输入内存连续，修复方式有副作用故回滚。教训：高性能 kernel 对输入 layout（连续性/对齐）有隐式假设，"加个 `.contiguous()`"看似无害却可能引入性能回退或掩盖真正的上游 bug |
| `#13750` | gptq_marlin 在 DeepSeek-V3 上出错 | 特定模型结构（如其 MoE/形状）触发 marlin 转换判定的边界。教训：`is_gptq_marlin_compatible` 这类"能否升级 kernel"的判定必须覆盖真实模型的非常规形状，否则升级路径会在新模型上崩 |

---

## 8. ASCII 全景图

```
                         ┌──────────────────────────────────────────────┐
启动期                    │  ModelConfig.quantization (CLI / HF config)   │
(config 解析)             │  config.py:_verify_quantization 遍历方法表     │
                         │   → override_quantization_method 升级 kernel   │
                         │   (gptq → gptq_marlin, awq → awq_marlin)       │
                         └───────────────────────┬──────────────────────┘
                                                 │ get_quant_config → from_config
                                                 ▼
                         ┌──────────────────────────────────────────────┐
                         │   QuantizationConfig 实例 (Fp8Config /          │
                         │   GPTQMarlinConfig / CompressedTensorsConfig…) │
                         └───────────────────────┬──────────────────────┘
                                                 │ 模型构造每层时
                  ┌──────────────────────────────┼──────────────────────────────┐
                  ▼                               ▼                              ▼
      LinearBase.__init__              FusedMoE.__init__              Attention.__init__
   get_quant_method(self,prefix)    get_quant_method(...)         get_quant_method(...)
            │                               │                              │
            ▼                               ▼                              ▼
   XxxLinearMethod              XxxMoEMethod                  BaseKVCacheMethod
   (或 UnquantizedLinearMethod)                              (加 _k_scale/_v_scale)
            │                               │                              │
            │   三阶段生命周期 (QuantizeMethodBase)                          │
            ▼                               ▼                              ▼
   ① create_weights        注册量化布局参数: qweight(int32) / scales / qzeros /
                           g_idx / weight(fp8) / weight_scale / input_scale ...
            │
            ▼  ← 权重从 checkpoint 加载 (weight_loader 按 TP 切分 weight 与 scale)
   ② process_weights_after_loading   ← loader.py:179, 一次性
        weight-only:  ops.gptq_marlin_repack 重排成 Tensor-Core 交错布局
                      marlin_permute_scales / marlin_zero_points 重排 scale/zp
        W8A8/FP8:     转置 weight, requantize_with_max_scale 合并逻辑分片 scale
        KV:           k_scale/v_scale 加载/标定 → _k_scale/_v_scale
            │
            ▼  forward 热路径
   ③ apply(layer, x, bias)
        weight-only:  gptq_marlin_gemm (边读边反量化, 直入 MMA)  / awq_dequantize+matmul
        W8A8/FP8:     scaled_fp8_quant(激活) → cutlass_scaled_mm → 反量化输出
        KV(在 attention backend): reshape_and_cache_flash(写 fp8) + q/k/v_descale(读)

────────────────────────────────────────────────────────────────────────────
三类方法的本质区别:
   weight-only (W4A16) :  低精度只在 HBM ↔ 反量化回 FP16 算    → 省访存, 加速 decode
   W8A8 / FP8          :  权重+激活都低精度, 算也用低精度 MMA  → 省算力, 加速 prefill
   KV cache FP8        :  只压 KV 读写, 不动权重/GEMM          → 扩容上下文/并发
```
