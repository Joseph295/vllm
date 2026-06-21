# 模块 08 · Quantization 量化 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的量化调用链：从启动期解析 `quant_config`，到加载时按方法替换每层的 `quant_method`，到 `process_weights_after_loading` 重排权重，再到 forward 的 `apply` 反量化/低精度 GEMM。
> 所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。代码根目录 `vllm/model_executor/layers/quantization/`（下文相对该目录或写全路径）。

---

## 1. 代码地图

```
抽象层 quantization/
  ├─ base_config.py             # QuantizationConfig / QuantizeMethodBase 两个 ABC
  ├─ __init__.py                # QUANTIZATION_METHODS 注册表 + get_quantization_config 名→类
  └─ kv_cache.py                # BaseKVCacheMethod：给 Attention 加 k/v/q_scale

接入点（无侵入替换发生处）
  ├─ ../linear.py               # LinearMethodBase / UnquantizedLinearMethod / LinearBase.__init__
  ├─ ../fused_moe/layer.py      # FusedMoEMethodBase
  └─ ../../../attention/layer.py# Attention.__init__ 装 BaseKVCacheMethod

weight-only 方法（W4A16/W8A16）
  ├─ gptq.py / gptq_marlin.py   # GPTQ（exllama）/ GPTQ-Marlin
  ├─ awq.py / awq_marlin.py     # AWQ / AWQ-Marlin
  ├─ marlin.py / gptq_marlin_24.py
  ├─ kernels/mixed_precision/   # MPLinearKernel + choose_mp_linear_kernel
  │    ├─ __init__.py           #   _POSSIBLE_KERNELS 优先级表
  │    ├─ MPLinearKernel.py     #   抽象基类 + MPLinearLayerConfig
  │    ├─ marlin.py machete.py exllama.py allspark.py
  └─ utils/marlin_utils.py      # 权重/scale/zp 重排（Tensor-Core 布局）

W8A8 / FP8 方法
  ├─ fp8.py                     # Fp8Config / Fp8LinearMethod / Fp8MoEMethod / Fp8KVCacheMethod
  ├─ modelopt.py                # ModelOpt FP8 / NVFP4
  └─ compressed_tensors/        # CompressedTensorsConfig + schemes/（int8/fp8/wNa16/2:4）

config 与加载
  ├─ ../../../config.py         # ModelConfig._verify_quantization / CacheConfig._verify_cache_dtype
  ├─ ../../model_loader/weight_utils.py  # get_quant_config：from_config 实例化
  └─ ../../model_loader/loader.py        # _process_weights_after_loading 调用点

kernel Python 绑定
  └─ ../../../_custom_ops.py    # gptq_marlin_gemm / awq_* / scaled_fp8_quant / cutlass_scaled_mm …
```

---

## 2. 端到端调用链

以一个 **GPTQ checkpoint 自动升级为 gptq_marlin + 可选 FP8 KV cache** 为主线，FP8/compressed-tensors 差异在各步内对比。

### 阶段 A：启动期 —— 解析 quant 方法、运行时升级 kernel

**A1.** `vllm/config.py:732` `ModelConfig._verify_quantization()`
- `:733` `supported_quantization = QUANTIZATION_METHODS`。
- `:743` `_parse_quant_hf_config()`（`:725-730`）：从 `hf_config.quantization_config` 或 `compression_config` 读出 checkpoint 自带的量化配置。
- `:749-756` **运行时升级 kernel 的核心遍历**：
  ```python
  for name in QUANTIZATION_METHODS:
      method = get_quantization_config(name)
      quantization_override = method.override_quantization_method(
          quant_cfg, self.quantization)
      if quantization_override:
          quant_method = quantization_override
          self.quantization = quantization_override
          break
  ```
  > 注意 `QUANTIZATION_METHODS`（`quantization/__init__.py:8`）里 marlin 系列**排在 gptq/awq 前面**，所以一个 `gptq` checkpoint 会先被 `GPTQMarlinConfig.override_quantization_method` 截胡升级。

**A2.** `gptq_marlin.py:133` `GPTQMarlinConfig.override_quantization_method()`
- `:135` `can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)`（`:168` 检查 quant_method=="gptq"、bits/group_size/sym/desc_act 齐全、`check_marlin_supported`、`current_platform.is_cuda()`）。
- `:137-144` 若可转且用户没强制别的，返回 `"gptq_marlin"`，并打印 "convertible ... Using gptq_marlin kernel"。
- `:146-150` 若用户显式 `quantization=gptq`，则尊重用户、不升级（提示用 gptq_marlin 更快）。

### 阶段 B：实例化 QuantizationConfig

**B1.** `vllm/model_executor/model_loader/weight_utils.py:143` `get_quant_config()`
- `:146` `quant_cls = get_quantization_config(model_config.quantization)`（名→类，走 `quantization/__init__.py:78`）。
- `:153-162` 取出 `hf_config.quantization_config`（或 vision 模型的 `text_config.quantization_config`、或 `compression_config`）。
- `:164` **`return quant_cls.from_config(hf_quant_config)`** —— 真正实例化。

**B2.** 各方法的 `from_config` 解析自家字段：
- `gptq_marlin.py:118` 读 `bits/group_size/desc_act/sym/lm_head/dynamic`。
- `fp8.py:101` 读 `quant_method`（判 `is_checkpoint_fp8_serialized`）、`activation_scheme`、`ignored_layers`、`weight_block_size`。
- `compressed_tensors/compressed_tensors.py:102` 读 `ignore/format`，并调 `_quantization_scheme_map_from_config`（`:143`）把 `config_groups` 的每个 `targets` 映射成 `{weights, input_activations}` 的 `QuantizationArgs`。

**B3.** `vllm/config.py:3712` `VllmConfig._get_quantization_config()`
- `:3720` `quant_config = get_quant_config(model_config, load_config)`。
- `:3721-3737` 校验 GPU capability ≥ `quant_config.get_min_capability()` 与激活 dtype 支持。
- 存进 `VllmConfig.quant_config`（字段 `config.py:3604`，`__post_init__` 赋值 `:3776-3778`），随后 `loader.py:125 configure_quant_config(...)` 给它注入 `packed_modules_mapping`，并把整个 `vllm_config` 传给模型构造（`loader.py:133`）。

### 阶段 C：构造模型 —— 每层选择 quant_method（无侵入替换发生处）

**C1.** `vllm/model_executor/layers/linear.py:207` `LinearBase.__init__()` —— **★最关键的插件替换点**
```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()          # :228-229
else:
    self.quant_method = quant_config.get_quant_method(self, prefix=prefix)  # :231-232
```
模型代码只是 `RowParallelLinear(..., quant_config=quant_config, prefix=...)`，这里就把该层的计算策略"插"了进去。

**C2.** `gptq_marlin.py:153` `GPTQMarlinConfig.get_quant_method()`
- `:155-164` 若 `isinstance(layer, FusedMoE)` → `GPTQMarlinMoEMethod`（不支持则 fallback `MoeWNA16Config`）。
- `:165` 否则 `get_linear_quant_method(self, layer, prefix, GPTQMarlinLinearMethod)` —— 内部按 `dynamic` 配置/`lm_head` 等决定返回 `GPTQMarlinLinearMethod` 还是 `UnquantizedLinearMethod`。

> 对比分派：
> - `awq.py:75`：`LinearBase` → `AWQLinearMethod`；命中 `modules_to_not_convert` → `UnquantizedLinearMethod`。
> - `fp8.py:114`：`LinearBase`→`Fp8LinearMethod`（`ignored_layers` 命中则 Unquantized），`FusedMoE`→`Fp8MoEMethod`，`Attention`→`Fp8KVCacheMethod`。
> - `compressed_tensors.py:77`：`should_ignore_layer`→Unquantized；`LinearBase`→`get_scheme()` 选出 per-layer scheme 挂到 `layer.scheme`，返回 `CompressedTensorsLinearMethod`；`Attention`→`CompressedTensorsKVCacheMethod`。

**C3.** `vllm/attention/layer.py:102-117` `Attention.__init__()` —— KV cache 量化接入
```python
quant_method = quant_config.get_quant_method(self, prefix=prefix) if quant_config else None
if quant_method is not None and not isinstance(quant_method, UnquantizedLinearMethod):
    assert isinstance(quant_method, BaseKVCacheMethod)        # :106
    ...
    self.quant_method = quant_method
    self.quant_method.create_weights(self)                    # :117 → 加 k/v/q_scale
```
`kv_cache_dtype` 来自 `cache_config.cache_dtype`（`layer.py:66`，存 `:83`），`_k_scale/_v_scale/_q_scale` 初值 1.0（`layer.py:85-94`）。

### 阶段 D：① create_weights —— 注册量化布局参数

**D1.** weight-only（GPTQ-Marlin）：`gptq_marlin.py:210` `create_weights()`
- `:224-235` 组装 `MPLinearLayerConfig` 并 `choose_mp_linear_kernel(...)` 选 kernel 类（Marlin/Machete/…）。
- `:249-260` **TP 下 scale 切分判定**：`marlin_repeat_scales_on_all_ranks(desc_act, group_size, is_row_parallel)` 为真则 `scales_and_zp_input_dim=None`（各 rank 复制完整 scale），否则 `=0`（沿输入维切分）。
- `:263-273` `qweight = PackedvLLMParameter(...)`：int32、`packed_dim=0`、`packed_factor=32//bits`（4-bit → 8 个塞一个 int32）。
- `:276-327` 注册 `g_idx`(RowvLLMParameter)、`scales`（`ChannelQuantScaleParameter` 或 `GroupQuantScaleParameter`）、`qzeros`(packed)。
- `:329-333` `self.kernel = kernel_type(config, w_q_param_name="qweight", w_s_param_name="scales", w_zp_param_name="qzeros", w_gidx_param_name="g_idx")`。

> 对比：
> - GPTQ(exllama) `gptq.py:133`：同样注册 `qweight/g_idx/qzeros/scales`，额外 `layer.exllama_state`（`:241`）。
> - AWQ `awq.py:98`：`qweight`(packed_dim=1)、`qzeros`(必有)、`scales`(GroupQuantScaleParameter)。
> - FP8 `fp8.py:186`：`weight` 为 `float8_e4m3fn`（serialized 时）；非-block 时 `weight_scale` 是 `PerTensorScaleParameter`（`:250-258`），block 时 `weight_scale_inv` 是 `BlockQuantScaleParameter`（`:259-274`，名字特意为 deepseekv3 兼容）；`activation_scheme=="static"` 才注册 `input_scale`（`:277-284`），dynamic 则 `None`。

### 阶段 E：weight_loader —— 加载并按 TP 切分权重与 scale

权重从 checkpoint 流入时，`ColumnParallelLinear.weight_loader`（`linear.py:419`）沿 `output_dim` 切，`RowParallelLinear.weight_loader`（`linear.py:1193`）沿 `input_dim` 切。**scale 是否跟着切**由 vLLM `Parameter` 子类的元数据决定：`GroupQuantScaleParameter` 带 `input_dim` → 沿输入维切；`PerTensorScaleParameter`（0 维标量）→ 复制不切（`linear.py:451-454` 的标量特例处理 AutoFP8）。这正是 D1 里 `scales_and_zp_input_dim` 的意义落地处。

### 阶段 F：② process_weights_after_loading —— 一次性重排/重量化（性能关键）

**F1.** 调用点：`vllm/model_executor/model_loader/loader.py:164` `_process_weights_after_loading()`
```python
for _, module in model.named_modules():
    ...
    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):           # :173
        with device_loading_context(module, target_device):   # :179 (CPU offload 时临时上 device)
            quant_method.process_weights_after_loading(module) # :180
```
MLA 的 `Attention` 模块在 `:185-190` 单独处理（晚于其他模块，便于解压 MLA 权重）。

**F2.** GPTQ-Marlin：`gptq_marlin.py:335` → `self.kernel.process_weights_after_loading(layer)`，实际在 `kernels/mixed_precision/marlin.py:48`：
- `:66-69` `transform_w_q`：把 qweight param 重排为 `{input_dim=0, output_dim=1, packed_dim=0}` 后调 **`ops.gptq_marlin_repack`**（`_custom_ops.py:765`）—— 把 int4/int8 权重 shuffle 成 Marlin Tensor-Core 交错布局（带 `g_idx_sort_indices` 的列重排）。
- `:76-79` `transform_w_s`：**`marlin_permute_scales`**（`marlin_utils.py:217`）按 MMA 访问模式重排 scale。
- `:85-89` 若 `has_g_idx`：`marlin_sort_g_idx` 排序并存 `g_idx_sort_indices`；`:94-104` 若 `zero_points`：`marlin_zero_points` 解包→重排→重打包 int32。

> **为何重排**：Marlin kernel 把量化权重直接喂进 NVIDIA MMA 指令，MMA 要求操作数按特定 lane 排布。一次性在加载后排好，forward 就零 reshuffle。`marlin_utils.py:250-251` 注释："zero-points 在每个 MMA 上都要用，所以也要按 MMA 模式重排"；`:255` "interleave 列维（给 dequantize 用）并 pack 成 int32"。

**F3.** GPTQ(exllama)：`gptq.py:243`
- `:245-248` 把 `Parameter` 子类换成普通 `Parameter`（**torch.compile 兼容**，见 [模块 07](../07-cuda-graph-compile/design.md)——Dynamo 不支持带元数据的 Parameter 子类）。
- `:252-261` 首次：act-order 时 `g_idx = argsort(g_idx)`（`:254`），否则清空 g_idx；`ops.gptq_shuffle` 原地 shuffle qweight。

**F4.** FP8 linear：`fp8.py:299` `process_weights_after_loading()`
- **block-quant 路径**（`:301-318`）：fnuz 归一化（ROCm，`:303-307`），把 Parameter 子类转普通 Parameter（`:314-317`，torch.compile），return。
- **非-serialized（fp16/bf16 权重）路径**（`:321-336`）：`qweight, scale = ops.scaled_fp8_quant(layer.weight, scale=None)`（`:322`）在线量化权重到 FP8；存转置 `qweight.t()`（`:334`），`input_scale=None`（`:336`）。
- **serialized fp8 路径**（`:340-382`）：W8A8 子路径先 fnuz 归一化（`:360-368`），再 **`requantize_with_max_scale`**（`:370`）把 qkv/gate_up 等**多个逻辑分片的不同 per-tensor scale 重量化合并成单一 per-tensor scale**——因为 `torch._scaled_mm` 只接受 per-tensor scale（`:353-354` 注释）；存 `weight.t()`（`:378`）。
- **marlin 收尾**（`:384-387`）：`prepare_fp8_layer_for_marlin(layer)` 后 `del layer.input_scale`（`:386` "Activations not quantized for marlin." —— 降级为 weight-only）。

**F5.** FP8 KV cache：`kv_cache.py:46` `BaseKVCacheMethod.process_weights_after_loading()`
- `:51` 若 `kv_cache_dtype != "auto"` 且非 dynamic 计算：从 checkpoint 的 `k_scale/v_scale` 读出（`:52-74`，含 fnuz `*2`、单 kv_scale 复制到 k/v 的兜底），写进 `_k_scale/_v_scale/_q_scale`（`:86-92`）。
- `:93-98` 若 scale==1.0 且非 e5m2，警告 "Using KV cache scaling factor 1.0 ... may cause accuracy issues"。
- `:100-101` 删除临时的 `k_scale/v_scale` Parameter。

### 阶段 G：③ apply —— forward 热路径

forward 统一经 `linear.py:321/474/1258` 的 `self.quant_method.apply(self, x, bias)` 触发。

**G1.** GPTQ-Marlin：`gptq_marlin.py:338` `apply()` → `self.kernel.apply_weights(layer, x, bias)` → `kernels/mixed_precision/marlin.py:110` → **`apply_gptq_marlin_linear`**（`marlin_utils.py:324`）：
- `:348` `ops.gptq_marlin_gemm(...)`（`_custom_ops.py:806`）—— **边读 int4 权重边在 kernel 内反量化，直接做 FP16 输出的 MMA**。`:307` `should_use_atomic_add_reduce` 决定归约方式，`:366` 原地加 bias。

**G2.** AWQ：`awq.py:162` `apply()` —— **两路启发式**：
```python
FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256   # :174
if FP16_MATMUL_HEURISTIC_CONDITION:
    out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)  # :177 先全反量化
    out = torch.matmul(reshaped_x, out)                         # :178 再 FP16 GEMM（大 batch 划算）
else:
    out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)  # :180 小 batch 用融合 W4A16 GEMM
```
> 体现 weight-only 本质：大 batch（compute-bound）时反量化成 FP16 再用普通 GEMM 反而更快；小 batch（memory-bound）用专用融合 kernel。

**G3.** GPTQ(exllama)：`gptq.py:263` `apply()` → `ops.gptq_gemm(reshaped_x, qweight, qzeros, scales, g_idx, exllama_ready, bits)`（`_custom_ops.py:261`）。

**G4.** FP8 W8A8：`fp8.py:389` `apply()`
- marlin 降级 → `apply_fp8_marlin_linear`（`:394-402`）。
- block-quant → `torch.ops.vllm.apply_w8a8_block_fp8_linear(...)`（`:406-414`，传 `weight_scale_inv` + 激活 per-token dynamic）。
- 默认 → `self.fp8_linear.apply(input=x, weight=layer.weight, weight_scale=layer.weight_scale, input_scale=layer.input_scale, ...)`（`:416-421`）。`Fp8LinearOp.apply` **内部**先 `scaled_fp8_quant` 量化激活（static 用 `input_scale`，dynamic 现算，CUTLASS 时 per-token），再 `cutlass_scaled_mm` 或 `torch._scaled_mm` 做 FP8 MMA，输出 `scale_a*scale_b` 反量化回 FP16。
> 这就是 W8A8 与 weight-only 的代码级区别：**激活在这里也被 `scaled_fp8_quant` 量化**，GEMM 两个操作数都是 FP8。

### 阶段 H：FP8 KV cache 在 attention backend 的读写

KV 量化不走 `apply`（`kv_cache.py:42` 的 `apply` 直接 raise），而在 attention backend 里用 `layer._k_scale/_v_scale/_q_scale`（paged KV 读写 / `reshape_and_cache` 的总体机制见 [模块 02](../02-paged-attention-kvcache/design.md)）。

**H1.** 写入：`vllm/v1/attention/backends/flash_attn.py:460`
```python
torch.ops._C_cache_ops.reshape_and_cache_flash(
    key, value, key_cache, value_cache,
    attn_metadata.slot_mapping,
    self.kv_cache_dtype,        # :466 "fp8" 等
    layer._k_scale, layer._v_scale)  # :467-468 量化用的 scale
```

**H2.** 读取/计算：`flash_attn.py:471-479` 把 cache `.view(torch.float8_e4m3fn)`，并 `ops.scaled_fp8_quant(query, ..., layer._q_scale)` 把 query 也量化；`:519-521` 把 `_q_scale/_k_scale/_v_scale` 作为 **descale** 传进 FA kernel 反量化。

**H3.** 动态 scale（`calculate_kv_scales`）：`vllm/attention/layer.py:182-185` forward 里若开启则 `calc_kv_scales(query,key,value)`（`:232-239` 按 `abs().max()/range` 现算 scale 后自我关闭）。

**H4.** `kv_cache_dtype` 字符串语义：`"fp8"/"fp8_e4m3"/"fp8_e5m2"` 经 `vllm/utils.py:156` `STR_DTYPE_TO_TORCH_DTYPE` 映射为 `torch.uint8`（存储），计算时再 `.view(float8_e4m3fn)`。校验在 `config.py:1320` `CacheConfig._verify_cache_dtype`。

---

## 3. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。理解这些才算真懂 vLLM 的量化层。

### T1 · 一份 checkpoint，运行时自动换更快的 kernel
- `config.py:749-756` 遍历 `QUANTIZATION_METHODS`（`quantization/__init__.py:8`），调每个方法的 `override_quantization_method`。**`QUANTIZATION_METHODS` 里 marlin 系列特意排在 gptq/awq 之前**（`__init__.py:18-26` 注释明示"顺序重要"），所以普通 `gptq` checkpoint 会被 `GPTQMarlinConfig` 先截胡升级（`gptq_marlin.py:140`）。用户体感："我下载的是 gptq 模型，vLLM 却用 gptq_marlin 跑得更快"，且无需改任何配置。

### T2 · `get_quant_method` 按 layer 类型多态分派，是无侵入的根
- `base_config.py:130` 的 `get_quant_method(layer, prefix)` 接收**层对象**，方法内 `isinstance(layer, FusedMoE/Attention/LinearBase)` 分流（如 `fp8.py:114-128`）。同一个 `Fp8Config` 既能给 Linear 返回 `Fp8LinearMethod`、给 MoE 返回 `Fp8MoEMethod`、给 Attention 返回 `Fp8KVCacheMethod`。模型代码只声明层类型，量化方法被"按类型"塞进去——**一个 config 覆盖三类层**。

### T3 · `prefix` 让"同一模型里部分层不量化"成为可能
- `get_quant_method(self, prefix=prefix)` 把**层的全限定名**传进来。`awq.py:78 is_layer_skipped_awq`、`compressed_tensors.py:86 should_ignore_layer`、`fp8.py` 的 `ignored_layers` 都用 prefix 做正则/子串匹配，命中就返回 `UnquantizedLinearMethod`。于是 `lm_head` 或某些敏感层可以保持 FP16，其余量化——精度/速度的细粒度取舍落在每一层。

### T4 · Marlin 重排在加载时一次性做，换 forward 零 reshuffle
- `kernels/mixed_precision/marlin.py:66-79` 用 `ops.gptq_marlin_repack` + `marlin_permute_scales` 把权重/scale 排成 Tensor-Core MMA 布局。**关键认知**：MMA 指令要求操作数按特定 lane 排布，若 forward 时排每次 GEMM 都要 reshuffle。把这"贵但只做一次"的重排放进 `process_weights_after_loading`，热路径 `gptq_marlin_gemm` 就能把 int4 直流 MMA。这是 weight-only 高吞吐的工程支点。

### T5 · scale 和 zero_point 的重排 permutation 不一样
- `marlin_utils.py:206 get_scale_perms` 给出**两套** permutation：`scale_perm`(64) 用于 grouped scale、`scale_perm_single`(32) 用于 channelwise(`group_size=-1`)。`marlin_zero_points`（`:248`）**不用 single 版**，因为 zero-point 在每个 MMA 都参与（`:250-251` 注释），还要额外 interleave 列维并 pack int32（`:255`，interleave 表 `[0,2,4,6,1,3,5,7]`/`[0,2,1,3]`）。grouped 与 channelwise 对应 MMA 不同的 fragment stride，所以 perm 不同。

### T6 · FP8 W8A8 必须把多个逻辑分片的 scale 重量化为单一 scale
- `fp8.py:370 requantize_with_max_scale`。qkv_proj、gate_up_proj 在 vLLM 里是**融合**成一个大权重的（多个逻辑分片），但每个分片在 checkpoint 里可能有自己的 per-tensor `weight_scale`。`torch._scaled_mm` 只接受**单个** per-tensor scale（`:353-354` 注释），所以这里取各分片 scale 的 max，按它把所有分片重新量化到统一 scale。weight-only 的 Marlin 路径则相反——可以 `convert_to_channelwise`（`:329/350`）保留 per-channel scale。

### T7 · AWQ 的双路启发式：大 batch 反量化、小 batch 融合 kernel
- `awq.py:174` `FP16_MATMUL_HEURISTIC_CONDITION = numel >= 256`。token 多（compute-bound）时 `awq_dequantize` 把权重全反量化成 FP16 再 `torch.matmul`（普通 GEMM 在大 M 下吞吐高）；token 少（memory-bound）时 `awq_gemm` 用专用 W4A16 融合 kernel（省去反量化的中间张量与额外访存）。**同一个 apply 里按负载切换两种数值路径**——直白地体现了 weight-only "省的是访存"的本质。

### T8 · process_weights_after_loading 普遍把 Parameter 子类换成普通 Parameter
- `gptq.py:245-248`、`fp8.py:314-317` 等。vLLM 的 `PackedvLLMParameter`/`GroupQuantScaleParameter` 携带 TP 切分元数据，**只在加载期有用**；加载完后 torch.compile/Dynamo 无法追踪这些子类，所以一律 `Parameter(tensor.data, requires_grad=False)` 退回纯 Parameter。这是"加载期元数据"与"运行期可编译"两个需求的接缝。

### T9 · FP8 在无原生 FP8 硬件的卡上优雅降级为 weight-only
- `fp8.py:171` `use_marlin = not cutlass_fp8_supported and capability < 89`。这种卡没有 FP8 MMA，于是 FP8 权重走 Marlin（weight-only），`fp8.py:386 del layer.input_scale`（激活不量化）。结果：**同一个 FP8 checkpoint 在 H100 上是 W8A8 算力翻倍，在 A100 上退化为 W8A16 只省显存**——对上层透明。

### T10 · KV cache scale 的三种来源与兜底
- `kv_cache.py:51-74`。优先用 checkpoint 里**分离的** `k_scale/v_scale`（`:52`）；只找到**单个** `kv_scale` 时复制成 k 和 v（`:64-71`）；都没有则用 **1.0** 并警告会掉点（`:59-63` + `:93-98`）。q_scale 缺失时复制 k_scale（`:81-86`，仅 flash-attn 后端需要）。ROCm fnuz 格式 scale 要 `*2`（`:56-57/72-73`）。一个看似简单的 scale 加载藏了 4 个分支。

### T11 · KV scale 可在 forward 动态计算，算一次就自我关闭
- `layer.py:182-185` + `:232-239`。开启 `calculate_kv_scales` 时，第一步 forward 用 `query.abs().max()/q_range` 现场标定 scale，写进 `_q/_k/_v_scale` 后**把自己 `calculate_kv_scales=False` 关掉**（避免每步重算）。这让没有标定 scale 的 checkpoint 也能跑 FP8 KV，代价是首步不准。

### T12 · weight-only kernel 选择器按性能优先级 + 能力 + 可实现性逐个试
- `kernels/mixed_precision/__init__.py:19-24` `_POSSIBLE_KERNELS = [Machete, AllSpark, Marlin, Exllama]`（性能降序）。`choose_mp_linear_kernel`（`:27`）按序：跳过 `VLLM_DISABLED_KERNELS` 里的（`:56`）、跳过 capability 不够的（`:61`）、调 `can_implement(config)`（`:68`）返回第一个可用。Machete 要 capability 90（Hopper），所以 A100 自动落到 Marlin。**同一份 GPTQ 权重在不同卡上用不同最优 kernel，create_weights 里就决定好**。

### T13 · compressed-tensors 用 (weight_quant, input_quant) 元组映射到具体 scheme
- `compressed_tensors.py:312 _get_scheme_from_parts`。它用一组谓词（`_is_wNa16_group_channel`/`_is_fp8_w8a8`/`_is_static_tensor_w8a8`/`_is_dynamic_token_w8a8`…）判断激活/权重的量化参数组合，分别返回 `CompressedTensorsWNA16`/`W8A8Fp8`/`W8A16Fp8`/`W8A8Int8`/`W4A16Sparse24`。**fp8 w8a8 若硬件不支持会自动 fallback 到 w8a16fp8**（`:334-346`）。一个 checkpoint 里不同层（不同 target）可以是不同 scheme，挂在 `layer.scheme` 上，`CompressedTensorsLinearMethod` 只是个转发壳（`:538-580`）。

### T14 · g_idx（act-order）使 row-parallel + Exllama 不兼容，需特判
- `gptq.py:165-169`。GPTQ 的 `desc_act` 会按 Hessian 重要性重排列（`g_idx`）。但当该层做 row-parallel（输入维被切）且 group_size≠-1 时，列重排会跨越分片边界，Exllama kernel 处理不了，于是 `exllama_state = UNUSED` 退回非-Exllama 路径。Marlin 路径则用 `g_idx_sort_indices` 在重排时统一处理（`gptq_marlin.py:509-530` MoE 里逐 expert `argsort`）。act-order 提精度，但给并行切分带来连锁约束。

### T15 · 块量化 scale 取名 `weight_scale_inv` 是为兼容 DeepSeek-V3
- `fp8.py:259-274`。block-wise（128×128）量化的 scale 注册成 `BlockQuantScaleParameter` 且命名 **`weight_scale_inv`**（`:273` 注释点明是为 deepseekv3 checkpoint 的命名约定），区别于 per-tensor 的 `weight_scale`。`apply` 时走 `apply_w8a8_block_fp8_linear`（`:406`）并传 `weight_scale_inv`。命名细节直接对接特定模型家族的 checkpoint 格式。

---

## 4. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| 某层在 ignore/skip 列表 | `awq.py:78`、`compressed_tensors.py:86`、`fp8.py:119-122` | 返回 `UnquantizedLinearMethod`，该层保持 FP16 |
| gptq 可升级 marlin 但用户强制 gptq | `gptq_marlin.py:146-150` | 尊重用户，不升级，提示用 gptq_marlin 更快 |
| FP8 无原生硬件（capability<89） | `fp8.py:171` | `use_marlin=True`，降级为 W8A16 weight-only，`del input_scale` |
| FP8 checkpoint 是 fp16/bf16（未序列化） | `fp8.py:321-336` | 加载后用 `scaled_fp8_quant` 在线量化权重，`input_scale=None`（dynamic 激活） |
| W8A8 多逻辑分片 scale 不一致 | `fp8.py:370` | `requantize_with_max_scale` 按 max 重量化成单一 per-tensor scale |
| KV cache 无 scale | `kv_cache.py:59-63` | 用 1.0 并警告精度风险 |
| 只有单个 kv_scale | `kv_cache.py:64-71` | 复制成 k_scale 和 v_scale |
| `kv_cache_dtype=auto` | `kv_cache.py:51` | 强制 k/v_scale=1.0，不处理 scale（FP16 KV） |
| fp8_e5m2 + fp8 checkpoint | `attention/layer.py:109-111` | raise（e5m2 KV 不支持 fp8 checkpoint 的 scale） |
| TP 过大致 input_size 不整除 group_size | `awq.py:103`、`gptq.py:145` | raise ValueError，提示 TP size 太大 |
| act-order + row-parallel + Exllama | `gptq.py:165-169` | `exllama_state=UNUSED`，退回非-Exllama |
| ROCm fnuz FP8 | `fp8.py:303-307/360-368`、`kv_cache.py:56` | `normalize_e4m3fn_to_e4m3fnuz`，KV scale `*2` |
| 没有可用 weight-only kernel | `kernels/mixed_precision/__init__.py:76` | raise ValueError，附各 kernel 的 `can_implement` 失败原因 |
| CPU offload 下重排权重 | `loader.py:179` | `device_loading_context` 临时把参数搬上 device 再搬回 |

---

## 5. 一图速查：量化调用链主干

```
[启动] config.py:_verify_quantization :732
        └─ 遍历 QUANTIZATION_METHODS, override_quantization_method :749  (gptq → gptq_marlin)
        └─ weight_utils.py:get_quant_config :143 → quant_cls.from_config :164  ⇒ QuantizationConfig 实例
              │
[构造] LinearBase.__init__ :227  self.quant_method = quant_config.get_quant_method(self, prefix)  ★无侵入替换
        ├─ LinearBase  → XxxLinearMethod        (gptq_marlin.py:153 / awq.py:75 / fp8.py:114)
        ├─ FusedMoE    → XxxMoEMethod
        └─ Attention   → BaseKVCacheMethod       (attention/layer.py:117 create_weights → +k/v/q_scale)
              │
[① create] XxxLinearMethod.create_weights        注册 qweight(int32)/scales/qzeros/g_idx
              │                                   或 weight(fp8)/weight_scale/input_scale
              │   ← weight_loader 按 TP 切分 weight 与 scale (linear.py:419/1193)
[② process] loader.py:_process_weights_after_loading :164 → quant_method.process_weights_after_loading :180
        ├─ weight-only(marlin.py:48): ops.gptq_marlin_repack + marlin_permute_scales  → Tensor-Core 布局
        ├─ gptq(exllama, gptq.py:243): Parameter 化 + ops.gptq_shuffle
        ├─ fp8(fp8.py:299): weight.t() + requantize_with_max_scale (W8A8) / scaled_fp8_quant (在线量化)
        └─ kv(kv_cache.py:46): 加载/标定 k/v_scale → _k_scale/_v_scale
              │
[③ apply] linear.py:321/474 self.quant_method.apply(self, x, bias)
        ├─ weight-only: gptq_marlin_gemm (边读边反量化直入 MMA)  /  awq_dequantize+matmul (大batch) | awq_gemm (小batch)
        ├─ W8A8/FP8:    scaled_fp8_quant(激活) → cutlass_scaled_mm/torch._scaled_mm(FP8 MMA) → scale_a*scale_b 反量化
        └─ KV (在 attention backend, flash_attn.py:460/471):
                reshape_and_cache_flash(写 fp8, 用 _k_scale/_v_scale)
                + cache.view(float8_e4m3fn) + q/k/v_descale (读时反量化)
```
