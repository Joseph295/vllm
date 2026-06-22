# 模块 15 · Mamba/SSM 与混合架构 + 位置编码 —— 实现文档

> 与 [`design.md`](design.md) 配套。给出**可逐行对照代码**的两条调用链。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
> 分两部分：**A 部分** Mamba mixer 的 conv + selective scan、状态读写；**B 部分** `get_rope → RotaryEmbedding.forward` 注入 Q/K。

---

## 1. 代码地图

```
A 部分：Mamba/SSM
  vllm/model_executor/layers/mamba/
    ├─ mamba_mixer.py        # MambaMixer（Mamba-1）：conv + selective_scan
    ├─ mamba_mixer2.py       # MambaMixer2（Mamba-2）：chunk scan + TP 分片
    ├─ mamba2_metadata.py    # prepare_mamba2_metadata（顶层算一次，各层共享）
    └─ ops/
        ├─ causal_conv1d.py  # causal_conv1d_fn / causal_conv1d_update
        ├─ mamba_ssm.py      # selective_scan_fn / selective_state_update（triton）
        └─ ssd_*.py          # Mamba-2 chunk scan 的 triton kernels
  vllm/model_executor/models/
    ├─ mamba_cache.py        # MambaCacheManager / MambaCacheParams
    ├─ constant_size_cache.py# ConstantSizeCache：状态槽分配/释放/拷贝
    ├─ mamba2.py / jamba.py  # 纯 SSM / 混合架构模型（SupportsV0Only）
    └─ interfaces.py         # HasInnerState / IsHybrid / IsAttentionFree / SupportsV0Only

B 部分：位置编码
  vllm/model_executor/layers/rotary_embedding.py   # 所有 RoPE 子类 + get_rope
  csrc/pos_encoding_kernels.cu                      # rotary_embedding CUDA kernel
  vllm/v1/worker/gpu_model_runner.py                # M-RoPE position 计算（V1）
```

---

## 2. 调用链 A：Mamba mixer 的 conv + selective scan、状态读写

以混合架构 **Jamba**（V0 路径）为主线，Mamba-1 mixer。Mamba-2 差异在 §2 末单列。

### A 阶段 0：状态缓存的懒初始化与每步取状态

**A0-1.** `vllm/model_executor/models/jamba.py:426` `JambaForCausalLM.forward()`
首次 forward 懒建 `MambaCacheManager`，按 mamba 层数和状态形状分配跨步张量：
```python
if self.mamba_cache is None:
    num_mamba_layers = self.model_config.get_num_layers_by_block_type(
        self.vllm_config.parallel_config, LayerBlockType.mamba)
    self.mamba_cache = MambaCacheManager(
        self.vllm_config, self.lm_head.weight.dtype, num_mamba_layers,
        *self._get_mamba_cache_shape())
mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)
```

**A0-2.** `vllm/model_executor/models/mamba_cache.py:59` `MambaCacheManager.current_run_tensors()` → 父类 `constant_size_cache.py:33`
为本步的请求集合算出 `state_indices_tensor`（每条序列在状态张量里的槽下标）：
- `:42` `_release_finished_requests(...)`：把已结束请求的槽还回 `free_cache_indices`。
- `:43` `_prepare_current_run_cache(...)`：对每条 `(req_id, seq_id)` 调 `_assign_seq_id_to_cache_index`（`:91`）——新请求 `free_cache_indices.pop()` 拿槽（`:101`）；`n>1` 子序列 `_copy_cache` 复制父槽（`:111-113`）。
- `:46` 打包成 `state_indices_tensor`（`int32`，device=cuda）。

返回 `MambaCacheParams(conv_state, ssm_state, state_indices_tensor)`（`mamba_cache.py:65`），其中 conv/ssm state 形状是 `[num_layers, max_batch, ...]`。

**A0-3.** `jamba.py:344` `mamba_cache_params.at_layer_idx(i)`（`mamba_cache.py:19`）
切出本层的状态视图 `conv_state[layer]`、`ssm_state[layer]`，`state_indices_tensor` 共享。混合架构在此按层类型分派（`jamba.py:340-346`）：只有 `JambaMambaDecoderLayer` 拿到非 `None` 的 `mamba_cache_params`，attention 层走 KV cache。

---

### A 阶段 1：门控投影 + 拆分

**A1.** `vllm/model_executor/layers/mamba/mamba_mixer.py:143` `MambaMixer.forward_cuda()`
```python
projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
hidden_states, gate = projected_states.chunk(2, dim=-2)   # 主路 x 与门控 gate
```
`in_proj` 是 `MergedColumnParallelLinear`，一次投影出 `[intermediate, intermediate]` 两份（`mamba_mixer.py:68-70`）。

---

### A 阶段 2：因果卷积（conv_state 读写）

**A2.** `mamba_mixer.py:150-175`：按是否 prefill 分两条路。
```python
if attn_metadata.query_start_loc is not None and \
   attn_metadata.context_lens_tensor is not None:        # prefill / chunked prefill
    hidden_states = causal_conv1d_fn(
        hidden_states, conv_weights, self.conv1d.bias,
        activation=self.activation,
        conv_states=mamba_cache_params.conv_state,        # ← 写回滑窗状态
        has_initial_state=attn_metadata.context_lens_tensor > 0,  # 是否带前一段状态
        cache_indices=mamba_cache_params.state_indices_tensor,    # 每序列状态槽
        query_start_loc=attn_metadata.query_start_loc)    # 变长 cu_seqlens
else:                                                      # decode（单 token）
    hidden_states = causal_conv1d_update(
        hidden_states.transpose(0, 1), mamba_cache_params.conv_state,
        conv_weights, self.conv1d.bias, self.activation,
        conv_state_indices=mamba_cache_params.state_indices_tensor)
```
- **解读**：`causal_conv1d_fn`（`ops/causal_conv1d.py`）是 varlen prefill 卷积，用 `query_start_loc` 处理 ragged batch、`has_initial_state` 决定是否拼接 chunked prefill 上一段的滑窗、`cache_indices` 指出每条序列把新滑窗写回哪个槽。decode 路径 `causal_conv1d_update` 只把 1 个新 token 推进滑窗（`conv_state` 原地更新）。这就是 conv_state 的**读 + 写回**。

---

### A 阶段 3：selective 参数（input-dependent Δ/B/C）

**A3.** `mamba_mixer.py:185-200`
```python
ssm_parameters = self.x_proj(hidden_states.transpose(-2, -1))[0]
time_step, B, C = torch.split(
    ssm_parameters,
    [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
# 可选的 dt/B/C layernorm
discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
```
- **解读**：`Δ(time_step)、B、C` 都从卷积后的 `hidden_states` 投影得到（`x_proj`，`mamba_mixer.py:73`）——这就是 **selective** 的实现：它们随输入变。`A`、`D` 是 `nn.Parameter`（`mamba_mixer.py:97-103`，`A` 由 `A_weight_loader` 存成 `−exp(...)`），输入无关。

---

### A 阶段 4：selective scan（ssm_state 读写）

**A4.** `mamba_mixer.py:205-234`：同样按 prefill/decode 分两条。
```python
if attn_metadata.query_start_loc is not None and ...:     # prefill
    scan_outputs = selective_scan_fn(
        hidden_states, mamba_cache_params.ssm_state,       # ← 状态读写
        discrete_time_step, self.A, B.transpose(-2,-1), C.transpose(-2,-1),
        self.D.float(), gate, time_proj_bias, delta_softplus=True,
        cache_indices=mamba_cache_params.state_indices_tensor,
        has_initial_state=attn_metadata.context_lens_tensor > 0,
        query_start_loc=attn_metadata.query_start_loc)
else:                                                      # decode
    scan_outputs = selective_state_update(
        mamba_cache_params.ssm_state,                      # ← 状态读写
        hidden_states.transpose(0,1), discrete_time_step.transpose(0,1),
        self.A, B, C, self.D, gate.transpose(0,1), time_proj_bias,
        dt_softplus=True,
        state_batch_indices=mamba_cache_params.state_indices_tensor)
```
- **解读**：
  - prefill：`selective_scan_fn`（`ops/mamba_ssm.py`）对整段序列做递归 `h_t = A_bar·h_{t-1} + B_bar·x_t`，把每条序列的**最终状态写回 `ssm_state` 的对应槽**（供后续 chunk/decode 续接）。
  - decode：`selective_state_update`（`ops/mamba_ssm.py:41` 的 triton kernel `_selective_scan_update_kernel`）拿出该序列的状态、用 1 个新 token 更新它、吐 1 个输出。kernel 用 `state_batch_indices_ptr`（`ops/mamba_ssm.py:108-112`）把"batch 里第 b 条序列"映射到"状态张量里第几个槽"——这正是 `state_indices_tensor` 在 GPU 里的落地：**状态张量是按 `max_batch` 索引的固定大数组，靠下标选中本序列的状态**。

**A5.** `mamba_mixer.py:242-243` `out_proj`：输出投影回 `hidden_size`，返回 `contextualized_states`。

---

### A 阶段 6（Mamba-2 差异）：chunk scan

**A6-1.** `vllm/model_executor/models/mamba2.py:144` `Mamba2Model.forward()`
顶层算一次元数据，所有层共享：
```python
mamba2_metadata = prepare_mamba2_metadata(
    chunk_size=self.config.chunk_size, input_ids=input_ids,
    attn_metadata=attn_metadata)
```

**A6-2.** `mamba2_metadata.py:61` `prepare_mamba2_metadata()`
- `:69-76` 算 `has_initial_states`（`context_lens > 0`）、`prep_initial_states`。
- `:82-90` prefill 时算 `seq_idx`（每 token 属于 batch 第几条序列）。
- `:100-101` `_seq_idx_to_chunk_indices_offsets`（`:27`）算 `chunk_indices`/`chunk_offsets`——处理"chunk 不整除序列边界"（`:36-37` 用 `ceil` + 边界修正插入额外 chunk）。

**A6-3.** `mamba_mixer2.py:382` `MambaMixer2.forward_cuda()`
- `:398-407` `in_proj` 一次投出 `[gate | hidden_states_B_C | dt]` 三段（Mamba-2 把 conv 输入和 B/C 合并）。
- `:413-443` 因果卷积（prefill `causal_conv1d_fn` / decode `causal_conv1d_update`），同 A2。
- `:457-487` prefill 走 `mamba_chunk_scan_combined`（`ops/ssd_combined.py:22`）：
  - `:459-465` 若有 initial states，从 `ssm_state` 按槽取出作为 `initial_states`。
  - `:467-487` 调 chunk scan，传 `chunk_size/seq_idx/chunk_indices/chunk_offsets/cu_seqlens`，`return_varlen_states=True`。
  - `:491-492` **把 `varlen_state` 写回 `ssm_state` 的对应槽**（状态写回）。
- `:496-531` decode 走 `selective_state_update`（同 A4 decode）。
- `:534-537` `Mixer2RMSNormGated`（门控 RMSNorm，`mamba_mixer2.py:35`）+ `out_proj`。

> **状态读写总结**：conv_state 在 A2 读写、ssm_state 在 A4/A6 读写。两者都是 `[num_layers, max_batch, ...]` 的固定张量，靠 `state_indices_tensor`/`state_batch_indices` 选槽。整条链里**没有 block table、没有 prefix hash**——这是与 [模块 02](../02-paged-attention-kvcache/design.md) KV 路径的根本不同。

---

## 3. 调用链 B：get_rope → RotaryEmbedding.forward 注入 Q/K

以 **Llama**（普通 RoPE，V1）为主线，M-RoPE 差异单列。

### B 阶段 0：构造期 —— get_rope 工厂分派

**B0-1.** `vllm/model_executor/models/llama.py:165` `LlamaAttention.__init__`
```python
self.rotary_emb = get_rope(
    head_size, rotary_dim=..., max_position=..., base=rope_theta,
    rope_scaling=rope_scaling, ...)
```

**B0-2.** `vllm/model_executor/layers/rotary_embedding.py:1151` `get_rope()`
- `:1163-1175` 把 `rope_scaling`（含 list 的项转 tuple）和其他参数组成 `key`，查 `_ROPE_DICT` 缓存（`:1176-1177`）——同参数 RoPE 实例全局复用。
- `:1179-1180` `rope_scaling is None` → 基础 `RotaryEmbedding`。
- `:1182-1276` 否则按 `rope_scaling["rope_type"]` 分派：`llama3`/`mllama4`/`default`(+`mrope_section`→`MRotaryEmbedding`)/`linear`/`dynamic`/`yarn`/`deepseek_yarn`/`longrope`，各自构造对应子类。
- `:1277` 存回 `_ROPE_DICT` 并返回。

**B0-3.** 构造期算 `cos_sin_cache`：`RotaryEmbedding.__init__`（`:99-102`）调 `_compute_cos_sin_cache`（`:114`）→ `_compute_inv_freq`（`:104`）。**各外推法只重写这两个方法**：
- YaRN：`:544` `_compute_inv_freq` 分段（外推/插值/ramp，`:551-559`）+ `:562` `_compute_cos_sin_cache` 乘 `mscale`（`:567-568`）。
- Dynamic NTK：`:451` 改 `base`（`:457-461`）。
- Linear：`:392` 多 scaling factor 拼接 + offset（`:398-424`）。
- Llama3：`:830` 按波长分段缩放频率（`:841-850`）。

---

### B 阶段 1：运行期 —— forward 注入 Q/K

**B1.** `vllm/model_executor/models/llama.py:205` `q, k = self.rotary_emb(positions, q, k)`
attention forward 里，在 QKV 投影之后、attention 之前调用，**原地旋转 Q/K**。

**B2.** `rotary_embedding.py:155` `RotaryEmbedding.forward_cuda()`（CUDA 路径）
```python
if self.cos_sin_cache.device != query.device or dtype 不符:
    self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
if offsets is not None:
    ops.batched_rotary_embedding(positions, query, key, self.head_size,
        self.cos_sin_cache, self.is_neox_style, self.rotary_dim, offsets)
else:
    ops.rotary_embedding(positions, query, key, self.head_size,
        self.cos_sin_cache, self.is_neox_style)     # 原地更新 query/key
return query, key
```
- `:164-169` 注释明示：`nn.Module.__setattr__` 很贵，仅在 device/dtype 不符时才迁移 cache（性能细节）。
- `:171-172` 注释：`ops.rotary_embedding` 是**原地操作**，直接改 query/key 张量。

**B3.** `csrc/pos_encoding_kernels.cu:124` `rotary_embedding(...)`（host launch）
- `:138-169` 校验 positions/query/key 形状一致，算 `num_heads`/`num_kv_heads`。
- `:171-174` `rot_dim = cos_sin_cache.size(1)`，算 `query_stride`/`key_stride`。
- `:176-177` `grid(num_tokens)`、`block(min(num_heads*rot_dim/2, 512))`——**每个 block 处理一个 token**。
- `:180-192` 按 `is_neox` 分派 `rotary_embedding_kernel<scalar_t, true/false>`。

**B4.** `pos_encoding_kernels.cu:71` `rotary_embedding_kernel`（device）
```cpp
const int token_idx = blockIdx.x;
int64_t pos = positions[token_idx];
const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;  // 按 position 查表
apply_rotary_embedding<scalar_t, IS_NEOX>(query, key, cache_ptr, ...);
```

**B5.** `pos_encoding_kernels.cu:37` `apply_rotary_embedding` → `:11` `apply_token_rotary_embedding`
```cpp
// cos_ptr = cache_ptr;  sin_ptr = cache_ptr + embed_dim;
const scalar_t x = arr[x_index];
const scalar_t y = arr[y_index];
arr[x_index] = x * cos - y * sin;     // 二维旋转
arr[y_index] = y * cos + x * sin;
```
- `:16-28` `IS_NEOX` 决定配对方式：Neox 用 `(rot_offset, embed_dim+rot_offset)`，GPT-J 用相邻 `(2i, 2i+1)`。这就是 §design B.3.1 的旋转矩阵在 GPU 上的落地：**每个二维子空间旋转 θ = position·freq_i**，cos/sin 已在 cache 里按 position 算好。

> **native 路径**（`forward_native`，`:125`）：`cos_sin_cache.index_select(0, positions)` 取 cos/sin，`_apply_rotary_emb`（`:49`）做同样的 `o1=x1·cos−x2·sin; o2=x2·cos+x1·sin`。CUDA 与 native 必须数值一致。

---

### B 阶段 6（M-RoPE 差异）：三维 position

**B6-1.** position 预计算（V1，prompt 段）：`vllm/v1/worker/gpu_model_runner.py:369-371` `_update_states`
```python
self.requests[req_id].mrope_positions, self.requests[req_id].mrope_position_delta = \
    MRotaryEmbedding.get_input_positions_tensor(...)
```
→ `rotary_embedding.py:1021` `get_input_positions_tensor`：
- `:1046-1050` 找 `vision_start_token`，数 image/video 数量。
- `:1089-1108` 对每个图像/视频块，文本段 `(p,p,p)` 三维同值（`:1095-1096`），视觉段 `t_index`（视频乘 `second_per_grid_t·tokens_per_second`，`:1098-1100`）、`h_index`/`w_index` 按 patch 二维坐标（`:1102-1105`）。
- `:1117-1119` 拼成 `[3, L]` 的 `llm_positions`，算 `mrope_position_delta`（多模态撑大后续文本 position 的量）。

**B6-2.** 每步拼 position：`gpu_model_runner.py:517-518` → `:701` `_calc_mrope_positions`
- `:724-734` prompt 部分：从预计算的 `req.mrope_positions` 切片拷进 `mrope_positions_cpu`。
- `:736-751` completion 部分：`MRotaryEmbedding.get_next_input_positions_tensor`（`rotary_embedding.py:1136`）用 `mrope_position_delta` 续算三维 position。
- `:561-564` 把 `mrope_positions_cpu` 拷到 GPU `mrope_positions` buffer。

**B6-3.** forward 选 position：`gpu_model_runner.py:1042-1045`
```python
if self.uses_mrope:
    positions = self.mrope_positions[:, :num_input_tokens]   # [3, L]
else:
    positions = self.positions[:num_input_tokens]            # [L]
```

**B6-4.** `rotary_embedding.py:942` `MRotaryEmbedding.forward()`（注意：M-RoPE 用 native forward，无 CUDA kernel）
- `:957` 断言 `positions.ndim == 1 or 2`。
- `:960-974` 若 `ndim==2`（多模态），按 `mrope_section` 把 cos/sin 切成 `[t,h,w]` 三段、**每段只取对应行**再拼回（`:965-974`）：
```python
cos = torch.cat([m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))], dim=-1)
```
- `:976-989` 之后旋转 Q/K，逻辑与基础 RoPE 相同。
- **解读**：M-RoPE 的本质就是"position 是 `[3,L]`，cos/sin 按 `mrope_section` 分段、每段用不同维度（t/h/w）的 position 查出来再拼起来"。文本 token 三维同值 → 退化成普通 RoPE。

---

## 4. 关键边界情况与容错

| 情况 | 代码位置 | 行为 |
|---|---|---|
| Mamba 模型开 prefix caching | `mamba2.py:180-181` | 直接 `assert not enable_prefix_caching`（状态不可按内容寻址）|
| Mamba2 量化 + TP>1 | `mamba_mixer2.py:250-251` | `assert tp_size==1 or quant_config is None`（`#14617`）|
| n_groups 不整除 TP | `mamba_mixer2.py:261-266` | 扩展额外 groups（`extra_groups_for_head_shards`）；仅当 `n_groups==1` 才支持不整除 |
| chunked prefill 带初始状态 | `mamba_mixer.py:164` / `mamba_mixer2.py:459-465` | `has_initial_state`/`prep_initial_states` 决定是否拼上一段状态 |
| n>1 并行采样的状态 | `constant_size_cache.py:104-115` | `_copy_cache` 真拷贝父序列状态到子槽（状态不可共享读）|
| 状态槽耗尽 | `constant_size_cache.py:101` | `free_cache_indices.pop()` 空则报错；并发上界 = `max_num_seqs` |
| decode + CUDA graph | `constant_size_cache.py:57-78` | `copy_inputs_before_cuda_graphs` 把 state_indices 写进 graph buffer，pad 到捕获尺寸 |
| RoPE cache device/dtype 不符 | `rotary_embedding.py:166-169` | 仅在不符时迁移 cache（避免昂贵的 `__setattr__`）|
| M-RoPE position 维度 | `rotary_embedding.py:957` | 断言 `ndim ∈ {1,2}`；1 维退化成普通 RoPE |
| mrope_section 和校验 | `rotary_embedding.py:939-940` | 断言 `sum(mrope_section) == rotary_dim // 2` |
| partial rotary（部分维度旋转）| `rotary_embedding.py:1172-1173` / `:142-143` | `rotary_dim < head_size` 时只旋转前 `rotary_dim` 维，`query_pass` 不动 |
| M-RoPE cache 预留 | `rotary_embedding.py:934` | `cache_max_position_num = max_pos × 4`（视频时间维可能放大位置）|

---

## 5. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · 状态张量靠"下标选槽"而非"指针/block table"

- **代码**：`ops/mamba_ssm.py:108-114`（kernel 用 `state_batch_indices_ptr` 把 `pid_b` 映射到状态槽）；`constant_size_cache.py:91-117`（CPU 侧维护 `{req_id:{seq_id:slot}}`）。
- **精妙之处**：KV cache 用 block table 做"逻辑块→物理块"的间接寻址（[模块 02](../02-paged-attention-kvcache/design.md)）；状态缓存因为**每序列恰好一个定长状态**，间接寻址退化成一个 `int32` 下标 `state_indices_tensor`。kernel 里 `state_ptr += state_batch_idx * stride_state_batch` 一句就选中了本序列的状态——比 block table 简单一个数量级，但也因此**丧失了共享/分页能力**。

### T2 · prefill 与 decode 是两套完全不同的 kernel

- **代码**：`mamba_mixer.py:150 vs 167`（conv：`causal_conv1d_fn` vs `causal_conv1d_update`）；`mamba_mixer.py:205 vs 221`（scan：`selective_scan_fn` vs `selective_state_update`）。
- **精妙之处**：prefill 要处理变长的整段序列（用 `query_start_loc` cu_seqlens 的 ragged kernel），decode 只处理"每序列 1 个新 token"（batched 单步更新）。二者数据布局、并行维度都不同，所以是两套 kernel。continuous batching 下"哪些序列在 prefill、哪些在 decode"由 `attn_metadata`/`mamba2_metadata.has_prefill` 决定走哪条。

### T3 · Mamba-2 用 chunk scan 把串行递归改写成矩阵乘

- **代码**：`ops/ssd_combined.py`（组合 `ssd_bmm`/`ssd_chunk_scan`/`ssd_chunk_state`/`ssd_state_passing`）；`mamba_mixer2.py:467`。
- **精妙之处**：朴素 scan 是 `h_t←f(h_{t-1})` 的严格串行，GPU 利用率低。SSD 把序列切成 `chunk_size` 块，**块内**当成结构化掩码的矩阵乘并行（喂饱 tensor core），**块间**只用 `ssd_state_passing` 传块边界状态（串行但量极小）。这是 Mamba-2 比 Mamba-1 prefill 快的根本原因。

### T4 · 元数据在 top-level 算一次，所有 mamba 层共享

- **代码**：`mamba2.py:144` `prepare_mamba2_metadata` 在 `Mamba2Model.forward` 顶层调一次；`mamba_mixer2.py:386` 各层接收同一个 `mamba2_metadata`。
- **精妙之处**：`seq_idx`/`chunk_indices`/`chunk_offsets` 对所有 mamba 层相同，且算它们要 `.item()` 触发 device sync（`mamba2_metadata.py:76`）。算一次共享，避免每层一次 device sync——`mamba2_metadata.py:92-101` 注释专门解释了这个"宁可在顶层无脑算、也不每层重算"的取舍。

### T5 · A 参数存成 −exp(...)，保证状态衰减稳定

- **代码**：`mamba_mixer.py:93-94` `A_weight_loader`（`−torch.exp(loaded_weight.float())`）；`mamba_mixer2.py:362-364` 同理。
- **精妙之处**：SSM 递归 `h_t = A_bar·h_{t-1}+...` 要稳定（不爆炸），`A` 必须是负实部。权重加载时就把原始参数过 `−exp(·)`，保证 `A<0`、离散化后 `A_bar=exp(Δ·A)∈(0,1)`——状态随时间**衰减**而非发散。这是 SSM 数值稳定性的硬约束，藏在 weight loader 里。

### T6 · 所有 RoPE 外推法只重写 `_compute_inv_freq`

- **代码**：`rotary_embedding.py:544`(YaRN)/`451`(NTK)/`392`(Linear)/`830`(Llama3) 各自的 `_compute_inv_freq`/`_compute_cos_sin_cache`，forward 全继承基类。
- **精妙之处**：五种外推法在数学上各异，但在代码里只是"换一种频率/cache 算法"，旋转逻辑和 CUDA kernel 完全复用。这把"长上下文外推"的复杂度全部锁在一个方法里，新外推法接入只需写一个子类——是位置编码模块最干净的抽象。

### T7 · `_ROPE_DICT` 全局缓存：同参数 RoPE 只构造一次

- **代码**：`rotary_embedding.py:1148,1174-1177,1277`。
- **精妙之处**：一个模型几十层 attention，若每层各建一个 RoPE（含预算 `cos_sin_cache`）会浪费显存和构造时间。`get_rope` 用全参数元组做 key 缓存实例，同配置的层共享一个 RoPE 对象。代价：key 必须涵盖所有影响参数，少一个就错误命中（历史 bug `#1867`）。

### T8 · cos_sin_cache 一张表存两半，kernel 切指针取 cos/sin

- **代码**：`rotary_embedding.py:122` `cache = torch.cat((cos, sin), dim=-1)`；kernel `pos_encoding_kernels.cu:48-49` `cos_ptr=cache_ptr; sin_ptr=cache_ptr+embed_dim`。
- **精妙之处**：把 cos 和 sin 拼成一张 `[max_pos, rot_dim]` 的连续表，按 `pos*rot_dim` 一次定位，kernel 内用指针偏移 `+embed_dim` 取 sin。一次访存定位、布局对 cache line 友好。

### T9 · 每个 CUDA block 处理一个 token，head 维并行展开

- **代码**：`pos_encoding_kernels.cu:84-85`（`token_idx = blockIdx.x`）；`:52` `for (i=threadIdx.x; i<nq; i+=blockDim.x)`。
- **精妙之处**：grid 维 = token 数，block 内线程沿"head×embed_dim/2"展开，每线程旋转一个二维子空间。这样 query 和 key 在同一 block 内顺序处理（`:51-67`），最大化每 token 的局部性。

### T10 · M-RoPE 走 native forward，不走 CUDA kernel

- **代码**：`rotary_embedding.py:942` `MRotaryEmbedding.forward`（覆盖基类，纯 PyTorch）。
- **精妙之处**：M-RoPE 要按 `mrope_section` 把 cos/sin 切三段、每段取不同维度（`:965-974`），这个"分段拼接"逻辑用 PyTorch 表达最自然，且多模态 batch 通常不在最热的 decode 路径。于是 M-RoPE 不复用 CUDA kernel，直接 native——简单优先于极致性能。

### T11 · mrope_position_delta：decode 续接位置的关键状态

- **代码**：`rotary_embedding.py:1118`（prefill 算 delta）；`gpu_model_runner.py:742-749`（decode 用 delta 续算）；`get_next_input_positions_tensor`（`:1136`）。
- **精妙之处**：多模态 token 把后续文本的 position 撑大了（一张图占 1 个 token 但贡献 `t·h·w` 个 position）。`mrope_position_delta = max_position+1 − num_tokens` 记下这个"撑大量"，decode 时新文本 token 的三维 position 直接 `delta + 当前下标` 续接，**无需重算整段**。这是多模态长生成正确性的命门（历史 bug `#10388`/`#10403` 都出在这）。

### T12 · 文本 token 在 M-RoPE 里三维同值，自动退化成普通 RoPE

- **代码**：`rotary_embedding.py:1095-1096`（文本段 `arange.expand(3,-1)`）；`gpu_model_runner.py:219-223` 注释。
- **精妙之处**：纯文本请求跑 M-RoPE 模型时，position 三维取相同值，旋转结果和普通 RoPE 完全一致。于是同一套 M-RoPE 代码无缝处理"纯文本"和"图文混合"，无需分支——M-RoPE 是 RoPE 的真·超集。

### T13 · partial rotary：只旋转一部分 head 维

- **代码**：`rotary_embedding.py:1172-1173`（`rotary_dim = int(rotary_dim * partial_rotary_factor)`）；`:142-145`（`query_rot`/`query_pass` 拆分，pass 部分不旋转）。
- **精妙之处**：有些模型（如 GPT-NeoX、部分 Phi）只对 head 维的前一部分做 RoPE，剩下的"原样通过"。`query[..., :rotary_dim]` 旋转、`query[..., rotary_dim:]` 直接 cat 回去。一个 `partial_rotary_factor` 配置就覆盖了这类变体。

### T14 · 状态缓存 max_batch 对齐 CUDA graph 捕获尺寸

- **代码**：`mamba_cache.py:32-34`（`max_batch = pad_for_cudagraph(max_num_seqs)`）；`mamba_cache.py:68` / `constant_size_cache.py:80` 的 `get_seqlen_agnostic_capture_inputs`。
- **精妙之处**：状态张量第二维 = `max_batch`，按 CUDA graph 捕获尺寸 pad。decode 时状态张量形状固定、`state_indices` 写进 graph 的固定 buffer（`constant_size_cache.py:57-78`），于是 mamba decode 也能被 CUDA graph replay——"seqlen-agnostic"正是为此。

---

## 6. 一图速查：两条调用链主干

```
A 部分：Mamba mixer（V0 路径，以 Jamba 为例）
─────────────────────────────────────────────────────────────
JambaForCausalLM.forward  jamba.py:426
  └─ MambaCacheManager.current_run_tensors  mamba_cache.py:59
        └─ ConstantSizeCache: 分配/复制/释放状态槽 → state_indices_tensor
  └─ 按层分派  jamba.py:340-352
        attention 层 → KV cache（PagedAttention，模块02）
        mamba 层    → mamba_cache_params.at_layer_idx(i)
              └─ MambaMixer.forward_cuda  mamba_mixer.py:143
                   ① in_proj → [x | gate]                       :143
                   ② causal_conv1d_fn/update  (conv_state 读写)  :150-175
                   ③ x_proj → [Δ | B | C]  (input-dependent)     :185
                   ④ selective_scan_fn/state_update (ssm_state)  :205-234
                        └─ triton kernel 用 state_batch_indices 选槽
                   ⑤ out_proj                                    :242
  ※ Mamba-2: 顶层 prepare_mamba2_metadata 算一次 → chunk scan (ssd_combined)

B 部分：RoPE（V1，以 Llama 为例）
─────────────────────────────────────────────────────────────
[构造期] LlamaAttention.__init__  llama.py:165
  └─ get_rope  rotary_embedding.py:1151
        按 rope_scaling["rope_type"] 分派子类 + _ROPE_DICT 缓存
        子类只重写 _compute_inv_freq / _compute_cos_sin_cache
              (YaRN 分段频率 | NTK 改 base | Linear 插值 | Llama3 波长分段)

[运行期] q, k = self.rotary_emb(positions, q, k)   llama.py:205
  └─ RotaryEmbedding.forward_cuda  rotary_embedding.py:155
        └─ ops.rotary_embedding (原地)  → pos_encoding_kernels.cu:124
              kernel: 每 block 一个 token  :84
                cache_ptr = cos_sin_cache + pos*rot_dim  (查表)  :87
                apply_token_rotary: arr[x]=x·cos−y·sin; arr[y]=y·cos+x·sin  :30-33

[M-RoPE 差异] positions: [L] → [3, L]
  gpu_model_runner: get_input_positions_tensor (prompt 段, t/h/w)  rotary_embedding.py:1021
                  + _calc_mrope_positions (每步拼接, delta 续接)   gpu_model_runner.py:701
  MRotaryEmbedding.forward (native): 按 mrope_section 切 cos/sin 三段拼接  rotary_embedding.py:942
```
