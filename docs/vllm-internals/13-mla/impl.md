# 模块 13 · MLA（Multi-head Latent Attention，DeepSeek）—— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的实现细节：从 DeepSeek MLA 层的投影压缩，到 latent 写入 paged KV cache，再到 decode（吸收）/ prefill（物化）两条路径完成 attention。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
> MLA 复用 [模块 02](../02-paged-attention-kvcache/design.md) 的分页 KV 机制（block / block table / slot_mapping），本文只追 MLA 特有的部分；通用分页寻址见 [模块 02 impl](../02-paged-attention-kvcache/impl.md)。

---

## 1. 代码地图

```
模型侧 MLA 层 vllm/model_executor/models/
  └─ deepseek_v2.py
       ├─ DeepseekV2MLAAttention (:335)   # MLA 层：q_a/q_b_proj、kv_a_proj_with_mqa、kv_b_proj、o_proj、rotary_emb
       │    └─ forward (:466)             # 投影压缩 → 调 self.mla_attn(Attention, use_mla=True)
       └─ DeepseekV2Attention (:189)      # 非 MLA 回退路径（model_config.use_mla=False 时用）

V1 MLA backend vllm/v1/attention/backends/mla/
  ├─ common.py                            # 全部公共逻辑（937 行，文件头注释是设计纲领）
  │    ├─ MLACommonBackend (:222)         # kv_cache_shape=单head latent、head_size=576、禁 cascade
  │    ├─ MLACommonMetadata (:291)        # prefill/decode 分段元数据
  │    ├─ MLACommonMetadataBuilder (:337) # reorder_batch(:380)、build(:450)、chunked workspace
  │    └─ MLACommonImpl (:570)            # forward(:855) 分流；吸收矩阵；_forward_prefill(:788)
  ├─ flashmla.py                          # FlashMLABackend：flash_mla_with_kvcache decode kernel
  └─ triton_mla.py                        # TritonMLABackend：Triton decode_attention_fwd

backend 选择 vllm/platforms/
  └─ cuda.py:178 get_attn_backend_cls     # use_mla → FlashMLA(block_size=64) / TritonMLA 分派

CUDA / kernel 绑定 vllm/_custom_ops.py
  ├─ concat_and_cache_mla (:1344)         # 把 latent+rope 写进 paged cache（单 op）
  └─ gather_cache (:1380)                 # chunked prefill 把 context 收集进 workspace
csrc/cache_kernels.cu
  └─ concat_and_cache_mla_kernel (:308)   # GPU 侧写入 kernel

worker 钩子 vllm/v1/worker/
  └─ gpu_model_runner.py:474-478          # 调 attn_metadata_builder.reorder_batch（MLA 用它分批）

attention 层 vllm/attention/layer.py
  └─ :193-205                             # use_mla 时跳过 q/k/v reshape（MLA 张量语义不同）
```

---

## 2. 端到端调用链：从投影压缩到输出

追踪一条 DeepSeek 请求在**一个 MLA 层、一个调度步**里的完整数据流。

### 阶段 A：模型侧投影压缩（latent + decoupled rope）

> **这一步在干嘛**：在进 attention backend 之前，模型层先把 hidden 向量"压扁"——一次投影同时挤出两样东西：一条要缓存的 latent（512 维，压缩态的 KV）和一小段要做位置旋转的 rope key（64 维）。这两样就是 MLA 唯一会写进 KV cache 的东西，完整多头 K/V 永不进 cache。

**A1.** `vllm/model_executor/models/deepseek_v2.py:466` `DeepseekV2MLAAttention.forward()` —— **产出 latent 和 rope key**
```python
if self.q_lora_rank is not None:
    ckq = self.q_a_proj(hidden_states)[0]                 # H → Lq (Q 低秩压缩)
    hidden_states_or_q_c = self.q_a_layernorm(ckq)
else:
    hidden_states_or_q_c = hidden_states
kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)   # 一次投影 → [latent | rope_key]
kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())      # latent 归一化
return self.mla_attn(hidden_states_or_q_c, kv_c_normed, k_pe,
                     output_shape=hidden_states.shape)
```
> 关键点：`kv_a_proj_with_mqa`（`:401-406`）**一次线性投影**同时产出 `kv_c`（latent，`kv_lora_rank=512`）和 `k_pe`（decoupled rope key，`qk_rope_head_dim=64`）。这两个就是将来唯一进 KV cache 的东西（design §3.1）。`q_b_proj`/`q_proj` 留给 backend 内部展开成多头 Q。

**A2.** `deepseek_v2.py:441-461` 构造 `Attention(use_mla=True)`，把 MLA 专属参数（`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、`rotary_emb`、以及三个投影矩阵 `q_proj`/`kv_b_proj`/`o_proj`）透传给 backend。**head_size 设为 `kv_lora_rank + qk_rope_head_dim = 576`**（`:443`），正是 paged cache 每 token 的存储维度——注释（`:435-440`）明示 `concat_and_cache_mla` 要求 `kv_lora_rank + qk_rope_head_dim == head_size`。

**A3.** `vllm/attention/layer.py:193-205` —— **MLA 跳过 q/k/v reshape**
```python
if not self.use_mla:
    query = query.view(-1, self.num_heads, self.head_size)
    ...    # MLA 的 query/key/value 张量语义不同（是 q_c/kv_c_normed/k_pe），不能按 MHA reshape
```
> MLA 把 `hidden_states_or_q_c`、`kv_c_normed`、`k_pe` 当作"query/key/value"三个位置参数传进 `impl.forward`（`:210`），但它们**不是**真正的多头 Q/K/V——展开成多头是 backend 内部的事。

### 阶段 B：reorder_batch —— prefill/decode 在 batch 内分段

> **这一步在干嘛**：MLA 的 decode 和 prefill 走两套完全不同的 kernel，没法在一个交错的 batch 上同时跑。所以开算前先把这一步里所有 decode 请求挪到 batch 前段、所有 prefill 请求挪到后段。排好序后，后面"取 decode 段 / prefill 段"就退化成简单的头尾切片，不用东挑西拣。

**B1.** `vllm/v1/worker/gpu_model_runner.py:474` `_prepare_inputs()` 每步先给 backend 一个分批钩子：
```python
# Some attention backends (namely MLA) may want to separate requests
# based on if the attention computation will be compute-bound or memory-bound.
modified_batch = self.attn_metadata_builder.reorder_batch(
    self.input_batch, scheduler_output)
if modified_batch:
    self.input_batch.refresh_sampling_metadata()   # 分批改了顺序 → 采样元数据要同步刷新
```
> `refresh_sampling_metadata` 这一步是 `#14253`（非法内存访问 + 精度 bug）补上的：reorder 打乱了请求顺序，采样元数据若不刷新就会错位（design §8.2）。

**B2.** `vllm/v1/attention/backends/mla/common.py:380` `reorder_batch()` —— **把 decode 挪前、prefill 挪后**
```python
for i, req_id in enumerate(input_batch.req_ids):
    num_tokens = scheduler_output.num_scheduled_tokens[req_id]
    if num_tokens == 1:                      # 只有1个scheduled token才算 decode
        decodes.append(i); num_decode_tokens += num_tokens
    else:
        prefills.append(i); num_prefill_tokens += num_tokens
...
for i in range(1, min(num_decodes, num_prefills) + 1):
    if decodes[num_decodes - i] >= num_decodes:        # decode 落在了"prefill 区"
        input_batch.swap_states(prefills[first_prefill], decodes[num_decodes - i])
        first_prefill += 1; modified_batch = True
    else:
        break
# 存给随后的 build() 用
self._num_decodes = num_decodes; self._num_prefills = num_prefills
self._num_decode_tokens = num_decode_tokens; self._num_prefill_tokens = num_prefill_tokens
return modified_batch
```
- `:399` **判据**：`num_tokens == 1` 才是 decode——因为 `_forward_decode` 当前只支持 query 长度 1（`:396-398` 注释承认这是临时策略，将来可能放宽到 `< 8`）。
- `:421-430` **最少交换**：decode 通常久驻 batch 前部（design §8.4），新请求 append 在后部，所以只需把"误落到 prefill 区的 decode"和"batch 前部的 prefill"两两交换，绝大多数步 `modified_batch=False`。
- `:435-438` 把四个计数存进实例，`build()` 直接读——`reorder_batch` 和 `build` 是同一 step 内的两次调用，靠实例字段传状态（`:432-434` 注释自承是 hack）。

### 阶段 C：build —— 切出 decode / prefill 两段元数据

> **这一步在干嘛**：照着 B 段排好的顺序，把这一步要用到的张量（block table、slot_mapping、positions、seq_lens）按"前 N 个是 decode、后面是 prefill"切成两段元数据，分别打包给两条路径。长 context 的 prefill 还要在这里规划好"分几块、每块多大"（chunked prefill 的 workspace）。

**C1.** `common.py:450` `build()` —— **按 decode 段 / prefill 段切分张量**
```python
assert self._num_decodes + self._num_prefills == num_reqs
block_table   = self.runner.input_batch.block_table.get_device_tensor()[:num_reqs]
slot_mapping  = self.runner.slot_mapping_cpu[:num_actual_tokens].to(device, non_blocking=True).long()
input_positions = self.runner.positions_cpu[:num_actual_tokens].to(device, non_blocking=True).long()
seq_lens      = self.runner.seq_lens_cpu[:num_reqs].to(device, non_blocking=True)
```
> `input_positions`（`:464`）是 MLA 特有的：因为 RoPE 在 backend 内部施加（design §3.2），positions 必须随 metadata 带下去。`:454-456` 注释专门告诫**避免 GPU→CPU 同步**（会阻塞前序 kernel），这是 `#14540` 性能优化的关注点。

**C2.** `common.py:471-546` —— **prefill 段 + chunked context**
```python
if self._num_prefills > 0:
    reqs_start = self._num_decodes        # prefill 请求从 decode 之后开始（B 段分批的产物）
    tokens_start = self._num_decode_tokens
    context_lens_cpu = num_computed_tokens_cpu_tensor[reqs_start:num_reqs]
    ...
    if self.chunked_prefill_enabled and max_context_len_cpu > 0:
        max_context_chunk = self.chunked_prefill_workspace_size // num_prefills_with_context_cpu
        max_context_chunk = round_down(max_context_chunk, self.page_size)   # 对齐 page_size
        num_chunks = cdiv(max_context_len_cpu, max_context_chunk)
        chunk_starts = arange(num_chunks)... * max_context_chunk            # 见 :504-507 注释举例
        ... 构造 cu_seq_lens / starts / workspace 进 ChunkedContextMetadata
    prefill_metadata = MLACommonPrefillMetadata(
        input_positions=input_positions[tokens_start:],         # prefill 段的 positions
        block_table=block_table[reqs_start:, ...],              # prefill 段的 block table
        query_start_loc=query_start_loc[reqs_start:] - query_start_loc[reqs_start],  # 重置起点
        max_query_len=max_query_len, chunked_context=chunked_context_metadata)
```
> `reqs_start = num_decodes`、`tokens_start = num_decode_tokens`：**正因为 B 段把 decode 排在前面**，prefill 段就是简单的尾部切片 `[num_decodes:]`。这是 reorder_batch 的直接回报——切分变成纯下标偏移，无需 gather/scatter。`:498` `round_down` 到 page_size 是因为 `gather_cache` kernel 要求 chunk 起点页对齐（`:495-497` 注释）。

**C3.** `common.py:548-554` —— **decode 段**（简单切片 `[:num_decodes]`）
```python
if self._num_decodes > 0:
    decode_metadata = self._build_decode(
        input_positions=input_positions[:self._num_decode_tokens],
        block_table=block_table[:self._num_decodes, ...],
        seq_lens=seq_lens[:self._num_decodes])
```
> FlashMLA 子类重写 `_build_decode`（`flashmla.py:61`）**多算一步** `get_mla_metadata`（tile 调度 + num_splits，给 FlashMLA kernel 用）：
> ```python
> tile_scheduler_metadata, num_splits = get_mla_metadata(seq_lens, self.num_q_heads, 1)  # 1 = MQA
> ```

### 阶段 D：写 KV —— 只把 latent + rope 存进 paged cache

> **这一步在干嘛**：先对那 64 维 rope 段做位置旋转（RoPE），再把"latent（512）+ rope（64）"拼成一条 576 维向量写进 paged KV cache 对应的 slot。对比普通 attention 写"完整多头 K + 完整多头 V"两份，MLA 只写一条压缩向量——这就是 KV cache 压缩在写路径上的兑现。

**D1.** `common.py:855` `MLACommonImpl.forward()` —— **总入口，切 decode/prefill 两段**
```python
num_decode_tokens = attn_metadata.num_decode_tokens
decode_hs_or_q_c = hidden_states_or_q_c[:num_decode_tokens]    # decode 段（前）
decode_k_pe      = k_pe[:num_decode_tokens]
prefill_hs_or_q_c = hidden_states_or_q_c[num_decode_tokens:]   # prefill 段（后）
prefill_k_pe      = k_pe[num_decode_tokens:]
prefill_k_c_normed = k_c_normed[num_decode_tokens:]
```

**D2.** `common.py:899-915` —— **两段各自施加 decoupled RoPE（在 backend 内部！）**
```python
if has_decode:
    decode_ql_nope, decode_q_pe = self._q_proj_and_k_up_proj(decode_hs_or_q_c)  # 吸收 W_UK（见 E）
    decode_q_pe[...], decode_k_pe[...] = self.rotary_emb(
        attn_metadata.decode.input_positions, decode_q_pe.contiguous(), decode_k_pe)
if has_prefill:
    prefill_q = self.q_proj(prefill_hs_or_q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
    prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]      # 切出 rope 段
    prefill_q_pe[...], prefill_k_pe[...] = self.rotary_emb(
        attn_metadata.prefill.input_positions, prefill_q_pe.contiguous(), prefill_k_pe)
```
> RoPE **只作用在 `*_q_pe` / `*_k_pe`（那 64 维 rope 段）**，nope 段完全不碰——这是 design §3.2 decoupled RoPE 的代码落点。`rotary_emb` 被预先取出 `forward_cuda`（`:620-622`），绕开 torch library 包装（`#14476` 精度修复）。

**D3.** `common.py:917-926` —— **写 KV cache：只存 latent + rope，单 op**
```python
if kv_cache.numel() > 0:
    ops.concat_and_cache_mla(
        k_c_normed,                      # latent [num_tokens, 512]
        k_pe.squeeze(1),                 # rope key [num_tokens, 64]
        kv_cache,                        # [num_blocks, block_size, 576]
        attn_metadata.slot_mapping.flatten(),
        kv_cache_dtype=self.kv_cache_dtype, scale=layer._k_scale)
```
> 对比 [模块 02](../02-paged-attention-kvcache/impl.md) §2 的 `reshape_and_cache_flash`（写完整 K + V 两份多头张量），MLA 只写**一条 576 维的合并向量**。这就是 KV cache 压缩在写路径上的体现。

**D4.** `csrc/cache_kernels.cu:308` `concat_and_cache_mla_kernel` —— **GPU 侧把 latent 和 rope 拼进同一 slot**
```cuda
const int64_t slot_idx = slot_mapping[token_idx];
if (slot_idx < 0) return;                          // padding token 跳过（同模块02的哨兵）
const int64_t block_idx = slot_idx / block_size;
const int64_t block_offset = slot_idx % block_size;
...
copy(kv_c,  kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);          // 前 512 维 = latent
copy(k_pe,  kv_cache, k_pe_stride, block_stride, pe_dim,  kv_lora_rank);    // 后 64 维 = rope，偏移 512
```
> slot → `(block_idx, block_offset)` 的分解与模块 02 完全一致（复用分页寻址）；MLA 的不同只在 **一个 slot 里拼了 `[latent(512) | rope(64)]`** 而非分 K/V 两个张量。

### 阶段 E：decode 路径 —— 吸收上投影，对 latent 跑 MQA

> **这一步在干嘛**：decode 是 memory-bound（瓶颈在读 KV cache 的显存带宽），所以**不**把 latent 还原成完整 KV——那会把压缩省下的带宽全吐回去。改用"吸收"：把 K 的上投影矩阵并进 query、把 V 的上投影矩阵并进输出，让整个 attention 直接在 512 维的 latent 上以 MQA（所有 query 头共享一份 KV）形式跑完。每个缓存 token 只读 576 维而非完整多头 K/V。

**E1.** `common.py:648` `_q_proj_and_k_up_proj()` —— **把 W_UK 吸收进 query**
```python
q_nope, q_pe = self.q_proj(x)[0].view(-1, self.num_heads, self.qk_head_dim)\
                  .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
q_nope = q_nope.transpose(0, 1)              # (B,N,P) → (N,B,P)
ql_nope = torch.bmm(q_nope, self.W_UK_T)     # (N,B,P)×(N,P,L) → (N,B,L)  ← 吸收！q_nope@W_UK^T
return ql_nope.transpose(0, 1), q_pe
```
> 这是 design §3.3 吸收的核心一行：`ql_nope = q_nope @ W_UK^T`，把 K 的上投影并进 query。结果 `ql_nope` 已经在 latent 维 `Lkv=512`，可直接和缓存里的 `kv_c` 点积，**无需还原 k_nope**。`W_UK_T` 在 `process_weights_after_loading`（`:709`）从 `kv_b_proj` 拆出并 permute 好。

**E2.** `flashmla.py:123` `FlashMLAImpl._forward_decode()` —— **FlashMLA kernel（MQA）**
```python
q = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)   # [B, 1(seqlen), N, Lkv+R=576]
o, _ = flash_mla_with_kvcache(
    q=q,
    k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),        # 加 head dim=1 → MQA
    block_table=attn_metadata.decode.block_table,     # 复用 paged block table！
    cache_seqlens=attn_metadata.decode.seq_lens,
    head_dim_v=self.kv_lora_rank,                      # V 头维 = 512（latent 维）
    tile_scheduler_metadata=attn_metadata.decode.tile_scheduler_metadata,
    num_splits=attn_metadata.decode.num_splits,
    softmax_scale=self.scale, causal=True)
return self._v_up_proj_and_o_proj(o)                  # 吸收 W_UV（见 E4）
```
> 关键：`q` 拼成 `[ql_nope | q_pe]`（576 维），cache 当作单 head（`unsqueeze(-2)`）→ **MQA**：所有 query head 共享同一份 latent KV。`block_table` 直接是模块 02 那张 paged 表——MLA decode 仍走分页寻址，只是每个 block 里存的是 latent。

**E3.** `triton_mla.py:69` `TritonMLAImpl._forward_decode()` —— **Triton 备选 kernel**
```python
q = torch.cat([q_nope, q_pe], dim=-1)
o = torch.zeros(B, self.num_heads, self.kv_lora_rank, ...)
kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)            # 加 head dim
kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]         # latent 部分单独切出做 V
decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o,
                     attn_metadata.decode.block_table, attn_metadata.decode.seq_lens,
                     attn_logits, num_kv_splits, self.scale, PAGE_SIZE)
return self._v_up_proj_and_o_proj(o)
```
> 同样是 MQA：K 用整条 `[latent|rope]`（576），V 只用 latent 部分（512）——因为 V 的"还原"被吸收进了后面的 `_v_up_proj_and_o_proj`。

**E4.** `common.py:638` `_v_up_proj_and_o_proj()` —— **把 W_UV 吸收进输出投影**
```python
x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)  # (B,N,L) → (N,B,L)
x = torch.bmm(x, self.W_UV)                                        # (N,B,L)×(N,L,V) → (N,B,V) 还原 V
x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
return self.o_proj(x)[0]
```
> attention 在 latent 维算完后，这里才用 `W_UV` 把结果上投影回 `v_head_dim`，紧接着 `o_proj`。design §3.3 的"输出侧吸收"：`(attn·kv_c)@W_UV` 推迟到 attention 之后做，整段 attention 因此全程在 latent 维运行。

### 阶段 F：prefill 路径 —— 物化完整 KV 跑 MHA

> **这一步在干嘛**：prefill 是 compute-bound（算力足够），所以走相反的路——把 latent 经 `kv_b_proj` **上投影还原成完整的多头 K/V**，然后直接喂给现成的 FlashAttention kernel 跑标准 MHA。长 context（已缓存很多 token）会让物化 OOM，于是把 context 切成固定大小的 chunk 逐块算、再用 online-softmax 合并。

**F1.** `common.py:788` `_forward_prefill()` —— **上投影还原完整 k_nope / v**
```python
kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)   # 物化完整 K/V！
k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)        # [k_nope|k_pe]
v_padded = F.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)                  # V 头维(128) 补齐到 QK 头维(192)
output = self.flash_attn_varlen_func(                                        # 标准 FA MHA
    q=q, k=k, v=v_padded,
    cu_seqlens_q=attn_metadata.prefill.query_start_loc,
    cu_seqlens_k=attn_metadata.prefill.query_start_loc,
    softmax_scale=self.scale, causal=True, return_softmax_lse=has_context)
```
> 与 decode 相反：prefill **不吸收**，而是 `kv_b_proj` 把 latent 上投影成完整多头 `k_nope/v`（design §3.4），直接喂给现成的 `flash_attn_varlen_func`。V 头维 128 < QK 头维 192，所以 `F.pad` 补 0 对齐（`:806-808`），输出再切掉 padding（`:838-841`）。

**F2.** `common.py:711` `_compute_prefill_context()` —— **chunked context：分块 + merge_attn_states**
```python
for i in range(iters):
    ops.gather_cache(                                   # 把第 i 段 context 从 paged cache 收进 workspace
        src_cache=kv_c_and_k_pe_cache, dst=workspace,
        block_table=prefill_metadata.block_table,
        cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
        batch_size=attn_metadata.num_prefills,
        seq_starts=prefill_metadata.chunked_context.starts[i])
    kv_c_normed = workspace[:toks][..., :self.kv_lora_rank]
    k_pe        = workspace[:toks][..., self.kv_lora_rank:].unsqueeze(1)
    kv_nope = self.kv_b_proj(kv_c_normed)[0]....         # 同 F1 上投影这一段 context
    attn_output, attn_softmax_lse = self.flash_attn_varlen_func(..., causal=False)  # context 不加 causal mask
    if output is None:
        output, output_lse = attn_output, attn_softmax_lse
    else:
        merge_attn_states(output=..., prefix_output=output, suffix_output=attn_output, ...)  # online-softmax 合并
```
> design §3.5：长 context 切成固定 `workspace` 大小的 chunk，逐块 `gather_cache` → 上投影 → 算 attention → `merge_attn_states` 合并。`_forward_prefill`（`:824-836`）最后把"对新 token 的 attention（suffix）"和"对已缓存 context 的 attention（prefix）"也用 `merge_attn_states` 合并。`merge_attn_states` 的 kernel 见 [模块 11](../11-gpu-kernels-memory/impl.md)。

### 阶段 G：写回 output

> **这一步在干嘛**：把 decode 段和 prefill 段各自算出的输出写回总输出张量的对应位置。因为 B 段已经把 decode 排在前、prefill 排在后，这里直接按 `[:num_decode_tokens]` / `[num_decode_tokens:]` 切片写入即可，无需再打散重排。

`common.py:928-937`：prefill 段写 `output[num_decode_tokens:]`、decode 段写 `output[:num_decode_tokens]`，正好对应 B 段的分批布局，无需再 scatter。返回 `output_padded`（CUDA graph 可能 padding 过）。

---

## 3. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · KV cache 形状退化为单 head，`num_kv_heads` 恒为 1
- **代码**：`common.py:239-245` `get_kv_cache_shape` 返回 `(num_blocks, block_size, head_size=576)`，注释 `num_kv_heads` "assumed to be 1 for MLA"。
- **精妙之处**：MHA 的 cache 是 `[2, num_blocks, block_size, num_kv_heads, head_size]`（[模块 02](../02-paged-attention-kvcache/design.md) §3.1，分 K/V 两份、多 head）。MLA 把它压成**一份、单 head、576 维**——因为缓存的是被所有 head 共享的 latent。这一行形状定义就是约 57× 压缩比的源头，且复用了模块 02 整套 block/block_table 机制，只换了"每个 slot 装什么"。

### T2 · 吸收靠"预转置好的 W_UK_T / W_UV" + bmm，不在热路径拆 kv_b_proj
- **代码**：`process_weights_after_loading`（`common.py:660-709`）一次性把 `kv_b_proj` 拆成 `W_UK`/`W_UV` 并 permute 成 `W_UK_T=(N,P,L)`（`:709`）、`W_UV=(N,L,V)`（`:707`）；decode 时只做 `bmm`（`:656`/`:642`）。
- **精妙之处**：吸收的数学是 `q@W_UK^T` 和 `attn@W_UV`，每 token decode 都要算。若每次现拆现转置 `kv_b_proj` 会很慢。这里在**权重加载后一次性**算好转置布局，热路径退化成两个 `bmm`，把代数恒等式（design §3.3）落成廉价的批量矩阵乘。

### T3 · decode 与 prefill 是"吸收 vs 物化"的镜像
- **代码**：decode `_q_proj_and_k_up_proj`（`:648`）+ `_v_up_proj_and_o_proj`（`:638`）吸收；prefill `_forward_prefill`（`:799`）物化。
- **精妙之处**：同一个 `kv_b_proj` 权重，decode 把它**拆开吸收进 Q/O**、prefill 把它**整体乘上去还原 KV**。一份权重、两种用法，分别服务 memory-bound / compute-bound（design §3.4）。这是 MLA 最核心的对称结构——读懂这对镜像就读懂了 MLA。

### T4 · RoPE 只作用 64 维 rope 段，且搬进 backend 内施加
- **代码**：`common.py:903-905`（decode）/`:911-915`（prefill）只对 `*_q_pe`/`*_k_pe` 调 `rotary_emb`；nope 段不碰。
- **精妙之处**：decoupled RoPE（design §3.2）的落点。RoPE 必须在 backend 内施加，因为只有这里才知道 latent 解压后哪些是 rope 段、positions 如何对齐——所以 `input_positions` 被特意塞进 metadata（`:270-272`/`:283`）。普通 attention 的 RoPE 在模型层就做完了，MLA 是少数把 RoPE 推迟到 attention backend 的设计。

### T5 · 一次 `kv_a_proj_with_mqa` 同时产出 latent 和 rope key
- **代码**：`deepseek_v2.py:476-477` `.split([kv_lora_rank, qk_rope_head_dim])`。
- **精妙之处**：`kv_c`（要压缩缓存的 latent）和 `k_pe`（要 RoPE 的 decoupled key）来自**同一个线性投影的两段切片**，省一次 matmul。命名里的 `with_mqa` 暗示 `k_pe` 是 MQA 式的（对所有 head 共享一份），这也是它能和 latent 一起进单 head cache 的原因。

### T6 · reorder_batch 后，prefill/decode 切分变成纯下标偏移
- **代码**：`common.py:473-474` `reqs_start=num_decodes; tokens_start=num_decode_tokens`；`:892-897` 用 `[:num_decode_tokens]` / `[num_decode_tokens:]` 切。
- **精妙之处**：因为 `reorder_batch`（B2）已把 decode 全排到 batch 前段，后续所有"取 decode 段 / prefill 段"都退化成 `tensor[:k]` / `tensor[k:]`，**零 gather/scatter**。分批的一次性成本（少量 swap）换来了 build/forward 全程的切片便宜。这是"前置排序简化后续逻辑"的经典工程取舍。

### T7 · `_forward_decode` 只支持 query 长度 1，故判据是 `num_tokens == 1`
- **代码**：`common.py:399` 判 decode；`flashmla.py:133` `unsqueeze(1)` 加 seqlen=1 维。
- **精妙之处**：decode kernel（FlashMLA/Triton）当前只处理单 query token 的 MQA。所以 reorder 的判据不是"理想的 compute/memory-bound 分界"，而是务实的"是否恰好 1 个 token"（`:396-398` 注释明说想改成 `<8` 但 kernel 还不支持）。这解释了为什么投机解码的多 token decode 当前会落到 prefill 路径。

### T8 · chunked prefill workspace 固定大小、上界 64K token
- **代码**：`common.py:354-371` workspace 大小 = `min(max(8×max_model_len, 4×max_num_seqs×block_size), 128×1024)`。
- **精妙之处**：prefill 物化 `k_nope=[Skv,N,P]` 会随 context 线性涨，长 context 必 OOM。固定 workspace（注释算出 144MB latent / 3GB 上投影的上界）把"内存随 context 涨"换成"迭代次数随 context 涨"（design §3.5）。64K 上限是对超长 context 模型不过度预占 cache 显存的权衡。

### T9 · gather_cache 的 chunk 起点必须 page 对齐
- **代码**：`common.py:498` `max_context_chunk = round_down(max_context_chunk, page_size)`。
- **精妙之处**：`gather_cache` kernel 无法处理非页对齐的 `context_chunk_starts`（`:495-497` 注释）。所以 chunk 大小被向下取整到 `page_size` 的倍数——一个"kernel 能力倒逼上层切分粒度"的约束，漏掉就会在 chunked prefill 长 context 时静默出错。

### T10 · V 头维补 0 对齐 QK 头维，再切掉
- **代码**：`common.py:806-808` `F.pad(v, [0, q.shape[-1]-v.shape[-1]])`；`:838-841` 输出 `[..., :v.shape[-1]]` 切回。
- **精妙之处**：MLA 的 V 头维（128）比 QK 头维（192）小，但 FlashAttention kernel 要求 Q/K/V 头维一致。于是把 V 用 0 补齐到 192 喂进 kernel，输出再切掉那段 padding。一个"为复用现成 kernel 而临时改形状"的适配 trick。

### T11 · merge_attn_states 统一服务 chunked context 与 suffix/prefix 合并
- **代码**：`common.py:775-782`（chunk 间合并）、`:829-836`（context 与新 token 合并）。
- **精妙之处**：无论是"context 分多 chunk"还是"已缓存 context vs 当前新 token"，本质都是"两段独立 softmax-attention 结果按 log-sum-exp 正确合并"。同一个 `merge_attn_states` 原语两处复用（也是 cascade attention 的同一原语，[模块 02](../02-paged-attention-kvcache/design.md) §3）。`#16173` 把它做成 3× 加速的 CUDA kernel（[模块 11](../11-gpu-kernels-memory/impl.md)）。

### T12 · FlashMLA 强制 block_size=64，平台层提前改 cache_config
- **代码**：`cuda.py:154-161` 若用 FlashMLA 且支持，则 `cache_config.block_size = 64`；`:185` block_size≠64 时回退 TritonMLA。
- **精妙之处**：FlashMLA kernel 硬要求 block_size=64（≠ 默认）。vLLM 在 **KV cache 分配之前**（平台 `check_and_update_config`）就强制改掉 block_size，否则后面 cache 形状对不上 kernel。这是"kernel 约束反向决定全局 KV cache 分页粒度"的一例——MLA 的 block_size 不是用户随便设的。

### T13 · W_UV/W_UK_T 强制 16-bit，不做量化 bmm
- **代码**：`common.py:685-687` 注释 + `:688` `get_and_maybe_dequant_weights` 对量化层做 O(N³) 反量化（仅离线）。
- **精妙之处**：吸收用的 `bmm` 当前没有量化 kernel，所以即便 `kv_b_proj` 是量化的，也把 `W_UK/W_UV` 反量化成 fp16/bf16 副本存着（多一份显存，但量小）。`:674-682` 用 `eye` 矩阵过一遍 `quant_method.apply` 来反量化——一个"借单位矩阵把量化权重还原成稠密权重"的离线技巧。

### T14 · get_mla_metadata 预算 tile 调度，把 kernel 调度搬到 build 期
- **代码**：`flashmla.py:64-69` `_build_decode` 调 `get_mla_metadata(seq_lens, num_q_heads, 1)`。
- **精妙之处**：FlashMLA 需要按 seq_lens 预先规划 tile 划分与 `num_splits`。vLLM 在每步 `build`（CPU 期、与 GPU 重叠）就算好 `tile_scheduler_metadata`，kernel 启动时直接用。把可在 CPU 提前做的调度决策从 GPU 关键路径挪开，是 MLA CPU 开销优化（`#14540`）的一部分。

### T15 · `num_decodes` 等字段从可选改必填，是非法内存访问的根因修复
- **代码**：`common.py:312-314` 三个字段无默认值（必填）；对比 `#14253` 前是 `Optional=None`。
- **精妙之处**：早期把 prefill/decode 计数设成可选，一旦某路径忘了填就会在切分时取到 `None` → 切片越界 / 静默错。`#14253`（design §8.2）把它们改为必填、并让 `reorder_batch` 返回 `modified_batch` 驱动 `refresh_sampling_metadata`。教训：分批是正确性的一等约束，不能用"可选字段 + 默认值"敷衍。

---

## 4. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| profiling run（attn_metadata=None） | `common.py:868-870` | 直接返回 output，不做 attention |
| padding token（slot<0） | `cache_kernels.cu:325-328` | `concat_and_cache_mla_kernel` 直接 return，不写 cache |
| CUDA graph padding | `common.py:874-879` | 先按 `num_actual_tokens` 切掉 padding，算完返回 `output_padded` |
| batch 全是 decode / 全是 prefill | `common.py:888-889`/`928-934` | `has_decode`/`has_prefill` 任一为 False 则跳过对应路径 |
| prefill 无已缓存 context | `common.py:798`/`821` | `has_context=False`，`_forward_prefill` 不调 `_compute_prefill_context`、不 merge |
| chunked prefill 关闭 | `common.py:482` | `chunked_context_metadata=None`，prefill 一次算完（context 大可能 OOM） |
| query 长度 >1（如投机解码） | `common.py:399` | 不算 decode，落到 prefill 路径（`_forward_decode` 仅支持 len=1） |
| FlashMLA 不支持的设备 / block≠64 | `cuda.py:185,196-204` | 回退 TritonMLABackend |
| FP8 KV cache | `flashmla.py:119-121`/`triton_mla.py:65-67` | 抛 NotImplementedError（MLA 暂不支持量化 KV） |
| alibi / sliding window / soft_cap | `flashmla.py:104-111`/`triton_mla.py:50-57` | 抛 NotImplementedError |
| encoder / cross-attention | `flashmla.py:113-117`/`triton_mla.py:59-63` | 抛 NotImplementedError（仅 DECODER） |
| head_dim ≠ 576 | `common.py:325-331` | `MLACommonMetadata.__post_init__` 抛 ValueError |
| 量化 kv_b_proj（吸收用） | `common.py:671-682` | 用 eye 矩阵离线反量化成 16-bit 副本（O(N³)，仅加载期） |

---

## 5. 一图速查：MLA 一个调度步

```
[模型层] deepseek_v2.py:466 forward
   kv_a_proj_with_mqa(hidden) ─split─► kv_c[Lkv=512] | k_pe[R=64]    (一次投影出 latent+rope)
   kv_a_layernorm(kv_c) → kv_c_normed
   q_a/q_b_proj → q_c                       ──► mla_attn(use_mla, head_size=576)
        ▼ ───────────────────────── attention 层 layer.py:193 跳过 q/k/v reshape ─────────────────
[reorder] gpu_model_runner.py:474 reorder_batch ──► common.py:380
              num_tokens==1 → decode(挪前)  ;  else → prefill(挪后)   (swap_states 最少交换)
        ▼
[build] common.py:450
   切 decode 段 [:num_decodes] / prefill 段 [num_decodes:]   (纯下标偏移，靠 reorder 回报)
   prefill: chunked_context(workspace固定, chunk页对齐 round_down)   decode: _build_decode(+get_mla_metadata)
        ▼
[forward] common.py:855
   ├─ decode/prefill 各自施加 decoupled RoPE (只动 *_pe 那64维, :903/:913)
   ├─ 写KV: concat_and_cache_mla :917 ──► cache_kernels.cu:308
   │         slot→(block,offset); 一个slot拼 [latent(512)|rope(64)]  (复用模块02分页寻址)
   │
   ├─ PREFILL :788  (compute-bound, MHA)
   │     kv_b_proj 上投影 → 物化完整 k_nope/v   :799
   │     V补0对齐QK头维 → flash_attn_varlen_func(causal) :811
   │     长context: gather_cache→上投影→attn→merge_attn_states 分块 :711
   │     o_proj :843
   │
   └─ DECODE :934  (memory-bound, MQA)
         _q_proj_and_k_up_proj: ql_nope = q_nope @ W_UK^T  (吸收K上投影) :656
         FlashMLA flash_mla_with_kvcache / Triton decode_attention_fwd  (block_table分页寻址)
         _v_up_proj_and_o_proj: bmm(o, W_UV) → o_proj  (吸收V上投影) :638
        ▼
   output[:num_decode_tokens]=decode ; output[num_decode_tokens:]=prefill   (对齐分批布局)
        ▼
   output [Sq, H]
```
