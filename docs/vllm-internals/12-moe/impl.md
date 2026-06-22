# 模块 12 · MoE 混合专家 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的端到端 MoE 调用链：router → select_experts → moe_align_block_size → grouped GEMM → combine。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
> 概念与并行组合见 design.md；EP 作为并行维度见 [模块 03](../03-distributed-parallel/impl.md)；kernel GPU 视角见 [模块 11](../11-gpu-kernels-memory/impl.md)；量化数值格式见 [模块 08](../08-quantization/design.md)。

---

## 1. 代码地图

```
Python 层 vllm/model_executor/layers/fused_moe/
  ├─ layer.py                  # FusedMoE 模块、FusedMoEMethodBase、determine_expert_map、
  │                            #   select_experts、weight_loader、forward_impl、naive_multicast
  ├─ fused_moe.py              # 核心：fused_topk / grouped_topk（路由）、
  │                            #   fused_moe_kernel（triton grouped GEMM）、invoke_fused_moe_kernel、
  │                            #   fused_experts / fused_experts_impl、moe_kernel_prepare_input
  ├─ moe_align_block_size.py   # moe_align_block_size：排序+padding 成分块（题眼）
  ├─ cutlass_moe.py            # cutlass_moe_fp8：CUTLASS grouped GEMM fp8 后端
  ├─ deep_gemm_moe.py          # deep_gemm_moe_fp8：DeepGEMM contiguous grouped GEMM 后端
  ├─ fused_marlin_moe.py       # fused_marlin_moe：int4/int8 wna16 Marlin 后端
  ├─ moe_pallas.py             # TPU：XLA Pallas gmm 路径
  └─ rocm_aiter_fused_moe.py   # ROCm AITER 后端

CUDA kernel csrc/moe/
  ├─ topk_softmax_kernels.cu   # ops.topk_softmax（fused_topk 用，改编自 TensorRT-LLM）
  ├─ moe_align_sum_kernels.cu  # ops.moe_align_block_size / sgl_moe_align_block_size / moe_sum
  ├─ moe_wna16.cu              # ops.moe_wna16_gemm（int4/int8 weight-only cuda kernel）
  └─ marlin_moe_*.cu           # ops.moe_wna16_marlin_gemm（Marlin MoE kernel）

量化方法（注入 quant_method）
  ├─ quantization/fp8.py                                  # Fp8MoEMethod
  ├─ quantization/compressed_tensors/compressed_tensors_moe.py  # CT 系列（fp8/cutlass/wna16）
  ├─ quantization/gptq_marlin.py                          # GPTQMarlinMoEMethod
  └─ quantization/awq_marlin.py                           # AWQMoEMethod

模型接线（举例）
  ├─ models/mixtral.py         # softmax top-2，reduce_results=True
  ├─ models/deepseek_v2.py     # grouped_topk + e_score_correction_bias + shared_experts
  └─ models/llama4.py          # custom_routing_function + apply_router_weight_on_input + shared_expert
```

---

## 2. 端到端调用链（一个 MoE 层的完整旅程）

以 **DeepSeek-V2/V3（grouped_topk + shared expert）** 为主线，因为它覆盖了最多机制；Mixtral 差异在 §3。

### 阶段 A：模型层 —— 算 router logits、并排 shared expert

> **这一步在干嘛**：在模型层里，先用一个小 Linear（gate）算出每个 token 对各专家的「偏好分数」（router logits）；DeepSeek 这类还在旁边并排放一个**共享专家**（普通 MLP，对所有 token 都激活，承载通用知识）。这一步还没选专家、没算路由专家，只是把「打分」和「共享专家输出」准备好。

**A1.** `vllm/model_executor/models/deepseek_v2.py:157` `DeepseekV2MoE.forward`
```python
router_logits, _ = self.gate(hidden_states)          # gate: ReplicatedLinear hidden→n_routed_experts
```
`gate` 是无 bias 的 `ReplicatedLinear`（`deepseek_v2.py:113-117`）。V3 时 gate 还挂一个负载均衡偏置：
```python
# deepseek_v2.py:118-122
if config.topk_method == "noaux_tc":
    self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts))
```

**A2.** `deepseek_v2.py:154-155` 共享专家在同一 hidden 上跑（始终激活）：
```python
shared_output = self.shared_experts(hidden_states)   # DeepseekV2MLP，reduce_results=False
```

**A3.** `deepseek_v2.py:124-137` `FusedMoE` 的构造（决定路由策略）：
```python
self.experts = FusedMoE(num_experts=config.n_routed_experts, top_k=config.num_experts_per_tok,
    hidden_size=..., intermediate_size=config.moe_intermediate_size,
    reduce_results=False,                          # 关键：延后 all-reduce
    renormalize=config.norm_topk_prob,
    use_grouped_topk=True, num_expert_group=config.n_group, topk_group=config.topk_group,
    scoring_func=config.scoring_func,              # V3 = "sigmoid"
    e_score_correction_bias=self.gate.e_score_correction_bias)
```

---

### 阶段 B：FusedMoE.forward —— 入口与 DP multicast

> **这一步在干嘛**：进 `FusedMoE` 层的总入口。单 DP 时直接往下走；多 DP 时要先把各 DP rank 的 token 聚到一起（因为专家计算要跨 DP 把 token 凑齐），算完再 all-reduce、切回本 rank。这里还决定 all-reduce 在哪做——DeepSeek 设 `reduce_results=False`，把 all-reduce 推迟到模型层和共享专家合并后只做一次。

**B1.** `layer.py:839` `FusedMoE.forward`
```python
if self.use_direct_call:                            # dp_size == 1
    return self.forward_impl(hidden_states, router_logits)
else:
    return torch.ops.vllm.moe_forward(hidden_states, router_logits, self.layer_name)
```
`use_direct_call = (dp_size == 1)`（`layer.py:443`）。DP>1 时走注册的 custom op `moe_forward`（`layer.py:933-953`），它从 forward_context 按 `layer_name` 取回 self 再调 `forward_impl`——这是为了让 MoE 能被 `torch.compile` 捕获（不能把 self 这种对象传进图，只传字符串 `layer_name`）。

**B2.** `layer.py:847` `FusedMoE.forward_impl`
- `:851-858` 若 `dp_size > 1`：`naive_multicast` 把各 DP rank 的 token 聚到一起（`layer.py:821`），因为专家计算需要跨 DP 把 token 凑齐。
- `:861-877` 调 `self.quant_method.apply(...)` —— 真正干活，传入 `global_num_experts`、`expert_map`、`top_k`、路由参数。
- `:879-885` DP 时 `all_reduce` 后切回本 rank 的 token 区间。
- `:887-890` 若 `reduce_results and (tp_size>1 or ep_size>1)`：TP/EP all-reduce。DeepSeek 设 `reduce_results=False`，所以这里**不**触发，留给模型层合并 shared_output 后再 reduce。

---

### 阶段 C：① 路由 —— select_experts

> **这一步在干嘛**：把 router 算出的分数变成实打实的「每个 token 去哪 top-k 个专家、各占多大门控权重」。三种打分/选择策略走同一个入口：Mixtral 的 softmax-topk、DeepSeek 的「先选组再选专家 + sigmoid 打分 + 负载均衡偏置」、Llama4/Phi 的自定义路由。输出就两个张量：`topk_ids`（选中的专家 id）和 `topk_weights`（门控权重）。

**C1.** `layer.py:199` `UnquantizedFusedMoEMethod.forward_cuda`（量化方法的 apply 同构）调：
```python
topk_weights, topk_ids = FusedMoE.select_experts(hidden_states=x, router_logits=router_logits,
    use_grouped_topk=..., top_k=..., renormalize=..., topk_group=..., num_expert_group=...,
    custom_routing_function=..., scoring_func=..., e_score_correction_bias=...)
```

**C2.** `layer.py:780` `FusedMoE.select_experts`（三分支，`layer.py:795-817`）
```python
if use_grouped_topk:                 # DeepSeek
    topk_weights, topk_ids = grouped_topk(...)
elif custom_routing_function is None:  # Mixtral
    topk_weights, topk_ids = fused_topk(...)
else:                                 # Llama4 / Phi
    topk_weights, topk_ids = custom_routing_function(...)
```

**C3a.** `fused_moe.py:853` `fused_topk`（softmax 门控）
- `:877` `gating_output.float()`（精度）。
- `:879-882` `vllm_topk_softmax`（`fused_moe.py:831`）→ `ops.topk_softmax(...)`（一个 CUDA kernel 融合 softmax+topk，`csrc/moe/topk_softmax_kernels.cu`）。
- renormalize 在 kernel 外：`topk_weights / topk_weights.sum(-1, keepdim=True)`（`fused_moe.py:841-842`）。

**C3b.** `fused_moe.py:890` `grouped_topk`（DeepSeek，`@torch.compile(dynamic=True)`）
```python
if scoring_func == "softmax": scores = softmax(gating_output)
elif scoring_func == "sigmoid": scores = gating_output.sigmoid()    # V3
...
if e_score_correction_bias is not None:                  # V3 noaux_tc
    original_scores = scores
    scores = scores + e_score_correction_bias.unsqueeze(0)           # 选择用带偏分数
    group_scores = scores.view(n, num_expert_group, -1).topk(2, -1)[0].sum(-1)  # 每组 top-2 之和
else:
    group_scores = scores.view(n, num_expert_group, -1).max(-1).values
group_idx = topk(group_scores, k=topk_group)[1]          # 先选组
group_mask → score_mask
tmp_scores = scores.masked_fill(~score_mask, float("-inf"))  # 屏蔽非选中组（-inf 而非 0！#13474）
if e_score_correction_bias is not None:
    topk_ids = topk(tmp_scores, k=topk)[1]
    topk_weights = original_scores.gather(1, topk_ids)   # 权重用原始未偏置分数
```
（`fused_moe.py:904-945`）这段是 §design 3.1(b) 的代码落点：**选择/权重分离 + 先组后专家**。

---

### 阶段 D：② 对齐分块 —— moe_align_block_size（题眼）

> **这一步在干嘛**：这是整个模块的题眼。路由完手上是「专家 A 有 37 个 token、专家 B 有 5 个」这种**变长锯齿**形状，GPU 的 GEMM kernel 吃不下。这一步把所有 token **按所属专家排序**、每个专家尾部补 padding 对齐到 `block_size` 的整数倍，产出两个张量：`sorted_token_ids`（排好序的 token 行布局）和 `expert_ids`（每个 block 属于哪个专家——grouped GEMM 的「权重选择器」）。从此 kernel 眼里就是规整的 tile，完全不知道路由原本是动态的。EP 时还顺手把全局专家 id 经 `expert_map` 重映射成本地 id（非本卡变 -1）。

由 `fused_experts_impl` 在每个 chunk 内调用（见 E2）：

**D1.** `moe_align_block_size.py:151` `moe_align_block_size(topk_ids, block_size, num_experts, expert_map)`
- `:201` `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)` —— 最坏情况每个专家都要补 `block_size-1` 个 padding。
- `:204-207` 分配 `sorted_ids`，**全部预填 `topk_ids.numel()`**（一个越界值，作为 padding 哨兵，kernel 用 `token_mask` 忽略）。
- `:211-213` `expert_ids` **预填 0**（注释 `:209-210`：必须清零，否则 EP 重映射时越界）。
- `:217-239` **按专家数选 kernel**：
  ```python
  if num_experts >= 224:
      if VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON or num_experts != 256:
          moe_align_block_size_triton(...)         # triton 版（共享内存放不下时）
      else:
          ops.sgl_moe_align_block_size(...)         # SGLang global-mem CUDA（需 num_experts==256，如 V3）
  else:
      ops.moe_align_block_size(...)                 # 标准 CUDA（Mixtral 8 专家走这）
  ```
  这个 224/256 的分叉正是 #12850 为 DeepSeek-V3 引入的（专家数太多，原 shared-memory kernel 失效）。
- `:240-241` **EP 重映射**：`if expert_map is not None: expert_ids = expert_map[expert_ids]` —— 把每块的全局专家 id 转成本地 id，非本卡专家变 -1。

**D2.** CUDA kernel `csrc/moe/moe_align_sum_kernels.cu:25` `moe_align_block_size_kernel`
三步（shared memory 内）：① 每线程数自己 token shard 里各专家的计数（`:49-51`）；② 前缀和求每专家的起始偏移并对齐到 `block_size`（`:67-76`，`cumsum[i] = cumsum[i-1] + CEILDIV(cnt, block_size)*block_size`）；③ 写 `expert_ids[block]`（`:84-89`）与 `sorted_token_ids`（`:98-111`）。docstring 例子在 `:91-97`。

产出三件套：`sorted_token_ids`（grouped GEMM 的行布局）、`expert_ids`（每块的权重选择器）、`num_tokens_post_pad`。

---

### 阶段 E：③ 专家 grouped GEMM —— fused_experts → fused_experts_impl

> **这一步在干嘛**：真正的专家计算——**两段 grouped GEMM 夹一个激活**。① 第一段 GEMM 用 `expert_ids` 给每个 token 块选对专家的 `w13` 权重算 gate+up；② `silu_and_mul` 激活；③ 第二段 GEMM 算 `w2`（down），并在这里把门控权重乘上去；④ 把每个 token 的 top-k 个专家输出加权求和成一行。全程一个融合 triton kernel 用 `expert_ids[block]` 切权重、一次 launch 算完所有专家；非本卡专家（EP）的块直接输出写 0。

**E1.** `layer.py:211` `forward_cuda` 调 `fused_experts(...)`（`fused_moe.py:1111`）
- `:1134-1151` 若 `allow_deep_gemm and use_fp8_w8a8 and _valid_deep_gemm(...)`：转 `deep_gemm_moe_fp8`（见 §4）。
- `:1152-1174` 否则 `dispatch_fused_experts_func(inplace)` → `inplace_fused_experts`（`:966`）/ `outplace_fused_experts`（`:1029`），二者都进 `fused_experts_impl`。inplace/outplace 各注册成 custom op（`:1018`/`:1082`）以便 torch.compile。

**E2.** `fused_moe.py:1230` `fused_experts_impl`（核心循环）
- `:1267-1272` 取形状：`E, N, _ = w1.shape`（N = 2*intermediate）、`K = w2.shape[1]`、`top_k_num`。`global_num_experts == -1` 时用 `E`。
- `:1275-1276` **分块执行**：`CHUNK_SIZE = VLLM_FUSED_MOE_CHUNK_SIZE`，`M = min(num_tokens, CHUNK_SIZE)`——绕过 issue #5938（token 太多时显存/数值问题）。
- `:1291` `config = get_config_func(M)`：查 tuned 的 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`（`try_get_optimal_moe_config`, `fused_moe.py:798`；按 `(E, N, dtype, device)` 查 `configs/*.json`）。
- `:1293-1304` **中间 cache 分配（显存复用，#13625）**：
  ```python
  cache13 = torch.empty(M * top_k_num * max(N, K), ...)
  intermediate_cache1 = cache13[:M*top_k_num*N].view(M, top_k_num, N)   # 第一段 GEMM 输出
  intermediate_cache3 = cache13[:M*top_k_num*K].view(M, top_k_num, K)   # 第二段 GEMM 输出（复用 cache13！）
  intermediate_cache2 = torch.empty((M*top_k_num, N//2), ...)           # silu 后，独立显存
  ```
  cache1 与 cache3 共用 `cache13`（"用 cache3 时 cache1 已算完"）。
- `:1320-1424` **每个 chunk 的五步**：
  ```python
  # chunk 切片（最后一个 chunk 调整 cache 形状——注意 cache2 要 * topk，#13693）
  intermediate_cache2 = intermediate_cache2[:tokens_in_chunk * topk_ids.shape[1]]   # :1336
  # ③-准备：量化激活（fp8/int8 时）
  qcurr_hidden_states, qa1_scale = moe_kernel_prepare_input(...)                    # :1344
  # ② 对齐分块
  sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
      curr_topk_ids, config['BLOCK_SIZE_M'], global_num_experts, expert_map)        # :1356
  # ③ 第一段 grouped GEMM：hidden @ w13
  invoke_fused_moe_kernel(qcurr_hidden_states, w1, intermediate_cache1, ...,
      apply_router_weight_on_input, top_k_num, ...)                                 # :1360
  # ④ 激活
  torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))  # :1382
  qintermediate_cache2, qa2_scale = moe_kernel_prepare_input(...)                   # :1390
  # ⑤ 第二段 grouped GEMM：cache2 @ w2，门控权重在此乘
  invoke_fused_moe_kernel(qintermediate_cache2, w2, intermediate_cache3, ...,
      not apply_router_weight_on_input, 1, ...)                                     # :1402
  # combine：把每 token 的 topk 个专家输出求和
  ops.moe_sum(intermediate_cache3.view(...), out_hidden_states[begin:end])         # :1423
  ```

**E3.** `fused_moe.py:466` `invoke_fused_moe_kernel`（launch triton kernel）
- `:490-502` 算 `EM`（grid 的 M 维）；小 batch 时优化 `EM = min(..., M*top_k*BLOCK_SIZE_M)`（假设每 token 的 topk 专家互异，可跳过空块）。
- `:504-534` int4/int8 wna16 时可能走 `ops.moe_wna16_gemm`（cuda）或 `fused_moe_kernel_gptq_awq`（triton）。
- `:573-621` 否则 launch `fused_moe_kernel[grid](...)`，传 A/B/C 指针、各 scale、`sorted_token_ids`、`expert_ids`、`num_tokens_post_padded`、`MUL_ROUTED_WEIGHT`、量化开关。

**E4.** `fused_moe.py:256` `fused_moe_kernel`（triton grouped GEMM 本体）
- `:349-351` `if pid_m*BLOCK_SIZE_M >= num_tokens_post_padded: return` —— 超出 padding 总长的块直接退出。
- `:354-355` `offs_token = sorted_token_ids[...]`；`token_mask = offs_token < num_valid_tokens`（屏蔽 padding）。
- `:357-365` **EP 跳过**：`off_experts = expert_ids[pid_m]`；`if off_experts == -1: write_zeros_to_output; return`（非本卡专家写 0）。
- `:370-374` **A 索引玄机**：`a_ptrs = a_ptr + (offs_token[:,None] // top_k * stride_am + ...)` —— `// top_k` 把"含 topk 复制的展平索引"还原成原始 token 行（激活不真复制）；`b_ptrs = b_ptr + off_experts*stride_be + ...`（按专家选权重矩阵）。
- `:406-441` fp32 累加的 K 维循环（量化时按 group/channel/tensor 加载 scale）。
- `:443-447` `if MUL_ROUTED_WEIGHT: accumulator *= moe_weight` —— 门控权重乘在这里（第二段调用时 `MUL_ROUTED_WEIGHT=True`）。
- `:457-463` 写回 C。

**E5.** combine：`ops.moe_sum`（`csrc/moe/moe_align_sum_kernels.cu`）把 `cache3` 的 `(M, top_k, hidden)` 沿 top_k 维求和成 `(M, hidden)`——一个 token 的 top_k 个专家输出（已各自乘过门控权重）相加。

---

### 阶段 F：combine 回模型层 —— shared 相加 + 单次 all-reduce

> **这一步在干嘛**：路由专家算完回到模型层，把它的输出（按 `routed_scaling_factor` 缩放）和阶段 A 算好的共享专家输出**相加**，因为两者各自都设了 `reduce_results=False`、都没 all-reduce，这里合并后**只做一次** all-reduce 就够（all-reduce 对加法可分配）——省一次跨卡集合通信。最后得到这一层的 `final_hidden_states`，接回 decoder 下一层。

回到 `deepseek_v2.py:159-177`：
```python
final_hidden_states = self.experts(...) * self.routed_scaling_factor   # 路由输出（非 fp16 路径）
if shared_output is not None:
    final_hidden_states = final_hidden_states + shared_output           # 合并共享专家
if self.tp_size > 1:
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 只 reduce 一次
```
fp16 时为防溢出改成 `+ shared_output * (1/routed_scaling_factor)`（`deepseek_v2.py:170-173`）。

---

## 3. 其它模型/路径差异

- **Mixtral**（`mixtral.py:88-106`）：`use_grouped_topk=False` → 走 `fused_topk`（softmax top-2）；`reduce_results=True` → all-reduce 在 `FusedMoE.forward_impl` 内部完成（`layer.py:887-890`），无 shared expert。
- **Llama4**（`llama4.py:73-105`）：`custom_routing_function=Llama4MoE.custom_routing_function`（`fast_topk` + sigmoid），`apply_router_weight_on_input=True`（门控权重提前乘到输入，于是两段 GEMM 都不乘），`reduce_results=False` + 并排 `shared_expert`，合并后 all-reduce。
- **EP 模式**（任意模型 + `enable_expert_parallel`）：`FusedMoE.__init__`（`layer.py:451-462`）把 `tp_size=1`、`ep_size=tp_size*dp_size`，`determine_expert_map` 生成 `expert_map`；weight loading 跳过非本卡专家（`layer.py:647-649`），kernel 内写 0。

---

## 4. 量化后端的分派（apply 落点）

| 后端 | 入口 file:line | 触发 | grouped GEMM 形式 |
|---|---|---|---|
| triton fused | `fused_experts_impl` `fused_moe.py:1230` | 默认 / fp8 / int8 / wna16 | `fused_moe_kernel`（排序+padding+expert_ids 索引） |
| DeepGEMM fp8 | `deep_gemm_moe_fp8` `deep_gemm_moe.py:117` | `allow_deep_gemm` 且 `_valid_deep_gemm`（`:22`，**`expert_map is None`、N>512、对齐**）| `dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(...)`（`:271`），block=`get_m_alignment_for_contiguous_layout()` |
| CUTLASS fp8 | `cutlass_moe_fp8` `cutlass_moe.py:11` | CT `...CutlassMethod` | `ops.cutlass_moe_mm(...)`（`:136`/`:146`），路由靠自算的 `a_map/c_map/expert_offsets`（`ops.get_cutlass_moe_mm_data`, `:126`），**无 expert_map 参数** |
| Marlin wna16 | `fused_marlin_moe` `fused_marlin_moe.py:167` | GPTQ/AWQ/CT-wna16-marlin | `ops.moe_wna16_marlin_gemm(...)`（`:282`/`:314`），仍用 `moe_align_block_size`（`:246`）；GPTQ 用 `g_idx`+`sort_indices`，AWQ 用 zero-points；门控权重折进第二段（`mul_topk_weights=True`） |
| moe_wna16 | `fused_experts(use_int4/int8_w16, block_shape=[0,group])` | CT `WNA16MoEMethod` `compressed_tensors_moe.py:1032` | `ops.moe_wna16_gemm` 或 `fused_moe_kernel_gptq_awq`（`fused_moe.py:504-572`） |
| TPU Pallas | `fused_moe` `moe_pallas.py:8` | TPU | `torch.ops.xla.gmm(x, w, group_sizes)`（`:56`/`:58`），argsort 分组、`group_sizes` 表变长 |

fp8 apply 的 load-bearing 调用（`fp8.py:799-819`）：
```python
return fused_experts(x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids,
    inplace=True, use_fp8_w8a8=True, global_num_experts=global_num_experts, expert_map=expert_map,
    w1_scale=(w13_weight_scale_inv if block_quant else w13_weight_scale), w2_scale=...,
    a1_scale=..., a2_scale=..., block_shape=weight_block_size, allow_deep_gemm=self.allow_deep_gemm)
```
Marlin apply 的 load-bearing 调用（`gptq_marlin.py:626-643`）：
```python
return torch.ops.vllm.fused_marlin_moe(x, layer.w13_qweight, layer.w2_qweight,
    layer.w13_scales, layer.w2_scales, router_logits, topk_weights, topk_ids,
    global_num_experts=..., expert_map=expert_map,
    g_idx1=layer.w13_g_idx, g_idx2=layer.w2_g_idx,
    sort_indices1=..., sort_indices2=..., num_bits=..., workspace=..., is_k_full=...)
```

---

## 5. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。下面每条都是读 MoE 代码时容易一眼滑过、但理解了才算真懂的点。

### T1 · `moe_align_block_size` 把"数据相关的稀疏"变成"kernel 眼里的稠密规整"
- **代码**：`moe_align_block_size.py:151`；CUDA `moe_align_sum_kernels.cu:25`。
- **精妙之处**：grouped GEMM kernel 完全不知道路由是动态的——它只看到一个对齐到 `block_size` 的 `sorted_token_ids` 和一个 `expert_ids[block]`。所有"哪个 token 去哪个专家、每专家多少 token"的数据相关性，被这一步**预先吸收进两个规整张量**。这是整个模块得以用"一个静态 kernel"跑"动态稀疏"的根本（design §3.2 的 MegaBlocks 思想落点）。

### T2 · padding 哨兵 = `topk_ids.numel()`，靠越界值 + token_mask 忽略
- **代码**：`moe_align_block_size.py:207` `sorted_ids.fill_(topk_ids.numel())`；kernel `fused_moe.py:354-355` `token_mask = offs_token < num_valid_tokens`。
- **精妙之处**：padding 不用单独的标志位，而是填一个**保证大于任何合法 token 索引**的值（token 总数）。kernel 里一个 `< num_valid_tokens` 的比较就同时屏蔽了所有 padding 行，load 时 `other=0.0`。零分支、零额外内存。

### T3 · `expert_map[expert_ids]` + kernel 内 `-1` 写零：EP 和 TP 共用一个 kernel
- **代码**：`moe_align_block_size.py:240-241`；`fused_moe.py:357-365` `if off_experts == -1: write_zeros_to_output`。
- **精妙之处**：EP 不需要第二套 dispatch kernel。把全局专家 id 经 `expert_map` 重映射，非本卡的块变 -1，kernel 见 -1 就写零返回。`expert_ids` 预填 0（`:211`）保证重映射前的索引不越界。代价是非本卡的块仍占 grid 空转一下，换来代码完全统一。

### T4 · A 的索引 `offs_token // top_k`：激活不真复制 top_k 份
- **代码**：`fused_moe.py:370` `a_ptrs = a_ptr + (offs_token[:,None] // top_k * stride_am + ...)`。
- **精妙之处**：`sorted_token_ids` 存的是"含 topk 复制的展平索引"（一个 token 出现 top_k 次，去不同专家）。但激活张量 `hidden_states` 只有 M 行、没复制。kernel 用 `// top_k` 把展平索引**映射回原始行**，于是 top_k 个专家块各自用不同权重读**同一行激活**——省掉了 top_k 倍的激活显存与拷贝。

### T5 · 门控权重乘在第二段 GEMM 的 fp32 累加器上
- **代码**：`fused_moe.py:1412`（第二段传 `mul_routed_weight=not apply_router_weight_on_input`）；kernel `:443-447`。
- **精妙之处**：`topk_weights` 不在路由后乘、不在第一段乘，而在 **w2 GEMM 末尾、accumulator 还是 fp32 时**乘，乘完再 cast 回低精度——数值最稳，且只乘一次。第一段 GEMM 的 `MUL_ROUTED_WEIGHT=False`（`fused_moe.py:1370` 传 `apply_router_weight_on_input`，常为 False）。

### T6 · cache1 与 cache3 复用同一块显存
- **代码**：`fused_moe.py:1293-1299` `cache13 = empty(M*topk*max(N,K)); cache1=cache13[:..N]; cache3=cache13[:..K]`。
- **精妙之处**：第一段输出 cache1（宽 N=2I）和第二段输出 cache3（宽 K=hidden）**生命周期不重叠**（算 cache3 时 cache1 已被 silu 消费进 cache2）。复用一块 `max(N,K)` 的 buffer 省下一份大显存（#13625）。但这正是 #13693 的温床——见 T7。

### T7 · cache2 含 topk 维、cache1/3 不含：chunk 切片必须区别对待
- **代码**：`fused_moe.py:1335-1338`：`cache1[:tokens]`、`cache3[:tokens]`，但 `cache2[:tokens * topk_ids.shape[1]]`。
- **精妙之处**：cache1/3 形状是 `(M, top_k, ·)`（topk 在中间维），cache2 是 `(M*top_k, N//2)`（topk 已展平进第 0 维）。最后一个不满的 chunk 切片时，cache2 必须 `* topk` 否则越界非法访存（#13693 修的就是漏了 `* topk`）。**显存复用的代价：形状一致性变成正确性负担**。

### T8 · 按专家数三路分派 moe_align kernel（8 → 256 的演进伤疤）
- **代码**：`moe_align_block_size.py:217-239`。
- **精妙之处**：Mixtral 8 专家走标准 CUDA kernel（shared memory 够用）；DeepSeek-V3 的 256 专家放不进 shared memory，于是 `num_experts>=224` 时走 SGLang 的 global-mem CUDA（`ops.sgl_moe_align_block_size`，仅支持 256）或 triton 变体。这是 #12850 留下的分叉——MoE 的 kernel 假设会被新模型的专家规模直接打破。

### T9 · DeepSeek-V3 路由的"选择分数 / 权重分数"分离
- **代码**：`fused_moe.py:913-935`。
- **精妙之处**：`e_score_correction_bias` 是训练时为负载均衡动态调的偏置。**带偏置分数只用于选专家（argmax），权重用 `original_scores.gather` 取原始分数**。若直接用带偏分数当权重，会把"调度均衡的扰动"混进数值、损害精度。这是无辅助损失负载均衡（DeepSeekMoE/V3）在推理侧的精确还原。

### T10 · grouped_topk 屏蔽用 `-inf` 而非 `0.0`
- **代码**：`fused_moe.py:929` `tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))`。
- **精妙之处**：sigmoid 分数恒为正，若用 `0.0` 屏蔽非选中组，被屏蔽专家的分数（0）仍可能大于某些选中专家、被错选进 top-k。必须用 `-inf` 才能保证 topk 只在选中组里选。#13474 修的正是 `0.0` → `-inf`。

### T11 · DP>1 走 custom op `moe_forward`，只为能进 torch.compile
- **代码**：`layer.py:839-845`、`:933-953` `direct_register_custom_op("moe_forward", ...)`。
- **精妙之处**：Dynamo 不能把 `self`（FusedMoE 对象）trace 进图。于是 DP 路径把 forward 包成 custom op，**只传 `layer_name` 字符串**，op 内部从 `forward_context.no_compile_layers[layer_name]` 取回 self（`:936`）再调 `forward_impl`。`use_direct_call=(dp_size==1)` 时才走直接调用绕过这层间接。同源思想见 [模块 03](../03-distributed-parallel/design.md) §4.2 的 all-reduce custom op。

### T12 · shared expert + reduce_results=False：把两次 all-reduce 合成一次
- **代码**：`deepseek_v2.py:124-149`（FusedMoE 与 shared MLP 都 `reduce_results=False`）+ `:175-177`（合并后单次 all-reduce）。
- **精妙之处**：routed 输出和 shared 输出**各自都不 all-reduce**，相加后再做一次 `tensor_model_parallel_all_reduce`。因为 all-reduce 对加法分配律成立（`AR(a)+AR(b)=AR(a+b)`），合成一次省一次集合通信。代价是 `FusedMoE.forward_impl` 的 `:887-890` 那个 reduce 分支被刻意跳过、责任转移到模型层。

### T13 · 权重堆叠成 `(E,N,K)` + 列/行并行的专家版 weight loading
- **代码**：`layer.py:583-617`（`_load_w13` 按 output_dim narrow、`_load_w2` 按 input_dim narrow）；`make_expert_params_mapping`（`:894`）。
- **精妙之处**：`w13` 是 gate(w1)+up(w3) 沿输出维拼接（列并行），`w2` 是 down（行并行）。weight_loader 对每个专家、每个 shard（w1/w2/w3）做精确的 `narrow` + `copy_`，把 checkpoint 里分散的 `experts.{i}.{gate,up,down}_proj` 搬进堆叠张量的正确切片。EP 时先经 `_map_global_expert_id_to_local_expert_id`（`:638`），非本卡（-1）直接 return 跳过加载。

### T14 · 量化激活在 grouped GEMM 前现场做（moe_kernel_prepare_input）
- **代码**：`fused_moe.py:1177` `moe_kernel_prepare_input`；在两段 GEMM 前各调一次（`:1344`、`:1390`）。
- **精妙之处**：fp8/int8 的**激活量化是动态的**（per-token / per-block / per-channel），必须在每段 GEMM 前对当前 chunk 的激活现算 scale（`scaled_fp8_quant` / `per_token_group_quant_*`）。第一段量化 `hidden`，第二段量化 silu 后的 cache2。权重 scale 则是加载期定好的。把"激活量化"塞进 chunk 循环内，是动态量化与分块执行结合的必然。

### T15 · 小 batch 时收缩 EM，跳过必然为空的块
- **代码**：`fused_moe.py:494-500`。
- **精妙之处**：`if A.shape[0] < BLOCK_SIZE_M:` 时，`EM = min(sorted_token_ids.shape[0], M*top_k*BLOCK_SIZE_M)`。decode 阶段 batch 很小（每步 1~几个 token），但 `sorted_token_ids` 是按"最坏 padding"分配的、很长。这里假设"每 token 的 topk 专家互异"，把 grid 的 M 维收缩到实际可能非空的范围，避免 launch 大量空块——decode 高频路径的实打实优化。

---

## 6. 关键边界情况与容错

| 情况 | 代码位置 | 行为 |
|---|---|---|
| 非本卡专家（EP） | `fused_moe.py:357-365` | `off_experts==-1` → 输出写 0 返回，由后续 all-reduce/combine 合并 |
| padding token | `fused_moe.py:354-355` | `token_mask = offs_token < num_valid_tokens`，load `other=0.0`，写回被 mask |
| 超出 padding 总长的块 | `fused_moe.py:349-351` | `pid_m*BLOCK_SIZE_M >= num_tokens_post_padded` → 直接 return |
| token 数过多 | `fused_moe.py:1275`、`:1320` | 按 `VLLM_FUSED_MOE_CHUNK_SIZE` 分块执行（绕 issue #5938） |
| 最后一个不满 chunk | `fused_moe.py:1330-1339` | 调整 cache 形状，**cache2 须 `* topk`**（#13693）；重查 config |
| 专家数 ≥ 224 / =256 | `moe_align_block_size.py:217-239` | 切到 SGLang global-mem 或 triton 对齐 kernel（#12850） |
| DeepGEMM 不支持 EP | `deep_gemm_moe.py:38-39`、`:172` | `_valid_deep_gemm` 见 `expert_map is not None` 返回 False；apply 断言 `expert_map is None` |
| DP>1 | `layer.py:851-885` | `naive_multicast` 聚 token → 计算 → `all_reduce` → 切回本 rank 区间 |
| grouped_topk 非 softmax 但非分组 | `layer.py:489-491` | `scoring_func != "softmax" and not use_grouped_topk` → ValueError |
| fp16 + shared expert 溢出 | `deepseek_v2.py:170-173` | 改用 `shared_output * (1/routed_scaling_factor)` 防溢出 |
| EP 专家不整除 | `layer.py:368-373` | 余数专家全给最后一个 rank（`local_num_experts` 重算） |

---

## 7. 一图速查：调用链主干

```
[模型层] deepseek_v2.py:157  router_logits = gate(hidden)
         deepseek_v2.py:154  shared_output = shared_experts(hidden)        (并排, 始终激活)
                │
[FusedMoE] forward :839 ─(dp==1)─► forward_impl :847
                │        └(dp>1)─► torch.ops.vllm.moe_forward :933 (torch.compile 友好)
                ▼
         quant_method.apply  (UnquantizedFusedMoEMethod.forward_cuda layer.py:181 / Fp8MoEMethod.apply fp8.py:766 / ...)
                │
         ① select_experts layer.py:780 ───► (M, top_k) topk_weights, topk_ids
                │   fused_topk:890 softmax / grouped_topk:890 sigmoid+bias / custom_routing
                ▼
         fused_experts fused_moe.py:1111 ─(allow_deep_gemm)─► deep_gemm_moe_fp8
                │
         fused_experts_impl :1230   ┌─ 每 chunk ───────────────────────────────────────┐
                │                   │ ② moe_align_block_size :151                        │
                │                   │    → sorted_token_ids / expert_ids(EP:expert_map→-1)│
                │                   │ ③ invoke_fused_moe_kernel :466 → fused_moe_kernel :256│
                │                   │      cache1 = grouped_GEMM(hidden, w13[expert_ids]) │
                │                   │ ④ silu_and_mul → cache2                            │
                │                   │ ⑤ invoke_fused_moe_kernel → cache3 = GEMM(cache2,w2)│
                │                   │      * topk_weights (门控权重乘在此, fp32 累加)      │
                │                   │    ops.moe_sum(cache3 over top_k) → out             │
                │                   └────────────────────────────────────────────────────┘
                ▼  routed_out
[模型层] deepseek_v2.py:159-177  routed_out * routed_scaling + shared_output
                                 → tensor_model_parallel_all_reduce (TP/EP 合并, 只一次)
                ▼
            final_hidden_states
```
