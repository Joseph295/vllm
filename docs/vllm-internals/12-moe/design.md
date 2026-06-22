# 模块 12 · MoE 混合专家 —— 设计文档

> 范围：vLLM 如何把一个 **MoE（Mixture-of-Experts，混合专家）层** 高效地跑在 GPU 上 —— 从 router 门控（选哪几个专家）、把变长的"每专家 token 数"对齐成可批量的 **grouped GEMM**，到专家并行（EP）下 `expert_map` 把全局专家映射到本卡、shared experts 与量化 MoE。
> 架构：**V1 为主**。MoE 与 **专家并行（EP）** 强相关，但二者分工不同：[模块 03](../03-distributed-parallel/design.md) 讲"EP 作为一种并行维度如何切分 world、和 TP/PP/DP 怎么组合"；**本模块专讲 MoE 机制本身**（路由、grouped GEMM、weight loading、量化后端）。kernel 的 GPU 视角（IPC、warp 归约等）指回 [模块 11](../11-gpu-kernels-memory/impl.md)，量化数值格式指回 [模块 08](../08-quantization/design.md)。
> 阅读前提：理解 TP 的列并行/行并行配对（[模块 03](../03-distributed-parallel/design.md) §3.1），因为 MoE 的 `w13`（gate_up）/`w2`（down）正是列并行/行并行的专家版。

---

## 1. 这个模块解决什么问题

MoE 的核心命题是 **稀疏激活（sparse activation）**：让模型 **总参数量很大**（容量高、知识多），但 **每个 token 只激活其中一小部分参数**（计算量低）。

以 Mixtral-8x7B 为例：它有 8 个专家、每个 token 只走 top-2，于是"参数像 47B 大模型，FLOPs 像 13B 小模型"。DeepSeek-V3 更极端：256 个路由专家 + 1 个共享专家，每 token 只激活 8 个路由专家。这把"参数规模"和"单 token 计算量"解耦开了。

但这个收益不是免费的，落到推理引擎上会撞出一串硬问题：

1. **路由是动态、数据相关的**：每个 token 去哪几个专家由 router（一个小 Linear + softmax/sigmoid）当场算出来，编译期不知道。于是每个专家实际要处理的 token 数 **运行时才确定、且各不相同**（负载不均）。
2. **变长分组 vs. 规整 GEMM 的矛盾**：GPU 的 GEMM kernel 要批量、要规整的 tile，但"专家 A 有 37 个 token、专家 B 有 5 个 token"这种 ragged（锯齿）形状没法直接喂给一个高效 kernel。
3. **专家多到单卡放不下**：256 个专家 × 每专家两个大矩阵，显存放不下 → 必须把 **整个专家** 分散到不同卡（EP），且每卡的融合 kernel 要"只算自己的专家、跳过别人的"。
4. **量化的组合爆炸**：fp8 / int8 / int4(GPTQ/AWQ) / 块量化各有各的 scale 布局，都得塞进同一个专家 GEMM 框架，还要选 triton / CUTLASS / Marlin / DeepGEMM 哪个后端。

vLLM 的 MoE 设计，本质就是围绕"**把数据相关的稀疏路由，变成 GPU 能批量吃的稠密 grouped GEMM**"来组织的，核心一招就是 `moe_align_block_size`（§3.2）。

---

## 2. 设计目标与约束

**目标**

- **把稀疏路由稠密化为可批量的 grouped GEMM**：不为每个专家单独 launch 一个小 GEMM（kernel launch 开销 + 利用率低），而是把所有 token 按专家排序、对齐到 `block_size` 的整数倍，让 **一个融合 triton kernel** 用 `expert_ids[block]` 切换权重矩阵，一次算完所有专家（§3.2、§3.3）。
- **两个专家投影融合 + 中间 buffer 复用**：`w13`（gate+up 合并成一个张量）与 `w2`（down）两段 GEMM 之间只过一次 `silu_and_mul`，中间 cache 显存尽量复用（§3.3、impl T6）。
- **EP 对融合 kernel 透明**：用 `expert_map`（全局专家 id → 本地专家 id，不在本卡的映射为 -1）让同一个 kernel 在 EP 下"自动跳过不属于自己的专家、对应输出写 0"，无需为 EP 写第二套 kernel（§3.4）。
- **路由策略可插拔**：softmax-topk（Mixtral）、grouped-topk + sigmoid + 修正偏置（DeepSeek-V3）、`custom_routing_function`（Llama4/Phi）共用一套 `select_experts` 入口（§3.1）。
- **量化后端可择优**：同一个 `FusedMoE` 层按量化方案选 triton / CUTLASS grouped GEMM / Marlin / DeepGEMM（§3.6）。

**约束 / 当前实现边界**

- **EP 下专家必须能整除**：`determine_expert_map` 把专家平均分给各 rank，余数全给最后一个 rank（`layer.py:358-373`）。
- **EP 用 `tp_size * dp_size` 作为并行度**：开 EP 时把 TP 折叠（`tp_size=1`），`ep_size = tp_size * dp_size`（`layer.py:451-462`）——这是 #16161 修过的易错点（§7.2）。
- **部分后端不支持 EP**：DeepGEMM 路径硬性 `expert_map is None`（`deep_gemm_moe.py:38-39`、`:172`）；CUTLASS 路径靠自己算的 `a_map`/`c_map` 路由、没有 `expert_map` 参数。
- **grouped_topk 走 `torch.compile`**：DeepSeek 的分组路由用 `@torch.compile(dynamic=True)`（`fused_moe.py:889`），逻辑较重。

---

## 3. 核心设计思想

### 3.0 一个 MoE 层在 vLLM 里长什么样

`FusedMoE`（`layer.py:377`）是 nn.Module，模型里这样用（以 Mixtral 为例，`mixtral.py:88-106`）：

```python
self.experts = FusedMoE(num_experts=8, top_k=2, hidden_size=..., intermediate_size=...,
                        reduce_results=True, renormalize=True)
# forward:
router_logits, _ = self.gate(hidden_states)          # 小 Linear：hidden → num_experts 个分数
final = self.experts(hidden_states, router_logits)   # 路由 + 专家 GEMM + combine
```

权重不是 8 个独立的 nn.Linear，而是 **堆叠成两个 3D 张量**（`UnquantizedFusedMoEMethod.create_weights`, `layer.py:79-100`）：

- `w13_weight`：`(num_experts, 2 * intermediate_size_per_partition, hidden_size)` —— gate_proj(w1) 与 up_proj(w3) **沿输出维拼在一起**（列并行的专家版），所以是 `2 *`。
- `w2_weight`：`(num_experts, hidden_size, intermediate_size_per_partition)` —— down_proj（行并行的专家版）。

把 8 个专家堆成一个 `(E, N, K)` 张量，是后面 grouped GEMM 用 `expert_id` 在第 0 维索引权重的前提。

### 3.1 路由：top-k 门控的三条路径

`FusedMoE.select_experts`（`layer.py:780`）是统一入口，输出 `(topk_weights, topk_ids)`——每个 token 选中的 top-k 专家 id 和对应的门控权重。三条路径：

**(a) `fused_topk`（Mixtral 风格，softmax 门控）** `fused_moe.py:853`
对 router logits 做 softmax → 取 top-k → 可选 renormalize（让选中的 k 个权重和为 1）。实际 topk+softmax 融合在一个 CUDA kernel 里（`ops.topk_softmax`，`csrc/moe/topk_softmax_kernels.cu`，改编自 TensorRT-LLM）。renormalize 在 kernel 外做（`fused_moe.py:841-842`）。

**(b) `grouped_topk`（DeepSeek-V2/V3 风格）** `fused_moe.py:890`
DeepSeek 的专家被分成 `num_expert_group` 组，**先选组、再在组内选专家**（减少跨节点 all-to-all 的目标范围）。关键设计点：
- **scoring_func 可为 `sigmoid`**（V3）而非 softmax —— 因为专家数极多时 softmax 会把概率摊得很薄。
- **`e_score_correction_bias`（DeepSeek-V3 的 noaux_tc 负载均衡偏置）**：这是 V3 的精髓。`fused_moe.py:912-935`——**用"加了偏置的分数"来选专家，但用"原始未加偏置的分数"作为路由权重**：
  ```python
  original_scores = scores
  scores = scores + e_score_correction_bias.unsqueeze(0)   # 选择用带偏置分数
  group_scores = scores.view(...).topk(2, dim=-1)[0].sum(dim=-1)  # 每组取 top-2 之和定组分
  ...
  topk_ids = torch.topk(tmp_scores, k=topk, ...)[1]
  topk_weights = original_scores.gather(1, topk_ids)        # 权重用原始分数
  ```
  这个偏置在训练中动态调整，把流量从热门专家推开、实现 **无辅助损失的负载均衡**（auxiliary-loss-free load balancing）。选择/权重分离是为了"调度均衡"不污染"数值权重"。

**(c) `custom_routing_function`（Llama4 / Phi-MoE）** `layer.py:813`
模型自带路由逻辑。如 Llama4 用 `fast_topk` 选专家后对选中分数做 `sigmoid`（`llama4.py:48-56`），并配 `apply_router_weight_on_input=True`（把门控权重提前乘到输入上，见 §3.5）。

### 3.2 为什么需要 `moe_align_block_size` —— 整个模块的"题眼"

路由完，我们手上是 `topk_ids`，形如 `[[2,3,4],[1,2,4],...]`：每个 token 去哪几个专家。问题来了：**专家 A 可能分到 37 个 token，专家 B 只有 5 个**。要让一个高效的 grouped GEMM kernel 批量处理，必须把这堆变长的"每专家 token 列表"**整理成规整的、对齐到 `block_size` 的分块**。

`moe_align_block_size`（`moe_align_block_size.py:151`）做的就是这件事，它产出三个张量：

- **`sorted_token_ids`**：把（展平后的）token **按所属专家排序**，并在每个专家的尾部补 padding，使每个专家的 token 数 **向上对齐到 `block_size` 的整数倍**。padding 位置填一个越界值（`= topk_ids.numel()`），kernel 里用 `token_mask` 忽略。
- **`expert_ids`**：长度 = 总块数，`expert_ids[i]` = 第 i 个 `block_size` 块属于哪个专家。**这就是 grouped GEMM 的"权重选择器"**——kernel 处理第 i 块时，去 `w[expert_ids[i]]` 取权重矩阵。
- **`num_tokens_post_padded`**：padding 后的总 token 数。

docstring（`moe_align_block_size.py:186-200`）给了一个具体例子：`topk_ids=[[2,3,4],[1,2,4],[1,3,4],[1,2,3]]`、`block_size=4`、4 个专家。展平 → 按专家排序 → 每专家补到 4 的倍数 → `sorted_token_ids = [3,6,9,12, 0,4,10,12, 1,7,11,12, 2,5,8,12]`（12 是 padding）。

**为什么这一步是关键**：它把"数据相关的 ragged 分组"一次性转换成"**对 kernel 而言完全规整的 tile 布局**"——每个 `BLOCK_SIZE_M` 行块只属于一个专家、长度对齐，于是同一个 GEMM kernel 可以用 `pid_m → expert_ids[pid_m]` 切换权重，把所有专家的计算合进一次 launch。**这就是 Megablocks（block-sparse MoE）思想在推理侧的体现**：不做 dense 的"每个 token × 每个专家"，而是 block-sparse 地只算实际命中的 (token, expert) 对。

> 设计动机出处：Gale et al., *"MegaBlocks: Efficient Sparse Training with Mixture-of-Experts"* (2022, arXiv:2211.15841) —— 用 block-sparse GEMM 避免 dropless MoE 里的 token 丢弃与 padding 浪费。vLLM 的对齐+分块 triton kernel 与之同源；CUDA 实现 `moe_align_block_size_kernel`（`csrc/moe/moe_align_sum_kernels.cu:25`）改编自 SGLang。

### 3.3 专家 GEMM 如何批量化（grouped GEMM）

对齐之后，专家计算是 **两段 grouped GEMM 夹一个激活**（`fused_experts_impl`, `fused_moe.py:1230`）：

```
对每个 chunk:
  ① sorted_token_ids/expert_ids = moe_align_block_size(topk_ids, BLOCK_SIZE_M, E, expert_map)
  ② intermediate_cache1 = grouped_GEMM(hidden, w13, expert_ids)   # (M*topk, 2*I)
  ③ intermediate_cache2 = silu_and_mul(intermediate_cache1)       # (M*topk, I)
  ④ intermediate_cache3 = grouped_GEMM(cache2, w2, expert_ids)    # (M*topk, hidden)
  ⑤ out = moe_sum(cache3 over topk)                               # 把每 token 的 topk 个专家输出加权求和
```

核心是融合 triton kernel `fused_moe_kernel`（`fused_moe.py:256`）。它的精妙之处在 `pid → block` 的映射（`fused_moe.py:333-374`）：

- 每个 program 负责输出 C 的一个 `[BLOCK_SIZE_M, BLOCK_SIZE_N]` 块。`pid_m` 决定它处理哪个 token 块，`off_experts = expert_ids[pid_m]` 决定 **用哪个专家的权重**（`b_ptr + off_experts * stride_be`，`:373`）。
- **A（激活）的索引玄机**：`a_ptrs = a_ptr + (offs_token // top_k) * stride_am`（`:370`）——因为一个 token 被 top_k 个专家复用，`sorted_token_ids` 里存的是"展平后含 topk 复制"的索引，`// top_k` 还原回原始 token 行。即 **激活不真的复制 top_k 份，只是被多个专家块用不同权重读同一行**。
- **门控权重在哪乘**：`MUL_ROUTED_WEIGHT` 为真时，kernel 末尾 `accumulator *= moe_weight`（`:443-447`）。vLLM 把它放在 **第二段 GEMM（w2）**（`fused_moe.py:1412` 传 `not apply_router_weight_on_input`），第一段不乘——少乘一次、且数值更稳。

grouped GEMM 还用 **L2 友好的块重排**（`GROUP_SIZE_M`，`:336-341`，标准 triton matmul 技巧）提升 cache 命中。

### 3.4 EP 下 `expert_map`：把全局专家映射到本卡

EP（专家并行）把 **整个专家** 分到不同 rank：rank `i` 只持有 `global_num_experts / ep_size` 个专家的完整权重。`FusedMoE.__init__`（`layer.py:451-470`）在开 EP 时：

```python
self.ep_size = self.tp_size * self.dp_size        # EP 并行度 = TP × DP
self.tp_size = 1                                   # EP 模式下把 TP 折叠掉
self.local_num_experts, self.expert_map = determine_expert_map(ep_size, ep_rank, global_num_experts)
```

`determine_expert_map`（`layer.py:332`）产出 `expert_map`：一个长度 `global_num_experts` 的张量，`expert_map[g]` = 全局专家 `g` 在本卡的本地索引，**不在本卡的专家映射为 -1**。

`expert_map` 在两个地方发挥作用：

1. **weight loading**：`weight_loader`（`layer.py:643-649`）先把 checkpoint 里的全局 `expert_id` 映射成本地 id，若为 -1 **直接 return 跳过**——本卡不加载别人的专家权重。
2. **kernel 内跳过**：`moe_align_block_size` 末尾 `expert_ids = expert_map[expert_ids]`（`moe_align_block_size.py:240-241`），把每个块的全局专家 id 重映射成本地 id（不在本卡的块变成 -1）。融合 kernel 检查 `if off_experts == -1: write_zeros_to_output; return`（`fused_moe.py:357-365`）——**不属于本卡的专家，对应输出直接写 0**，再由后续 all-reduce / combine 把各卡结果合起来。

这套设计让 **同一个融合 kernel 既能跑 TP-on-MoE 也能跑 EP**，区别只是 `expert_map` 传不传。注意区分 `global_num_experts`（全局，决定 router 输出维和对齐空间）与 `local_num_experts = global // ep_size`（本地，决定本卡权重张量第 0 维）——二者混用正是 #13784 的 bug（§7.2）。

> EP 作为并行维度如何与 TP/PP/DP 组合、dispatch/combine 的 all-to-all 通信，见 [模块 03](../03-distributed-parallel/design.md) §3.3。本模块只关注"`expert_map` 如何让本地 kernel 跳过非本卡专家"。

### 3.5 Shared experts（共享专家）

DeepSeek-V2/V3、Llama4 等在路由专家之外，还有 **始终对所有 token 激活的共享专家**（一个普通 MLP），用来承载"通用知识"、让路由专家专注于专精化。

vLLM **不把 shared expert 塞进 `FusedMoE`**，而是让模型层并排放一个普通 MLP，forward 里把两者的输出相加（`deepseek_v2.py:154-177`）：

```python
shared_output = self.shared_experts(hidden_states)   # 普通 MLP，始终激活
final = self.experts(hidden_states, router_logits)   # FusedMoE，稀疏激活
final = final * routed_scaling_factor + shared_output
if self.tp_size > 1:
    final = tensor_model_parallel_all_reduce(final)  # 合并后只 all-reduce 一次
```

关键设计：**`FusedMoE` 和 shared MLP 都设 `reduce_results=False`**（`deepseek_v2.py:124-149`），把 all-reduce **推迟到两者相加之后只做一次**——省一次集合通信。`apply_router_weight_on_input`（Llama4 用）则是把门控权重提前乘到输入、让 shared/routed 输出能直接相加的相关开关。

### 3.6 量化 MoE：一个层，多个后端

量化时 `FusedMoE` 从 `quant_config.get_quant_method()` 拿到一个 `FusedMoEMethodBase` 子类（`layer.py:498-521`），它的 `create_weights`/`apply` 决定权重布局和走哪个后端：

| 量化方案 | 方法类（file:line） | 后端 | 关键点 |
|---|---|---|---|
| 无量化 | `UnquantizedFusedMoEMethod` (`layer.py:76`) | triton `fused_experts` | bf16/fp16 |
| fp8（per-tensor / block） | `Fp8MoEMethod` (`fp8.py:424`) | triton `fused_experts(use_fp8_w8a8=True)`，可 `allow_deep_gemm` 转 DeepGEMM | block_quant 时用 `*_weight_scale_inv` + dynamic 激活 |
| fp8 + CUTLASS | `CompressedTensorsW8A8Fp8MoECutlassMethod` (`compressed_tensors_moe.py:305`) | `cutlass_moe_fp8` | CUTLASS grouped GEMM，自算 `a_map/c_map` |
| int8 w8a8 | `CompressedTensorsW8A8Fp8MoEMethod` (`:79`) | `fused_experts(use_int8_w8a8, per_channel_quant)` | per-channel 权重 + per-token 激活 |
| int4/int8 wna16（GPTQ/AWQ/CT） | `GPTQMarlinMoEMethod`/`AWQMoEMethod`/`...WNA16MarlinMoEMethod` | `torch.ops.vllm.fused_marlin_moe` | GPTQ 用 `g_idx`+`sort_indices`(act-order)；AWQ 用 zero-points |
| int4/int8 wna16（非 marlin） | `CompressedTensorsWNA16MoEMethod` (`:871`) | `fused_experts(use_int4_w4a16 / use_int8_w8a16, block_shape=[0, group_size])` | moe_wna16 triton/cuda kernel |

后端选择本质是在"硬件支持 / 数值格式 / 矩阵形状"之间择优。各量化数值格式（scale 布局、零点、act-order）见 [模块 08](../08-quantization/design.md)；kernel 的 GPU 视角见 [模块 11](../11-gpu-kernels-memory/impl.md)。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `FusedMoE` | `layer.py:377` | MoE 层 nn.Module。持 `global_num_experts`/`local_num_experts`/`expert_map`/`top_k`/quant_method；EP↔TP 切换 |
| `FusedMoEMethodBase` | `layer.py:45` | 量化方法抽象基类，定义 `create_weights` + `apply`；各量化派生 |
| `w13_weight` | `layer.py:83` | `(E, 2*I_per_partition, hidden)` 堆叠的 gate+up 专家权重（列并行版） |
| `w2_weight` | `layer.py:93` | `(E, hidden, I_per_partition)` 堆叠的 down 专家权重（行并行版） |
| `expert_map` | `layer.py:332` `determine_expert_map` | `(global_num_experts,)`，全局→本地专家 id，非本卡为 -1 |
| `topk_weights` / `topk_ids` | `select_experts` (`layer.py:780`) | `(M, top_k)`：每 token 选中的门控权重 / 专家 id |
| `e_score_correction_bias` | 传入 `grouped_topk` (`fused_moe.py:912`) | DeepSeek-V3 负载均衡偏置（选择用带偏分数、权重用原始分数） |
| `sorted_token_ids` | `moe_align_block_size` (`moe_align_block_size.py:151`) | 按专家排序+padding 的 token 索引（grouped GEMM 的行布局） |
| `expert_ids` | 同上 | 每个 `block_size` 块属于哪个专家（grouped GEMM 的权重选择器） |
| `num_tokens_post_padded` | 同上 | padding 后总 token 数 |
| `intermediate_cache1/2/3` | `fused_experts_impl` (`fused_moe.py:1295-1304`) | 两段 GEMM 的中间 buffer，cache1/3 复用同一块显存 |

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| 专家权重堆叠成 `(E,N,K)` 3D 张量 | grouped GEMM 可用 `expert_id` 索引权重；weight loading 统一 | 权重 loader 复杂（per-expert/per-shard narrow，`layer.py:583-636`） |
| `moe_align_block_size` 排序+padding | 把 ragged 分组变规整 tile，所有专家合一次 kernel launch | padding 带来计算浪费（专家越不均、`block_size` 越大，浪费越多）；多一趟排序 kernel |
| 融合 triton kernel + `expert_map` 跳过 | 一套 kernel 同时支持 TP-on-MoE 和 EP | 非本卡专家的块仍占 grid（写 0 后返回），EP 极度不均时有空转 |
| `w13` 合并 + 中间 cache 复用 | 省一次 kernel、省显存 | cache1/3 复用要求"用 cache3 时 cache1 已不需要"，chunk 边界易错（#13693） |
| 门控权重乘在第二段 GEMM | 少乘一次、数值更稳 | `apply_router_weight_on_input` 时语义翻转，需小心 |
| shared expert 放模型层、`reduce_results=False` | 合并后只 all-reduce 一次 | 模型层要手写"相加 + 缩放 + all-reduce"，每个 MoE 模型重复 |
| grouped_topk 用 `@torch.compile` | 复杂分组逻辑由编译器优化 | 编译开销；动态 shape 下偶有兼容问题 |
| 多量化后端（triton/CUTLASS/Marlin/DeepGEMM） | 各形状/硬件择优 | 维护面大；后端间能力不一（如 DeepGEMM 不支持 EP） |

---

## 6. 设计动机出处（小结）

- **稀疏激活 / top-k 路由 / 专家容量**：Shazeer et al., *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"* (2017, arXiv:1701.06538)。
- **大规模 MoE + 专家并行（EP）+ 组路由**：Lepikhin et al., *"GShard"* (2020, arXiv:2006.16668)。
- **单专家路由 + 简化稳定训练**：Fedus et al., *"Switch Transformer"* (2021, arXiv:2101.03961)。
- **稠密-MoE 推理标杆**：Jiang et al., *"Mixtral of Experts"* (2024, arXiv:2401.04088) —— 8 专家 top-2，vLLM 的 `fused_topk` 路径直接服务它。
- **细粒度专家 + 共享专家 + 无辅助损失负载均衡（`e_score_correction_bias`）**：DeepSeek-AI, *"DeepSeekMoE"* (2024, arXiv:2401.06066) 与 *"DeepSeek-V3 Technical Report"* (2024, arXiv:2412.19437)。
- **block-sparse grouped GEMM（`moe_align_block_size` 的思想源头）**：Gale et al., *"MegaBlocks"* (2022, arXiv:2211.15841)。
- **topk_softmax CUDA kernel** 改编自 NVIDIA TensorRT-LLM；**moe_align CUDA kernel** 与 triton 版改编自 SGLang（见文件头注释）。

---

## 7. 设计背后的考量与历史教训

> 本节比 §5 更进一步：挖掘**为什么是现在这个设计**（否决了什么、被什么倒逼、随哪些重构演进），以及从真实 bugfix 读出的教训。所有 PR 号均来自本仓库 `git log -- vllm/model_executor/layers/fused_moe/ csrc/moe/` 的真实记录。

### 7.1 设计背后的考量（深化"为什么"）

1. **为什么是"排序+padding 成 block"而非"每专家一个独立 GEMM"或"dense one-hot"**。最朴素的两条路都被否决：(a) 为每个专家单独 launch 一个小 GEMM——专家多时 kernel launch 开销爆炸、且每个 GEMM 的 M 太小利用率极低；(b) 把路由展开成 `(token, expert)` 的 dense one-hot 再做大矩阵乘——计算量是稀疏激活的 `num_experts/top_k` 倍，等于把 MoE 的省算优势全抵消。vLLM 选 MegaBlocks 式的 **block-sparse**：`moe_align_block_size` 把命中的 (token,expert) 对按专家聚成对齐块，一个 kernel 用 `expert_ids[block]` 切权重，**只算实际命中、又只 launch 一次**。padding 的浪费（每专家最多补 `block_size-1` 个假 token）是为"规整 tile"付的小税。

2. **`expert_map` 用 -1 哨兵 + kernel 内写零，而不是为 EP 写第二套 kernel**。EP 下每卡只有部分专家，最直接的实现是给 EP 单独写 dispatch/gather kernel。vLLM 反而复用同一个融合 kernel：`moe_align` 末尾把全局专家 id 经 `expert_map` 重映射，非本卡的块变 -1，kernel 见 -1 就 `write_zeros_to_output` 后返回（`fused_moe.py:357-365`）。代价是这些"别人的块"仍占 grid、空转一下；收益是 **EP 和 TP-on-MoE 共用一份 kernel 与一份 host 代码**，`expert_map` 传不传就是唯一区别。这是"用一点 GPU 空转换巨大的代码统一"的取舍。

3. **DeepSeek-V3 的"选择分数 / 权重分数"为何要分离**。`e_score_correction_bias` 是训练时为负载均衡动态调整的偏置。如果直接用"加了偏置的分数"当路由权重，会把"调度均衡的扰动"混进"专家输出的加权系数"里、损害数值。vLLM 严格分离：**带偏置分数只用于 argmax 选专家，原始分数 gather 出来当权重**（`fused_moe.py:913-935`）。这让"无辅助损失负载均衡"这一训练侧机制，在推理侧被精确还原而不串味。

4. **门控权重为什么乘在第二段（w2）GEMM 而不是第一段或路由后**。可以在 router 后就把 `topk_weights` 乘到激活上，或在第一段 GEMM 后乘。vLLM 选在 **w2 GEMM 末尾** 乘（`fused_moe.py:1412` 传 `not apply_router_weight_on_input`）：此时累加器还是 fp32，乘完再 cast，数值最稳；且只乘一次。`apply_router_weight_on_input`（Llama4）是反过来的特例——top_k 路由权重提前乘到输入，让 shared/routed 输出能直接相加，于是两段 GEMM 都不再乘。这条开关的存在本身说明"权重乘在哪"是被不同模型结构倒逼出的可配置点。

5. **shared expert 为何留在模型层、而非并进 `FusedMoE`**。共享专家对所有 token 稠密激活、且常需要和路由输出做模型特定的缩放（`routed_scaling_factor`、fp16 溢出特判，`deepseek_v2.py:167-174`）。把它塞进 `FusedMoE` 会让这个本应"纯稀疏 grouped GEMM"的层背上一堆模型分支。于是设计选择：`FusedMoE` 只管路由专家，shared MLP 并排放、`reduce_results=False`、**两者相加后只 all-reduce 一次**。代价是每个 MoE 模型都要手写这段"相加+缩放+reduce"，但换来 `FusedMoE` 的纯粹。

6. **演进脉络：从 Mixtral 的 softmax-topk 到 DeepSeek/Llama4 的可插拔路由**。最早 MoE 只服务 Mixtral（softmax + top-2，`fused_topk`）。DeepSeek-V2 带来 grouped-topk（#12583 加 EP），V3 带来 sigmoid 打分 + `e_score_correction_bias`（#13474 修了组分计算），Llama4 带来 `custom_routing_function` + `apply_router_weight_on_input`（#16113）。`select_experts` 这个统一入口（三分支）正是被这条"路由策略越来越花"的演进逼出来的——把"路由"从"专家 GEMM"里彻底解耦，新模型加一条路由分支即可，grouped GEMM 一行不改。

### 7.2 重要 bug 修复（真实、精选）

> 下列 PR 均可在本仓库 `git log -- vllm/model_executor/layers/fused_moe/ csrc/moe/` 查到；每条点出它**暴露的设计点 / 教训**。

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| **#13784** `[Bugfix][Quantization] Fix FP8 + EP` | EP 下 `num_experts` 一个字段同时被当"全局专家数"和"本地专家数"用，FP8+EP 时权重切分/scale 错位（改动横跨 layer.py + fp8/gptq/awq/compressed/quark 五个量化方法） | 凡被并行切分的"数量"，**全局值 `global_num_experts` 与本地值 `local_num_experts = global // ep_size` 必须命名上就分开**。这也是 §3.4 `expert_map` 与 `global_num_experts` 并存的根本原因 |
| **#16161** `fix use-ep bug to enable ep by dp/tp size > 1` | `use_ep` 判定写成 `enable_expert_parallel and self.tp_size > 1`，导致**纯 DP（tp_size==1）下 EP 开不起来** | EP 的并行度来自 `tp_size * dp_size` 而非只看 TP（`layer.py:439-440/456`）。教训：MoE 的 EP/DP/TP 三维耦合，判定条件不能只盯一维 |
| **#13474** `Fix deepseekv3 grouped topk error` | grouped_topk 在有 `e_score_correction_bias` 时仍用 `max` 算组分、且 mask 填 `0.0`；V3 应当每组取 **top-2 之和** 定组分、mask 填 **-inf** | 路由数值细节（组分聚合方式、屏蔽用 0 还是 -inf）直接决定选对哪些专家。`masked_fill(..., 0.0)` 在 sigmoid 分数下会让被屏蔽专家仍可能被选中——必须 -inf（`fused_moe.py:929`）。教训：稀疏路由的数值边界比想象中脆 |
| **#13693** `Illegal memory access for MoE On H20` | chunked 执行的最后一个 chunk，`intermediate_cache2` 只截到 `tokens_in_chunk` 而非 `tokens_in_chunk * topk`，越界非法访存 | cache2 形状是 `(M*topk, N//2)`（每 token 复制 topk 份），cache1/3 是 `(M, topk, ·)`。三个 cache 的"是否含 topk 维"不一致，chunk 切片时极易切错（`fused_moe.py:1336-1337`）。教训：中间 buffer 复用省了显存，但把"形状一致性"变成了正确性负担 |
| **#13625** `Optimize moe intermediate_cache usage` | 中间 cache 显存占用过高 | 把 cache1 与 cache3 **复用同一块 `cache13` 显存**（"用 cache3 时 cache1 已算完"，`fused_moe.py:1293-1299`）。这是性能优化，但正是它制造了 #13693 那类 chunk 切片陷阱——优化与脆弱性同源 |
| **#12850** `Optimize moe_align_block_size for deepseek_v3` | 原 moe_align CUDA kernel 在 V3 的 256 专家下共享内存放不下/慢 | 专家数从 Mixtral 的 8 跳到 V3 的 256，原本"塞进 shared memory"的对齐 kernel 失效，催生了 `num_experts>=224` 走 SGLang 的 global-mem / triton 变体（`moe_align_block_size.py:217-239`、`csrc/moe/moe_align_sum_kernels.cu:114`）。教训：MoE 的 kernel 假设（专家数级别）会被新模型直接打破 |
| **#13772** `Fix precommit fail in fused_moe intermediate_cache2 chunking` | 紧随 #13693 的 cache2 chunk 修复的收尾 | 同一处 chunk 边界逻辑反复出问题，印证上面"cache 复用 → chunk 一致性"是真实的长期维护痛点 |

### 7.3 一图速览：MoE 层数据流

```
                router_logits = gate(hidden)        hidden: (M, hidden_size)
                        │
   ┌────────────────────┴───────────── select_experts (layer.py:780) ───────────────┐
   │  fused_topk(softmax)   /   grouped_topk(sigmoid + e_score_correction_bias)      │
   │                        /   custom_routing_function (Llama4/Phi)                 │
   └────────────────────┬───────────────────────────────────────────────────────────┘
                        ▼  topk_weights, topk_ids : (M, top_k)
              ┌──── moe_align_block_size (moe_align_block_size.py:151) ────┐
              │  按专家排序 + padding 到 block_size 整数倍                  │
              │  → sorted_token_ids   (grouped GEMM 行布局)                │
              │  → expert_ids[block]  (权重选择器；EP: expert_map 重映射→-1)│
              └────────────────────────┬───────────────────────────────────┘
                                       ▼
  ┌───────────────────── fused_experts_impl (fused_moe.py:1230) ─────────────────────┐
  │  ① cache1 = grouped_GEMM(hidden, w13[expert_ids])   # gate+up,  off_experts==-1→写0│
  │  ② cache2 = silu_and_mul(cache1)                                                   │
  │  ③ cache3 = grouped_GEMM(cache2, w2[expert_ids]) * topk_weights  # down + 门控权重 │
  │  ④ out    = moe_sum(cache3 over top_k)            # 每 token 的 topk 专家输出求和   │
  └────────────────────────┬──────────────────────────────────────────────────────────┘
                           ▼  routed_out : (M, hidden)
        ( + shared_experts(hidden) )  → ×routed_scaling → tensor_model_parallel_all_reduce
                           ▼
                     final_hidden_states

  EP 时：本卡只算自己的专家（expert_map），其余写 0；combine 由 all-reduce/all-to-all 完成（见模块 03）
```

> 逐行调用链、`file:line` 对照、实现精妙之处与边界情况，见同目录 [`impl.md`](impl.md)。
