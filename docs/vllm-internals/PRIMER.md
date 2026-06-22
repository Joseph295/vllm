# 背景与全景：给普通程序员的前置导读

> **这份文档给谁看**：你会编程、大致知道 Transformer / attention 是什么，但**没做过"推理服务（inference serving）"**。读完这一篇，你会建立起一套心智模型——再去读 00~15 各模块时就不会被术语劝退。
>
> 阅读顺序建议：**先读这篇 → 再看 [README](README.md) 的端到端全景图 → 然后挑模块深入**。

---

## 第一部分：LLM 推理服务 101（先建立心智模型）

这一部分不涉及 vLLM 代码，只讲"为什么 LLM 服务这么难做"。后面所有模块的设计，根子都在这里。

### 1.1 一次推理 = "循环吐 token"

LLM 一次前向（forward）**只产出一个 token**：它输出"下一个 token 的概率分布"，采样得到 1 个 token。然后把这个新 token 接到输入末尾，再 forward 一次，再吐 1 个……直到遇到结束符（EOS）或达到长度上限。这叫**自回归（autoregressive）生成**。

> 关键推论：**生成 N 个 token = N 次串行 forward**。这是后面一切"慢"的来源，也是投机解码等加速手段要对付的对象。

### 1.2 prefill 与 decode：两个性质截然不同的阶段

把一次请求的生成拆成两段，它们的计算特征完全相反：

| 阶段 | 在干什么 | 计算特征 | 形象比喻 |
|---|---|---|---|
| **prefill**（预填充） | 第一次 forward，一次性处理**整个 prompt**（比如 1000 个 token） | 这 1000 个 token 能**并行**算完 → **计算密集（compute-bound）**，GPU 算力打满 | 一锅端，吃算力 |
| **decode**（解码） | 之后每次只新增 **1 个** token，但要 attention 到前面所有 token | 一次只算 1 个 token 的"瘦长"运算 → **访存密集（memory-bound）**，显存带宽打满、算力闲置 | 挤牙膏，吃带宽 |

> **一句话记住：prefill 吃算力，decode 吃显存带宽。** vLLM 里无数设计都源于这个区别——**chunked prefill**（把一条长 prompt 的 prefill **切成多步**来算，别让它一次占满整步算力、把其他请求的 decode 卡住；详见 §2.4 与 [模块 01](01-scheduler-batching/design.md)）、**PD 分离**（干脆把 prefill 和 decode 两阶段拆到不同机器上跑，互不干扰；详见 [模块 04](04-pd-disaggregation/design.md)）、**投机解码**（让显存受限的 decode 一步多吐几个 token；详见 [模块 05](05-speculative-decoding/design.md)）都在对付它。

### 1.3 KV cache：它为什么存在、为什么是瓶颈

attention 的本质：当前 token 要和**前面所有 token** 的 K（key）、V（value）做运算。

- 如果每生成一步都重算前面所有 token 的 K/V，就是 O(N²) 的重复劳动。
- 所以把每个 token 算过的 K、V **存下来复用**——这就是 **KV cache**。

代价：KV cache 的大小 ≈ `序列长度 × 层数 × KV头数 × head维度 × 2(K和V) × 字节数`，**随序列长度线性增长**。一个几千 token 的请求就可能占几 GB 显存。同时要服务很多并发请求时，**KV cache 而不是模型权重，才是显存的头号消耗者**。

> 于是"怎么省 KV 显存、怎么高效管理这块缓存"成了 vLLM 的**核心战场**：[模块 02](02-paged-attention-kvcache/design.md)（PagedAttention 分页 + 前缀复用）、[模块 08](08-quantization/design.md)（量化 KV）、[模块 13](13-mla/design.md)（MLA 把 KV 压成低秩）全在解决它。

### 1.4 为什么 decode 是 memory-bound（性能直觉）

decode 每一步：要把**整个模型的权重 + 这条请求的全部 KV** 从显存读到计算单元，**只为算出 1 个 token**。算得少、搬得多 → GPU 算力大量闲置，瓶颈在"搬数据"。

由此，给 decode 提速只有两条路：
1. **一次多算几个 token**（摊薄"搬一次权重"的成本）：把多条请求塞进一个 batch 一起算（continuous batching）、或一步验证多个候选 token（投机解码）。
2. **少搬点数据**：量化（权重/KV 用更少字节）、MLA（把 KV 压小）。

### 1.5 服务的两个指标：吞吐 vs 延迟

做"服务"就要同时盯两个互相拉扯的指标：

- **吞吐（throughput）**：每秒能处理多少 token / 多少请求 → 决定**成本**（同样的卡服务更多用户）。
- **延迟（latency）**：**TTFT**（首 token 多久到，受 prefill 影响）、**TPOT**（后续每个 token 的间隔，受 decode 影响）→ 决定**体验**。

**batching（多请求一起算）能大幅提吞吐**，因为"搬一次权重，服务 batch 里所有请求"。但朴素 batching 有个老问题：必须等一整批里**最慢的那条**算完才能开始下一批，短请求被长请求拖死。

> vLLM 的招牌 **continuous batching（连续批处理）**：不等整批结束——**谁生成完谁就退出、空出的位置立刻让新请求补进来**。这是 vLLM 高吞吐的根基，见 [模块 01](01-scheduler-batching/design.md)。

### 1.6 把以上串起来：vLLM 要解决的问题

> **在有限的显存里，尽量多塞并发请求、让 GPU 别闲着、同时兼顾首 token 延迟和成本。**

后面 16 个模块，全是在这句话的不同侧面上做文章。带着这个总目标去读，就不会迷路。

---

## 第二部分：代码全景地图（每个模块/文件/类干什么 + 区别与联系）

### 2.1 两条 track：先分清"骨架"和"血肉"

| Track | 模块 | 是什么 | 特点 |
|---|---|---|---|
| **引擎基础设施** | 00–11 | 与具体模型无关的"服务骨架"：收请求、调度、管显存、跑分布式、出 token、GPU kernel | 横向、通用 |
| **模型架构技术** | 12–15 | 具体模型/注意力变体的实现：MoE、MLA、注意力变体、Mamba、位置编码 | 垂直、随模型而异 |

读的时候先问自己："我现在看的是**通用骨架**，还是**某类模型特有的技术**？"——这决定了它在整张图里的位置。

### 2.2 一个请求穿过哪些东西（30 秒速览）

```
客户端 → [前端] tokenize 文本(00,10) → 跨进程进 EngineCore(00)
       → [调度] 决定这步算哪些请求的哪些 token(01；受 KV 02 / LoRA 09 / 多模态 10 / 投机 05 约束)
       → [执行] Executor 下发到 Worker(03) → GPU 跑模型 forward
                 (编译 07 · 量化 08 · LoRA 09 · attention kernel 02/11/13/14 · MoE 12)
       → [采样] 出 token(06；投机 05) → 回填检测结束(01)
       → [前端] detokenize 成文本(00,06) → 流式返回客户端
```

> 这张图的详细版见 [README](README.md) 的「端到端全景图」。

### 2.3 关键类速查（最容易混的那几个，先认人）

读 vLLM 代码最大的障碍是"一堆名字不知道谁是谁"。先记住这张表：

| 类 / 概念 | 在哪 | 一句话职责 | 容易和谁混 |
|---|---|---|---|
| `AsyncLLM` / `LLMEngine` | `v1/engine/` | **前端引擎**：收请求、把结果流式吐回 | `EngineCore`（前者在 API 进程，后者在后台进程） |
| `EngineCore` | `v1/engine/core.py` | **后端引擎**：独立进程里跑 `schedule→execute→update` 主循环 | `LLMEngine`（见上） |
| `Processor` / `OutputProcessor` | `v1/engine/` | 文本↔token 的转换：前者 tokenize，后者 detokenize | 互为反向 |
| `Scheduler` | `v1/core/sched/scheduler.py` | **决定每一步算哪些请求的哪些 token** | `Executor`（调度 vs 执行，一个动脑一个动手） |
| `KVCacheManager` / `BlockPool` | `v1/core/` | 管 KV 显存**块**的分配/复用/驱逐 | `Executor`（管显存 vs 管计算） |
| `Executor`（Uniproc/Multiproc/Ray） | `v1/executor/` | 把"一步执行"下发到一个或多个 Worker | `Worker`（下发者 vs 干活者） |
| `Worker` / `GPUModelRunner` | `v1/worker/` | **真正在 GPU 上跑模型 forward + 采样** | `Executor`（见上） |
| Attention backend（FlashAttn/Triton/MLA） | `v1/attention/backends/` | attention 那一层到底用哪个 kernel 怎么算 | 模型本身（backend 只管 attention，不管整个模型） |

> 一句话串起来：**`Scheduler` 动脑（算什么）→ `Executor` 派活 → `Worker`/`GPUModelRunner` 动手（在 GPU 上算）→ `KVCacheManager` 在旁边管显存账本**。

### 2.4 易混概念对照（区别与联系——这部分专治"听过但分不清"）

| 一组易混概念 | 区别（各自在解决什么） | 联系 |
|---|---|---|
| **chunked prefill** vs **prefix caching** | chunked prefill =「把**自己**的长 prompt 拆成几步算」，避免一次占太多算力、阻塞别人；prefix caching =「**复用别人/自己**已经算过的相同前缀 KV」，避免重复计算 | 都和"prefill 太贵"有关；前者切自己，后者蹭现成。见 [模块 01](01-scheduler-batching/design.md) / [模块 02](02-paged-attention-kvcache/design.md) |
| **continuous batching** vs **静态 batching** | 静态 = 等一整批算完再换下一批（被最慢请求拖死）；continuous = 逐 token 调度，谁完谁退、随到随加 | 都是"多请求一起算提吞吐"，continuous 解决了静态的队头阻塞。见 [模块 01](01-scheduler-batching/design.md) |
| **TP** vs **PP** vs **EP** vs **DP** | TP=切每一层的矩阵；PP=按层切成段；EP=把 MoE 专家分到不同卡；DP=整个引擎复制多份 | 四个**正交**维度，可任意叠加。见 [模块 03](03-distributed-parallel/design.md) |
| **MLA** vs **GQA** vs **MQA** | 都是"省 KV cache"，手法不同：MQA=所有 query 头共享 1 组 KV 头（最省但掉点）；GQA=分组共享（折中）；MLA=把 KV 压成一条低秩 latent 向量（DeepSeek，省得最狠且质量好） | 同一目标的三代手法。GQA/MQA 见 [模块 14](14-attention-variants/design.md)，MLA 见 [模块 13](13-mla/design.md) |
| **SWA**（滑动窗口注意力） vs **full attention** | full=看前面**全部** token；SWA=只看**最近 W 个** token → KV 不再随长度无限增长 | SWA 是用"看得短"换"省显存"。见 [模块 14](14-attention-variants/design.md) |
| **PagedAttention** vs **普通 attention** | 唯一区别：KV 不是连续大块，而是切成定长**块（page）**、用**块表**间接寻址 → 消除显存碎片、支持前缀共享 | 见 [模块 02](02-paged-attention-kvcache/design.md) |
| **投机解码** vs **普通 decode** | 普通=一步一个 token；投机=小模型先猜 k 个，大模型一次验证、接受多少前进多少 | 用"一次验证多个"摊薄 decode 的访存成本。见 [模块 05](05-speculative-decoding/design.md) |
| **MoE** vs **稠密模型** | 稠密=每个 token 过全部参数；MoE=每个 token 只激活少数"专家"→ 参数量大但每 token 计算少 | 见 [模块 12](12-moe/design.md) |
| **V0** vs **V1**（两套引擎） | V1 是对引擎核心的重写（默认），V0 是遗留 fallback | 详见 [README](README.md) 的「如何区分 V0/V1」 |

### 2.5 完整模块索引

各模块的主题、代码锚点、文档链接见 [README](README.md) 的「模块索引」。建议路径：

1. **先读本篇 PRIMER** 建立心智模型；
2. 读 [模块 00](00-request-lifecycle/design.md) 看一个请求的完整旅程（骨架）；
3. 按 `01 调度 → 02 显存 → 03 分布式` 打通主干；
4. 按需读 04~11 的专题；
5. 想了解具体模型技术再读 12~15。

---

## 第三部分：一个贯穿全文的玩具示例（强烈建议先看懂这个）

> ⚠️ **这是"玩具尺寸"**：为了能手算、看清每个矩阵长什么样，下面的维度被**刻意缩到极小**（真实模型大几个数量级：hidden 几千、层数几十、词表几万）。数值用来**理解数据流与形状**，不是真实模型的输出。各阶段末尾标了 `→ 归哪个模块`，方便你带着这个例子去读对应模块。

### 玩具配置

| 超参 | 值 | 真实模型量级（对比） |
|---|---|---|
| hidden 维度 `d` | **4** | 4096~8192 |
| 注意力头数 `h` | **2**（每头 `head_dim=2`） | 32~64 |
| 层数 `L` | **1**（真实是逐层重复，原理一样） | 32~80 |
| KV block 大小 `block_size` | **2**（每块存 2 个 token 的 KV） | 16 |

**请求**：prompt 是 3 个 token，tokenize 后 `token_ids = [11, 7, 4]`，采样参数 `temperature=0`（贪心，便于复现）。

---

### 阶段① 进 prefill 前：tokenize 与调度  → 归 [模块 00](00-request-lifecycle/design.md) / [模块 01](01-scheduler-batching/design.md)

- 前端把文本切成 `token_ids = [11, 7, 4]`（3 个 token）。
- 调度器要决定**这一步给这条请求算几个 token**。这里先认识三个贯穿全文的量：
  - **`num_computed_tokens`** = 这条请求**已经算过**的 token 数（此刻 = 0，还没开算）。
  - **`num_new_tokens`** = 这一步**打算新算**的 token 数（此刻 = 3，整段 prompt 都还没算）。
  - **`token_budget`** = **这一步 forward、所有请求加起来最多能算多少 token 的总预算**，上限是配置项 `max_num_batched_tokens`。
    - *用途/设计意图*：单步算的 token 越多越吃算力和显存，而一个 batch 里通常有多条请求要分这块算力。用一个"预算"给单步封顶，既能防止某条超长 prompt 一次占满、把别人饿死，又能在"多塞 prefill 提吞吐"和"别拖慢正在 decode 的请求"之间调节——**`token_budget` 正是 chunked prefill 得以成立的前提**（预算不够就把长 prompt 切成几步算）。
- 于是：若 `token_budget` 还够 3，这一步就调度这 3 个 token（**一次完整 prefill**）；若 prompt 很长、预算不够，就只调度其中一截（**chunked prefill**）。
- 调度器再向 KV 管理器要 **block**（KV cache 被切成的定长小块，见阶段③）：3 个 token、每块装 2 个 → 需要 **2 个块**（逻辑块 L0 装 token0/1，L1 装 token2）。假设分到**物理块 `[7, 3]`**。

### 阶段② Prefill 前向：算 Q/K/V 与 attention  → 归 [模块 02](02-paged-attention-kvcache/design.md)(KV) / [模块 11](11-gpu-kernels-memory/design.md)(kernel)

输入经 embedding 得到 `X`（形状 `[3, 4]` = 3 token × hidden 4）。再分别乘三个投影权重得到 Q、K、V（各 `[3,4]`，reshape 成 `[3, 2头, 2]`）。**为便于手算，下面直接给出第 0 个 head 投影后的 k/v/q 向量**（真实里是 `X·W` 算出来的）：

```
       k（key）   v（value）   q（query）
token0  [1, 0]     [2, 0]       [1, 0]
token1  [0, 1]     [0, 2]       [0, 1]
token2  [1, 1]     [1, 1]       [1, 1]
```

注意力（causal：token i 只能看 ≤ i 的 token），`score = q·kᵀ/√head_dim`，softmax 后加权 v：

| 算谁的输出 | 看哪些 token | scores（÷√2 后）| softmax ≈ | 输出 o（`[2]`）|
|---|---|---|---|---|
| token0 | {0} | [0.71] | [1.0] | [2.0, 0.0] |
| token1 | {0,1} | [0, 0.71] | [0.33, 0.67] | [0.66, 1.34] |
| token2 | {0,1,2} | [0.71,0.71,1.41] | [0.25,0.25,0.50] | [1.0, 1.0] |

> 这就是"**算哪个阶段的什么矩阵的什么值**"的核心：prefill 一次并行算出 **3×3 的 score 矩阵**（下三角有效）、softmax、再 ×V 得到 3 个 token 的输出。`token2`（最后一个）的输出会用来预测下一个 token。

### 阶段③ KV 写进分页缓存  → 归 [模块 02](02-paged-attention-kvcache/design.md)

把上面 3 个 token 的 k、v 存进物理块（`block_size=2`，物理块 `[7,3]`）。每个 token 的全局位置 `slot = 物理块号 × block_size + 块内偏移`：

```
token0 → 逻辑块L0 偏移0 → 物理块7 偏移0 → slot = 7*2+0 = 14   存 k0,v0
token1 → 逻辑块L0 偏移1 → 物理块7 偏移1 → slot = 7*2+1 = 15   存 k1,v1
token2 → 逻辑块L1 偏移0 → 物理块3 偏移0 → slot = 3*2+0 = 6    存 k2,v2
                          物理块3 偏移1 → slot = 7        （空着，留给下一个 token）
所以 slot_mapping = [14, 15, 6]
block_table（这条请求的"逻辑块→物理块"页表）= [7, 3]
```

> 这张 **block_table 就是 PagedAttention 的灵魂**：逻辑上 token 连续，物理上可以乱放（这里用了物理块 7 和 3）。前缀复用（prefix caching）= 别的请求若前缀相同，其 block_table 也指向物理块 7，省去重算。

### 阶段④ 采样下一个 token  → 归 [模块 06](06-sampling-structured-output/design.md)

token2 的输出过 `lm_head`（**把隐藏向量映射到整个词表的输出层**，形状 `[d, vocab]`）得到词表上的 logits（每个 token 一个分数，形状 `[vocab]`）。`temperature=0` 表示贪心、直接取分数最大的（argmax），假设采样出 **token id = 9**。这是生成的第 1 个 token。

### 阶段⑤ Decode 第 1 步：只算 1 个 token  → 归 [模块 01](01-scheduler-batching/design.md)(调度) / [模块 02](02-paged-attention-kvcache/design.md)(读KV)

- 调度器：这条请求现在 `num_computed_tokens=3`，要算的 `num_new_tokens = 1`（就是刚生成的 token id=9，位置 pos=3）。**这就是 decode——prefill 的退化情形（只算 1 个 token）**。
- 给定它投影后的 `q3=[1,0]`、`k3=[0,1]`、`v3=[0,2]`。
- attention：新的 `q3` 要对 **KV cache 里已有的全部 4 个位置**（pos0~3）算 score。kernel 通过 block_table 取物理 KV：

```
pos0 → 逻辑块0 → 物理块7 偏移0 → slot14   pos2 → 逻辑块1 → 物理块3 偏移0 → slot6
pos1 → 逻辑块0 → 物理块7 偏移1 → slot15   pos3 → 逻辑块1 → 物理块3 偏移1 → slot7（本步刚写入）
```

scores = `q3·[k0,k1,k2,k3]ᵀ/√2` = [0.71, 0, 0.71, 0] → softmax ≈ [0.34,0.16,0.34,0.16] → `o3 ≈ [1.0, 1.0]`。

- 新 token 的 `k3,v3` 写进**物理块3 偏移1 = slot 7**（block_table 不变，仍是 `[7,3]`，块表不需新增，因为 L1 还有空位）。
- o3 → logits → 采样出第 2 个生成 token……如此循环，直到采样到 EOS 或达长度上限（由 [模块 01](01-scheduler-batching/design.md) 的 `check_stop` 判定）。

---

### 这个例子如何贯穿各模块

| 模块 | 在这个例子里对应什么 |
|---|---|
| [00 请求生命周期](00-request-lifecycle/design.md) | `[11,7,4]` 怎么从 HTTP 进来、tokenize、跨进程进 EngineCore，token id=9 怎么 detokenize 流式回去 |
| [01 调度器](01-scheduler-batching/design.md) | prefill 步 `num_new_tokens=3`、decode 步 `num_new_tokens=1`、`token_budget` 怎么扣、长 prompt 时怎么切成 chunk |
| [02 PagedAttention/KV](02-paged-attention-kvcache/design.md) | `block_table=[7,3]`、`slot_mapping=[14,15,6]`、前缀复用怎么让两条请求共享物理块 7 |
| [05 投机解码](05-speculative-decoding/design.md) | decode 步若开投机，一次不是算 1 个而是验证 k+1 个候选 |
| [06 采样](06-sampling-structured-output/design.md) | token2/o3 的输出 → logits → 怎么取出 id=9 |
| [11 GPU kernel](11-gpu-kernels-memory/design.md) | attention kernel 内部怎么用 `block_table[pos//2]` 把逻辑位置翻成物理 slot 去取 KV |
| [13 MLA](13-mla/design.md) / [14 注意力变体](14-attention-variants/design.md) | 同一个例子里，KV 那部分若换成 MLA（压成 latent）或 SWA（只看最近 W 个）会怎样 |

> 后面读任何模块时，都可以把它套回这个 `[11,7,4] → 9 → …` 的小例子，问自己："**这个模块在这个例子的哪一步、动了哪个矩阵/哪块缓存？**"

---

> 每个模块文档内部还有一节 **「背景与定位」**（开头）：进一步交代该模块的设计背景、实现背景，以及它和相邻/相似模块的区别与联系。本篇是全局导读，模块内的「背景与定位」是局部聚焦，配合着读。
