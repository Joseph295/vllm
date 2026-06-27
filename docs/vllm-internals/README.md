# vLLM 核心模块设计与实现（V1 架构为主）

> 本系列文档基于仓库当前代码（`vllm/v1/` 为主的 **V1 架构**）梳理 vLLM 的核心设计与实现。
> 信息来源以**仓库实际代码为唯一事实来源**，网络资料（官方 blog / 论文 / PR）仅用于补充"为什么这么设计"的动机，并在引用处标注出处。
> 每个模块拆成两份文档：
> - **`design.md`** —— 解决什么问题、设计目标与约束、核心设计思想（含 V0→V1 对比）、关键数据结构、权衡取舍。
> - **`impl.md`** —— 代码地图、关键类/函数职责、端到端调用链（带 `file:line`，可逐行对照）、关键片段解读、边界情况。

## 阅读前提：V0 与 V1 两套架构

仓库里同时存在两套引擎实现，二者设计差异很大，阅读代码时务必先确认自己在看哪一套：

| | V0（legacy） | V1（current / 默认） |
|---|---|---|
| 引擎入口 | `vllm/engine/llm_engine.py`、`async_llm_engine.py` | `vllm/v1/engine/llm_engine.py`、`async_llm.py` |
| 调度器 | `vllm/core/scheduler.py` | `vllm/v1/core/sched/scheduler.py` |
| KV/Block 管理 | `vllm/core/block_manager.py` | `vllm/v1/core/kv_cache_manager.py`、`block_pool.py` |
| Worker | `vllm/worker/` | `vllm/v1/worker/gpu_model_runner.py`、`gpu_worker.py` |
| 进程模型 | 引擎与 API server 同进程 | **EngineCore 独立进程**，ZMQ + msgpack 解耦前后端 |

V1 是 2024 下半年开始的重写，开关由环境变量 `VLLM_USE_V1` 控制。本系列**以 V1 为主线**，在关键处对比 V0 说明"为什么要重构"。

> 📖 **新手必读的两份前置导读**：
> - [PRIMER.md](PRIMER.md) —— **业务背景**：LLM 推理服务 101（prefill/decode、KV cache、连续批处理…）+ 代码全景地图 + 一个贯穿全文的玩具示例。
> - [PYTHON-PRIMER.md](PYTHON-PRIMER.md) —— **语法背景**：vLLM 用到的 Python 进阶特性（类型系统、`async`、`@dataclass`/`msgspec`、ABC、`weakref`、海象运算符…），每条配真实代码。
> - [RAY-PRIMER.md](RAY-PRIMER.md) —— **框架背景（进阶）**：读多机/PP 分布式执行（`RayDistributedExecutor`、compiled DAG、placement group）前先扫一眼 Ray 的 actor/task/object store 模型。
> - [RAY-MASTERY.md](RAY-MASTERY.md) —— **成为 Ray 专家（深入）**：Ray 官方源码梳理（`src/ray` 各组件）+ 一份 10–12 周可执行学习计划 + 权威资料清单。
>
> 不懂推理服务先读 PRIMER，被 Python 语法卡住读 PYTHON-PRIMER，看多机分布式（Ray）那部分读 RAY-PRIMER。

### V0 与 V1 的关系，以及如何区分一段代码属于哪套

**关系不是"平级二选一"，而是"重写版 + 遗留版"**：

```
V1 = 对 vLLM 引擎核心的整体重写（默认启用，VLLM_USE_V1 默认=1）
      ├─ 重写：调度器、KV/前缀缓存、进程模型(EngineCore 独立进程)、
      │        投机解码、采样、CUDA graph(piecewise) ……
      └─ 复用：模型定义层、大部分 kernel(csrc/)、分布式原语、量化/LoRA 算子
V0 = 遗留实现(legacy)，现在主要作为 fallback：请求了 V1 尚不支持的特性时自动回退
```

两者**共享"下半身"**（`vllm/model_executor/models/`、`csrc/`、`vllm/distributed/`、`quantization/`、`lora/` 的底层算子），**分叉在"上半身"**（引擎编排、调度、KV 管理、worker 执行流程）。

**运行时如何决定走哪套**（代码路径）：
1. `VLLM_USE_V1` 默认 `1`（`vllm/envs.py:528-529`）。
2. `EngineArgs._is_v1_supported_oracle(...)`（`vllm/engine/arg_utils.py:1368`）逐项检查配置，遇到 V1 尚不支持的特性就 `return False` → 回退 V0。典型触发：`--kv-transfer-config`（PD 分离，仅 V0）、部分 encoder-decoder 模型等。
3. `LLMEngine.__new__`（`vllm/engine/llm_engine.py:518`）据此把对象重定向到 `vllm.v1.engine.llm_engine.LLMEngine`。
   > ⚠️ 注意：V0 的 `LLMEngine.__new__` 在 `VLLM_USE_V1` 时直接返回 V1 实例——`from vllm.engine.llm_engine import LLMEngine` 拿到的运行时类型可能是 V1 的，**光看 import 路径会被骗**。

**判别清单（按可靠度排序）**：

① **看路径（最可靠）**：

| 属于 V1 | 属于 V0 | 公用（都不算） |
|---|---|---|
| `vllm/v1/**` 全部 | `vllm/core/scheduler.py`、`block_manager.py`<br>`vllm/worker/`<br>`vllm/engine/llm_engine.py`<br>`vllm/spec_decode/` | `vllm/model_executor/`、`csrc/`<br>`vllm/distributed/`<br>`vllm/lora/`、`quantization/` 算子 |

一句话：**路径里有 `/v1/` 就是 V1；`vllm/core/scheduler.py`、`vllm/worker/`、`vllm/engine/llm_engine.py` 是 V0。**

② **看 import**：`from vllm.v1.xxx` → V1；`from vllm.core.scheduler` / `from vllm.worker.xxx` → V0。

③ **看结构特征（标志物）**：

| 特征 | V1 | V0 |
|---|---|---|
| 引擎进程 | **EngineCore 独立进程** + ZMQ/msgpack | 引擎与 API server 同进程 |
| 调度抽象 | 无 prefill/decode 阶段，统一 `num_computed_tokens` | 分 prefill/decode batch，有 swap in/out |
| KV 管理 | `KVCacheManager` + `block_pool` + hash 化 prefix cache | `BlockSpaceManager` + `evictor` |
| worker 执行 | `GPUModelRunner.execute_model(scheduler_output)` | `ModelRunner` + `Worker` 多步流程 |
| 投机解码 | 原位 MQA 验证 + Triton `rejection_sampler` | `SpecDecodeWorker` + batch expansion |

④ **运行时确认**：V1 启动日志打印 `Initializing a V1 LLM engine (v...)`（`vllm/v1/engine/core.py:57`）；或设 `VLLM_USE_V1=0` 强制 V0、`=1` 强制 V1。

### V0 vs V1 执行流程对比

```
═══════════════════ V0（legacy · 单进程）════════════════ │ ═════════════════ V1（current · 双进程）═══════════════
                                                          │
┌───────────── 单个 Python 进程（共享 GIL）─────────────┐ │  ┌────────── 进程 1：API server（前端·asyncio）──────────┐
│  API server (FastAPI)                                 │ │  │  Processor.process_inputs：tokenize/多模态(CPU)        │
│     │ add_request                                     │ │  │  OutputProcessor：detokenize(CPU,后台 output_handler) │
│     ▼                                                 │ │  └───────────────┬───────────────────────▲─────────────┘
│  LLMEngine.step()  ◀──── 同步循环，反复调用           │ │   EngineCoreRequest│ ZMQ+msgpack         │EngineCoreOutputs
│   ┌─────────────────────────────────────────────┐    │ │   (ROUTER/DEALER)  ▼  零拷贝          (PUSH/PULL)│
│   │ ① Scheduler.schedule()                       │    │ │  ┌────────── 进程 2：EngineCore（后端·同步 busy loop）──┐
│   │    三队列：waiting / running / swapped        │    │ │  │  run_busy_loop: while True → step():                 │
│   │    · prefill 与 decode 分阶段（分开成批）      │    │ │  │   ┌────────────────────────────────────────────────┐ │
│   │    · 显存不足→抢占：SWAP(换出到CPU) 或 RECOMP  │    │ │  │   │ ① schedule()  无 prefill/decode 之分            │ │
│   │    · 产物 SeqGroupMetadata + SchedulerOutputs │    │ │  │   │    统一 num_computed_tokens 追 num_tokens       │ │
│   │ ② executor.execute_model()                   │    │ │  │   │    prefill+decode 混批；抢占只 RECOMPUTE(无SWAP) │ │
│   │    Worker → ModelRunner.forward + sample      │    │ │  │   │    产物 SchedulerOutput                          │ │
│   │ ③ _process_model_outputs()  同进程 detokenize │    │ │  │   │ ② executor.execute_model()  MessageQueue 广播   │ │
│   └─────────────────────────────────────────────┘    │ │  │   │    Worker → GPUModelRunner.execute_model         │ │
│   tokenize/detokenize/调度/采样后处理 全抢同一 GIL    │ │  │   │ ③ update_from_output()  append/check_stop/回退  │ │
└───────────────────────────────────────────────────────┘ │  │   └────────────────────────────────────────────────┘ │
                                                          │  │   ＋ 输入/输出各一 IO 线程跑 ZMQ（释放 GIL 与 GPU 重叠）│
  特点：实现直观、单进程易调试；                          │  └──────────────────────────────────────────────────────┘
  但 CPU 后处理周期性"卡住"GPU，长 prefill 阻塞 decode。   │   特点：GPU 关键路径独占进程、CPU 活并行不挡道；
                                                          │         统一抽象天然支持 chunked prefill / prefix cache / spec。
```

| 维度 | V0 | V1 |
|---|---|---|
| 进程模型 | 单进程，引擎与 API server 同 GIL | **EngineCore 独立进程** + ZMQ/msgpack，CPU/GPU 重叠 |
| 调度抽象 | prefill / decode **分阶段**，三队列(`waiting/running/swapped`) | **无阶段之分**，统一 `num_computed_tokens` 推进 |
| 长 prompt | prefill 独占整步 → 阻塞 decode（head-of-line） | `chunked prefill` 混批，不阻塞 decode |
| 抢占策略 | RECOMPUTE **或** SWAP（换出 KV 到 CPU） | 只 RECOMPUTE，**砍掉 swap** |
| 驱动方式 | 主线程同步 `step()` 循环 | 后端 busy loop + 前端单一 `output_handler` |
| KV 管理 | `BlockSpaceManager` + `evictor` | `KVCacheManager` + `block_pool` + hash `prefix caching` |

> 代码锚点：V0 `step()` 在 `vllm/engine/llm_engine.py`、三队列在 `vllm/core/scheduler.py:469-475`、抢占模式(RECOMPUTE/SWAP)在 `vllm/core/scheduler.py:487` 一带；V1 `step()` 在 `vllm/v1/engine/core.py:195`、统一调度在 `vllm/v1/core/sched/scheduler.py:122`（详见 [模块 00](00-request-lifecycle/design.md) 与 [模块 01](01-scheduler-batching/design.md)）。

## 模块索引

| # | 模块 | 主要代码锚点 | 文档 |
|---|------|------------|------|
| 00 | **请求全生命周期总览** —— 从 API 进来到 token 出去 | `entrypoints/` → `v1/engine/{async_llm,core,processor,output_processor}.py` → `v1/executor/` → `v1/worker/gpu_model_runner.py` | [design](00-request-lifecycle/design.md) · [impl](00-request-lifecycle/impl.md) |
| 01 | **调度器：Continuous Batching + Chunked Prefill** | `v1/core/sched/scheduler.py` | [design](01-scheduler-batching/design.md) · [impl](01-scheduler-batching/impl.md) |
| 02 | **PagedAttention + KV Cache 管理 + Prefix Caching** | `csrc/attention/`、`v1/core/kv_cache_manager.py`、`block_pool.py` | [design](02-paged-attention-kvcache/design.md) · [impl](02-paged-attention-kvcache/impl.md) |
| 03 | **分布式并行（TP/PP/EP/DP）+ 分布式执行** | `distributed/parallel_state.py`、`v1/executor/` | [design](03-distributed-parallel/design.md) · [impl](03-distributed-parallel/impl.md) |
| 04 | **PD 分离（Prefill-Decode Disaggregation）** | `distributed/kv_transfer/` | [design](04-pd-disaggregation/design.md) · [impl](04-pd-disaggregation/impl.md) |
| 05 | **Speculative Decoding 投机解码** | `v1/spec_decode/`、`spec_decode/` | [design](05-speculative-decoding/design.md) · [impl](05-speculative-decoding/impl.md) |
| 06 | **Sampling + Structured/Guided Output** | `v1/sample/`、`v1/structured_output/` | [design](06-sampling-structured-output/design.md) · [impl](06-sampling-structured-output/impl.md) |
| 07 | **CUDA Graph / torch.compile 编译优化** | `vllm/compilation/`、`forward_context.py` | [design](07-cuda-graph-compile/design.md) · [impl](07-cuda-graph-compile/impl.md) |
| 08 | **Quantization 量化** | `model_executor/layers/quantization/` | [design](08-quantization/design.md) · [impl](08-quantization/impl.md) |
| 09 | **LoRA 多适配器** | `vllm/lora/` | [design](09-lora/design.md) · [impl](09-lora/impl.md) |
| 10 | **Multimodal 多模态** | `vllm/multimodal/`、`v1/core/encoder_cache_manager.py` | [design](10-multimodal/design.md) · [impl](10-multimodal/impl.md) |
| 11 | **GPU 执行底层：Kernel 与显存** —— 深入 csrc/ CUDA 源码 | `csrc/attention/`、`csrc/cache_kernels.cu`、`csrc/custom_all_reduce.cu`、`csrc/quantization/` | [design](11-gpu-kernels-memory/design.md) · [impl](11-gpu-kernels-memory/impl.md) |
| — | **—— 模型架构技术 track（12~15）——** | 横向引擎之外的"模型/注意力变体"垂直技术 | |
| 12 | **MoE 混合专家** —— 路由 / fused_moe / 分组 GEMM / EP | `model_executor/layers/fused_moe/`、`csrc/moe/` | [design](12-moe/design.md) · [impl](12-moe/impl.md) |
| 13 | **MLA 潜在注意力（DeepSeek）** —— latent KV 压缩 / 吸收矩阵 | `v1/attention/backends/mla/` | [design](13-mla/design.md) · [impl](13-mla/impl.md) |
| 14 | **注意力变体：SWA / Cascade / GQA·MQA** | `v1/core/specialized_manager.py`、`v1/attention/backends/` | [design](14-attention-variants/design.md) · [impl](14-attention-variants/impl.md) |
| 15 | **Mamba/SSM 与混合架构 + 位置编码（RoPE/M-RoPE/YaRN）** | `model_executor/layers/mamba/`、`rotary_embedding.py` | [design](15-mamba-rope/design.md) · [impl](15-mamba-rope/impl.md) |

## 端到端全景图（一条请求如何穿过 11 个模块）

下图把"请求的完整旅程"与"每个模块在哪一步介入"对应起来。`[NN]` 标注表示该处的实现细节属于哪个模块，点进对应文档即可深读。

```
                        客户端 (HTTP / OpenAI 兼容 API)
                                   │  prompt + sampling_params
                                   ▼
╔═══════════════════ 进程 1：API server（前端 · asyncio）═══════════════════╗
║                                                                            ║
║   ① Processor.process_inputs            ② OutputProcessor (后台 output_handler)
║      tokenize / 参数校验                    detokenize → 按 req_id 分发 → yield ║
║      多模态预处理 [10]                        采样输出转文本                      ║
║      └─ mm 镜像缓存(P0) [10]                                  ▲               ║
║                │ EngineCoreRequest                           │ EngineCoreOutputs
╚════════════════│═════════════════════════════════════════════│═══════════════╝
                 │  ZMQ(ROUTER/DEALER)+msgpack 零拷贝   ZMQ(PUSH/PULL)
                 ▼  [00]                                        │  [00]
╔═══════════════════ 进程 2：EngineCore（后端 · 同步 busy loop）═════════════════╗
║   run_busy_loop:  while True →  step()  ＝  ① schedule → ② execute → ③ update ║
║                                                                              ║
║  ┌── ① schedule  [01 调度器]──────────────────────────────────────────────┐  ║
║  │   token 预算混合 prefill+decode / chunked prefill / 抢占                 │  ║
║  │   · 查 prefix cache 命中、分配 block        ........................ [02] │  ║
║  │   · 多模态 encoder budget 约束              ........................ [10] │  ║
║  │   · LoRA max_loras 约束                     ........................ [09] │  ║
║  │   · 投机 draft token 纳入预算               ........................ [05] │  ║
║  │   · 结构化输出 grammar bitmask 生成         ........................ [06] │  ║
║  └────────────────────────────┬───────────────────────────────────────────┘  ║
║                               │ SchedulerOutput                               ║
║  ┌── ② execute ──────────────▼───────────────────────────────────────────┐  ║
║  │   Executor 广播 → Worker(s)            分布式并行 TP/PP/EP/DP ...... [03] │  ║
║  │      └─ GPUModelRunner.execute_model                                    │  ║
║  │           · (多模态) 跑 encoder → scatter 进 input embeds ........ [10]  │  ║
║  │           · CUDA graph 捕获/replay + torch.compile .............. [07]  │  ║
║  │           · 模型各层 forward：列/行并行 [03] · 量化 GEMM [08] · LoRA [09]│  ║
║  │           · attention：PagedAttention 读 block table ........... [02]  │  ║
║  │           · logits → grammar bitmask 掩码 + 采样 ............... [06]  │  ║
║  │           · 投机解码 propose / 拒绝采样 verify ................. [05]  │  ║
║  └────────────────────────────┬───────────────────────────────────────────┘  ║
║                               │ ModelRunnerOutput (sampled tokens)            ║
║  ┌── ③ update_from_output [01]▼───────────────────────────────────────────┐  ║
║  │   append token / check_stop / 投机被拒回退 [05] / 推进 grammar FSM [06]   │  ║
║  └──────────────────────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════════════════╝
                 ┊
                 ┊  [04] PD 分离（可选）：prefill 实例算完 KV 经 connector
                 ┊        跨实例搬到 decode 实例续算。⚠ 当前仅接入 V0，
                 ┊        设 --kv-transfer-config 会从 V1 回退 V0。
   prefill 实例 ━━━━━━━━━━━━━━━━━━━━━━▶ decode 实例
                 (KV cache 跨节点传输)

  横切关注点：[07] 编译/CUDA graph、[08] 量化、[09] LoRA 在"模型 forward"内被各层透明调用；
              [03] 并行是整个 ② execute 的承载方式；[00] 贯穿前后端编排与跨进程通信。
```

> 一句话串联：**前端把文本变 token [00/10] → 跨进程进 EngineCore [00] → 调度器决定这步算什么 [01]（受 KV[02]/encoder[10]/LoRA[09]/spec[05]/grammar[06] 约束）→ Executor 分布式执行 [03] 跑模型 forward（编译[07]·量化[08]·LoRA[09]·PagedAttention[02]）→ 采样[06]/投机[05] 出 token → 回填检测 stop [01] → 跨进程回前端 detokenize [00/06] → 流式返回**；PD 分离 [04] 则把 prefill 与 decode 拆到不同实例。

## 重点机制特写：投机解码（Speculative Decoding）流程

投机解码是"在调度 + KV + 采样之上叠加 draft/verify"的一步循环，下图是它一步的数据流（完整推导见 [模块 05](05-speculative-decoding/design.md)）：

```
        ┌─ 上一步 proposer 已为每条请求产出 k 个 draft：req.spec_token_ids = [d1,d2,…,dk]
        ▼
  ① 调度  scheduler.schedule()
        把 k 个 draft 一并纳入 token 预算（一步要算 k+1 个位置）   ...... [01 调度]
        ▼
  ② Target 一次 forward（注意：是一次，不是 k+1 次）
        对 k 个 draft 位置 + 1 个 bonus 位置，算出 target logits p(x)  ...... [00/03 执行]
        ▼
  ③ RejectionSampler（Triton）逐位置比较 draft 分布 q 与 target 分布 p
        ┌───────────────────────────────────────────────────────────────┐
        │  for i in 1..k:                                                 │
        │     r ~ U(0,1)                                                  │
        │     若 r < min(1, p(di)/q(di))  → accept di                     │
        │     否则 → reject：从修正分布 p'=norm(max(0,p−q)) 重采样 1 个    │
        │            token，并丢弃 d(i+1..k)（建立在被推翻前缀上）  ──┐    │
        │  若 1..k 全 accept → 再白送 bonus token（第 k+1 位置）       │    │
        └────────────────────────────────────────────────────────────┘    │
        ▼  本步产出 1 ~ k+1 个「严格服从 target 分布 p」的 token  ◀────────┘
           （最坏=普通 decode 一步；最好=一次前进 k+1）        ...... [06 采样]
        ▼
  ④ 回填  update_from_output
        按"被拒位置"回退 num_computed_tokens（接受 m 个就只认 m 个）  ...... [01 调度]
        ▼
  ⑤ 为下一步重新提议 k 个 draft（ngram / eagle / draft model）
        写回 request.spec_token_ids ──► 回到 ①                       ...... [05 proposer]
```

**关键直觉**：draft 的好坏只影响**接受率（即速度）**，不影响输出分布（**正确性由拒绝采样在数学上保证 = target 分布**）。所以"接受率"是投机解码的第一指标；proposer 越重→提议越准→但每步 draft 开销越大，最优点取决于"省下的 target forward × 接受率"能否盖过"draft 固定开销"。

## 重点机制特写：PagedAttention 的 block table 寻址

vLLM 把 OS 的虚拟内存分页搬到 GPU 管 KV cache：每条序列的 KV 切成定长 `block`（页，典型 `block_size=16`），逻辑连续的 block 经 **block table（页表）** 映射到**物理上可不连续、甚至跨序列共享**的物理块——这就是消灭显存碎片、实现前缀共享的物理基础（完整机制见 [模块 02](02-paged-attention-kvcache/design.md)）。

```
逻辑视角（每条序列连续编号）          block table（每请求一行：逻辑块→物理块）
  req A: [L0][L1][L2][L3]   ──────►    A: [7, 3, 9, 1]
  req B: [L0][L1]          ──────►     B: [7, 5]          ← B 的 L0 也指向物理块 7
            │                                  │              = 与 A 共享前缀（prefix caching）
            ▼                                  ▼
物理 KV cache（num_gpu_blocks 个定长块，乱序复用）
  ┌phys0┐┌phys1┐┌phys2┐┌phys3┐┌…┐┌phys5┐┌…┐┌phys7┐┌…┐┌phys9┐
  │     ││ A.L3││     ││ A.L1││ ││ B.L1││ ││A.L0 ││ ││ A.L2│   phys7 被 A、B 同时指向
  └─────┘└─────┘└─────┘└─────┘└─┘└─────┘└─┘│B.L0 ││ │└─────┘
                                            └─────┘└─┘   ref_cnt=2 → 写时触发 copy-on-write

单个 token 的全局地址：  slot = physical_block_id * block_size + (position % block_size)
  · 写 KV：reshape_and_cache 按 slot_mapping[token] 写入 cache[slot]
  · 读 KV：attention kernel 用 block_table[logical] → physical 反查 —— 这是 PagedAttention
           与朴素 flash-attention 的唯一本质区别：多一次"逻辑块→物理块"的间接寻址。
```

> 由此，**抢占重分配 / 前缀共享 / copy-on-write 都只是改 block table，不搬数据**。前缀命中靠对 block 的 token 内容做 hash、沿 hash 链查字典实现（命中前缀直接计入 `num_computed_tokens`、跳过计算）。

## 重点机制特写：PD 分离的跨实例 KV 搬运

PD 分离把 **prefill（计算密集）** 与 **decode（访存密集）** 拆到不同实例，prefill 算好的 KV cache 经 connector 跨实例搬到 decode 实例续算，互不干扰（完整三层抽象见 [模块 04](04-pd-disaggregation/design.md)）。**⚠️ 当前 KV 传输仅接入 V0**：设 `--kv-transfer-config` 会使引擎从 V1 回退 V0，且仅支持 1P1D。

```
   client ──► Proxy：① 复制请求(max_tokens=1) 发 prefill ② 等完成 ③ 原请求发 decode，流式回
                       │                                    │
        ┌──────────────▼─────────────┐        ┌────────────▼──────────────────┐
        │  PREFILL 实例  kv_producer  │        │  DECODE 实例  kv_consumer       │
        │  execute_model：算 prefill KV│        │  need_recv_kv? → 收 KV 写回本地  │
        │  按 seq_lens 切请求          │        │  paged cache，bypass 掉 prefill  │
        │  按 layer×slot 抽 K/V/hidden │        │  forward，用 hidden 直接采样首   │
        │  buffer.insert(...)         │        │  token，之后正常 decode          │
        └──────────────┬─────────────┘        └────────────▲──────────────────┘
                       │ KV cache (按 token×layer 切分)      │
                       ▼                                     │
            ┌──────────────── KVPipe ─────────────────────────┘
            │  signal pipe(CPU 元数据) + data pipe(GPU 张量)
            │  PyNcclPipe(NCCL) / MooncakePipe(RDMA)，send 非阻塞与计算重叠
            └─────────────────────────────────────────────────
```

> 为什么不能用裸 FIFO 管道：decode 实例取 KV 时要按"哪条请求/哪段前缀"**按需查询**，所以中间隔一层 **LookupBuffer**（producer 侧 deque + 后台 `drop_select` 线程）做按内容选取，而非先进先出。

## 重点机制特写：TP / PP 并行的张量切分与通信

TP 与 PP 是两种正交的切法：**TP 把"每一层"横向切开（切矩阵），PP 把"层与层"纵向切段**（切深度）。二者通信代价与适用场景截然不同（完整四种并行见 [模块 03](03-distributed-parallel/design.md)）。

**① TP（张量并行，Megatron 式"列并行→行并行"配对，每个 block 只 2 次 all-reduce）：**

```
   一个 Transformer block 在 TP=2 下的数据流（中间激活全程切分、不通信）

   X ──┬──────────────► ColumnParallel (up_proj)   权重按【输出维】切
       │                 rank0: Y0 = X·W0   rank1: Y1 = X·W1   ← 各持输出的不同列，无需通信
       │                      │                  │
       │                   激活 act            激活 act
       │                      ▼                  ▼
       │                 RowParallel (down_proj)  权重按【输入维】切，输入恰好是上游切好的列
       │                 rank0: Z0=Y0·W0'  rank1: Z1=Y1·W1'   ← 各得"部分和"
       │                      └────────► all-reduce(Z0+Z1) ◄───┘   ★唯一一次通信
       └──────────────────────────────────┘
   Attention 同理：QKV 列并行(每 rank 算一部分 head) → attn → O 行并行 → all-reduce
   ⇒ 每个 block 仅 2 次 all-reduce（attn 后 + mlp 后），但【高频、在关键路径】→ 依赖 NVLink，通常不跨节点
```

**② PP（流水线并行，按层切段，段间点对点只传一次中间张量）：**

```
   层 0..L 切成 P 段，每张卡一段；段间传 IntermediateTensors（hidden state）

   GPU0 [layers 0..k]  ──send/recv──►  GPU1 [layers k..2k]  ──►  …  ──►  GPUP [last] → logits/sample
        每段边界只 1 次点对点通信（远小于 TP 的 all-reduce），对跨节点带宽不敏感 → 适合多机扩层

   朴素 PP 有"气泡"(bubble)：单批数据时同一时刻只有一段在算。vLLM 用 batch queue 叠多个
   micro-batch 填气泡（1F1B 思想）：
        t1: GPU0[b1]
        t2: GPU0[b2]  GPU1[b1]
        t3: GPU0[b3]  GPU1[b2]  GPU2[b1]   ← 流水线填满，各段都在干活
```

> ⚠️ V1 现状：`MultiprocExecutor` 当前断言 `world_size == tensor_parallel_size`，**V1 的多进程后端尚不支持 PP**（`multiproc_executor.py:56-59`）；V1 的 PP 需走 **Ray Executor**（`max_concurrent_batches = pipeline_parallel_size`，靠 batch queue 叠流水）。TP 则多进程/Ray 都支持。四者可正交组合：`world = ExternalDP × DP × PP × TP`。

## 模块依赖关系

```
                       ┌─────────────────────────────────────────┐
                       │  00 请求全生命周期（贯穿所有模块的主干）   │
                       └─────────────────────────────────────────┘
                                          │
         ┌────────────────┬───────────────┼───────────────┬────────────────┐
         ▼                ▼               ▼               ▼                ▼
   01 调度器         02 KV/Paged     03 并行/执行     06 采样/结构化   10 多模态
  (batching/        Attention       (TP/PP/EP/DP)   (sampler +       (encoder
   chunked prefill)  + prefix cache                  guided decode)   cache)
         │                │               │
         │                │               └──────► 04 PD 分离（跨实例搬运 KV）
         │                │
         └──────┬─────────┘
                ▼
          05 投机解码（在调度+KV+采样之上叠加 draft/verify）

   横切关注点（被多个模块使用）：07 CUDA Graph/编译、08 量化、09 LoRA

   GPU 底层（02/03/07/08 的 CUDA 源码延伸）：11 GPU Kernel 与显存
     ├─ PagedAttention kernel 内幕  ◀── 02 的 GPU 实现
     ├─ KV 显存 profiling / 布局      ◀── 02
     ├─ CUDA graph 显存池 / stream    ◀── 07
     ├─ custom all-reduce GPU IPC     ◀── 03
     └─ 量化 GEMM（Marlin）kernel     ◀── 08

   模型架构技术 track（垂直层，挂在 02/03/10 之上）：
     ├─ 12 MoE 混合专家        ── 路由+分组 GEMM，与 03 的 EP 并行结合
     ├─ 13 MLA 潜在注意力       ── 02 PagedAttention 的变体，latent KV 压缩
     ├─ 14 注意力变体 SWA/Cascade/GQA ── 02 KV 管理 + 调度公共前缀(01) 的特化
     └─ 15 Mamba/SSM + 位置编码  ── 常量状态替代 KV；RoPE/M-RoPE(10) 注入位置
```

> 建议阅读顺序：先读 **00 总览**建立全局执行骨架，再按 `01 → 02 → 03` 打通"调度—显存—分布式"主干，按需阅读 04~10 的专题模块；想钻 GPU 底层 / CUDA kernel 则读 **11**。
