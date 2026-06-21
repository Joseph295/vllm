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
```

> 建议阅读顺序：先读 **00 总览**建立全局执行骨架，再按 `01 → 02 → 03` 打通"调度—显存—分布式"主干，最后按需阅读 04~10 的专题模块。
