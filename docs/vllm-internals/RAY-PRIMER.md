# Ray 在 vLLM 中的用法导读

> **这份文档给谁看**：你想看懂 vLLM 的**多机/多卡分布式执行**那部分代码（`RayDistributedExecutor`、compiled DAG、placement group），但**没怎么用过 Ray**。本篇先用 30 秒讲清 Ray 的核心概念，再逐条对到 vLLM 真实代码（`file:line`）。
>
> 配套：分布式并行的整体设计见 [模块 03](03-distributed-parallel/design.md)，进程/执行模型见 [模块 00](00-request-lifecycle/design.md)。本篇只解决"**Ray 这层是怎么回事**"。

---

## 1. 先认识 Ray（没用过也能懂）

Ray 是一个 **Python 分布式计算框架**。你只需要记住它的 4 个核心概念：

| Ray 概念 | 一句话 | 类比 |
|---|---|---|
| **Task**（任务） | `f.remote(x)` 把一个普通函数丢到**别的进程/机器**上异步执行 | "远程调用一个函数" |
| **Actor**（有状态的远程对象） | `Cls.remote()` 在远端起一个**长期存活、带状态**的对象，之后 `actor.method.remote()` 调它的方法 | "远程的一个对象实例" |
| **ObjectRef**（远程结果的引用） | `.remote()` 不返回结果本身，而是一个**未来引用**；`ray.get(ref)` 才真正取回值 | Future / Promise |
| **Placement Group**（资源预留） | 提前向集群**预订一批资源**（如 8 个 GPU bundle），并控制它们落在哪些节点上 | "先把 8 张卡圈下来" |

> 和普通 `multiprocessing` 的关键区别：**Ray 能跨多台机器**（不止本机多进程），且自带对象存储、资源调度、actor 生命周期管理。这正是 vLLM 需要它的原因。

---

## 2. vLLM 为什么用 Ray（而不是只用多进程）

vLLM 有两套分布式执行后端（`distributed_executor_backend`）：

- **`MultiprocExecutor`**（`vllm/v1/executor/multiproc_executor.py`）：**单机多卡**，用 Python `multiprocessing` + 共享内存。轻量，但**只能在一台机器内**，且 V1 下**不支持 PP**（断言 `world_size == tensor_parallel_size`，见 [模块 03](03-distributed-parallel/design.md)）。
- **`RayDistributedExecutor`**（`vllm/v1/executor/ray_distributed_executor.py`）：**跨多机** + **支持 PP（流水线并行）**。

所以选 Ray 的两个核心动机：
1. **跨节点**：模型大到一台机器装不下，要把 worker 摊到多台机器——只有 Ray 能统一调度多机 GPU。
2. **流水线并行（PP）**：Ray 的 **Compiled Graph** 能把"PP 各段 worker 的依赖关系"编译成一张高效执行图，自动在段间传中间张量（见 §5）。V1 的 PP 目前**只能走 Ray**。

> 默认在多节点或显式 `--distributed-executor-backend ray` 时启用 Ray；单机多卡默认用更轻的 `MultiprocExecutor`。

---

## 3. vLLM 里的 Ray：关键概念 → 代码映射

### 3.1 起集群 + 预订 GPU：`initialize_ray_cluster` 与 placement group

```python
# vllm/executor/ray_utils.py:269  initialize_ray_cluster(...)
ray.init(address=ray_address, num_gpus=parallel_config.world_size)   # :297 连上/起 Ray 集群
current_placement_group = ray.util.get_current_placement_group()      # :311 拿到/创建 placement group
```
- **placement group** 把 `world_size` 个 **GPU bundle**（每个 `{"GPU": 1}`）预订下来，确保有足够卡、且能控制它们的**节点分布**。
- `_verify_bundles`（`ray_utils.py:165`）会**校验 bundle 落在正确的节点**上（比如同一 TP 组的卡最好在同一台机器，吃 NVLink）。
- `_wait_until_pg_ready`（`ray_utils.py:211`）等资源就绪。

> 怎么读：placement group ＝ "先把 N 张卡圈好、按拓扑摆好位置"，是后面起 worker 的前提。

### 3.2 把 Worker 变成 Ray Actor：`RayWorkerWrapper`

```python
# vllm/executor/ray_utils.py:37
class RayWorkerWrapper(WorkerWrapperBase):
    """Ray wrapper for vllm Worker, allowing Worker to be
    lazily initialized after Ray sets CUDA_VISIBLE_DEVICES."""
```
- 每个 GPU 上的 `Worker`（[模块 00](00-request-lifecycle/design.md) 里"真正跑 forward 的人"）被包成一个 **Ray actor**，由 Ray 调度到某张卡上。
- **懒初始化**很关键：actor 先被 Ray 放到某卡、设好 `CUDA_VISIBLE_DEVICES`，**之后**才真正构造里面的 `Worker`（否则会抢错卡）。`get_node_and_gpu_ids`（`:56`）让每个 actor 上报"我在哪个节点、占了哪些 GPU id"，用来排 world 拓扑。

> 怎么读：把 `RayWorkerWrapper` 理解成"**远程 GPU 上的一个 Worker 替身**"，主控进程通过 `actor.方法.remote(...)` 指挥它。

### 3.3 用 Compiled Graph 编排 PP 流水线（vLLM 用 Ray 最精妙的地方）

朴素做法是主控进程挨个 `ray.get` 每个 worker 的结果——但 PP 下 worker 之间有依赖（第 0 段的输出是第 1 段的输入），来回 `ray.get` 开销大。vLLM 改用 Ray 的 **Compiled Graph（加速图）**：把整个执行依赖**编译成一张静态 DAG**，一次提交、自动流转。

```python
# vllm/executor/ray_distributed_executor.py:556  _compiled_ray_dag(...)
from ray.dag import InputNode, MultiOutputNode
with InputNode() as input_data:                         # :581 图的输入：SchedulerOutput
    outputs = [input_data for _ in self.pp_tp_workers[0]]
    for pp_rank, tp_group in enumerate(self.pp_tp_workers):
        outputs = [
            worker.execute_model_ray.bind(outputs[i])   # :604 把每个 worker 的方法"绑"进图
            for i, worker in enumerate(tp_group)
        ]
        # PP 段间：指定中间张量怎么传（nccl / shm）
        outputs = [o.with_tensor_transport(transport=...) for o in outputs]  # :623
    forward_dag = MultiOutputNode(outputs)              # :627 图的输出：各 rank 的 ModelRunnerOutput
return forward_dag.experimental_compile(...)            # :629 编译成加速图
```

注释里画出的 DAG（`:590-594`，PP=2 TP=4）：
```
SchedulerOutput → worker0 → (SchedulerOutput, IntermediateTensors) → worker4 → ModelRunnerOutput
SchedulerOutput → worker1 → (...) → worker5 → ModelRunnerOutput
SchedulerOutput → worker2 → (...) → worker6 → ModelRunnerOutput
SchedulerOutput → worker3 → (...) → worker7 → ModelRunnerOutput
        └── PP stage 0（4 个 TP rank）──┘   └── PP stage 1 ──┘
```
- **`bind`** ＝ "把这个 actor 方法接进图里"（声明依赖，还没执行）。
- **`with_tensor_transport("nccl"/"shm")`**（`:623`）：PP 段之间的中间张量走 NCCL（跨卡直传）还是共享内存——这是 PP 性能的关键。
- **`experimental_compile`**：把图编译好，之后**每步 forward 只提交一次输入**，Ray 自动驱动整条流水线。

> 怎么读：`InputNode → 一串 bind → MultiOutputNode` 就是"**用数据流图描述 PP 各段怎么串**"，编译后高效重放。这是 vLLM 把 PP 跑得不卡的核心。

### 3.4 一步执行：提交输入、拿回 ObjectRef

```python
# vllm/v1/executor/ray_distributed_executor.py:50
if self.forward_dag is None:
    self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)  # 首次构建图
refs = self.forward_dag.execute(scheduler_output)                    # :53 提交一步，拿回引用
if self.max_concurrent_batches == 1:        # 无 PP：阻塞取结果
    return refs[0].get()                                              # :57
return FutureWrapper(refs[0])               # 有 PP：立即返回 future，让调度器去排下一批  # :61
```
- `execute(scheduler_output)` 把这一步的输入丢进编译图，返回 **ObjectRef**（结果的引用，还没算完）。
- **无 PP**：直接 `.get()` 阻塞等结果。
- **有 PP**：包成 `FutureWrapper` 立刻返回——这样 EngineCore 能**马上调度下一批**塞进流水线填气泡（呼应 [模块 03](03-distributed-parallel/design.md) 的 batch queue 与 [模块 00](00-request-lifecycle/impl.md) 的 T10）。`max_concurrent_batches = pipeline_parallel_size`（`:31`）就是"流水线里同时能有几批在飞"。

---

## 4. V1 与 V0 的 Ray 执行器

V1 的 `RayDistributedExecutor`（`vllm/v1/executor/ray_distributed_executor.py:27`）**直接继承 V0 的实现**，只覆写两点：
- `max_concurrent_batches` 返回 `pipeline_parallel_size`（声明支持 PP 并发）。
- `execute_model` 用编译图 + `FutureWrapper`（上面 §3.4）。

集群初始化、placement group、worker actor、compiled DAG 构建这些**重活都复用 V0 的 `vllm/executor/ray_distributed_executor.py` 与 `ray_utils.py`**。compiled DAG 里靠 `self.use_v1` 分支选择调 `execute_model_ray`（V1）还是 `execute_model_spmd`（V0）（`ray_distributed_executor.py:602-613`）。

---

## 5. 关键环境变量速查（`vllm/envs.py`）

| 变量 | 作用 |
|---|---|
| `VLLM_USE_RAY_COMPILED_DAG` | 是否用 Ray 编译图（PP 时基本必开，`envs.py:50`） |
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | PP 段间张量传输通道：`auto`/`nccl`/`shm`（`ray_distributed_executor.py:566`） |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | 是否让通信与计算重叠 |
| `VLLM_USE_RAY_SPMD_WORKER` | SPMD 模式 worker（`envs.py:49`） |
| `RAY_CGRAPH_get_timeout` | 编译图取结果的超时（vLLM 默认调到 300s，`:577`） |

---

## 6. 怎么读 vLLM 的 Ray 代码（实用建议）

1. **先分清两个后端**：`MultiprocExecutor`（单机、无 PP）vs `RayDistributedExecutor`（多机、有 PP）。看到 Ray 就知道是"多机或 PP"场景。
2. **`RayWorkerWrapper` ＝ 远程 GPU 上的 Worker 替身**；主控通过 `actor.方法.remote()` 指挥，`ray.get` 取结果。
3. **看懂 compiled DAG 那段（§3.3）就懂了一大半**：`InputNode → bind → MultiOutputNode → experimental_compile` 是 PP 编排的骨架。
4. **`.remote()`/`bind()` 都不立即执行**，返回的是引用/图节点；真正算是在 `execute`/`get` 时。
5. Ray 相关概念（actor/ObjectRef/placement group）属 Ray 框架本身，**vLLM 只是使用者**——卡在 Ray API 时查 Ray 官方文档，卡在"vLLM 怎么用它"时回本篇。

> 配套阅读：[模块 03](03-distributed-parallel/design.md)（四种并行 + Executor 抽象）、[PRIMER.md](PRIMER.md)（业务背景）、[PYTHON-PRIMER.md](PYTHON-PRIMER.md)（Python 语法）。
