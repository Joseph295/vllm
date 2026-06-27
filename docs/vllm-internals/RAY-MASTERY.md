# 成为 Ray 专家：源码梳理 + 可执行学习计划

> **这份文档给谁看**：已经会用 Ray（看过 [RAY-PRIMER](RAY-PRIMER.md)、能写 task/actor），想**深入 Ray 内部、读懂它的源码、成为专家**的人。
>
> ⚠️ **重要前提**：Ray 的源码**不在本仓库**（vLLM 只是它的使用者）。下面的目录/文件路径指的是 **`ray-project/ray` 官方仓库**（`master` 分支，随版本会有漂移，以你 clone 下来的实际代码为准）。本文是"地图 + 学习路线"，不是逐行代码注释。
>
> 配套：[RAY-PRIMER](RAY-PRIMER.md)（vLLM 怎么用 Ray）、[模块 03](03-distributed-parallel/design.md)（分布式执行）。

---

## 第一部分：先建立 Ray 的整体心智模型

### 1.1 Ray 是什么（一句话）

Ray 是一个**通用分布式计算框架**：用 `@ray.remote` 把普通 Python 函数（**task**）和类（**actor**）调度到集群里成百上千个进程/机器上执行，靠一个**分布式对象存储**在它们之间高效传数据。上层再叠了 AI 库（Train / Tune / Serve / Data / RLlib）。

### 1.2 三个核心抽象（务必先吃透语义）

| 抽象 | 是什么 | 关键语义 |
|---|---|---|
| **Task** | 无状态远程函数 `f.remote(x)` | 返回 `ObjectRef`（未来值）；由 Ray 调度到任意有资源的 worker 上跑 |
| **Actor** | 有状态远程对象 `C.remote()` | 一个长期存活的进程，方法串行执行（默认）；状态留在该进程内 |
| **Object** | 不可变数据，存在分布式对象存储里 | `ray.put` 放、`ray.get` 取；按 `ObjectRef` 寻址；**ownership** 模型管它的生命周期 |

> 成为专家的标志之一：能清楚说出 **task vs actor 的取舍**、**ObjectRef 的所有权（ownership）与引用计数**、**为什么对象不可变**。

### 1.3 集群里的四类进程（Ray 架构的骨架）

```
┌──────────────────────────── Ray 集群 ────────────────────────────┐
│  Head 节点                          Worker 节点(们)                │
│  ┌─────────────────────┐           ┌─────────────────────┐        │
│  │ GCS (Global Control │           │                     │        │
│  │ Service)：集群元数据  │◄─────────►│  (无 GCS)            │        │
│  │ 节点/actor/资源/job   │           │                     │        │
│  └─────────────────────┘           │                     │        │
│  ┌─────────────────────┐           ┌─────────────────────┐        │
│  │ Raylet：本节点守护进程 │           │ Raylet              │        │
│  │  · 本地资源管理        │           │                     │        │
│  │  · 任务调度(bottom-up) │           │                     │        │
│  │  · Worker 进程池       │           │                     │        │
│  │  · 本地对象管理        │           │                     │        │
│  └─────────┬───────────┘           └─────────┬───────────┘        │
│   ┌────────┴────────┐                ┌────────┴────────┐           │
│   │ Worker 进程       │  ...           │ Worker 进程      │  ...      │
│   │  └ Core Worker   │                │  └ Core Worker  │           │
│   │    (C++ 引擎)     │                │                 │           │
│   │  └ 你的 Python    │                │                 │           │
│   │    task/actor 代码 │                │                 │           │
│   └─────────────────┘                └─────────────────┘           │
│   每个节点本地有一块共享内存对象存储（Plasma，归 object_manager 管）    │
└────────────────────────────────────────────────────────────────────┘
```

- **GCS（Global Control Service）**：集群的"大脑/元数据中心"——节点成员、actor 注册表、资源总账、job 信息。**只在 head 节点**。
- **Raylet**：**每个节点一个**的守护进程。三大职责：本地资源管理 + 任务调度（自底向上 bottom-up scheduler）+ Worker 进程池 + 本地对象管理。
- **Core Worker**：**嵌在每个 worker 进程里的 C++ 引擎**——负责任务提交、引用计数、任务执行、对象存取。你的 Python task/actor 代码就跑在它上面。
- **对象存储（Plasma / object_manager）**：每个节点本地一块**共享内存**对象存储；跨节点取对象时由 object_manager 拉取。

> 一句话串起来：**Core Worker（提交任务/管引用）→ Raylet（调度到哪跑/给资源）→ Worker（执行）；数据走 Object Store；元数据走 GCS。**

---

## 第二部分：源码地图（`ray-project/ray`）

先 clone：`git clone https://github.com/ray-project/ray`。Ray 是 **C++ 核心 + Python/Java/C++ 前端**：性能关键路径在 `src/ray/`（C++），用户 API 与胶水在 `python/ray/`。

### 2.1 C++ 核心：`src/ray/`（性能与分布式逻辑都在这）

| 目录 | 干什么 | 阅读优先级 |
|---|---|---|
| **`core_worker/`** | **Core Worker 引擎**：任务提交（`task_submission/`）、actor 管理（`actor_management/`）、引用计数、任务执行。`core_worker.cc/.h` 是总入口 | ⭐⭐⭐ 先读 |
| **`raylet/`** | **本节点守护进程**：本地任务调度、worker 进程池、资源记账、本地对象管理 | ⭐⭐⭐ |
| **`object_manager/`** | **分布式对象存储**：Plasma 共享内存、跨节点对象拉取、object spilling | ⭐⭐⭐ |
| **`gcs/`** | **全局控制服务**：`gcs_server`、节点/actor/资源/job 表、容错 | ⭐⭐ |
| **`common/`** | 共享类型与配置：`ray_config_def.h`（所有可调参数）、ID 类型、RayObject | ⭐⭐ 常翻 |
| **`rpc/`、`*_rpc_client/`** | 各组件间的 gRPC 框架与客户端（core_worker / raylet / gcs / object_manager 各一套） | ⭐ 按需 |
| **`pubsub/`** | 发布订阅（GCS 通知、引用计数消息等） | ⭐ |
| **`ray_syncer/`** | 节点间状态同步（资源视图等） | ⭐ |
| **`protobuf/`、`flatbuffers/`** | 消息定义（读它能看清各组件交换什么数据） | ⭐ 查协议用 |
| **`observability/`、`stats/`** | 指标、日志、追踪 | ⭐ |
| **`asio/`、`util/`、`internal/`** | 异步 IO、工具函数 | — |

### 2.2 Python 前端：`python/ray/`

| 路径 | 干什么 |
|---|---|
| `python/ray/_private/worker.py` | **用户进程的总控**：`ray.init`、`ray.get/put`、driver 逻辑 |
| `python/ray/_private/services.py` | 启动 GCS / raylet / dashboard 等进程 |
| `python/ray/_raylet.pyx`（Cython） | **Python ↔ C++ Core Worker 的桥**——`@ray.remote` 调用最终落到这里 |
| `python/ray/actor.py`、`remote_function.py` | `@ray.remote` 装饰器、actor handle、`.remote()` 调用 |
| `python/ray/dag/` | **Compiled Graph / aDAG**（`InputNode`/`MultiOutputNode`/`experimental_compile`）——vLLM 的 PP 就靠它（见 [RAY-PRIMER §3.3](RAY-PRIMER.md)） |
| `python/ray/util/placement_group.py` | placement group（vLLM 用它预订 GPU） |
| `python/ray/train|tune|serve|data/`、`rllib/` | 上层 AI 库（成为 Core 专家后按需） |

> **读码切入点建议**：从 Python 侧 `@ray.remote`/`.remote()` 入手，顺着 `_raylet.pyx` 钻进 C++ `core_worker/`——这条"一个 task 怎么被提交并执行"的链路是理解 Ray 的脊梁。

---

## 第三部分：必须吃透的 6 个核心机制

成为专家＝能讲清下面每一个的**完整链路**（建议每个都画一张时序图）：

1. **Task 生命周期**：`f.remote()` → Core Worker 生成 `ObjectRef` 与 TaskSpec → 提交给 Raylet → Raylet 调度（本地够资源就本地起 worker，否则转发/借调）→ worker 执行 → 结果存入本地 object store → 调用方 `ray.get` 取回。（源码：`core_worker/task_submission/`、`raylet/scheduling/`）
2. **Actor 生命周期**：`C.remote()` 在 GCS 注册 actor → 调度一个专属 worker 常驻 → `actor.m.remote()` 把方法调用发到该 worker 串行执行 → actor 失败时的重建策略。（`core_worker/actor_management/`、`gcs/` actor 表）
3. **对象存储 + Ownership + 引用计数**：对象存共享内存（Plasma）；**owner**（创建该 ObjectRef 的 worker）负责追踪它的引用与元数据；引用计数归零才回收；跨节点取对象走 object_manager 拉取；内存不够时 **spilling** 到磁盘。（`object_manager/`、`core_worker/reference_count`；**配合读 Ownership 论文**）
4. **分布式调度（bottom-up）**：Raylet 本地先尝试调度，资源不足才向其他 Raylet/GCS 询问——这是 Ray 高吞吐低延迟的关键设计。（`raylet/scheduling/`）
5. **GCS 与容错**：GCS 存集群关键状态、可做 HA；节点/worker/actor 失败后的检测与恢复。（`gcs/`）
6. **Compiled Graph（aDAG）**：把一组 actor 方法调用的依赖编译成静态图、一次提交反复高效执行、段间张量可走 NCCL/共享内存——**这正是 vLLM 跑 PP 的机制**（[RAY-PRIMER §3.3](RAY-PRIMER.md)）。（`python/ray/dag/`、`experimental` 通道）

---

## 第四部分：可执行学习计划（约 10–12 周，每周 ~8h）

每个阶段都给了 **目标 / 读什么 / 动手做什么 / 验收标准**。验收标准是"能不能讲清/做出来"，不是"读没读完"。

### 阶段 0 · 热身：会用才能谈懂（3–5 天）
- **读**：官方 [Ray Core Quickstart](https://docs.ray.io/en/latest/ray-core/walkthrough.html) + 《Learning Ray》第 2 章。
- **做**：写一个并行程序——N 个 task 并行算 + 1 个 actor 聚合结果；用 `ray.get` 取回。再写一个有状态计数器 actor。
- **验收**：能说清 `f.remote()` 返回的不是结果而是 `ObjectRef`；能解释 task 与 actor 何时各用。

### 阶段 1 · 编程模型彻底吃透（1 周）
- **读**：[Ray Core User Guides](https://docs.ray.io/en/latest/ray-core/user-guide.html) 全部（resources、placement group、runtime env、actor 并发、对象管理、容错语义）。
- **做**：用 placement group 把 actor 钉到指定资源；试 `max_concurrency` 让 actor 并发；用 `ray.put` 共享大对象给多个 task（体会零拷贝）。
- **验收**：能解释 ObjectRef 所有权直觉、`ray.get` 的阻塞语义、actor 默认串行 vs 并发、placement group 的 bundle/策略。

### 阶段 2 · 架构总览：先看全局再钻代码（1 周）
- **读**：[OSDI 2018 论文](https://www.usenix.org/conference/osdi18/presentation/moritz)（**精读**，Ray 的设计纲领）+ [Ray 2.0 Architecture 白皮书](https://docs.ray.io/en/latest/ray-contribute/whitepaper.html)（Google Doc）。辅助：[DeepWiki: ray-project/ray](https://deepwiki.com/ray-project/ray)。
- **做**：**画两张图**——(1) 集群进程拓扑（GCS/Raylet/CoreWorker/ObjectStore）；(2) 一个 task 从 `.remote()` 到结果取回的端到端时序。
- **验收**：能向别人讲清这两张图，并说出"为什么是 bottom-up 调度""为什么对象不可变""GCS 存什么"。

### 阶段 3 · 源码：Core Worker + Task/Actor（2 周）
- **读**：clone 仓库。从 `python/ray/_raylet.pyx` 找到 `.remote()` 的落点，钻进 `src/ray/core_worker/core_worker.cc`、`core_worker/task_submission/`、`core_worker/actor_management/`。配合 [DeepWiki: Core Worker and Task Execution](https://deepwiki.com/ray-project/ray/2.1-core-worker-and-task-execution)。
- **做**：本地 debug 版编译 Ray（`pip install -e` + 源码构建）；在 task 提交路径加日志，实测跟踪一个 `f.remote()` 走了哪些函数。
- **验收**：能在源码里指出"TaskSpec 在哪构造、提交到哪、引用计数在哪 +1/-1"。

### 阶段 4 · 源码：对象存储 + Ownership + 引用计数（2 周）
- **读**：[Ownership 论文（NSDI 2021，PDF）](https://www.usenix.org/system/files/nsdi21-wang.pdf)（Ray 对象系统的理论基础，**精读**）+ `src/ray/object_manager/` + `core_worker` 的 reference counting。
- **做**：写实验观察 object spilling（put 很多大对象撑爆内存看溢写到磁盘）；用 `ray memory` 观察引用；制造一个 ObjectRef 跨节点取，观察 object_manager 拉取。
- **验收**：能讲清 owner 的职责、引用计数何时归零回收、跨节点取对象的链路、spilling 触发条件。

### 阶段 5 · 源码：Raylet 调度 + GCS（2 周）
- **读**：`src/ray/raylet/`（scheduling、worker pool、本地资源记账）+ `src/ray/gcs/`（actor 表、节点管理、资源广播）+ `src/ray/common/ray_config_def.h`（所有可调旋钮）。
- **做**：做调度实验——提交超过本地资源的 task 看它怎么排队/转发；杀掉一个 worker 看 raylet/GCS 如何检测与恢复；改一两个 `ray_config` 参数看行为变化。
- **验收**：能讲清 bottom-up 调度的决策流程、worker 池如何复用、actor 失败重建、GCS 高可用思路。

### 阶段 6 · 高级：Compiled Graph + 容错 + 综合实战（2 周）
- **读**：`python/ray/dag/`、`experimental_compile` 路径、tensor transport（NCCL/shm）。**对照本仓库**：[RAY-PRIMER §3.3](RAY-PRIMER.md) 与 `vllm/executor/ray_distributed_executor.py:556` 的 `_compiled_ray_dag`——看一个真实大项目（vLLM）如何用 aDAG 跑 PP。
- **做（综合项目，二选一）**：
  1. **复刻一个 mini-vLLM-executor**：用 Ray actor + compiled graph 把"两段流水线"串起来，段间传张量，测吞吐。
  2. **给 Ray 提一个改进/修 bug**：在 [good first issue](https://github.com/ray-project/ray/labels/good%20first%20issue) 里挑一个，走通"读相关源码→改→测→提 PR"。
- **验收**：能独立用 aDAG 编排一个有依赖的分布式计算；能讲清 vLLM 为什么用 aDAG 跑 PP、它比朴素 `ray.get` 强在哪。

> **加速建议**：每个源码阶段都配一个"**改一行加日志/断点**"的小实验——读 100 行不如亲手让一个调用在你眼前流过一遍。

---

## 第五部分：权威学习资料清单

**论文（按重要性）**
- ⭐ [Ray: A Distributed Framework for Emerging AI Applications（OSDI 2018）](https://www.usenix.org/conference/osdi18/presentation/moritz) —— 设计纲领，必读。
- ⭐ [Ownership: A Distributed Futures System for Fine-Grained Tasks（NSDI 2021）](https://www.usenix.org/conference/nsdi21/presentation/wang) —— 对象系统/引用计数的理论基础。
- [Exoshuffle（SIGCOMM 2023, arXiv:2203.05072）](https://arxiv.org/abs/2203.05072) —— 数据面/shuffle 的扩展。

**官方文档与白皮书**
- [Ray 文档](https://docs.ray.io/)（Ray Core User Guide 部分对内部语义讲得最细）。
- [Ray 2.0 Architecture 白皮书](https://docs.ray.io/en/latest/ray-contribute/whitepaper.html)（Google Doc，最权威的内部设计总览）。
- [Ray 源码贡献指南](https://docs.ray.io/en/latest/ray-contribute/)（教你怎么本地编译、调试、提 PR）。

**书与导览**
- 《Learning Ray》(Max Pumperla 等, O'Reilly)，[在线版](https://maxpumperla.com/learning_ray/)。
- [DeepWiki: ray-project/ray](https://deepwiki.com/ray-project/ray) —— AI 生成的源码结构导览，配合读源码很省力。

**社区**
- Ray 仓库的 `src/ray/design_docs/`、GitHub Discussions、Ray Slack。

---

## 第六部分：借 vLLM 当实战练习场

你手上正好有一个**深度使用 Ray 的真实大项目**（vLLM）。把它当 Ray 知识的应用题：

| Ray 知识点 | 在 vLLM 里的落点（本仓库真实代码） |
|---|---|
| Actor | `RayWorkerWrapper`（`vllm/executor/ray_utils.py:37`）把每个 GPU Worker 包成 actor |
| Placement Group | `initialize_ray_cluster`（`vllm/executor/ray_utils.py:269`）预订 GPU bundle |
| ObjectRef / `.get()` | `execute_model` 拿 ref 再取（`vllm/v1/executor/ray_distributed_executor.py:53-57`） |
| Compiled Graph (aDAG) | `_compiled_ray_dag`（`vllm/executor/ray_distributed_executor.py:556`）编排 PP |
| 跨进程序列化 | `RayWorkerWrapper` 用 msgspec 编解码（`ray_utils.py:49-51`） |

> 学完阶段 6 再回头读 vLLM 的这几处，你会从"看不懂这堆 Ray API"变成"一眼看穿它在用 aDAG 填 PP 气泡"——这就是从使用者到专家的跨越。

---

> 配套阅读：[RAY-PRIMER.md](RAY-PRIMER.md)（vLLM 怎么用 Ray，先看这个）、[模块 03](03-distributed-parallel/design.md)（分布式并行设计）。
