# 模块 10 · Multimodal 多模态 —— 设计文档

> 范围：图像 / 音频 / 视频等非文本模态如何与文本**统一进入同一个 LLM**——从前端的「merged processor 把多模态展开成占位 token」，到调度器的「encoder budget + encoder cache 双约束」，再到执行层的「encoder 输出按占位范围 scatter 进 decoder 的 input embeddings」，以及 M-RoPE 多模态位置编码。
> 架构：**V1 为主**。本模块是 [模块 00](../00-request-lifecycle/design.md)（请求全生命周期）里多模态分支的系统化深入——00 已点到 Processor 预处理、`mm_input_cache` 镜像、scheduler 的 `_try_schedule_encoder_inputs`、runner 的 `_execute_mm_encoder` / `_gather_mm_embeddings`，这里把它们讲透。

---

## 1. 这个模块解决什么问题

一个支持多模态的 LLM（LLaVA、Qwen-VL、Qwen2-VL……）本质是「**vision/audio encoder + 文本 decoder**」的拼接：图像先过一个 ViT 类编码器变成一串「视觉 token 的 embedding」，再和文本 token 的 embedding 拼在一起喂进 decoder。要把这套东西塞进一个为**纯文本**设计、且追求**高吞吐 continuous batching** 的推理引擎，会撞上几组具体矛盾：

1. **两种 token，一条序列**：decoder 的输入是一串「位置」，其中一部分位置来自文本 token id，另一部分来自图像 encoder 的输出向量。引擎必须让这两类位置在**同一个序列、同一套 KV cache、同一套 attention** 里无缝对齐——decoder 并不知道某个位置「原本是图像」。

2. **占位符与 KV 必须对齐**：prompt 里的图像在 tokenize 后会变成一段**占位 token**（如 `<image>` 重复 N 次）。这段占位 token 的**数量必须精确等于** encoder 输出的向量条数，否则位置错位、KV 写偏，结果全错。N 由图像分辨率 / patch 数动态决定，不能写死。

3. **encoder 计算重、且可复用**：跑一遍 ViT 对一张高分辨率图可能产出上千个视觉 token，计算量堪比一段长 prefill。同一张图（同一个 prompt 前缀）在 chunked prefill 里会被分多步算 decoder，但 encoder **只需要算一次**——算完缓存起来，后续 chunk 直接取。

4. **大张量不该反复过界**：pixel values 动辄几 MB，前端进程（P0）和 EngineCore 进程（P1）之间若每条请求都把原始像素 msgpack 过一遍 ZMQ，IPC 直接成为瓶颈。

5. **位置编码要懂二维结构**：图像是 2D 的，简单地给视觉 token 排 1D 递增位置会丢掉空间结构。Qwen2-VL 等用 **M-RoPE（Multimodal RoPE）** 给文本/图像/视频分配不同语义的 3D 位置 id。

V1 多模态设计的主线，就是围绕「**把 encoder 的重计算拆出来、缓存复用，并让它产出的 embedding 精确落到占位符位置**」来组织数据流与调度。

> 设计动机出处：LLaVA（*Visual Instruction Tuning*, arXiv:2304.08485）确立了「vision encoder → projector → LLM」的主流范式；Qwen-VL / Qwen2-VL（arXiv:2308.12895 / arXiv:2409.12191）引入动态分辨率与 M-RoPE。vLLM 的多模态支持把这套范式落进 continuous batching 引擎，核心代码在 `vllm/multimodal/` 与 `vllm/v1/core/encoder_cache_manager.py`。

---

## 2. 设计目标与约束

- **占位符即真相**：tokenize 阶段产出的占位 token 段，其**长度**（`PlaceholderRange.length`）就是该 mm item 在序列里占据的位置数；encoder 输出条数必须与之匹配。所有后续对齐都以 `PlaceholderRange(offset, length)` 为准。
- **encoder 算一次，缓存复用**：同一 mm item 在 chunked prefill / prefix caching 下不重复跑 encoder；输出存进 `EncoderCacheManager`（按 token 数计容量），消费完即释放。
- **双约束决定 chunk 边界**：一步内能跑多少 encoder，受 **encoder compute budget**（计算上限）与 **encoder cache space**（缓存空间）两个约束；放不下就把 decoder token 的 chunk 截断到 mm item 之前，绝不**部分计算**一个 mm item（encoder 多为双向注意力，必须整块算）。
- **大张量只过界一次**：P0/P1 各持一份按 `mm_hash` 索引的**镜像缓存**；命中后只把 hash 过界，后端凭 hash 还原。
- **统一用 embedding 作 decoder 输入**：多模态模型的 decoder 输入恒为 `inputs_embeds`（而非 token id），使「文本 token 的 embedding」与「视觉 soft token」能在同一张量里拼接。
- **不支持 encoder-decoder 架构**：V1 目前只支持「encoder 输出注入 decoder 输入嵌入」这一类（decoder-only + mm encoder），真正的 cross-attention encoder-decoder（如 Whisper 原生形态）暂未支持（`processor.py:244` 直接 `NotImplementedError`）。

---

## 3. 核心设计思想

### 3.1 Merged processor：把多模态「展开」成占位 token

vLLM 不让模型代码自己去 tokenize 和插占位符，而是在前端用一个 **merged multi-modal processor**（`BaseMultiModalProcessor`，`processing.py:1055`）统一干三件事：

1. **跑 HF processor**：把 `mm_data`（PIL 图像、音频波形……）喂给 HuggingFace 的 `ImageProcessor`/`AudioProcessor`，得到模型真正需要的张量（pixel values、grid_thw 等），打包成 `MultiModalKwargs`。
2. **展开 prompt**：把 prompt 里每个 mm 占位符（如一个 `<image>`）按该 item 实际占多少位置，**替换/扩展**成对应数量的占位 token。这一步通过 `PromptReplacement` / `PromptInsertion`（`processing.py:209/276`）声明「target → 占位 token 序列」的规则，由 `apply()`（`processing.py:1570`）执行。
3. **产出对齐元数据**：扫描展开后的 token 序列，定位每个 mm item 的占位段，产出 `PlaceholderRange(offset, length)`（`inputs.py:112`），并为每个 item 算 `mm_hash`。

最终 `apply()` 返回一个 `MultiModalInputs`，里面 `prompt_token_ids` 已含占位 token、`mm_kwargs` 是 encoder 的输入、`mm_placeholders` 是每个 item 的位置、`mm_hashes` 是身份标识。**关键不变式**：`PlaceholderRange.length`（或其 `is_embed` mask 选中的数量）== encoder 将产出的向量条数。

> 「merged」的含义：把「tokenize 文本」和「处理多模态并对齐占位符」合并到同一个 processor 里一次完成，而不是文本一套、图像一套各自为政——这样占位符长度与 encoder 输出能在同一处保证一致。

### 3.2 占位 token、is_embed mask 与位置对齐

占位段不一定全是「要填 embedding」的位置。某些模型的占位段里混有 `<img_start>`/`<img_break>`/`<img_end>` 这类**结构 token**，它们仍走普通文本 embedding。`PlaceholderRange.is_embed`（`inputs.py:135`）是一个 `(length,)` 的布尔 mask，标出段内**哪些位置真正接收 encoder 向量**。`get_num_embeds()`（`inputs.py:141`）据此算出真正的视觉 token 数。

于是「对齐」分两层：
- **粗对齐**：`offset` / `length` 圈定占位段在序列中的范围（决定 KV cache 怎么分块、encoder 怎么算；KV cache 的 block 分块机制见 [模块 02](../02-paged-attention-kvcache/design.md)）。
- **细对齐**：`is_embed` 在段内挑出要被 encoder 向量覆盖的位置（决定 scatter 时填哪几行）。

### 3.3 Encoder 输出缓存复用（EncoderCacheManager）

encoder 是重计算，且在 chunked prefill 下一个 mm item 可能横跨多个调度步。设计上**把 encoder 计算与 decoder 计算解耦**：

- 调度器维护一个 `EncoderCacheManager`（`encoder_cache_manager.py:15`），**以 token 数为单位**记账可用空间（`num_free_slots`）。
- 某 mm item 被决定要算时，`allocate`（`:33`）扣掉 `length` 个 slot；其 encoder 输出存进 runner 的 `self.encoder_cache[req_id][input_id]`。
- 后续 chunk 需要这段视觉 token 时，直接从缓存取（`_gather_mm_embeddings`），不重算。
- 当该 mm item 的所有位置都已被 decoder「算过」并写进 KV cache 后（`start_pos + length <= num_computed_tokens`），encoder 输出就没用了，`free_encoder_input`（`:43`）回收空间，runner 侧 pop 掉张量（`gpu_model_runner.py:302`）。

注意：**当前 encoder cache 只在单个请求生命周期内复用**（按 `req_id` 索引），尚未做跨请求的视觉 token 共享——`encoder_cache_manager.py:96` 的注释明确说「目前 compute 和 space 用同一份 budget，等做了跨请求 embedding 缓存再拆开」。跨请求复用由 `prefix caching`（见 [模块 02](../02-paged-attention-kvcache/design.md)）在 token 层面间接覆盖（mm_hash 进 token hash）。

### 3.4 双约束决定 chunk 边界

调度一条含图请求时，引擎要同时尊重两个 encoder 侧约束（外加 KV / token budget）：

| 约束 | 含义 | 来源 |
|---|---|---|
| **encoder compute budget** | 这一步最多跑多少 encoder token | `max_num_encoder_input_tokens`（`scheduler.py:110`，由 `compute_encoder_budget` 算）|
| **encoder cache space** | 缓存还能存多少 encoder token | `EncoderCacheManager.num_free_slots` |

`_try_schedule_encoder_inputs`（`scheduler.py:489`）遍历请求的 mm_positions：若某 mm item 的 `length` 既装不进 cache（`can_allocate` 失败）又超 compute budget，就**把本步的 `num_new_tokens` 截断到该 item 起点之前**（`:551-554`），保证这一步只算它前面的 decoder token，等下一步腾出预算再算这个 mm item。这就是「encoder 把 `chunked prefill` 的 chunk 边界往前推」的机制：**decoder 永远不会越过一个还没算 encoder 的 mm item**。这里的 `chunked prefill` 总框架与调度器的 token budget 调度逻辑属 [模块 01](../01-scheduler-batching/design.md)，本模块讲的是它在多模态分支上叠加的 encoder 双约束。

### 3.5 Embedding 合并：把 encoder 输出 scatter 进 decoder 输入

decoder 的输入嵌入是一张 `(num_tokens, hidden)` 的张量。多模态路径下（`gpu_model_runner.py:1014-1034`）：
1. `_execute_mm_encoder` 跑 encoder，输出按 `is_embed` mask scatter 进 `encoder_cache`（`:885`）。
2. `_gather_mm_embeddings` 找出与「本步调度 token 范围」重叠的占位段，从缓存切片取出对应的 encoder 向量（`:917-933`）。
3. `self.model.get_input_embeddings(input_ids, mm_embeds)`（`:1027`）先把所有 token id 过 embedding 层得到文本嵌入，再用 `merge_multimodal_embeddings`（`models/utils.py:441`）**把 mm_embeds 按占位 token id 覆盖到对应行**。

于是 decoder 看到的就是一张「文本 embedding 与视觉 embedding 拼好」的统一张量，attention 一视同仁。注意 V1 刻意让多模态模型**恒用 embedding 输入**（连纯文本步也是，`:1022` 注释），代价是 embedding 层不在 CUDA graph 里，换来 token id 与 soft token 的统一处理。

### 3.6 M-RoPE：多模态的 3D 位置编码

普通 RoPE 给每个位置一个 1D 标量。Qwen2-VL 的 **M-RoPE**（`uses_mrope`，`gpu_model_runner.py:144`）给每个位置一个 **3D 位置 id**（时间 / 高 / 宽三个轴），让视觉 token 的位置能编码 2D 空间结构、视频还能编码时间轴。runner 为此把 `mrope_positions` 做成 `(3, max_num_tokens+1)`（`:212-231`），prompt 部分位置预计算、completion 部分按 `mrope_position_delta` 增量推（`_calc_mrope_positions`，`:701`）。纯文本输入时 3 个轴取相同值，退化为 1D RoPE。

> 出处：Qwen2-VL 论文（arXiv:2409.12191）第 5 页定义了 M-RoPE；vLLM 代码注释（`gpu_model_runner.py:213-220`）直接引用了这篇。

---

## 4. 关键数据结构

| 数据结构 | 定义位置 | 角色 |
|---|---|---|
| `MultiModalDataDict` | `inputs.py:104` | 用户输入：`{"image": [...], "audio": [...]}`，每模态一到多个 item |
| **`MultiModalKwargs`** | `inputs.py:542` | merged processor 产出，encoder 的实际输入（pixel values、grid_thw 等张量），按 modality 分组；`get_items(modality)` 取单 item |
| `MultiModalKwargsItem` | `inputs.py:523` | 单个 mm item 的 kwargs（一张图/一段音频）|
| `MultiModalFieldConfig` | `inputs.py:369` | 声明某个字段如何 batch（batched / flat / shared）——决定多 item 怎么拼/拆 |
| **`PlaceholderRange`** | `inputs.py:112` | `(offset, length, is_embed)`：mm item 在序列中的占位段及其 embedding mask。**对齐的核心** |
| `MultiModalHashDict` | `hasher.py:19` | `{modality: [hash, ...]}`，每个 item 一个 blake3 hash |
| **`EncoderCacheManager`** | `encoder_cache_manager.py:15` | 以 token 数计容量的 encoder 输出缓存簿记（cached/freed/num_free_slots）|
| **`MirroredProcessingCache`** | `mm_input_cache.py:33` | P0/P1 镜像缓存，按 `mm_hash` 决定大张量是否过界 |
| `EngineCoreRequest.{mm_inputs,mm_hashes,mm_placeholders}` | `processor.py:308` | 过界载荷里的三件套：encoder 输入（可为 None）、hash、占位范围 |
| `Request.{mm_inputs,mm_positions,has_encoder_inputs}` | `v1/request.py:58-62` | 后端内部状态机里的 mm 字段；`get_num_encoder_tokens(i)`（`:126`）取某 item token 数 |
| `self.encoder_cache` | `gpu_model_runner.py:158` | runner 侧 `req_id -> {input_id -> encoder_output_tensor}` |

---

## 5. 权衡取舍

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| merged processor 统一展开占位符 | tokenize 与占位符对齐一处完成，长度不变式有单一保证点 | processor 逻辑复杂（match/replace/insert + bind tokenizer），每个新模型要写 `_get_prompt_updates` |
| 占位 token + `is_embed` mask 双层对齐 | 既能描述「占位段」又能在段内排除结构 token | mask 是张量，需随 PlaceholderRange 一路传递、序列化 |
| encoder 输出按 token 数缓存复用 | chunked prefill 下 encoder 只算一次；解耦 encoder/decoder 计算 | 缓存容量是硬约束，反过来限制了 chunk；当前仅请求内复用，未跨请求 |
| 双约束截断 chunk（不部分算 mm item） | 适配 encoder 双向注意力；正确性优先 | 大图可能让整步只能算很少 decoder token，吞吐受限；`disable_chunked_mm_input` 时更严格 |
| 镜像缓存 + hash 过界 | 大像素张量只过界一次，省 IPC 带宽 | P0/P1 缓存须严格同步同容量；hash 碰撞需 blake3 保证可忽略 |
| 多模态恒用 embedding 输入 | 文本 token 与视觉 soft token 统一处理 | embedding 层被排除出 CUDA graph，纯文本步也走 embedding 路径略慢 |
| M-RoPE 3D 位置 | 编码图像 2D / 视频时间结构 | runner 要额外维护 `(3, N)` 位置张量与 delta 推进逻辑 |

---

## 6. 一图速查：多模态数据流（V1）

```
                       用户输入  {"image": <PIL>, prompt: "<image>描述这张图"}
                                 │
   ┌─────────────────────────── 前端进程 P0 ───────────────────────────────────┐
   │ Processor.process_inputs (processor.py:194)                               │
   │   └─ input_preprocessor.preprocess → BaseMultiModalProcessor.apply        │
   │        (processing.py:1570)                                               │
   │        ① HF processor  ─► MultiModalKwargs(pixel_values, grid_thw…)       │
   │        ② PromptReplacement 展开 "<image>" → 占位token × N                  │
   │        ③ 扫描得 PlaceholderRange(offset,length,is_embed) + mm_hash         │
   │   └─ merge_and_sort_multimodal_metadata (utils.py:295) 按 offset 排序      │
   │   └─ MirroredProcessingCache.get_and_update_p0 (mm_input_cache.py:40)      │
   │        命中 hash ⇒ mm_input=None（只过界 hash），未命中⇒缓存并过界张量     │
   │   └─ EngineCoreRequest{prompt_token_ids(含占位), mm_inputs, mm_hashes,     │
   │                        mm_placeholders}                                    │
   └──────────────────────────────┬───────────────────────────────────────────┘
                       ZMQ/msgpack │  (大像素张量仅首次过界)
   ┌──────────────────────────────▼─── EngineCore 进程 P1 ─────────────────────┐
   │ EngineCore.add_request → get_and_update_p1 (mm_input_cache.py:62) 还原张量 │
   │ Request{mm_positions, mm_inputs, has_encoder_inputs}                       │
   │                                                                           │
   │ Scheduler.schedule (scheduler.py:122)                                      │
   │   每条含图请求 → _try_schedule_encoder_inputs (scheduler.py:489)           │
   │     遍历 mm_positions：                                                    │
   │       cache 命中? KV 已算? → skip                                          │
   │       can_allocate? 且 length ≤ encoder_budget?                           │
   │         否 ⇒ num_new_tokens 截断到 mm item 之前（决定 chunk 边界）         │
   │         是 ⇒ allocate + 记入 scheduled_encoder_inputs，扣 budget           │
   │   SchedulerOutput{scheduled_encoder_inputs, free_encoder_input_ids, …}     │
   │                                                                           │
   │ GPUModelRunner.execute_model (gpu_model_runner.py:987)                     │
   │   if is_multimodal_model:                                                  │
   │     _execute_mm_encoder (:828) ─ batch by modality ─► model.get_           │
   │        multimodal_embeddings ─► scatter_mm_placeholders ─► encoder_cache   │
   │     _gather_mm_embeddings (:890) ─ 取与本步token范围重叠的占位段切片        │
   │     get_input_embeddings(input_ids, mm_embeds) (:1027)                     │
   │        ─► merge_multimodal_embeddings 按占位token id覆盖对应行             │
   │     positions = mrope_positions (3D, :1043)                               │
   │   model(inputs_embeds=…) → decoder forward（视觉/文本 token 一视同仁）     │
   │                                                                           │
   │   mm item 全部 token 进 KV 后 → free_encoder_input，下步 free_encoder_     │
   │   input_ids 通知 runner pop 缓存                                           │
   └───────────────────────────────────────────────────────────────────────────┘
```

> 实现层面的逐行调用链、`file:line` 对照与边界情况，见同目录 [`impl.md`](impl.md)。

---

## 7. 设计背后的考量与历史教训

### 7.1 设计背后的考量（深化"为什么"）

1. **为什么把"占位符长度"立为唯一真相、而非让模型代码自己对齐**。多模态最容易错的地方就是"encoder 输出条数 ≠ 占位 token 数"，一旦错位 KV 全偏、结果静默错误。vLLM 否决了"每个模型自己 tokenize + 自己插占位符"的分散方案，改用 merged processor（`processing.py:1055`）把 tokenize 与占位符展开**合并到同一处**，让 `PlaceholderRange.length == encoder 输出条数`这个不变式有**单一保证点**。代价是 processor 逻辑复杂、每个新模型要写 `_get_prompt_updates`，但换来"对齐错误只可能发生在一个地方"。

2. **encoder cache 为何按 token 数计账、且当前只在请求内复用**。encoder 是重计算且在 chunked prefill 下横跨多步，所以必须把 encoder 与 decoder 计算解耦、缓存 encoder 输出。设计上**有意先不做跨请求的视觉 token 共享**：`encoder_cache_manager.py:96` 注释明确说"目前 compute 和 space 共用同一份 budget，等做了跨请求 embedding 缓存再拆开"。这是一个刻意的"先简单后复杂"演进决策——跨请求复用暂由 prefix caching 在 token 层面间接覆盖（mm_hash 进 token hash），避免过早引入跨请求生命周期管理的复杂度。

3. **双约束截断 chunk：正确性优先于吞吐的硬选择**。一个 mm item 的 encoder 多为双向注意力，**必须整块算**，不能像 decoder 那样部分 chunk。于是 `_try_schedule_encoder_inputs`（`scheduler.py:489`）遇到放不下的 mm item 时，宁可把本步 decoder token 截断到该 item 之前，也绝不部分计算。这是"encoder 把 chunk 边界往前推"的根因——一个被否决的替代方案是"允许 encoder 跨步续算"，但那会破坏双向注意力的正确性，所以被放弃。这个演进还派生出 #14361 把 encoder profiling 与 decoder profiling **解耦**，避免大图把显存峰值算爆。

4. **大像素张量"只过界一次"——P0/P1 镜像缓存的动机**。pixel values 动辄几 MB，若每条请求都 msgpack 过 ZMQ，IPC 直接成瓶颈。`MirroredProcessingCache`（`mm_input_cache.py:33`）让 P0/P1 各持一份按 `mm_hash` 索引的镜像缓存，命中后只把 hash 过界、后端凭 hash 还原张量。这要求两侧缓存**严格同容量同步**——后面 §7.2 的 #16273 正是这条约束没守住时暴露的 bug。

5. **多模态恒用 embedding 输入：为统一处理付出 CUDA graph 的代价**。V1 让多模态模型的 decoder 输入恒为 `inputs_embeds`（连纯文本步也是，`gpu_model_runner.py:1022` 注释），这样"文本 token embedding"与"视觉 soft token"能在同一张量里拼接、attention 一视同仁。代价是 embedding 层被排除出 CUDA graph、纯文本步也略慢——这是"统一数据通路"压倒"逐步最优性能"的取舍。

### 7.2 重要 bug 修复（真实、精选）

下列 PR 号均经 `git log`/`git show` 在本仓库实际历史中核对。

| PR | 问题 | 暴露的设计点 / 教训 |
|---|---|---|
| **#13173** [V1][Bugfix] Copy encoder input ids to fix set iteration issue during VLM abort | `EncoderCacheManager.free` 直接遍历 `get_cached_input_ids(request)` 返回的 set，又在循环里 `free_encoder_input` 修改它——"迭代时改集合"在 VLM abort 路径上崩溃 | 一行 `.copy()` 修复。教训：缓存簿记的"释放"操作会原地改动正在迭代的容器，abort（异常路径）最容易把这类隐患引爆；释放语义要对快照迭代 |
| **#12545** [V1][BugFix] Free encoder cache for aborted requests | 请求被 abort 时，它占用的 encoder cache slot 没被回收，长期跑会泄漏 encoder 缓存空间 | encoder cache 以 token 数为硬容量，任何不回收的路径都会让 `num_free_slots` 单调减少直至无法调度新 mm item。教训：cache 的生命周期必须覆盖**所有**终止路径（正常完成 + abort），不能只在 happy path 释放 |
| **#16273** [Bugfix] Avoid transferring cached multi-modal items from P0 to P1 | 已被 P0 缓存命中的 mm item，本应只把 hash 过界，却仍把大张量传给了 P1——镜像缓存"只过界一次"的承诺被违背 | 直接命中 §7.1.4 那条"P0/P1 严格同步"约束。教训：镜像缓存的正确性依赖两侧对"命中即不传张量"的判断完全一致，任一侧判断偏差就退化成每次都过界、IPC 带宽白白浪费 |
| **#12259** [V1][Bugfix] Fix data item ordering in mixed-modality inference | 混合模态（同一 batch 同时有 image+audio 等）推理时，mm 数据项的顺序被打乱，导致 embedding scatter 到错误位置 | mm item 必须按 `offset` 排序后再 scatter（催生 `merge_and_sort_multimodal_metadata`）。教训：当一个序列里有多模态、多 item 时，"占位段在序列中的物理顺序"是 scatter 正确性的前提，不能假设输入顺序即序列顺序 |
| **#16593** [Bugfix] Multi-modal caches not acting like LRU caches | `mm_input_cache` 用的缓存实现并非真正的 LRU，命中不更新访问顺序，热项可能被误驱逐 | 修复触及 `vllm/utils.py` 的 LRUCache 实现本身。教训："缓存"和"LRU 缓存"不是一回事——若命中时不刷新顺序，缓存退化为 FIFO，多模态大张量的命中率会显著下降 |

无"挖不到 bug"的情况：本模块（含 `encoder_cache_manager.py` / `mm_input_cache.py`）的 bug 修复历史相当丰富。
