# 模块 10 · Multimodal 多模态 —— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的多模态数据流：输入预处理（占位符扩展 + hash）→ 调度 encoder budget → 运行 encoder → embedding 合并进 decoder。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名 / 逻辑为准。
> 阅读方式：每步标注 `文件:行号 函数名`，摘录关键代码并解读。

---

## 1. 代码地图

```
多模态预处理 vllm/multimodal/
  ├─ inputs.py        # PlaceholderRange / MultiModalKwargs / MultiModalFieldConfig
  ├─ processing.py    # BaseMultiModalProcessor（merged processor）/ ProcessingCache
  │                   #   PromptReplacement / PromptInsertion / apply()
  ├─ hasher.py        # MultiModalHasher（blake3）—— mm_hash
  ├─ parse.py         # MultiModalDataItems 解析
  ├─ registry.py      # MultiModalRegistry：注册 processor、算 max tokens/budget
  ├─ utils.py         # merge_and_sort_multimodal_metadata / group_mm_inputs_by_modality
  └─ profiling.py     # dummy data 用于显存 profiling

V1 前端输入路径 vllm/v1/engine/
  ├─ processor.py         # Processor.process_inputs：多模态分支
  └─ mm_input_cache.py    # MirroredProcessingCache：get_and_update_p0 / p1（镜像缓存）

V1 调度层 vllm/v1/core/
  ├─ encoder_cache_manager.py  # EncoderCacheManager / compute_encoder_budget
  └─ sched/scheduler.py        # _try_schedule_encoder_inputs / scheduled_encoder_inputs

V1 执行层 vllm/v1/worker/
  └─ gpu_model_runner.py       # _execute_mm_encoder / _gather_mm_embeddings
                               #   get_input_embeddings merge / M-RoPE

模型侧合并 vllm/model_executor/models/
  └─ utils.py                  # merge_multimodal_embeddings（按占位 token id scatter）
```

---

## 2. 端到端调用链

### 阶段 A：前端输入预处理 —— 占位符扩展 + hash（P0 进程）

> **这一步在干嘛**：用户发来「文字 + 图片」，这一步在前端把它们一起处理好：文字 tokenize 成 token id，图片先过 HF 的 image processor 变成模型要的张量、再在 prompt 里把 `<image>` 这种符号**展开成 N 个占位 token**（N = 这张图会产出多少视觉 token），同时记下「每张图占序列哪一段」和「这张图的 hash」。最后还把大像素张量按 hash 决定要不要真的传给后端。

**A1.** `vllm/v1/engine/processor.py:194` `Processor.process_inputs()`
- `:225-230` 调 `input_preprocessor.preprocess(..., return_mm_hashes=self.use_hash)`：tokenize 文本 **并** 对多模态走 merged processor（见 A2）。`use_hash`（`processor.py:54`）在「未禁用 mm 预处理缓存」或「开启 prefix caching」时为真——后者要把 mm hash 喂进 token hash 以支持含图前缀的 prefix 复用。
- `:241-245` `split_enc_dec_inputs`；V1 不支持真正的 encoder-decoder，`:245` 直接 `NotImplementedError`。

**A2.** `vllm/multimodal/processing.py:1570` `BaseMultiModalProcessor.apply()` —— **merged processor 主入口**
```python
def apply(self, prompt, mm_data, hf_processor_mm_kwargs,
          return_mm_hashes=False) -> MultiModalInputs:
```
内部三步：
- **跑 HF processor 并缓存**：`_cached_apply_hf_processor`（`processing.py:1341`）把 `mm_data` 喂 HF 的 image/audio processor，得到 `MultiModalKwargs`（pixel_values、grid_thw 等）；命中 `ProcessingCache`（`processing.py:878`，LRU keyed by hash）则复用。
- **展开 prompt 占位符**：`_apply_prompt_updates`（`processing.py:1459`）按子类声明的 `PromptReplacement`/`PromptInsertion`（`processing.py:276`/`:209`）把 prompt 里的 mm 占位符替换/插入成对应数量的占位 token。
- **定位占位段**：`find_mm_placeholders`（`processing.py:866`）→ `_iter_placeholders`（`processing.py:805`）扫描展开后的 token 序列，产出每个 item 的 `PlaceholderFeaturesInfo`，再 `to_range()`（`processing.py:642`）转成 `PlaceholderRange(offset, length, is_embed)`（`inputs.py:112`）。
- `:1596-1608` 若 `return_mm_hashes`，对每个 item 调 `MultiModalHasher.hash_kwargs(model_id=…, **{modality:item}, **kwargs)`（`hasher.py:70`）算 blake3 hash。

> **占位符长度即对齐真相**：`PlaceholderRange.length`（`inputs.py:132`）== 展开出的占位 token 数 == encoder 将产出的向量条数（细到 `is_embed` mask 选中的位置）。这是后续所有对齐的根。

**A3.** `vllm/multimodal/hasher.py:70` `MultiModalHasher.hash_kwargs()`
```python
hasher = blake3()
for k, v in kwargs.items():
    for k_bytes, v_bytes in cls.item_to_bytes(k, v):
        hasher.update(k_bytes); hasher.update(v_bytes)
return hasher.hexdigest()
```
- `serialize_item`（`:28`）：图像走 `Image.tobytes()`、tensor/ndarray 走 `.tobytes()`，递归展开 list/dict（`item_to_bytes`，`:52`）。hash 含 `model_id` 与 processor kwargs，确保「同图不同模型/不同处理参数」hash 不同。

**A4.** `vllm/v1/engine/processor.py:264-306` —— **多模态分支：拍平 + 排序 + 镜像缓存**
- `:270-277` `merge_and_sort_multimodal_metadata`（`utils.py:295`）把 `{modality: [PlaceholderRange...]}` 字典**按每个 item 的 `offset` 排成单一交错序列**（`utils.py:343` `sort(key=lambda x: x[1].offset)`），返回 `(sorted_item_modalities, sorted_mm_positions, sorted_mm_hashes)`——使「序列里第 k 个 mm item」有确定顺序，与后续 encoder 输出一一对应。
- `:284-300` 把 `MultiModalKwargs` 按上面排好的 modality 顺序**拍平成单 item 列表** `orig_sorted_mm_inputs`（多模态混排时逐个 `get_items(modality)` 取）。
- `:302-304` **镜像缓存过滤**：
```python
if sorted_mm_hashes is not None:
    sorted_mm_inputs = self.mm_input_cache_client.get_and_update_p0(
        orig_sorted_mm_inputs, sorted_mm_hashes)
```

**A5.** `vllm/v1/engine/mm_input_cache.py:40` `MirroredProcessingCache.get_and_update_p0()`（P0 侧）
```python
for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
    if self.mm_cache.get(mm_hash) is not None:
        mm_input = None          # 已缓存 → 置 None，不再过界大张量
    else:
        self.mm_cache[mm_hash] = mm_input
    full_mm_inputs.append(mm_input)
```
- 命中则把该 item 的张量替换为 `None`，**只让 hash 过界**；未命中则存入本地缓存并保留张量过界一次。

**A6.** `processor.py:308-319` 构造 `EngineCoreRequest`，带 `mm_inputs`（可含 None）、`mm_hashes`、`mm_placeholders`（=`sorted_mm_positions`）。随后经 ZMQ/msgpack 送入 EngineCore 进程（见 [模块 00](../00-request-lifecycle/impl.md) 阶段 C）。

---

### 阶段 B：后端还原 —— P1 镜像缓存

> **这一步在干嘛**：请求跨进程到了后端（EngineCore）。如果前端为了省 IPC 把某张图的张量置成了 None（只传了 hash），后端就凭这个 hash 从自己那份「镜像缓存」里把张量取回来，补回 `Request` 里——这样后端拿到的就是完整的 mm 输入了。

**B1.** `vllm/v1/engine/mm_input_cache.py:62` `MirroredProcessingCache.get_and_update_p1()`（P1 侧，由 `EngineCore.add_request` 调用，见 [模块 00](../00-request-lifecycle/impl.md) 阶段 D2）
```python
for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
    if mm_input is None:
        mm_input = self.mm_cache[mm_hash]   # 凭 hash 从本地镜像还原
    else:
        self.mm_cache[mm_hash] = mm_input   # 首次见到则缓存
    full_mm_inputs.append(mm_input)
```
- P0/P1 **同容量同 hash** 是镜像不变式（`mm_input_cache.py:28` 注释）；正因 P0 命中即置 None，P1 此处必有对应缓存可还原。还原后落进 `Request.mm_inputs`（`v1/request.py:59`）。

`Request` 侧 mm 字段（`v1/request.py:58-65`）：`mm_positions`、`mm_inputs`、`mm_hashes`、`has_encoder_inputs = num_encoder_inputs > 0`；`get_num_encoder_tokens(i)`（`:126`）返回 `mm_positions[i].length`。

---

### 阶段 C：调度 —— encoder budget + encoder cache 双约束

> **这一步在干嘛**：调度器在决定「这一步给这条请求算多少 token」时，多模态请求还要额外问两件事——这一步能不能跑得起这张图的 encoder（**compute 预算**够不够）、缓存里还放不放得下它的输出（**cache 空间**够不够）。如果放不下，就把这一步的 token 截断到这张图**之前**，让这步只算图前面的文本，等下一步腾出预算再整张算这张图。因为图的 encoder 是双向注意力、必须整块算，绝不能算半张。

> 本阶段叠加在调度器的 `chunked prefill` / token budget 主框架之上（scheduler 主干、`num_new_tokens` 与 FCFS 调度见 [模块 01](../01-scheduler-batching/design.md)），这里只讲多模态分支额外引入的 encoder compute / cache 双约束。

**C0.** 预算初始化：`vllm/v1/core/sched/scheduler.py:101` `compute_encoder_budget(...)`（`encoder_cache_manager.py:67`）算出 `(encoder_compute_budget, encoder_cache_size)`：
- `encoder_cache_manager.py:144-147`：两者各取 `max(scheduler_config 配置, 单 item 最大 token 数)`，保证「至少能装下一个最大的 mm item」。
- `scheduler.py:110` `self.max_num_encoder_input_tokens = encoder_compute_budget`；`:114` `self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)`。
- 纯文本模型时两者为 0（`encoder_cache_manager.py:87`），encoder cache 不初始化。

**C1.** `scheduler.py:152` `schedule()` 起始：`encoder_budget = self.max_num_encoder_input_tokens`（每步的可用 compute 预算）。

**C2.** RUNNING 队列调度时（`scheduler.py:177-194`）：
```python
if request.has_encoder_inputs:
    (encoder_inputs_to_schedule, num_new_tokens,
     new_encoder_budget) = self._try_schedule_encoder_inputs(
         request, request.num_computed_tokens, num_new_tokens, encoder_budget)
    if num_new_tokens == 0:
        req_index += 1; continue   # encoder 受限 → 跳过这条（故意放松 FCFS，:186-190）
```

**C3.** `scheduler.py:489` `_try_schedule_encoder_inputs()` —— **决定 chunk 边界的核心**
```python
for i, pos_info in enumerate(mm_positions):
    start_pos = pos_info.offset
    num_encoder_tokens = pos_info.length
    if start_pos >= num_computed_tokens + num_new_tokens: break   # :523 还没到这个item
    if start_pos + num_encoder_tokens <= num_computed_tokens: continue  # :526 已进KV
    if self.encoder_cache_manager.has_cache(request, i): continue       # :531 已缓存
    # disable_chunked_mm_input：禁止部分调度一个 mm item
    if (self.scheduler_config.disable_chunked_mm_input
            and num_computed_tokens < start_pos
            and (num_computed_tokens + num_new_tokens) < (start_pos + num_encoder_tokens)):
        num_new_tokens = start_pos - num_computed_tokens; break        # :538-543
    # 双约束：cache 空间 OR compute 预算
    if (not self.encoder_cache_manager.can_allocate(request, i)
            or num_encoder_tokens > encoder_budget):                   # :545-546
        if num_computed_tokens < start_pos:
            num_new_tokens = start_pos - num_computed_tokens           # :554 截到 item 前
        else:
            num_new_tokens = 0                                         # :560 prefix 命中越过，本步算不了
        break
    encoder_budget -= num_encoder_tokens                              # :563 扣 compute 预算
    encoder_inputs_to_schedule.append(i)                             # :564
return encoder_inputs_to_schedule, num_new_tokens, encoder_budget
```
解读：
- **两个约束**`can_allocate`（cache 空间，`encoder_cache_manager.py:29`：`length <= num_free_slots`）与 `num_encoder_tokens > encoder_budget`（compute 预算）任一不满足，就把 `num_new_tokens` 截断到该 mm item 起点之前（`:554`），**保证 decoder 不越过未算 encoder 的 mm item**。
- 注释 `:548-550` 点明：encoder 输入必须**整块算**（双向注意力），所以不做部分调度——这正是 chunk 边界被 encoder「往前推」的原因。
- `:555-560` 的边界：prefix caching 让 `num_computed_tokens` 越过了 `start_pos`、但该 item 的 encoder 输出又不在缓存里，此时这步一个 token 也调不了（`num_new_tokens=0`）。

**C4.** 调度成功后分配缓存（RUNNING：`scheduler.py:254-260`；WAITING：`:367-373`）：
```python
if encoder_inputs_to_schedule:
    scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
    for i in encoder_inputs_to_schedule:
        self.encoder_cache_manager.allocate(request, i)   # 扣 num_free_slots
    encoder_budget = new_encoder_budget
```
WAITING 队列对应 `scheduler.py:319-330`（与 C2 同构，但 `num_new_tokens==0` 时 `break` 而非 continue，因为新请求按 FCFS）。

**C5.** 打包进 `SchedulerOutput`（`scheduler.py:428-444`）：
- `:434` `scheduled_encoder_inputs`（`output.py` 中 `dict[str, list[int]]`）—— 哪些请求的哪些 mm item 要在这步跑 encoder。
- `:441` `free_encoder_input_ids = self.encoder_cache_manager.get_freed_ids()` —— 上一步标记可释放的 encoder 缓存，下发给 runner。

**C6.** 回填阶段释放缓存：`scheduler.py:613-625`（在 `update_from_output` 内）
```python
for input_id in list(cached_encoder_input_ids):
    start_pos = request.mm_positions[input_id].offset
    num_tokens = request.mm_positions[input_id].length
    if start_pos + num_tokens <= request.num_computed_tokens:   # mm item 全进 KV
        self.encoder_cache_manager.free_encoder_input(request, input_id)
```
- 一旦某 mm item 的所有占位位置都被 decoder 算过（写进 KV），encoder 输出就无用，`free_encoder_input`（`encoder_cache_manager.py:43`）归还 slot 并记入 `freed`，下一步通过 `free_encoder_input_ids` 通知 runner pop。请求结束时 `_free_request`（`scheduler.py:735`）调 `encoder_cache_manager.free` 清空。

---

### 阶段 D：执行 —— 跑 encoder + scatter 进 decoder（GPU）

> **这一步在干嘛**：到 GPU 上真正干活了。先把这一步要算的图喂给 vision encoder（ViT 之类）跑出 embedding 存进缓存；再从缓存里**切出本步 token 窗口覆盖到的那段视觉 embedding**；然后把所有 token id 过 embedding 层得到文本 embedding，并把视觉 embedding**按占位 token id 覆盖到对应行**——拼成一张「文本 + 视觉」混合的输入嵌入喂给 decoder。Qwen2-VL 这类还要算 M-RoPE 的 3D 位置。

入口：`gpu_model_runner.py:1014` `if self.is_multimodal_model:`（`:103` 设置）。

**D1.** `gpu_model_runner.py:828` `_execute_mm_encoder()` —— 跑 encoder 并缓存
```python
scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
if not scheduled_encoder_inputs: return
# 收集本步要算的 (req_id, input_id, pos_info)
for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
    for mm_input_id in encoder_input_ids:
        mm_inputs.append(req_state.mm_inputs[mm_input_id])             # :840
        req_ids_pos.append((req_id, mm_input_id, req_state.mm_positions[mm_input_id]))
grouped = group_mm_inputs_by_modality(mm_inputs)                       # :851 同模态才能 batch
for g in grouped:
    batched = MultiModalKwargs.as_kwargs(MultiModalKwargs.batch(g), device=self.device)
    curr_group_outputs = self.model.get_multimodal_embeddings(**batched)   # :866 跑 ViT/encoder
    sanity_check_mm_encoder_outputs(curr_group_outputs, expected_num_items=len(g))  # :869
    for output in curr_group_outputs: encoder_outputs.append(output)
# 缓存
for (req_id, input_id, pos_info), output in zip(req_ids_pos, encoder_outputs):
    self.encoder_cache.setdefault(req_id, {})[input_id] = scatter_mm_placeholders(
        output, is_embed=pos_info.is_embed)                           # :885
```
- `group_mm_inputs_by_modality`（`utils.py:354`）：只把**连续同模态**的 item 拼成一批跑（混模态会拆开以保序，`:847` FIXME 注释承认这是 hack）。
- `scatter_mm_placeholders`（`:885`）按 `is_embed` mask 把 encoder 输出**铺进占位段大小的张量**——段内非 embed 位置留空，使缓存张量的行号与占位段位置一一对齐。

**D2.** `gpu_model_runner.py:890` `_gather_mm_embeddings()` —— 取与本步 token 范围重叠的占位段
```python
for req_id in self.input_batch.req_ids:
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
    num_computed_tokens = req_state.num_computed_tokens
    for i, pos_info in enumerate(req_state.mm_positions):
        start_pos = pos_info.offset; num_encoder_tokens = pos_info.length
        if start_pos >= num_computed_tokens + num_scheduled_tokens: break    # :909 不重叠（在后）
        if start_pos + num_encoder_tokens <= num_computed_tokens: continue   # :912 不重叠（已过）
        start_idx = max(num_computed_tokens - start_pos, 0)                  # :917
        end_idx = min(num_computed_tokens - start_pos + num_scheduled_tokens,
                      num_encoder_tokens)                                    # :918
        encoder_output = self.encoder_cache[req_id][i]                       # :924
        if (is_embed := pos_info.is_embed) is not None:
            is_embed = is_embed[start_idx:end_idx]                           # :927
        mm_embeds.append(gather_mm_placeholders(encoder_output[start_idx:end_idx],
                                                is_embed=is_embed))          # :929
return mm_embeds
```
- 核心是**两个区间求交**：本步调度的 token 范围 `[num_computed_tokens, +num_scheduled_tokens)` 与占位段 `[start_pos, +length)`。`start_idx`/`end_idx` 把交集换算成 encoder_output 内的切片下标——这就是 chunked prefill 下「一个 mm item 跨多步、每步只取自己那段视觉 token」的实现。
- `gather_mm_placeholders` 用 `is_embed` 过滤掉段内非 embed 行，只取真正要注入的视觉向量。

**D3.** `gpu_model_runner.py:1021-1034` —— merge 进 input embeddings
```python
input_ids = self.input_ids[:num_scheduled_tokens]
if mm_embeds:
    inputs_embeds = self.model.get_input_embeddings(input_ids, mm_embeds)   # :1027
else:
    inputs_embeds = self.model.get_input_embeddings(input_ids)
self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)             # :1032 → 持久 buffer
inputs_embeds = self.inputs_embeds[:num_input_tokens]
input_ids = None
```
- 模型的 `get_input_embeddings` 内部先把 token id 过 embedding 层，再调 `merge_multimodal_embeddings`（`model_executor/models/utils.py:441`）**按占位 token id 把 `mm_embeds` 覆盖到对应行**（`input_ids == placeholder_token_id` 处替换，`:483`）。占位 token id 的顺序必须与 mm_embeds 顺序一致（`:454` 注释强调）。
- `:1022` 注释：多模态模型**恒用 embedding 输入**（连纯文本步），以统一 token id 与 soft token——代价是 embedding 层不进 CUDA graph（对比纯文本路径 `:1040` 直接用 token id）。

**D4.** M-RoPE 位置：`gpu_model_runner.py:1042-1045`
```python
if self.uses_mrope:
    positions = self.mrope_positions[:, :num_input_tokens]   # 3D 位置
else:
    positions = self.positions[:num_input_tokens]
```
- `mrope_positions` 张量 `(3, max_num_tokens+1)` 在 `:212-231` 分配（额外一个 dummy 位置使其非连续以兼容 torch.compile，`:214` 注释）。
- 每条请求的 mrope 位置预计算：`:351-377` 从 `image_grid_thw`/`video_grid_thw`/`second_per_grid_ts` 调 `MRotaryEmbedding.get_input_positions_tensor`，存进 `req.mrope_positions` + `mrope_position_delta`。
- 每步组 batch：`_calc_mrope_positions`（`:701-751`）prompt 段拷预计算值、completion 段用 delta 在线推（`get_next_input_positions_tensor`）；`:561-565` 拷到 GPU。

**D5.** decoder forward：`gpu_model_runner.py:1063-1068`（见 [模块 00](../00-request-lifecycle/impl.md) 阶段 F）`self.model(input_ids=None, positions=positions, inputs_embeds=inputs_embeds)`——视觉/文本 token 在同一 attention 里一视同仁。

**D6.** 释放 runner 侧缓存：`gpu_model_runner.py:302-307`
```python
for req_id, input_id in scheduler_output.free_encoder_input_ids:
    encoder_outputs = self.encoder_cache.get(req_id)
    if encoder_outputs is not None:
        encoder_outputs.pop(input_id, None)
        if not encoder_outputs: self.encoder_cache.pop(req_id, None)
```
请求结束时 `:286-288` 按 `finished_req_ids` 整体 pop。

---

## 3. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · 占位符长度是唯一对齐真相，is_embed 做段内二次过滤
- **代码**：`PlaceholderRange(offset, length, is_embed)`（`inputs.py:112-145`），`get_num_embeds`（`:141`）。
- **精妙之处**：调度用 `length` 圈定占位段（决定 KV 分块、encoder budget），但段内可能混有 `<img_start>`/`<img_break>` 等仍走文本 embedding 的结构 token。`is_embed` 这个 `(length,)` 布尔 mask 在 scatter（`gpu_model_runner.py:885`）和 gather（`:927`）两端都参与，确保 encoder 向量只落在「真·视觉位置」。粗对齐与细对齐分离，使「占位段连续」与「embedding 稀疏」两件事解耦。

### T2 · merge_and_sort 把多模态拍平成单一有序序列
- **代码**：`utils.py:295` `merge_and_sort_multimodal_metadata`，按 `offset` 排序（`:343`）。
- **精妙之处**：一个 prompt 可能图、音频交错。后端、调度、runner 全部假定「第 k 个 mm item」有确定顺序且与 encoder 输出第 k 条对应。在前端就按序列位置 `offset` 排好序、并把 `MultiModalKwargs` 同步拍平（`processor.py:284-300`），下游所有 `enumerate(mm_positions)` 才能与 `mm_inputs[i]` 一一对应，不必再带 modality 标签去对。

### T3 · 镜像缓存：大像素张量全程只过界一次
- **代码**：P0 `mm_input_cache.py:40` `get_and_update_p0`（命中即置 `None`），P1 `:62` `get_and_update_p1`（凭 hash 还原）。
- **精妙之处**：两进程各持一份**同容量、同 hash key** 的 LRU。P0 一旦发现某 hash 已缓存，就把该 item 的张量替换成 `None` 只发 hash；P1 收到 `None` 必能从自己的镜像里取回。由此「同一张图的第二次请求」零张量过界。镜像不变式靠「两边都用 `VLLM_MM_INPUT_CACHE_GIB`」（`mm_input_cache.py:28`）保证。这与 [模块 00](../00-request-lifecycle/impl.md) 的「prompt 文本置 None」是同源思想：后端用不到 / 已有的，就不过界。

### T4 · encoder 必须整块算，于是它「往前推」chunk 边界
- **代码**：`scheduler.py:545-561`，注释 `:548-550`。
- **精妙之处**：encoder（ViT）通常是双向注意力，无法只算半个 mm item。所以当一个 mm item 既装不进 cache 又超 compute budget 时，调度器不是去「算半张图」，而是把本步 decoder token 截断到这张图**之前**（`num_new_tokens = start_pos - num_computed_tokens`）。这把「chunked prefill 的 chunk 边界」和「encoder 能否整块装下」绑定了——decoder 永远不会跑到一段没有 encoder 输出的占位 token 上。

### T5 · 双 budget：compute 与 space 现在共用、但已为分家留口
- **代码**：`encoder_cache_manager.py:144-147`（两者分别 max）；`scheduler.py:96-101` 注释 "we use the same budget for both compute and space"。
- **精妙之处**：`can_allocate`（空间）与 `num_encoder_tokens > encoder_budget`（计算）是**两个独立检查**（`scheduler.py:545-546`），尽管当前两个 budget 数值常相等。注释明说等将来做「跨请求 embedding 缓存」时，space 会远大于 compute（缓存可以攒很多但每步只能算一点），届时两者分离。API 已经按双约束写好，改的只是数值来源。

### T6 · encoder 受限时故意放松 FCFS
- **代码**：`scheduler.py:183-191`，注释 `:186-190`。
- **精妙之处**：RUNNING 队列里若高优请求被 encoder budget 卡住（`num_new_tokens==0`），用 `continue` 跳过它去调度后面的请求，而非 `break` 整步停摆。这是**有意**违反严格 FCFS：与其让一张大图把整个 GPU 步空着，不如让别的请求先填上预算。WAITING 队列则保持 `break`（`:325`）——新请求按入队序，不抢跑。

### T7 · 缓存释放看「是否已进 KV」，而非「是否消费过」
- **代码**：`scheduler.py:613-625`，判据 `start_pos + length <= num_computed_tokens`。
- **精妙之处**：encoder 输出能丢弃的真正条件，是这个 mm item 的**所有占位位置都已被 decoder 算过并写进 KV**——之后 attention 只读 KV，不再需要原始视觉 embedding。所以释放判据是位置区间被 `num_computed_tokens` 完全覆盖，而不是「gather 过一次」（chunked prefill 下会 gather 多次）。释放分两段：scheduler 标记 freed（`encoder_cache_manager.py:43`）→ 下步 `free_encoder_input_ids` 下发 → runner pop（`gpu_model_runner.py:302`）。

### T8 · 区间求交把 encoder 输出切到「本步该用的那段」
- **代码**：`gpu_model_runner.py:917-920`。
- **精妙之处**：一个 1000-token 的大图在 chunked prefill 下跨 N 步，每步只算其中一段。`start_idx = max(num_computed_tokens - start_pos, 0)`、`end_idx = min(... + num_scheduled_tokens, length)` 正是「本步 token 窗口」与「占位段」两区间的交集，换算成 encoder_output 内的切片下标。同一套区间逻辑在调度侧（`scheduler.py:520-529`）也出现，两端用同一几何判断，保证调度决定与执行取数一致。

### T9 · 只有连续同模态才 batch，混模态拆开保序
- **代码**：`gpu_model_runner.py:851` `group_mm_inputs_by_modality`，FIXME 注释 `:847`。
- **精妙之处**：把多个 item 拼一批跑 encoder 能提效，但不同模态（图 vs 音频）encoder 不同、且必须保持 item 在序列里的顺序。于是按「连续同模态」分组，组内 batch、组间顺序执行。注释坦承这是 hack（正解应是「encoder 输出后重排」），但在常见「全图」或「图在前音频在后」场景已够用。

### T10 · 多模态恒用 embedding 输入，牺牲一点 graph 换统一
- **代码**：`gpu_model_runner.py:1021-1034`，对比纯文本 `:1035-1041`，注释 `:1022` 与 `:1037-1039`。
- **精妙之处**：纯文本模型用 token id 作输入（embedding 层在 CUDA graph 内，更快）；多模态模型**即使这一步全是文本** token，也走 `get_input_embeddings` 把它变成 embedding 再拼视觉向量。原因：token id 和「soft token（视觉 embedding）」无法在同一个 id 张量里共存，只能在 embedding 空间合并。代价是 embedding 层被排除出 graph——这是一个被显式注释承认的取舍。

### T11 · scatter 先铺满占位段，gather 再按 is_embed 取
- **代码**：`_execute_mm_encoder` 末尾 `scatter_mm_placeholders`（`:885`）、`_gather_mm_embeddings` 里 `gather_mm_placeholders`（`:929`）。
- **精妙之处**：缓存里存的不是「紧凑的视觉向量」，而是**已按 is_embed 铺进占位段形状**的张量。这样 gather 时只需用占位段坐标 `[start_idx:end_idx]` 做普通切片，再过一遍 is_embed 取真正的 embedding 行——切片下标与序列位置天然对齐，无需额外索引映射。scatter/gather 成对，把「稀疏 embedding 在连续占位段里的位置」这件麻烦事固化在缓存形状里。

### T12 · M-RoPE 用「prompt 预算 + completion 增量」两段拼位置
- **代码**：`_calc_mrope_positions`（`gpu_model_runner.py:701-751`）。
- **精妙之处**：M-RoPE 的 3D 位置里，prompt 段（含图像的 2D/时间结构）依赖 grid 信息、计算复杂，故**整段预计算**存在 `req.mrope_positions`；生成段（纯文本，3 轴同步递增）只需一个 `mrope_position_delta` 在线外推（`get_next_input_positions_tensor`）。chunked prefill 下用 `prompt_part_len`/`completion_part_len` 切分本步窗口，分别从预算缓冲拷贝、或在线生成。预计算重的、外推轻的，精确踩在「图像位置贵、文本位置便宜」上。

### T13 · mrope_positions 故意多一个 dummy 列以兼容 torch.compile
- **代码**：`gpu_model_runner.py:212-218`，张量 shape `(3, max_num_tokens+1)`，注释 `:214`。
- **精妙之处**：`+1` 那一列纯粹是为了让张量**内存非连续**，从而绕过 torch.compile/CUDA graph 对某些连续性假设的处理。一个看似浪费一列的设计，实为编译兼容的 workaround——不读注释会以为是 off-by-one。

### T14 · hash 把模型与处理参数都吃进去
- **代码**：`hasher.py:70` `hash_kwargs(model_id=…, **{modality:item}, **hf_processor_mm_kwargs)`（`processing.py:1601`）。
- **精妙之处**：mm_hash 不只是「图像内容」的 hash，还含 `model_id` 与 HF processor kwargs。于是「同一张图，喂给不同模型 / 用不同分辨率参数」会得到不同 hash——既防止镜像缓存跨模型误命中，也让含图 prefix 在 prefix caching 里只在「图+参数完全一致」时复用。blake3 over `tobytes()`（`:35/:43`）对大像素也足够快。

### T15 · 纯文本模型短路，encoder cache 干脆不建
- **代码**：`encoder_cache_manager.py:87` `if not is_multimodal_model: return 0, 0`；`scheduler.py` 据此建零容量 cache。
- **精妙之处**：所有 encoder 相关检查都用 `request.has_encoder_inputs`（`v1/request.py:62`）门控，纯文本请求直接 `encoder_inputs_to_schedule = None`（`scheduler.py:193`），零开销。多模态模型若用户用 `limit_mm_per_prompt` 关掉所有非文本模态，`compute_encoder_budget` 也会返回 0 并打 warning（`encoder_cache_manager.py:126-131`），退化成纯文本路径。

---

## 4. 关键边界情况

| 情况 | 代码位置 | 行为 |
|---|---|---|
| mm item 超 cache + budget | `scheduler.py:545-554` | 截断 `num_new_tokens` 到 item 起点前，本步只算前面的 decoder token |
| prefix 命中越过 mm item 但其 encoder 未缓存 | `scheduler.py:555-560` | `num_new_tokens = 0`，本步该请求一个 token 都调不了 |
| `disable_chunked_mm_input` 且 chunk 只能盖半个 item | `scheduler.py:538-543` | 回退到 item 前，整块下次算 |
| 单 item 比 `max_num_batched_tokens` 还大且禁 chunk | `encoder_cache_manager.py:136-142` | 启动即 `ValueError`，要求调大 batch token |
| 高优请求被 encoder 卡住 | `scheduler.py:183-191` | RUNNING 队列 `continue` 跳过（放松 FCFS）；WAITING 队列 `break` |
| 镜像缓存禁用（`disable_mm_preprocessor_cache` 且无 prefix cache）| `processor.py:305-306` / `mm_input_cache.py:47` | 不走缓存，所有张量直接过界 |
| mm item 全进 KV | `scheduler.py:621-625` → `gpu_model_runner.py:302` | scheduler 标记 freed，runner pop encoder 缓存张量 |
| 本步无 encoder 要算 | `gpu_model_runner.py:830` | `_execute_mm_encoder` 直接 return |
| mm 占位段与本步 token 窗口不重叠 | `gpu_model_runner.py:909/912` | `break`/`continue` 跳过，不取该 item |
| 纯文本模型 / 全模态被禁 | `encoder_cache_manager.py:87/126` | budget=0，不建 encoder cache，走纯文本路径 |
| encoder-decoder 架构（cross-attn） | `processor.py:244-245` | `NotImplementedError`，V1 暂不支持 |
| 纯文本步（多模态模型）| `gpu_model_runner.py:1029-1030` | `mm_embeds` 为空，仍走 `get_input_embeddings(input_ids)` 用 embedding 输入 |

---

## 5. 一图速查：多模态调用链主干

```
[前端 P0] processor.process_inputs :194
   └─ input_preprocessor.preprocess ─► BaseMultiModalProcessor.apply :1570
        ① _cached_apply_hf_processor :1341  ─► MultiModalKwargs
        ② _apply_prompt_updates :1459  (PromptReplacement :276) ─► 占位token扩展
        ③ find_mm_placeholders :866 ─► PlaceholderRange(offset,length,is_embed)
        ④ hash_kwargs :70 (blake3) ─► mm_hashes
   └─ merge_and_sort_multimodal_metadata :295  按 offset 排序拍平
   └─ MirroredProcessingCache.get_and_update_p0 :40  命中hash⇒张量置None
   └─ EngineCoreRequest{prompt_token_ids(含占位), mm_inputs, mm_hashes, mm_placeholders}
   ▼ ───────────── ZMQ/msgpack（大张量仅首次过界）─────────────
[后端 P1] EngineCore.add_request ─► get_and_update_p1 :62  凭hash还原张量
   └─ Request{mm_positions, mm_inputs, has_encoder_inputs}

   Scheduler.schedule :122
     └─ _try_schedule_encoder_inputs :489  (双约束 :545-546)
          can_allocate? length≤encoder_budget?  否⇒chunk截到item前 :554
          是⇒allocate :259/372 + scheduled_encoder_inputs[req]=ids
     └─ SchedulerOutput{scheduled_encoder_inputs :434, free_encoder_input_ids :441}
     └─ update_from_output: mm item 全进KV ⇒ free_encoder_input :624

   GPUModelRunner.execute_model :987   (is_multimodal_model :1014)
     └─ _execute_mm_encoder :828
          group_mm_inputs_by_modality :851 ─► model.get_multimodal_embeddings :866
          ─► scatter_mm_placeholders :885 ─► encoder_cache[req][id]
     └─ _gather_mm_embeddings :890
          区间求交 :917-920 ─► encoder_cache 切片 ─► gather_mm_placeholders :929 ─► mm_embeds
     └─ get_input_embeddings(input_ids, mm_embeds) :1027
          ─► merge_multimodal_embeddings (models/utils.py:441) 按占位token id覆盖
     └─ positions = mrope_positions(3D) :1043  (_calc_mrope_positions :701)
     └─ model(inputs_embeds=…) :1063  decoder forward（视觉/文本 token 同等对待）
     └─ free_encoder_input_ids ⇒ encoder_cache.pop :302
```
