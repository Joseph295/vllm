# 模块 04 · PD 分离（Prefill-Decode Disaggregation）—— 实现文档

> 与 [`design.md`](design.md) 配套。本文给出**可逐行对照代码**的 KV 跨实例传输调用链。所有 `file:line` 基于仓库当前版本，行号若与本地略有出入，以函数名/逻辑为准。
>
> **再次强调架构现状**：这套 KV 传输代码挂在 **V0 worker**（`vllm/worker/model_runner.py`）上。`vllm/v1/` 下无任何引用，且 `--kv-transfer-config` 会触发 V1→V0 回退（`vllm/engine/arg_utils.py:1526-1530`）。整套机制标注为 experimental，且仅支持 1P1D。下文调用链全部基于 V0 路径。

---

## 1. 代码地图

```
配置与入口
  vllm/config.py:3157                         KVTransferConfig（kv_connector/kv_role/kv_rank/...）
  vllm/engine/arg_utils.py:941                --kv-transfer-config CLI 解析
  vllm/engine/arg_utils.py:1526               V1 → V0 fallback（PD 分离尚不支持 V1）

总控与初始化
  vllm/worker/worker.py:510                    ensure_kv_transfer_initialized(vllm_config)
  vllm/distributed/parallel_state.py:965       建立进程级单例 _KV_TRANSFER
  vllm/distributed/parallel_state.py:778       get_kv_transfer_group()（worker 取用入口）
  vllm/distributed/kv_transfer/kv_transfer_agent.py:24   KVTransferAgent（薄 shim）

接入点（V0 worker forward 前后两个钩子）
  vllm/worker/model_runner.py:1742             need_recv_kv → recv_kv_caches_and_hidden_states
  vllm/worker/model_runner.py:1786             need_send_kv → send_kv_caches_and_hidden_states

三层抽象 vllm/distributed/kv_transfer/
  kv_connector/base.py:22                       KVConnectorBase（send/recv 抽象）
  kv_connector/factory.py:12                    KVConnectorFactory（懒加载 + 注册表）
  kv_connector/simple_connector.py:30           SimpleConnector（PyNccl/Mooncake 共用）
  kv_connector/lmcache_connector.py:25          LMCacheConnector（委托 LMCache）
  kv_connector/mooncake_store_connector.py:26   MooncakeStoreConnector（KV store 式）
  kv_connector/utils.py:15                       model_aware_kv_ops_helper（抽/写 paged KV）
  kv_lookup_buffer/base.py:39                    KVLookupBufferBase / KVStoreBufferBase:122
  kv_lookup_buffer/simple_buffer.py:26           SimpleBuffer（deque + 背压 + 后台线程）
  kv_pipe/base.py                                KVPipeBase（send_tensor/recv_tensor）
  kv_pipe/pynccl_pipe.py:41                      PyNcclPipe（NCCL + stateless group）
  kv_pipe/mooncake_pipe.py                       MooncakePipe（RDMA）

外置编排（不在 vllm 包内）
  examples/online_serving/disaggregated_prefill.sh           1P1D 启动脚本
  benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py 请求编排 proxy
```

---

## 2. 端到端调用链：一条请求如何 prefill → 传 KV → decode 续算

以 `PyNcclConnector` / `SimpleConnector` 的 1P1D 为主线。

### 阶段 0：启动期 —— 两实例各自初始化 KV transfer

**0a.** 启动参数（`examples/online_serving/disaggregated_prefill.sh:50-65`）：prefill 实例 `kv_role=kv_producer, kv_rank=0`，decode 实例 `kv_role=kv_consumer, kv_rank=1`，二者 `kv_parallel_size=2`，共享同一 `kv_ip:kv_port`。

**0b.** `vllm/engine/arg_utils.py:941`
```python
parser.add_argument('--kv-transfer-config', type=KVTransferConfig.from_cli, ...)
```
JSON 串被 `KVTransferConfig.from_cli`（`config.py:3212`）反序列化成 `KVTransferConfig`。注意 `config.py:1526-1530` 此时若引擎本想用 V1，会因该 config 非默认而 fallback 到 V0。

**0c.** `vllm/worker/worker.py:510` `ensure_kv_transfer_initialized(vllm_config)`（在 `init_distributed_environment` + `ensure_model_parallel_initialized` 之后）。

**0d.** `vllm/distributed/parallel_state.py:965` `ensure_kv_transfer_initialized()`
```python
if all([vllm_config.kv_transfer_config.is_kv_transfer_instance, _KV_TRANSFER is None]):
    _KV_TRANSFER = kv_transfer.KVTransferAgent(
        rank=get_world_group().rank, local_rank=get_world_group().local_rank, config=vllm_config)
```
> 解读：每个 worker 进程建一个**进程级单例** `_KV_TRANSFER`。`is_kv_transfer_instance`（`config.py:3232`）要求 `kv_connector` 非空且 `kv_role` 合法。

**0e.** `kv_transfer_agent.py:49` `KVTransferAgent.__init__` → `KVConnectorFactory.create_connector(rank, local_rank, config)`。

**0f.** `kv_connector/factory.py:28` `create_connector`
```python
connector_name = config.kv_transfer_config.kv_connector  # "PyNcclConnector"
connector_cls = cls._registry[connector_name]()          # 懒加载真正的类
return connector_cls(rank, local_rank, config)
```
注册表在文件底部（`factory.py:42-60`）：`PyNcclConnector`/`MooncakeConnector` 都映射到 `SimpleConnector`，`LMCacheConnector`、`MooncakeStoreConnector` 各自独立。

**0g.** `simple_connector.py:32` `SimpleConnector.__init__`：按 `is_kv_producer`（`config.py:3238`）**只建发送侧 pipe + producer buffer**（`:80-104`），consumer 实例则**只建接收侧 pipe + consumer buffer**（`:106-133`）。每 rank 用 2 个 pipe，端口偏移 `port_offset_base = 2*rank`（`:76`），一个传数据（GPU）一个传信号（CPU）。

---

### 阶段 P：Prefill 实例算出 KV

Proxy 把 `max_tokens=1` 的请求发给 prefill 实例（`disagg_prefill_proxy_server.py` 的 `prefill_request['max_tokens'] = 1`）。请求照常走 V0 生命周期到 model runner。

**P1.** `vllm/worker/model_runner.py:1767` 正常 forward 计算 prefill KV：
```python
if not bypass_model_exec:
    with set_forward_context(model_input.attn_metadata, self.vllm_config, virtual_engine):
        hidden_or_intermediate_states = model_executable(input_ids=model_input.input_tokens, ...)
```
prefill 实例 `bypass_model_exec` 始终为 `False`（它是 producer，不收 KV），所以真正跑了前向，paged KV cache 被填满，`hidden_or_intermediate_states` 是 prompt 的 hidden states。

**P2.** `model_runner.py:1786` forward 之后判断是否要发送：
```python
if self.need_send_kv(model_input, kv_caches):
    get_kv_transfer_group().send_kv_caches_and_hidden_states(
        model_executable, model_input, kv_caches, hidden_or_intermediate_states)
```
**P3.** `model_runner.py:1889` `need_send_kv`
```python
return self.vllm_config.kv_transfer_config.is_kv_producer and (not is_profile_run) and is_prefill_run
```
- `is_profile_run = (kv_caches[0].numel() == 0)`（`:1907`）—— profiling 时 KV cache 空，跳过。
- `is_prefill_run = prefill_meta is not None`（`:1909`）—— 只在 prefill 步发。

**P4.** `kv_transfer_agent.py:52` `KVTransferAgent.send_kv_caches_and_hidden_states` → 直接转给 `self.connector.send_...`（纯 shim）。

**P5.** `simple_connector.py:151` `SimpleConnector.send_kv_caches_and_hidden_states` —— **核心切分逻辑**
```python
input_tokens_tensor = model_input.input_tokens
seq_lens = model_input.attn_metadata.seq_lens
slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
start_layer = model_executable.model.start_layer
end_layer = model_executable.model.end_layer
num_heads, head_size = self.kv_helper.get_model_args(model_executable)

for idx, slen in enumerate(seq_lens):                    # ① 按请求切
    start_pos = sum(seq_lens[:idx]); end_pos = start_pos + slen
    if start_pos >= num_prefill_tokens:                  # 遇到 decode token 段就停
        logger.warning("... Their KVCache won't be sent."); break
    current_tokens = input_tokens_tensor[start_pos:end_pos]
    keys, values = [], []
    for layer_id in range(start_layer, end_layer):       # ② 按 layer 切（PP 范围）
        kv_cache = kv_caches[layer_id - start_layer]
        key_cache, value_cache = self.kv_helper.get_kv_from_cache(kv_cache, num_heads, head_size)
        current_slot_mapping = slot_mapping_flat[start_pos:end_pos]
        keys.append(key_cache[current_slot_mapping].unsqueeze(0))     # ③ 按 slot 抽出本请求 KV
        values.append(value_cache[current_slot_mapping].unsqueeze(0))
    keys = torch.cat(keys, dim=0); values = torch.cat(values, dim=0)
    self.insert(current_tokens, torch.ones_like(current_tokens, dtype=bool),
                keys, values, hidden_or_intermediate_states[start_pos:end_pos])  # ④ 入 buffer
```
> 解读三个切分维度（paged KV / 物理 slot 概念见 [模块 02](../02-paged-attention-kvcache/design.md)，PP 切分见 [模块 03](../03-distributed-parallel/design.md)）：
> - **请求维度**：`seq_lens` 把 batch 拆成每条请求的 token 段（`:171-173`）。`FIXME(Kuntai)`（`:170`）坦承"假设全是 prefill"，遇到 decode token 段直接 `break`（`:175-181`）。
> - **layer 维度**：只遍历本 worker 的 `start_layer..end_layer`（PP 切分），所以 KV 是**分 PP 段**传的。
> - **slot 维度**：`key_cache[current_slot_mapping]` 用物理 slot 索引从 paged 显存里**抽出**这条请求这些 token 的 KV，堆叠成紧凑张量 —— 不传整块 paged block。
> - `roi` 当前固定全 1（`torch.ones_like`），即"这些 token 的 KV 全都有效"。

**P6.** `simple_connector.py:142` `insert` → `simple_buffer.py:214` `SimpleBuffer.insert`
```python
self._add_to_buffer(input_tokens, roi, key, value, hidden)
if self.request_handling_thread is None:                 # 首次 insert 起后台线程
    self.request_handling_thread = threading.Thread(target=self.drop_select_handler)
    self.request_handling_thread.start()
```
**P7.** `simple_buffer.py:102` `_add_to_buffer`：对每个张量 `.clone()`（防发送前被释放，`:106-115`），算大小，**buffer 满则阻塞等待**（背压）：
```python
with self.buffer_cv:
    while self.buffer_size + data_size > self.buffer_size_threshold:
        self.buffer_cv.wait()        # ← decode 慢时反压 prefill
    self.buffer_size += data_size; self.buffer.append(buffer_item); self.buffer_cv.notify()
```
> 至此 prefill 侧的工作完成：KV + hidden 已进 producer buffer，等 consumer 来取。**发送是非阻塞的**（`model_runner.py:1785` 注释），prefill 实例随即可处理下一请求。

---

### 阶段 T：跨实例传输（producer buffer ↔ consumer）

**T1.** Producer 后台线程 `simple_buffer.py:135` `drop_select_handler`（阶段 P6 启动）
```python
while True:
    signal = self.signal_pipe.recv_tensor()              # 阻塞等 consumer 的查询信号（CPU pipe）
    if self._is_end_signal(signal): break
    input_tokens = self.data_pipe.recv_tensor()          # 收 consumer 要查的 tokens
    roi = self.data_pipe.recv_tensor(); roi = (roi > 0.5)
    with self.buffer_cv:
        while not is_buffer_available(tokens_roi_recver): # O(n) 前缀匹配（见 T3）
            self.buffer_cv.wait()
        matched_item = self.buffer.popleft()
        for tensor in matched_item:
            self._send_tensor_and_dec_size(tensor)        # 命中：把 5 元组逐个发回
        self.buffer_cv.notify()
```
**T2.** `simple_buffer.py:53` `_matches`：做**公共前缀匹配** —— 取 `min_length` 长度，`torch.allclose` 比对 token，相同返回匹配长度（`:75-78`）。`is_buffer_available`（`:153-165`）对 deque 做 O(n) 轮转扫描找命中项。
> `FIXME`（`:157`）：匹配是 O(n) 非 O(1)，但 buffer 不会太大故暂不优化。这就是 README 说的"用 lookup buffer 把 FIFO 翻译成按 token 查找"以容忍乱序。

**T3.** 底层传输 `kv_pipe/pynccl_pipe.py`：
- `send_tensor`（`:225`）非阻塞 —— 提交进单线程 `ThreadPoolExecutor`（`:233-247`），先 `block_if_full`（`:216-223`）背压。
- `_send_impl`（`:169`）先发 metadata（dtype+shape，`:117-132`），GPU 张量用 `device_send_func`（PyNCCL `comm.send`，`:91-95`）发；CPU 信号用 stateless group 的 `send_obj`（`:101`）。
- 连接由 `StatelessProcessGroup.create(host=kv_ip, port=kv_port+offset, rank=kv_rank, world_size=kv_parallel_size)` 建立（`:63-69`），send/recv 目标 rank 是环形邻居 `(kv_rank±1) % kv_parallel_size`（`:75-76`）。

---

### 阶段 D：Decode 实例接收 KV 并续算

Proxy 把**原请求**（完整 `max_tokens`）发给 decode 实例。请求走 V0 生命周期，第一步是 prefill 步（因为 decode 实例还没有这条请求的 KV），进入 model runner。

**D1.** `model_runner.py:1742` forward **之前**判断是否要收：
```python
bypass_model_exec = False
if self.need_recv_kv(model_input, kv_caches):
    hidden_or_intermediate_states, bypass_model_exec, model_input = \
        get_kv_transfer_group().recv_kv_caches_and_hidden_states(model_executable, model_input, kv_caches)
```
**D2.** `model_runner.py:1864` `need_recv_kv`
```python
return self.vllm_config.kv_transfer_config.is_kv_consumer and (not is_profile_run) and is_prefill_run
```
> 对称于 `need_send_kv`：consumer 角色 + 非 profiling + prefill 步。**接收是阻塞的**（`:1740` 注释）。

**D3.** `simple_connector.py:207` `recv_kv_caches_and_hidden_states` —— **逐请求查询 + 写回 + 决定 bypass**
```python
bypass_model_exec = True
for idx, slen in enumerate(seq_lens):
    start_pos = sum(seq_lens[:idx]); end_pos = start_pos + slen
    if start_pos >= num_prefill_tokens:                  # 碰到 decode token（chunked prefill 等）
        logger.warning("set --enable_chunked_prefill=False ...")
        bypass_model_exec = False; assert start_pos == num_prefill_tokens; break
    current_tokens = input_tokens_tensor[start_pos:end_pos]
    ret = self.select(current_tokens, torch.ones_like(current_tokens, dtype=bool))  # → drop_select
    if ret[0] is None:                                   # 没查到 → 必须回退重算
        bypass_model_exec = False; num_computed_tokens_list.append(0); continue
    roi, keys, values, hidden = ret[1], ret[2], ret[3], ret[4]
    num_computed_tokens = roi.shape[0]
    if not all([(num_computed_tokens == num_tokens), hidden is not None]):  # 部分命中 → 回退
        bypass_model_exec = False
    end_pos = start_pos + num_computed_tokens
    for cur_layer in range(start_layer, end_layer):       # 把 KV 写回本地 paged cache
        layer_id = cur_layer - start_layer
        remote_k, remote_v = keys[layer_id], values[layer_id]
        self.kv_helper.put_kv_to_cache(model_executable, remote_k, remote_v,
                                       model_executable.model.layers[cur_layer],
                                       kv_caches[layer_id], slot_mapping, start_pos, end_pos)
    hidden_or_intermediate_states_for_one_req.append(hidden)

if not bypass_model_exec:
    hidden_or_intermediate_states = None                  # 回退：交给正常 forward 重算
else:
    hidden_or_intermediate_states = torch.cat(hidden_or_intermediate_states_for_one_req, dim=0)
return hidden_or_intermediate_states, bypass_model_exec, model_input
```
**D4.** `simple_connector.py:135` `select` → `simple_buffer.py:185` `drop_select`（consumer 侧主动查询）
```python
self.signal_pipe.send_tensor(self.normal_signal)   # 经 CPU 信号 pipe 通知 producer
self.data_pipe.send_tensor(input_tokens)           # 发要查的 tokens
self.data_pipe.send_tensor(roi)
input_tokens = self.data_pipe.recv_tensor()        # 收回 producer 匹配到的 5 元组
roi = self.data_pipe.recv_tensor(); roi = (roi > 0.5)
key = self.data_pipe.recv_tensor(); value = self.data_pipe.recv_tensor()
hidden = self.data_pipe.recv_tensor()
return [input_tokens, roi, key, value, hidden]
```
> 这正是与阶段 T1 producer 后台线程的对手戏：consumer `drop_select` 发查询 → producer `drop_select_handler` 匹配并回传。

**D5.** 写回 paged KV：`kv_connector/utils.py:61` `put_kv_to_cache`
```python
# 非 MLA 路径：
key_cache, value_cache = kv_cache[0], kv_cache[1]
ops.reshape_and_cache_flash(keys.to(key_cache.device), values.to(value_cache.device),
                            key_cache, value_cache, slot_mapping[start_pos:end_pos],
                            layer.self_attn.attn.kv_cache_dtype, _k_scale, _v_scale)
```
> 用**本地** `slot_mapping` 把收到的紧凑 KV `reshape_and_cache` 回 decode 实例**自己分配的** paged block —— 两实例 block 布局独立，必须各按各的 slot 写。MLA（DeepSeek）走另一分支 `concat_and_cache_mla`（`utils.py:66-78`）。

**D6.** 回到 `model_runner.py:1767`：
```python
if not bypass_model_exec:                       # 全部命中时为 False → 跳过这整段 forward
    with set_forward_context(...):
        hidden_or_intermediate_states = model_executable(...)
```
> **bypass 生效**：当 `recv` 返回 `bypass_model_exec=True`，decode 实例**完全跳过 prefill 前向**，`hidden_or_intermediate_states` 直接用收到的 hidden states。

**D7.** 续算第一个 token：`model_runner.py:1816` `compute_logits` → `:1826` `self.model.sample(...)`。decode 实例用收到的 hidden states 算出 logits、采样**第一个生成 token**，之后这条请求就在 decode 实例上正常自回归 decode（后续步的 KV 都在本地 paged cache 里，不再走传输层）。

---

## 3. 实现中的精妙之处（Tricks 与非显然设计）

> 格式：**现象/问题 → 代码位置 → 精妙之处**。

### T1 · 信号管道放 CPU、数据管道放 GPU —— 规避"on-device recv 阻塞全进程"

- **代码**：`simple_buffer.py:30-39` 的 docstring；构造时 `signal_pipe` device="cpu"、`data_pipe` device="cuda"（`simple_connector.py:88-93`）。
- **精妙之处**：注释点破一个隐蔽坑 —— **GPU 上的 NCCL `recv` 会阻塞进程内所有线程**，使 producer 在传 KV 时无法监听新查询请求；而 **CPU `recv` 只阻塞当前线程**。于是 producer 后台线程用 **CPU 信号管道** `recv` 来"等查询"（不卡住主推理线程），真正的 KV 数据才走 GPU 管道。这是"传输与监听并行"的关键。

### T2 · 发送非阻塞、接收阻塞 —— 与下一步 forward 重叠

- **代码**：`model_runner.py:1785`（send 非阻塞）vs `:1740`（recv 阻塞）；`pynccl_pipe.py:233-247` 把 send 丢进 `ThreadPoolExecutor`。
- **精妙之处**：prefill 实例 `send` 立即返回，主线程继续算下一批，KV 实际发送在后台线程进行 —— **传输与计算重叠**。decode 实例 `recv` 必须阻塞（要拿到 KV 才能决定是否 bypass），但因为它本来就处于这条请求的第一步，阻塞代价被 bypass 省下的整段 prefill 前向远远盖过。

### T3 · `bypass_model_exec` 的"全有或全无"语义

- **代码**：`simple_connector.py:218`（初值 True）、`:262`/`:278`（任一请求缺失即置 False）、`model_runner.py:1767`。
- **精妙之处**：只要 batch 里**有一条请求**没完整收到 KV（`ret[0] is None` 或部分命中），整个 batch 的 `bypass_model_exec` 就翻成 False，回退到**对整批正常 forward**。这是个保守但安全的设计：宁可重算整批，也不混合"部分 bypass + 部分重算"那种容易错的状态。代价是浪费已收到的部分 KV（注释 `:302-304` 提示理论上可只对缺失 token 重算，但未实现）。

### T4 · KV 按 paged slot 抽取、再按本地 slot 写回 —— 两实例显存布局解耦

- **代码**：发送 `key_cache[current_slot_mapping]`（`simple_connector.py:194`）；接收 `reshape_and_cache_flash(..., slot_mapping[start_pos:end_pos], ...)`（`utils.py:81-89`）。
- **精妙之处**：prefill 与 decode 实例**各自独立分配 paged block**，同一条请求在两边的物理 slot 完全不同。所以传输的不是"block 拷贝"，而是 **"按 token 逻辑序抽取 → 紧凑张量 → 按接收方 slot 重新 cache"**。这让 KV 传输与两端的 block 分配策略彻底解耦。

### T5 · `clone()` 防"发送前张量被释放"

- **代码**：`simple_buffer.py:106-115`（入 buffer 前 clone 五元组）、`:172-176`（popleft 后发送，注释"in case the tensor is freed before sending finishes"）、`drop_select` 里 `:193-196`。
- **精妙之处**：KV 抽取自 paged cache，而 paged block 随时可能被 vLLM 回收复用。因为发送是异步的，若不 clone，张量可能在后台线程真正发出前就被覆盖。这里在入 buffer 时就**深拷贝隔离**，牺牲一点内存换正确性。

### T6 · 背压在两个层级同时存在

- **代码**：buffer 级 `simple_buffer.py:120-130`（Condition `wait` 直到有空间）；pipe 级 `pynccl_pipe.py:216-223` `block_if_full`（`while buffer_size > thresh: time.sleep(0.05)`）。
- **精妙之处**：decode 慢导致 KV 堆积时，**两道闸**都会把 prefill 实例的 `insert`/`send` 卡住，防止 KV 在显存里无限膨胀撑爆 producer。`kv_buffer_size`（默认 1e9 ≈ 1GB，`config.py:3169`）就是这道水位线。

### T7 · 懒加载 connector，只 import 当前用到的那一个

- **代码**：`factory.py:22-26` 的 `loader` 闭包（`importlib.import_module` 延迟到 `create_connector` 时才执行）；注释 `:40-41` "only load the files corresponding to the current connector"。
- **精妙之处**：LMCache、Mooncake 都是**可选第三方依赖**。若在模块顶层 import，没装这些库的用户连 `import vllm` 都会炸。工厂用字符串 + 懒加载，保证只有真正选用某 connector 时才 import 它的实现文件。

### T8 · 同一个 `SimpleConnector` 服务两种底层管道

- **代码**：`factory.py:42-50` 把 `PyNcclConnector` 和 `MooncakeConnector` 都映射到 `SimpleConnector`；`simple_connector.py:42-64` 按 `kv_connector` 字符串选 `PyNcclPipe` 或 `MooncakePipe`。
- **精妙之处**：connector 层（切分 KV、bypass 逻辑）与 pipe 层（NCCL vs RDMA）正交。换传输介质**不用重写** send/recv 切分逻辑，只换底层 pipe 类。Mooncake 的 `signal_pipe` 还直接复用 `data_pipe`（`:100`、`:127`），因为它自带 key-value 语义、不需要单独的信号通道。

### T9 · MooncakeStore 用 `blake2b(tokens)` 当 key —— 天然乱序无关

- **代码**：`mooncake_store_connector.py:89,147` `store_key_prefix = self.tensor_hash(current_tokens)`；`:104` `f"{store_key_prefix}_{self.local_tp_rank}"`；`:195-199` `tensor_hash`。
- **精妙之处**：与 SimpleConnector 的"O(n) 前缀匹配 + 点对点管道"不同，MooncakeStore 把 KV 直接 **put/get 进数据库式 store**，key 是 token 序列的哈希。这样 **producer 和 consumer 无需在线对接、无需顺序对齐** —— consumer 拿 token 哈希一查即得，是更彻底的解耦（对应 design §3.2 的 KVStoreBufferBase 抽象）。key 里带 `local_tp_rank` 保证 TP 各分片互不串。

### T10 · `roi`（region of interest）：为"部分命中 / TP-PP 部分持有"预留的扩展点

- **代码**：抽象定义 `kv_lookup_buffer/base.py:46-54`；当前实现固定全 1（`simple_connector.py:201,259`）。
- **精妙之处**：`roi` 是 token 上的二值掩码，语义是"KV 只对这些 token 可用"。当前 PD 分离每条请求 KV 都完整，所以恒为全 1；但抽象层特意保留它，为两类未来场景铺路：(a) 连外部 KV store 时只命中部分 token（base.py:48-50）；(b) TP/PP 下每个进程只持有 KV 的一部分（base.py:51-54，注释"not implemented for now"）。

### T11 · MLA（DeepSeek）走完全不同的 KV 抽取/写回路径

- **代码**：`utils.py:39-50`（MLA 下 `num_heads=1`、head_size 用 `kv_lora_rank+qk_rope_head_dim`）、`:53-59`（key/value 共用同一 reshape）、`:66-78`（`concat_and_cache_mla`）。
- **精妙之处**：DeepSeek 的 Multi-head Latent Attention 用**压缩的潜在 KV**（一份 latent 而非分头 K/V），且 KV cache 形状随 `VLLM_MLA_DISABLE` 变化。`model_aware_kv_ops_helper` 用 `is_deepseek_mla` + `use_mla_opt` 分支屏蔽这些差异，让上层 connector 的切分循环对 MLA / 普通注意力**写法一致**。

### T12 · prefill 实例 `max_tokens=1` —— "只 prefill 不真生成"的省法

- **代码**：`disagg_prefill_proxy_server.py` 的 `prefill_request['max_tokens'] = 1`。
- **精妙之处**：vLLM 实例本身不知道自己"只该做 prefill"。proxy 通过把 `max_tokens` 设成 1，让 prefill 实例算完 prompt、发完 KV、生成 1 个 token 就结束。真正的生成全交给 decode 实例。这是用**现有 API 拼出 PD 分离**、而不改 vLLM 核心调度的取巧办法（API 标注 subject to change，`disaggregated_prefill.sh:77-78`）。

### T13 · `is_profile_run` 用 `kv_caches[0].numel()==0` 判定

- **代码**：`model_runner.py:1882`、`:1907`。
- **精妙之处**：vLLM 启动时会跑一次 dummy forward 做显存 profiling，此时 KV cache 尚未真正分配（numel 为 0）。send/recv 钩子用这个廉价判据跳过 profiling 步，避免在没有真实 KV 时误触发传输。

### T14 · `kv_rank` 环形邻居寻址，为多实例预留但当前限 1P1D

- **代码**：`pynccl_pipe.py:75-76` `target_rank_for_send/recv = (kv_rank ± 1) % kv_parallel_size`；`config.py:3176-3177` 注释"Currently only 1P1D is supported"。
- **精妙之处**：传输用环形拓扑寻址（rank0↔rank1），数学上能推广到多实例环，但上层编排（proxy、buffer 匹配）目前只验证了 1P1D。这是"底层留口子、上层先做最简"的渐进策略。

---

## 4. 关键边界情况与容错

| 情况 | 代码位置 | 行为 |
|---|---|---|
| profiling 步（KV cache 空） | `model_runner.py:1882,1907` | `is_profile_run=True`，send/recv 都跳过 |
| 非 prefill 步（纯 decode） | `model_runner.py:1884,1909` | `is_prefill_run=False`，不触发传输（decode token 的 KV 留本地） |
| batch 里混入 decode token 段 | `simple_connector.py:175`(send)/`:239`(recv) | 遇到 `start_pos >= num_prefill_tokens` 就 `break`；recv 侧置 `bypass_model_exec=False` 并告警建议关 chunked prefill |
| consumer 查无此 KV | `simple_connector.py:260-264` | `ret[0] is None` → `bypass_model_exec=False`，回退正常 forward 重算 |
| 部分命中（KV 全、hidden 缺等） | `simple_connector.py:276-278` | `bypass_model_exec=False`，整批回退重算 |
| producer buffer 满 | `simple_buffer.py:120-130`；`pynccl_pipe.py:216-223` | `insert`/`send` 阻塞等待，反压 prefill |
| consumer 查询但 buffer 暂无匹配 | `simple_buffer.py:167-171` | producer 后台线程在 Condition 上 `wait`，等新 `insert` 唤醒后再匹配 |
| 连接被对端关闭 | `simple_buffer.py:179-181` | `drop_select_handler` 吞掉 `'Connection closed by peer'`，其余 RuntimeError 重抛 |
| 关闭 / 收尾 | `simple_buffer.py:227-237`；`simple_connector.py:319-328` | requester 侧 `join` 后台线程；非 requester 发 `end_signal`；Mooncake 复用 data_pipe 故只关一个 |
| MLA 模型（DeepSeek） | `utils.py:39-78` | 走 `concat_and_cache_mla` 分支，KV 形状/写回方式不同 |
| 用 V1 引擎 + 设了 kv-transfer-config | `arg_utils.py:1526-1530` | 触发 fallback，引擎退回 V0 跑（V1 暂不支持 PD 分离） |
| 选 Mooncake 但没设环境变量 | `simple_connector.py:54-57`；`mooncake_store_connector.py:44-47` | 抛 `ValueError`，要求 `MOONCAKE_CONFIG_PATH` |

---

## 5. 一图速查：KV 传输调用链主干

```
[启动] worker.py:510 ensure_kv_transfer_initialized
        └► parallel_state.py:965 → KVTransferAgent :49 → Factory.create_connector :28
              └► SimpleConnector.__init__ :32  (producer 建发送 pipe+buffer / consumer 建接收)

════════════════ PREFILL 实例 (kv_producer) ════════════════
 model_runner.execute_model
   forward (计算 prefill KV)  ──► hidden_states
   need_send_kv :1889 ? ──► get_kv_transfer_group().send... :1786
     └► KVTransferAgent.send :52 ─► SimpleConnector.send :151
           按 seq_lens 切请求 → 按 layer 切 → key_cache[slot_mapping] 抽 KV
           insert(tokens, roi, K, V, hidden) :200
             └► SimpleBuffer.insert :214 ─► _add_to_buffer :102 (clone + 背压)
                  └► 首次起后台 drop_select_handler 线程 :135
                                                 │
        signal_pipe(CPU) ◄───────────┐          │ data_pipe(GPU, PyNCCL)
        ════════ 跨实例 ════════      │          ▼
                              ┌───────┴──────────────────────────┐
                              │ PyNcclPipe :41 (StatelessGroup    │
                              │   @ kv_ip:kv_port, rank 0↔1)      │
                              └───────┬──────────────────────────┘
════════════════ DECODE 实例 (kv_consumer) ════════════════
 model_runner.execute_model
   need_recv_kv :1864 ? ──► get_kv_transfer_group().recv... :1742  (阻塞)
     └► KVTransferAgent.recv :68 ─► SimpleConnector.recv :207
           逐请求 select(tokens, roi) :135
             └► SimpleBuffer.drop_select :185
                  signal_pipe.send → producer 后台线程匹配 :135 (O(n)前缀)
                  data_pipe 收回 [tokens, roi, K, V, hidden]
           put_kv_to_cache :293 (utils.py:61 reshape_and_cache 写回本地 paged)
           全命中? ──► bypass_model_exec=True, hidden=cat(...)
   if not bypass_model_exec: forward 重算 :1767    ← 全命中则跳过整段 prefill forward
   compute_logits :1816 ─► sample :1826  → 第一个生成 token
   之后在本实例正常 decode（KV 已在本地，不再走传输层）
```

> 设计动机、权衡取舍与论文出处见同目录 [`design.md`](design.md)。
