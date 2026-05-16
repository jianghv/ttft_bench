# **0** **1**

# **背景：大模型推理中的"记忆断层"**

随着 AI Agent 在多轮对话、RAG 检索增强、长上下文推理等场景中广泛应用，KV Cache 的复用效率成为制约 TTFT（首 Token 延迟）的核心瓶颈。以 LMCache 为代表的 KV Cache 引擎已经构建了 GPU HBM → CPU DDR → Local SSD 的多级存储体系，但现有的缓存策略存在一个根本性的**"认知断层"**：

> 推理引擎只知道"有没有缓存"，但上层 Agent 才知道"该不该用缓存"。

试想一个典型的多轮对话场景：

**➢ 语义盲区：**

用户说"继续讨论刚才的反向传播算法"，Agent 知道要复用上一轮的 KV Cache；但 vLLM 和 LMCache 只看到一串 token，不理解这是"延续对话"还是"新话题"。

**➢ 被动缓存污染：**

当用户切换话题问"今天天气怎么样"，旧对话的 KV Cache 仍然滞留在宝贵的 DDR 空间中，挤占了真正需要被复用的缓存。

**➢ 调度不可控：**

Agent 无法在推理发起前告诉 LMCache"这几个 chunk 不需要了，驱逐掉"或"这几个 chunk 马上要用，提前从 SSD 拉到 DDR"。

针对上述痛点，**我们设计了一套语义感知的 KV Cache 协同框架——通过 chunk_hash 作为全局标识符，在 Agent 与 LMCache 之间建立直接的管控通道**，让上层记忆系统能够在推理发起前主动调度 KV Cache 的驱逐与预取，从而将 TTFT 从秒级压缩到毫秒级。

# **0** **2**

# **KV Cache 协同框架：四层架构串联"感知→调度→执行→反馈"**

框架从语义到硬件形成四层闭环。**上层记忆系统掌握语义，直接指挥 LMCache 的存储操作；vLLM 作为推理引擎居中调度，并在请求结束时将新产生的 chunk_hash 回传给记忆系统。**

```
┌───────────────────────────────────────────────────────────────┐
│   Layer 1   语义感知引擎 ─ 理解"用户在聊什么"                   │
│   · Embedding 相似度  · 对话树追踪  · Session 管理              │
│   · Prompt → 意图识别 → 复用预测 → 语义标签                     │
└───────────────────────────────┬───────────────────────────────┘
                                │ 意图 + chunk_hash 映射查询
                                ▼
┌───────────────────────────────────────────────────────────────┐
│   Layer 2   chunk_hash 映射 & 调度决策 ─ 决定"缓存怎么管"       │
│   · 映射表：语义标签 → [hash_a, hash_b, ...]                   │
│   · 策略引擎：DDR 容量感知 → evict / prefetch / keep            │
└───────────────────────────────┬───────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │ evict           │ prefetch         │
              ▼                 ▼                  │
┌─────────────────────────────────────────────────┼─────────────┐
│              Layer 3   vLLM 推理引擎 ─ 负责"模型怎么跑"         │
│   · lookup(hash) → cache_engine.retrieve() → generate(tokens)  │
│   · request_finished() → 重算 chunk_hashes → SSE 回写          │
└───────────────────────────────┬───────────────────────────────┘
                                │ retrieve(KV)
                                ▼
┌───────────────────────────────────────────────────────────────┐
│   Layer 4   LMCache ─ 执行"缓存存哪里"                            │
│   · /memory/evict → batched_remove()    DDR / SSD 数据清除       │
│   · /memory/prefetch → async_lookup_and_prefetch()  SSD→DDR    │
│   · 存储层级：GPU HBM ◄ PCIe ► CPU DDR ◄ Disk I/O ► Local SSD │
└───────────────────────────────────────────────────────────────┘
```

**核心概念**

| 概念 | 定义 | 作用 |
|---|---|---|
| **chunk_hash** | 基于 blake3 滚动前缀哈希的 KV Cache 唯一标识符 | 跨层通信的统一语言 |
| **语义感知引擎** | 通过 Embedding、对话树等手段理解 prompt 意图的上层模块 | 产出"复用/废弃"的语义标签 |
| **映射表** | 维护 intent / session_id → chunk_hash 列表的双向索引 | 语义 ↔ 缓存的翻译层 |
| **策略引擎** | 结合语义标签、chunk 列表、DDR 容量的决策模块 | 产出 evict_list / prefetch_list |
| **记忆系统 API** | LMCache 暴露的 REST 接口（`/memory/evict` + `/memory/prefetch`） | Agent 直连 KV Cache 的管控通道 |
| **chunk_hashes 回写** | 推理结束后通过 SSE 最后一个 chunk 返回新产生的 hash | 闭环反馈，持续更新映射表 |

**工作流程**

```
上层 Agent               vLLM 推理引擎                LMCache
   │                         │                          │
   │ ① 语义分析               │                          │
   │   prompt → intent        │                          │
   │   查映射表 → [hash_x,    │                          │
   │              hash_y]     │                          │
   │                         │                          │
   │ ② 驱逐：hash_z 不再需要  │                          │
   │ ─ POST /memory/evict ───┼─────────────────────────▶│ batched_remove()
   │                         │                          │
   │ ③ 预取：hash_y 即将复用  │                          │
   │ ─ POST /memory/prefetch ─┼────────────────────────▶│ async_lookup_and_
   │                         │                          │   prefetch()
   │                         │                          │   SSD→DDR (异步)
   │                         │                          │
   │ ④ 发起推理               │                          │
   │ ─ POST /chat ──────────▶│                          │
   │                         │ lookup(hash_x, hash_y)   │
   │                         │ ──── retrieve ──────────▶│
   │                         │                          │  cache_engine
   │                         │                          │  .retrieve()
   │                         │                          │  DDR→GPU
   │ ◄─ SSE stream ──────────│                          │
   │ ◄─ [DONE] +             │                          │
   │    kv_transfer_params:  │                          │
   │    {chunk_hashes:<br>    │                          │
   │     ["hash_x","hash_y"]} │                          │
   │                         │                          │
   │ ⑤ 写回映射表              │                          │
   │   intent → [新 hash]     │                          │
```

> **关键设计**：上层 Agent 的 evict 和 prefetch 请求**直连 LMCache**，不经过 vLLM。这意味着缓存调度可以在推理发起之前完成，完全不受模型生成流程的影响。chunk_hashes 的回写则通过 vLLM 的 SSE 流最后一个 chunk 返回，确保映射表始终与最新的推理结果同步。

# **0** **3**

# **关键模块详解**

### 3.1 chunk_hash：KV Cache 的全局身份证

chunk_hash 是整个协同框架的通信基石。LMCache 通过 `ChunkedTokenDatabase` 将 token 序列按 `chunk_size`（默认 256 tokens）切块，对每个 chunk 执行滚动前缀哈希：

```
chunk_hash[i] = blake3( chunk_hash[i-1] || token_ids[cs:ce] )
```

**核心特性**

- **确定性**：相同 token 序列、相同 `PYTHONHASHSEED` → 永远产出相同 hash，天然支持跨请求复用
- **前缀依赖**：hash[i] 依赖 hash[i-1]，形成不可篡改的链式结构，保证 chunk 顺序
- **纯 CPU 计算**：blake3 在 CPU 上微秒级完成，在 `request_finished()` 中重算零开销

### 3.2 `/memory/evict`：语义驱动的精细化驱逐

当 Agent 判定某个话题的 KV Cache 不再需要时，直接调用 LMCache 的驱逐接口，释放 DDR 或 SSD 空间：

```
POST /memory/evict
{
    "chunk_hashes": ["a1b2c3..."],
    "locations": ["LocalCPUBackend"]
}
→ storage_manager.batched_remove(keys, locations)
→ 指定 tier 数据清除
```

在 vLLM 多进程模式（TP 多卡）下，Scheduler 进程收到请求后会自动通过 `_forward_to_all_workers()` 广播到所有 Worker 进程，确保每个 TP Rank 的 KV 分片都被清理。

### 3.3 `/memory/prefetch`：预测性预取

当 Agent 预判某个历史会话即将被复用时，在推理发起前调用预取接口，将 KV Cache 从 SSD 异步加载到 DDR：

```
POST /memory/prefetch
{
    "chunk_hashes": ["a1b2c3...", "d4e5f6..."],
    "lookup_id": "req_001"
}
→ async_lookup_and_prefetch(lookup_id, keys)
→ LocalDiskBackend.batched_get_non_blocking()
→ SSD → DDR 异步传输
→ EventType.LOADING 注册，retrieve 时直接取用
```

**关键特性**

- **异步非阻塞**：数据在后台从 SSD 传输到 DDR，不阻塞主流程
- **Event 驱动**：预取完成后注册 `EventType.LOADING`，推理时 `cache_engine.retrieve()` 直接通过 `event_manager` 获取数据
- **多 tier 遍历**：按 LocalDiskBackend → RemoteBackend 顺序查找，自动回退

### 3.4 流式响应回写：打通数据闭环

chunk_hash 需要在 Scheduler 进程中重算（Worker 进程的 `Session` 中有原始数据，但存在进程边界），再通过流式 SSE 的最后一个 chunk 返回：

```
LMCache-Ascend adapter:
  request_finished() → ChunkedTokenDatabase.process_tokens()
                    → return_params["chunk_hashes"] = [...]

vLLM-Ascend patch:
  流式生成器的最后一个 chunk → ChatCompletionStreamResponse
    .kv_transfer_params = res.kv_transfer_params
```

**改动量**：`lmcache-ascend` + `vllm-ascend` 合计 **+12 行**，不触碰 `lmcache` 和 `vllm` 主线代码。

### 3.5 cache_engine.retrieve()：三段式加载的计时语义

LMCache 的 `retrieve()` 函数按三段执行，其计时直接反映了不同存储命中对 TTFT 的影响：

```
retrieve() 计时 = process_tokens_time + broadcast_time + to_gpu_time
                  ─────────────────   ─────────────   ───────────
                  存储层读取 (SSD/DDR)  多 Rank 广播    DDR→HBM
```

| 命中层级 | process_tokens 耗时 | to_gpu 耗时 | 总 retrieve_time | TTFT |
|---|---|---|---|---|
| DDR Hit | 微秒级 (memcpy) | ~2.5ms (PCIe) | ~2.5ms | ~285ms |
| SSD Load | 数百毫秒 (磁盘 I/O) | ~2.5ms (PCIe) | 数百毫秒 | ~1520ms |
| **Prefetch** | **微秒级 (已预取到 DDR)** | **~2.5ms (PCIe)** | **~2.5ms** | **~311ms** |

**prefetch 将 SSD→DDR 的磁盘 I/O 完全移出推理关键路径**，TTFT 从 1520ms 降至 311ms，接近纯 DDR 命中水平。

# **0** **4**

# **Agent 接入指南：三行代码即可调度**

上层 Agent 接入此框架只需三步，以下为完整的交互伪代码：

```python
class MemoryAgent:
    def __init__(self):
        self.mapping = {}              # intent → [chunk_hashes]
        self.lmcache_api = "http://localhost:6999"
        self.vllm_api = "http://localhost:8080"

    def on_user_prompt(self, prompt: str):
        # ① 语义分析 + 查映射表
        intent = self.semantic_engine.classify(prompt)
        related_hashes = self.mapping.get(intent, [])
        stale_hashes = self.mapping.get(prev_intent, [])

        # ② 驱逐旧话题的 KV Cache（直连 LMCache，不经 vLLM）
        if stale_hashes:
            requests.post(f"{self.lmcache_api}/memory/evict", json={
                "chunk_hashes": stale_hashes,
                "locations": ["LocalCPUBackend"]
            })

        # ③ 预取复用 chunk（直连 LMCache，异步 SSD→DDR）
        if related_hashes:
            requests.post(f"{self.lmcache_api}/memory/prefetch", json={
                "chunk_hashes": related_hashes,
                "lookup_id": f"prefetch_{uuid4()}"
            })

        # ④ 发起流式推理
        response = requests.post(f"{self.vllm_api}/v1/chat/completions",
            json={"model":"glm4.7","messages":[...],"stream":True},
            stream=True
        )

        # ⑤ 提取 chunk_hashes，回写映射表
        for line in response.iter_lines():
            chunk = json.loads(line[6:])
            kv = chunk.get("kv_transfer_params")
            if kv and kv.get("chunk_hashes"):
                self.mapping[intent] = kv["chunk_hashes"]
                break
```

# **0** **5**

# **总结与展望**

通过 **chunk_hash** 作为统一标识符，我们在上层 Agent 和 LMCache KV Cache 引擎之间建立了一条完整的**语义感知管控通道**。Agent 掌握了 LMCache 存储层的"遥控器"——可以在推理发起之前驱逐无用的 chunk、预取即将复用的 chunk，并在推理结束后自动更新语义→hash 的映射表，形成自进化的闭环。

**框架核心价值**

| 能力 | 实现方式 | 收益 |
|---|---|---|
| **语义驱逐** | Agent 直连 `/memory/evict` | DDR 空间零污染，命中率更高 |
| **预测性预取** | Agent 直连 `/memory/prefetch` | SSD→DDR 移出关键路径，TTFT 降 80% |
| **自动闭环** | SSE 最后 chunk 携带 chunk_hashes | 映射表自动更新，无需额外请求 |
| **零侵入** | 仅改 `-ascend` fork，合计 +12 行 | 主线代码完全不受影响 |

**后续方向**

- **跨模型语义复用**：一个模型产出的 chunk_hash 语义映射，在 hash 算法一致的前提下，可直接指导另一个模型的 prefetch 策略
- **预测性预取池化**：Agent 根据历史对话模式预判下一轮可能出现的 chunk，在用户打字间隙批量预取到 DDR
- **语义权重驱逐**：为 DDR 中的 chunk 附加语义权重（高频/低频），实现比纯 LRU 更智能的空间管理
- **分布式记忆共享**：将语义→hash 映射表扩展到多节点，实现跨实例的 KV Cache 协同
