# LMCache TTFT 基准测试

本目录包含三个 TTFT（Time To First Token）基准测试脚本，用于测量 LMCache KV Cache 在不同存储层级命中时的首 Token 延迟。

## 三个脚本概览

| 脚本 | 用途 | 场景数 | 提示词长度 |
|---|---|---|---|
| `ttft_bench_v1.py` | 快速单次 TTFT 测量 | 1（不区分命中层级） | 选一个 |
| `ttft_bench_v2.py` | 单长度 3 场景对比 | 3（DDR Hit / SSD Load / Prefetch） | 选一个 |
| `ttft_bench_ssd_sync.py` | 多长度 DDR vs SSD 对比表 | 2（DDR Hit / SSD Load）× 8 长度 | 全部 8 个 |

---

## 环境要求

1. **vLLM + LMCache** 服务已启动，监听 `localhost:8080`（推理 API）和 `localhost:6999`（LMCache 内存管理 API）。
2. 模型名在脚本中配置为 `glm4.7`，可修改 `MODEL` 变量。
3. 需要 `root` 权限执行 `echo 3 > /proc/sys/vm/drop_caches`，或者当前用户有 `sudo` 的 NOPASSWD 权限。
4. Python 依赖：`requests`。

### 针对 `ttft_bench_ssd_sync.py` 的特殊要求

该脚本需要在 **同步加载模式** 下运行，LMCache 配置文件中需设置：

```yaml
enable_async_loading: False
```

---

## ttft_bench_v1.py — 快速单次 TTFT

**最快上手**的脚本，仅发一次 streaming 请求，测量 TTFT 和总延迟，无需操作 KV Cache。

### 使用方法

```bash
python ttft_bench_v1.py
```

### 选择提示词长度

编辑脚本中的 `PROMPT` 行，取消注释你需要的长度：

```python
PROMPT = ("wait " * 1024)   # ~1K  tokens
# PROMPT = ("set " * 2048)  # ~2K  tokens
# PROMPT = ("buy " * 4096)  # ~4K  tokens
# ...
# PROMPT = ("king " * 131072)  # ~131K tokens
```

### 输出示例

```
TTFT: 285.3ms
Hello, this is GLM-4.7...
Total: 1845.2ms
```

### 注意

- 不区分 DDR 命中还是 SSD 命中，只测量 raw TTFT。
- 首次运行数据不在缓存中，第二次运行（同一 prompt）才是 DDR 命中。
- 要测试 SSD 命中，需先手动 evict DDR 并 drop_caches。

---

## ttft_bench_v2.py — 单长度 3 场景对比

**核心对比脚本**，对**同一个 prompt 长度**依次执行三个场景：

| 阶段 | 场景 | 数据路径 | 说明 |
|---|---|---|---|
| Stage 2 | **DDR Hit** | DDR → HBM | 数据在 warmup 后已在 DDR 中 |
| Stage 3 | **SSD Load** | SSD → DDR → HBM | Evict DDR + drop_caches 后强制从磁盘读取 |
| Stage 4 | **Prefetch** | SSD → DDR（预取） → DDR 命中 → HBM | 先异步预取到 DDR，再命中 |

### 使用方法

```bash
# 确认启动配置参数enable_async_loading为True
python ttft_bench_v2.py
```

### 选择提示词长度

同 v1，取消注释需要的 `PROMPT` 行：

```python
PROMPT = ("go " * 4096)     # ~4K  tokens
# PROMPT = ("why " * 131072) # ~131K tokens  (当前激活)
```

### 输出示例

```
============================================================
LMCache TTFT Test: DDR / SSD / Prefetch
============================================================
Prompt: ~131072 tokens

[Stage 0] Clean start
----------------------------------------
  Cleared LMCache cache

[Stage 1] Warmup (non-streaming, populate DDR + Disk)
----------------------------------------
  Sending warmup request...
  chunk_hashes: 128
  Waiting 60s for async disk writes...

[2/3] DDR Hit (data in DDR from warmup)
----------------------------------------
  DDR-hit request (streaming)...
  TTFT: 285.3ms | Total: 1845.2ms

[3/3] SSD Load (evict DDR, read from physical disk)
----------------------------------------
  Evicting 128 chunks from LocalCPUBackend...
  Dropped OS page cache (to force physical disk reads)
  SSD-load request (streaming)...
  TTFT: 1520.6ms | Total: 3080.4ms

[4/4] Prefetch to DDR (evict DDR -> prefetch SSD->DDR -> DDR hit)
----------------------------------------
  Prefetching 128 chunks (lookup_id=prefetch_test_xxx)...
  Prefetch OK
  Waiting 120s for prefetch to finish...
  Prefetch-DDR-hit request (streaming)...
  TTFT: 310.8ms | Total: 1870.1ms

============================================================
Summary
============================================================
  Scenario                    TTFT      Total
  ---------------------- ---------- ----------
  DDR Hit                  285.3ms   1845.2ms
  SSD Load                1520.6ms   3080.4ms
  Prefetch (SSD->DDR)      310.8ms   1870.1ms

  SSD overhead vs DDR: +1235.3ms
  Prefetch overhead vs DDR: +25.5ms
```

### 关键设计说明

- **drop_caches 策略**：每个 `send_request_stream()` 前都调用 `drop_caches`，确保 OS page cache 不干扰结果。SSD Load 之前也 evict DDR，强制走物理磁盘。
- **Prefetch 流程**：先 evict DDR → 调用 `/memory/prefetch` API 异步加载 SSD→DDR → 等待完成后发 streaming 请求 → 此时数据已在 DDR 中，等同于 DDR 命中。

---

## ttft_bench_ssd_sync.py — 多长度 DDR vs SSD 同步对比

**批量对比脚本**，遍历 8 个 prompt 长度，对每个长度分别测量 **DDR Hit** 和 **SSD Load** 的 TTFT，生成一张对比表格。

### 使用方法

```bash
# 确认启动配置参数enable_async_loading为False
python ttft_bench_ssd_sync.py
```

### 提示词长度

内置 8 个长度（不可配置，如需修改请编辑 `PROMPTS` 列表）：

| Prompt | Token 数 |
|---|---|
| `"rise " * 1024` | ~1K |
| `"set " * 2048` | ~2K |
| `"go " * 4096` | ~4K |
| `"man " * 8192` | ~8K |
| `"what " * 16384` | ~16K |
| `"key " * 32768` | ~32K |
| `"fly " * 65536` | ~64K |
| `"king " * 131072` | ~131K |

### 输出示例

```
======================================================================
Sync SSD Load TTFT Test  (enable_async_loading: False)
======================================================================
    Tokens  KV Size  Chunks  DDR TTFT  SSD TTFT   Overhead
----------  --------  -------  ---------  ----------  ----------
      1024    0.7G       2   158.2ms    321.5ms    160.1ms
      2048    1.4G       4   185.1ms    450.8ms    265.7ms
      4096    2.9G       8   242.1ms    785.2ms    543.1ms
      8192    5.8G      16   320.5ms   1450.3ms   1129.8ms
     16384   11.5G      32   510.2ms   2800.1ms   2289.9ms
     32768   23.0G      64   980.5ms   5480.4ms   4499.9ms
     65536   46.0G     128  1950.3ms  10890.5ms   8940.2ms
    131072   92.0G     256  3920.1ms  21780.8ms  17860.7ms
  (iteration took 350s)

Done.
```

### 与 v2 的区别

| 特性 | v2 | ssd_sync |
|---|---|---|
| Prompt 长度 | 选 1 个 | 全部 8 个 |
| 场景数 | 3（含 Prefetch） | 2（DDR + SSD） |
| 异步加载 | 兼容 | 要求 `enable_async_loading: False` |
| 输出格式 | 分阶段详细日志 | 紧凑表格 |
| 单次运行耗时 | ~5 分钟 | ~30-60 分钟 |
| KV Size 估算 | 无 | 近似估算显示 |
| 每轮清理 | 只开头清理一次 | 每轮循环后清理 |

---

## 常用操作

### 清空 LMCache

```bash
curl -X DELETE http://localhost:8080/cache/clear
```

### 手动 evict DDR

```bash
curl -X POST http://localhost:6999/memory/evict \
  -H "Content-Type: application/json" \
  -d '{"chunk_hashes": ["hash1", "hash2", ...], "locations": ["LocalCPUBackend"]}'
```

### 手动 drop_caches

```bash
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

---

## 结果解读

- `TTFT` = 从发送请求到收到第一个有效 content token 的时间。
- `Total` = 从发送请求到流结束的总时间。
- `Overhead` = SSD Load TTFT - DDR Hit TTFT，衡量从磁盘加载 KV Cache 相比直接从 DDR 加载的额外延迟。
- **DDR Hit 场景**的 TTFT 主要由 DDR→HBM 的 PCIe 传输时间决定。
- **SSD Load 场景**的 TTFT 还额外包含 SSD→DDR 的磁盘 I/O 时间。
- **Prefetch 场景**将 SSD→DDR 提前到请求之外完成，因此 TTFT 接近 DDR Hit 的水平。
- 同步模式下（`ttft_bench_ssd_sync.py`），SSD Load 的 `retrieve_time` 包含**完整的** SSD→DDR 耗时；异步模式下只包含**未来得及完成的残留** SSD→DDR。
