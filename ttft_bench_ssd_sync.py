#!/usr/bin/env python3
"""Test synchronous SSD load TTFT.

Requires lmcache_config.yaml:
  enable_async_loading: False

Only LocalDiskBackend reads use drop_caches to force physical disk I/O.
DDR Hit does NOT use drop_caches to avoid IOMMU page-table cold-start penalty.
"""

import time
import json
import sys
import os

import requests

MODEL = "glm4.7"
ENDPOINT = "http://localhost:8080/v1/chat/completions"
CACHE_API = "http://localhost:6999"
# fmt: off
PROMPTS = [
    ("rise " * 1024),
    ("set " * 2048),
    ("go " * 4096),
    ("man " * 8192),
    ("what " * 16384),
    ("key " * 32768),
    ("fly " * 65536),
    ("king " * 131072),
]
# fmt: on
MAX_TOKENS = 128


def _fmt(val: float | None, suffix: str = "ms") -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}{suffix}"


def _get_first_content(delta: dict) -> str | None:
    content = delta.get("content")
    if content:
        return content
    reasoning = delta.get("reasoning")
    if reasoning:
        return reasoning
    return None


def send_request_stream(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    t0 = time.perf_counter()
    first_token_time = None

    with requests.post(
        ENDPOINT,
        json=payload,
        headers={"Authorization": "Bearer sk-odd-0425"},
        stream=True,
        timeout=600,
    ) as resp:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = _get_first_content(delta)
                if content and first_token_time is None:
                    first_token_time = time.perf_counter()

    total = time.perf_counter() - t0
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else None
    return {"ttft_ms": ttft_ms, "total_ms": total * 1000}


def send_request_nonstream(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(
        ENDPOINT,
        json=payload,
        headers={"Authorization": "Bearer sk-odd-0425"},
        timeout=600,
    )
    total = time.perf_counter() - t0
    data = r.json()
    chunk_hashes = data.get("kv_transfer_params", {}).get("chunk_hashes", [])
    return {"total_ms": total * 1000, "chunk_hashes": chunk_hashes}


def cache_evict(chunk_hashes: list[str]) -> None:
    if not chunk_hashes:
        return
    payload = {"chunk_hashes": chunk_hashes, "locations": ["LocalCPUBackend"]}
    requests.post(f"{CACHE_API}/memory/evict", json=payload, timeout=30)


# ── Main ──────────────────────────────────────────────────────
os.system("echo 3 > /proc/sys/vm/drop_caches")
try:
    requests.delete("http://localhost:8080/cache/clear", timeout=30)
except Exception:
    pass
time.sleep(3)

print("=" * 70)
print("Sync SSD Load TTFT Test  (enable_async_loading: False)")
print("=" * 70)
print(f"{'Tokens':>10} {'KV Size':>8} {'Chunks':>7} {'DDR TTFT':>9} {'SSD TTFT':>10} {'Overhead':>10}")
print(f"{'----------':>10} {'--------':>8} {'-------':>7} {'---------':>9} {'----------':>10} {'----------':>10}")

for prompt in PROMPTS:
    num_tokens = len(prompt.split())
    kv_gb = num_tokens * 46 / 65536  # approximate scaling from measured ~46GB@65536
    t0 = time.time()

    # ── Warmup (populate DDR + Disk) ──
    warmup = send_request_nonstream(prompt)
    chunk_hashes = warmup["chunk_hashes"]
    chunk_count = len(chunk_hashes)
    if not chunk_hashes:
        print(f"  ERROR: no chunk_hashes for {num_tokens} tokens!")
        continue

    # Wait for async disk writes
    time.sleep(max(15, chunk_count * 0.2))

    # ── DDR Hit ──
    ddr = send_request_stream(prompt)

    # ── SSD Load ──
    cache_evict(chunk_hashes)
    time.sleep(1)
    os.system("echo 3 > /proc/sys/vm/drop_caches")
    ssd = send_request_stream(prompt)

    overhead = (ssd.get("ttft_ms") - ddr.get("ttft_ms") if ssd.get("ttft_ms") and ddr.get("ttft_ms") else None)

    print(
        f"{num_tokens:>10} {kv_gb:>7.1f}G {chunk_count:>7} "
        f"{_fmt(ddr['ttft_ms'], ''):>9} {_fmt(ssd['ttft_ms'], ''):>10} "
        f"{_fmt(overhead, ''):>10}"
    )

    # Cleanup for next iteration
    try:
        requests.delete("http://localhost:8080/cache/clear", timeout=30)
    except Exception:
        pass
    os.system("echo 3 > /proc/sys/vm/drop_caches")
    time.sleep(5)

    elapsed = time.time() - t0
    print(f"  (iteration took {elapsed:.0f}s)")

print()
print("Done.")
