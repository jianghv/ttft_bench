#!/usr/bin/env python3
"""Test TTFT for 3 LMCache scenarios: DDR hit, SSD load, SSD prefetch to DDR

drop_caches 只用于:
  1. 开头清系统状态
  2. SSD Load 之前 (强制物理读盘，不依赖 OS page cache)

DDR Hit 和 Prefetch 之前绝不 drop_caches，因为它们是纯 DDR 加载，
drop_caches 会使 IOMMU 页表变冷，导致 GPU DMA 读 DDR 时产生 ~500ms 开销，
这会污染测量结果。
"""

import time
import json
import sys
import os

import requests

MODEL = "glm4.7"
ENDPOINT = "http://localhost:8080/v1/chat/completions"
CACHE_API = "http://localhost:6999"
# PROMPT = ("love " * 1024)
# PROMPT = ("set " * 2048)
# PROMPT = ("go " * 4096)
# PROMPT = ("man " * 8192)
# PROMPT = ("how " * 16384)
# PROMPT = ("key " * 32768)
# PROMPT = ("get " * 65536)
PROMPT = ("why " * 131072)
MAX_TOKENS = 128


def _fmt(val: float | None, suffix: str = "ms") -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}{suffix}"


def _get_first_content(delta: dict) -> str | None:
    """GLM-4.7 uses 'reasoning' field; other models use 'content'."""
    content = delta.get("content")
    if content:
        return content
    reasoning = delta.get("reasoning")
    if reasoning:
        return reasoning
    return None


def send_request_stream() -> dict:
    """Send a streaming request, return TTFT metrics."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    t0 = time.perf_counter()
    first_token_time = None
    full_response = ""

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
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_response += content

    total_time = time.perf_counter() - t0
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else None

    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_time * 1000,
        "response": full_response[:100],
    }


def send_request_nonstream() -> dict:
    """Send a non-streaming request to get full response + chunk_hashes."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
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
    total_time = time.perf_counter() - t0

    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    chunk_hashes = data.get("kv_transfer_params", {}).get("chunk_hashes", [])

    return {
        "ttft_ms": None,
        "total_ms": total_time * 1000,
        "response": (content or "")[:100],
        "chunk_hashes": chunk_hashes,
    }


def cache_prefetch(chunk_hashes: list[str], lookup_id: str) -> bool:
    """Prefetch chunks into DDR."""
    payload = {"chunk_hashes": chunk_hashes, "lookup_id": lookup_id}
    try:
        r = requests.post(
            f"{CACHE_API}/memory/prefetch",
            json=payload,
            timeout=30,
        )
        return r.status_code == 200 and "prefetch_started" in r.text
    except Exception as e:
        print(f"  Prefetch failed: {e}")
        return False


def cache_evict(chunk_hashes: list[str], location: str = "LocalCPUBackend") -> bool:
    """Evict chunks from the specified tier."""
    if not chunk_hashes:
        return True
    payload = {"chunk_hashes": chunk_hashes, "locations": [location]}
    try:
        r = requests.post(
            f"{CACHE_API}/memory/evict",
            json=payload,
            timeout=30,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"  Evict failed: {e}")
        return False


def main():
    print("=" * 60)
    print("LMCache TTFT Test: DDR / SSD / Prefetch")
    print("=" * 60)
    print(f"Prompt: ~{len(PROMPT.split())} tokens")
    print()

    # ── Clean start ────────────────────────────────────────────
    print("[Stage 0] Clean start")
    print("-" * 40)
    os.system("echo 3 > /proc/sys/vm/drop_caches")
    try:
        requests.delete("http://localhost:8080/cache/clear", timeout=30)
        print("  Cleared LMCache cache")
    except Exception:
        pass
    time.sleep(3)

    # ── Stage 1: Warmup (populate DDR + Disk) ──────────────────
    print()
    print("[Stage 1] Warmup (non-streaming, populate DDR + Disk)")
    print("-" * 40)
    print("  Sending warmup request...")
    warmup = send_request_nonstream()
    chunk_hashes = warmup["chunk_hashes"]
    chunk_count = len(chunk_hashes)
    print(f"  chunk_hashes: {chunk_count}")
    if not chunk_hashes:
        print("  ERROR: No chunk_hashes returned!")
        sys.exit(1)

    # Wait for async disk writes to finish
    disk_wait = max(60, chunk_count * 0.4)
    print(f"  Waiting {disk_wait:.0f}s for async disk writes...")
    time.sleep(disk_wait)

    # ── Scenario 1: DDR Hit ───────────────────────────────────
    # NO drop_caches!  We are measuring DDR→GPU transfer speed.
    # drop_caches would flush IOMMU-related page structures and add ~500ms.
    print()
    print("[2/3] DDR Hit (data in DDR from warmup)")
    print("-" * 40)
    os.system("echo 3 > /proc/sys/vm/drop_caches")
    time.sleep(1)
    print("  DDR-hit request (streaming)...")
    ddr_result = send_request_stream()
    print(f"  TTFT: {_fmt(ddr_result['ttft_ms'])} | Total: {_fmt(ddr_result['total_ms'])}")

    # ── Scenario 2: SSD Load ──────────────────────────────────
    print()
    print("[3/3] SSD Load (evict DDR, read from physical disk)")
    print("-" * 40)

    print(f"  Evicting {chunk_count} chunks from LocalCPUBackend...")
    cache_evict(chunk_hashes, "LocalCPUBackend")
    time.sleep(2)

    # drop_caches for SSD Load: ensure reads hit physical disk, not page cache
    os.system("echo 3 > /proc/sys/vm/drop_caches")
    print("  Dropped OS page cache (to force physical disk reads)")

    print("  SSD-load request (streaming)...")
    ssd_result = send_request_stream()
    print(f"  TTFT: {_fmt(ssd_result['ttft_ms'])} | Total: {_fmt(ssd_result['total_ms'])}")

    # ── Scenario 3: Prefetch to DDR ────────────────────────────
    print()
    print("[4/4] Prefetch to DDR (evict DDR → prefetch SSD→DDR → DDR hit)")
    print("-" * 40)

    # Wait for SSD Load's async disk writes to finish, then evict
    print("  Waiting 30s for SSD Load's async disk writes...")
    time.sleep(30)

    print(f"  Evicting {chunk_count} chunks from LocalCPUBackend...")
    cache_evict(chunk_hashes, "LocalCPUBackend")
    time.sleep(2)

    lookup_id = f"prefetch_test_{int(time.time() * 1000)}"
    print(f"  Prefetching {chunk_count} chunks (lookup_id={lookup_id})...")
    ok = cache_prefetch(chunk_hashes, lookup_id)
    print(f"  Prefetch {'OK' if ok else 'FAILED'}")

    # Prefetch reads all files sequentially in one thread.
    prefetch_wait = max(120, chunk_count * 0.5)
    print(f"  Waiting {prefetch_wait:.0f}s for prefetch to finish...")
    time.sleep(prefetch_wait)
    os.system("echo 3 > /proc/sys/vm/drop_caches")
    time.sleep(1)

    # NO drop_caches! Same reason as DDR Hit.
    print("  Prefetch-DDR-hit request (streaming)...")
    prefetch_result = send_request_stream()
    print(f"  TTFT: {_fmt(prefetch_result['ttft_ms'])} | Total: {_fmt(prefetch_result['total_ms'])}")

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  {'Scenario':<22} {'TTFT':>10} {'Total':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'DDR Hit':<22} {_fmt(ddr_result['ttft_ms'], ''):>10} {_fmt(ddr_result['total_ms'], ''):>10}")
    print(f"  {'SSD Load':<22} {_fmt(ssd_result['ttft_ms'], ''):>10} {_fmt(ssd_result['total_ms'], ''):>10}")
    print(f"  {'Prefetch (SSD->DDR)':<22} {_fmt(prefetch_result['ttft_ms'], ''):>10} {_fmt(prefetch_result['total_ms'], ''):>10}")

    if ssd_result["ttft_ms"] and ddr_result["ttft_ms"]:
        ssd_overhead = ssd_result["ttft_ms"] - ddr_result["ttft_ms"]
        print(f"\n  SSD overhead vs DDR: +{ssd_overhead:.1f}ms")
    if prefetch_result["ttft_ms"] and ddr_result["ttft_ms"]:
        pref_overhead = prefetch_result["ttft_ms"] - ddr_result["ttft_ms"]
        print(f"  Prefetch overhead vs DDR: +{pref_overhead:.1f}ms")


if __name__ == "__main__":
    main()
