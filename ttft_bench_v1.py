#!/usr/bin/env python3
"""测试 GLM-4.7 streaming TTFT"""
import time, json, requests

PROMPT = ("wait " * 1024)
# PROMPT = ("set " * 2048)
# PROMPT = ("buy " * 4096)
# PROMPT = ("you " * 8192)
# PROMPT = ("hit " * 16384)
# PROMPT = ("get " * 32768)
# PROMPT = ("fly " * 65536)
# PROMPT = ("king " * 131072)

payload = {
    "model": "glm4.7",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 128,
    "stream": True,
}

t0 = time.perf_counter()
first_content_time = None

with requests.post(
    "http://localhost:8080/v1/chat/completions",
    json=payload,
    headers={"Authorization": "Bearer sk-odd-0425"},
    stream=True,
) as resp:
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue

        chunk = json.loads(line[6:])
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})

        # GLM-4.7 uses 'reasoning' for thinking tokens, 'content' for final answer
        token = delta.get("content") or delta.get("reasoning")
        if token:
            if first_content_time is None:
                first_content_time = time.perf_counter()
                ttft_ms = (first_content_time - t0) * 1000
                print(f"TTFT: {ttft_ms:.1f}ms")
            print(token, end="", flush=True)

print()
print(f"Total: {(time.perf_counter() - t0) * 1000:.1f}ms")
