import requests
import time
import json

url = "http://127.0.0.1:8000/ask"

questions = [
    "显卡PCB一般多少层？",
    "什么是底部填充胶？",
    "显存和数据线之间为什么要等长绕线？"
]

results = []
for q in questions:
    start = time.time()
    resp = requests.post(url, json={"question": q, "temperature": 0.1})
    elapsed = (time.time() - start) * 1000
    data = resp.json()
    results.append({
        "question": q,
        "time_ms": elapsed,
        "answer_length": len(data.get("answer", "")),
        "status": resp.status_code
    })
    print(f"✅ {q[:20]}... → {elapsed:.1f}ms")

# 统计
avg_time = sum(r["time_ms"] for r in results) / len(results)
print(f"\n平均响应时间: {avg_time:.1f}ms")
