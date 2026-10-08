"""Manual end-to-end checks for a running RAG-Agent API service."""

import os
import time

import requests


BASE_URL = os.getenv("RAG_AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("RAG_AGENT_API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def run_check(name, method="POST", path="/ask", payload=None):
    started = time.time()
    response = requests.request(
        method,
        f"{BASE_URL}{path}",
        headers=HEADERS,
        json=payload,
        timeout=120,
    )
    elapsed = (time.time() - started) * 1000
    try:
        data = response.json()
    except ValueError:
        data = {"response": response.text}

    print(f"\n{'OK' if response.ok else 'FAIL'} {name}")
    print(f"  状态码: {response.status_code} | 耗时: {elapsed:.0f}ms")
    if response.ok:
        if "answer" in data:
            print(f"  回答: {data['answer'][:150]}...")
        if "tools_used" in data:
            print(f"  使用工具: {data['tools_used']}")
        if "steps" in data:
            print(f"  推理步数: {data['steps']}")
        if "status" in data:
            print(f"  状态: {data['status']}")
    else:
        print(f"  错误: {data}")
    return response.ok


def main():
    checks = [
        ("健康检查", "GET", "/health", None),
        ("RAG: 显卡PCB层数", "POST", "/ask", {"question": "显卡PCB一般多少层？各层结构是怎样的？", "session_id": "test_rag"}),
        ("RAG: 底部填充胶", "POST", "/ask", {"question": "什么是底部填充胶？它起什么作用？", "session_id": "test_rag"}),
        ("RAG: 等长绕线", "POST", "/ask", {"question": "显存和数据线之间为什么要等长绕线？", "session_id": "test_rag"}),
        ("计算: 基础四则运算", "POST", "/ask", {"question": "计算 125 + 378 等于多少？", "session_id": "test_calc"}),
        ("计算: 平方根和幂", "POST", "/ask", {"question": "计算 sqrt(144) + 2**10 等于多少？", "session_id": "test_calc"}),
        ("计算: 三角函数", "POST", "/ask", {"question": "计算 sin(30) 的值是多少？", "session_id": "test_calc"}),
        ("计算: 除法/零", "POST", "/ask", {"question": "计算 10 ÷ 0 等于多少？", "session_id": "test_calc"}),
        ("时间: 当前时间", "POST", "/ask", {"question": "现在几点？", "session_id": "test_time"}),
        ("时间: 当前日期", "POST", "/ask", {"question": "今天几号？", "session_id": "test_time"}),
        ("多轮: 第一问", "POST", "/ask", {"question": "我叫小明，请记住我的名字", "session_id": "test_multi"}),
        ("多轮: 第二问", "POST", "/ask", {"question": "我叫什么名字？", "session_id": "test_multi"}),
        ("多轮: 不同 session", "POST", "/ask", {"question": "我叫什么名字？", "session_id": "test_multi_other"}),
        ("删除会话 test_multi", "DELETE", "/session/test_multi", None),
        ("最终健康检查", "GET", "/health", None),
    ]

    print("=" * 60)
    print("  AI Agent 全面功能测试")
    print("=" * 60)
    results = [run_check(*check) for check in checks]
    print(f"\n完成：{sum(results)}/{len(results)} 项请求成功")


if __name__ == "__main__":
    main()
