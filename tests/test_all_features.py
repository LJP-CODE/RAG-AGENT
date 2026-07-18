"""全面测试 AI Agent 的所有功能"""
import requests
import json
import time

BASE = "http://127.0.0.1:8000"

def test(name, method="POST", path="/ask", payload=None):
    url = f"{BASE}{path}"
    start = time.time()
    try:
        if method == "GET":
            resp = requests.get(url, timeout=60)
        elif method == "DELETE":
            resp = requests.delete(url, timeout=60)
        else:
            resp = requests.post(url, json=payload, timeout=60)
        elapsed = (time.time() - start) * 1000
        data = resp.json()
        ok = resp.status_code in (200, 201)
        status = "✅" if ok else "❌"
        print(f"\n{status} {name}")
        print(f"  状态码: {resp.status_code} | 耗时: {elapsed:.0f}ms")
        if ok:
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
        return data
    except Exception as e:
        print(f"\n❌ {name} — 请求失败: {e}")
        return None


print("=" * 60)
print("  AI Agent 全面功能测试")
print("=" * 60)

print("\n" + "─" * 50)
print("  🩺 健康检查")
print("─" * 50)
test("健康检查", "GET", "/health")

print("\n" + "─" * 50)
print("  📚 RAG 知识库搜索（3 个问题）")
print("─" * 50)
test("RAG: 显卡PCB层数", payload={
    "question": "显卡PCB一般多少层？各层结构是怎样的？",
    "session_id": "test_rag"
})
test("RAG: 底部填充胶", payload={
    "question": "什么是底部填充胶？它起什么作用？",
    "session_id": "test_rag"
})
test("RAG: 等长绕线", payload={
    "question": "显存和数据线之间为什么要等长绕线？",
    "session_id": "test_rag"
})

print("\n" + "─" * 50)
print("  🧮 计算器功能（4 个测试）")
print("─" * 50)
test("计算: 基础四则运算", payload={
    "question": "计算 125 + 378 等于多少？",
    "session_id": "test_calc"
})
test("计算: 平方根和幂", payload={
    "question": "计算 sqrt(144) + 2**10 等于多少？",
    "session_id": "test_calc"
})
test("计算: 三角函数", payload={
    "question": "计算 sin(30) 的值是多少？",
    "session_id": "test_calc"
})
test("计算: 除法/零", payload={
    "question": "计算 10 ÷ 0 等于多少？",
    "session_id": "test_calc"
})

print("\n" + "─" * 50)
print("  🕐 时间查询功能（2 个测试）")
print("─" * 50)
test("时间: 当前时间", payload={
    "question": "现在几点？",
    "session_id": "test_time"
})
test("时间: 当前日期", payload={
    "question": "今天几号？",
    "session_id": "test_time"
})

print("\n" + "─" * 50)
print("  💬 多轮对话记忆测试")
print("─" * 50)
test("多轮: 第一问", payload={
    "question": "我叫小明，请记住我的名字",
    "session_id": "test_multi"
})
test("多轮: 第二问（应该记得名字）", payload={
    "question": "我叫什么名字？",
    "session_id": "test_multi"
})
test("多轮: 不同 session（不应记得）", payload={
    "question": "我叫什么名字？",
    "session_id": "test_multi_other"
})

print("\n" + "─" * 50)
print("  🗑️ 会话管理")
print("─" * 50)
test("删除会话 test_multi", method="DELETE", path="/session/test_multi")
test("健康检查（最终）", "GET", "/health")

print("\n" + "=" * 60)
print("  测试完成 ✅")
print("=" * 60)
