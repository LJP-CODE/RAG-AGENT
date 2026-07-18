import requests
import time
import concurrent.futures
import statistics

URL = "http://localhost:8000/ask"
SESSION_ID = "load_test_user"

def send_request(question: str) -> float:
    """发送单次请求，返回耗时（毫秒）"""
    start = time.time()
    try:
        resp = requests.post(URL, json={
            "question": question,
            "session_id": SESSION_ID,
            "temperature": 0.1
        }, timeout=60)
        elapsed = (time.time() - start) * 1000
        return elapsed if resp.status_code == 200 else None
    except:
        return None

def run_load_test():
    questions = [
        "显卡 PCB 一般多少层？",
        "什么是底部填充胶？",
        "等长绕线的作用是什么？"
    ] * 10  # 每个问题重复10次，共30次请求

    times = []
    failures = 0

    print(f"🚀 开始压测：{len(questions)} 次请求...")

    # 并发执行（5个并发）
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(send_request, questions)

    for t in results:
        if t is None:
            failures += 1
        else:
            times.append(t)

    # 统计
    print(f"\n📊 压测结果（{len(times)} 成功 / {failures} 失败）")
    print(f"  平均响应时间: {statistics.mean(times):.2f} ms")
    print(f"  中位数响应时间: {statistics.median(times):.2f} ms")
    print(f"  最大响应时间: {max(times):.2f} ms")
    print(f"  最小响应时间: {min(times):.2f} ms")
    if len(times) > 1:
        print(f"  标准差: {statistics.stdev(times):.2f} ms")
    print(f"  成功率: {(len(times)/(len(times)+failures)*100):.2f}%")

if __name__ == "__main__":
    run_load_test()
