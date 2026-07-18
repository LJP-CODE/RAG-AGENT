"""
读取 TXT 文件 → 调用 DeepSeek API 处理 → 生成统计报告
=====================================================
统计指标:
  - API 调用成功率
  - 平均单次请求耗时
  - 平均 token/请求
"""

import json
import os
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 自动加载项目根目录的 .env 文件
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


# ===================== DeepSeek 配置 =====================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "未设置 DEEPSEEK_API_KEY 环境变量。\n"
        "请在项目根目录 .env 文件中配置，或设置系统环境变量：\n"
        "  set DEEPSEEK_API_KEY=你的密钥    (Windows CMD)\n"
        "  $env:DEEPSEEK_API_KEY='你的密钥'  (PowerShell)"
    )
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

base_dir = Path(__file__).resolve().parent.parent  # scripts/ -> root
input_dir = base_dir / "data" / "day1_data" / "day1_data" / "input"
output_dir = base_dir / "data" / "day1_data" / "day1_data" / "output"
output_json = output_dir / "translation_report.json"

output_dir.mkdir(parents=True, exist_ok=True)
files = sorted(input_dir.glob("*.txt"))


# ===================== Token 统计回调 =====================

class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs):
        for gen in response.generations:
            for g in gen:
                meta = g.message.response_metadata if g.message else {}
                usage = meta.get("token_usage", {})
                if usage:
                    self.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.completion_tokens += usage.get("completion_tokens", 0)

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


# ===================== 初始化 LLM =====================

counter = TokenCounter()
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
    temperature=0.1,
    callbacks=[counter],
)


# ===================== 处理文件 =====================

results = []
failures = []
total_calls = 0
success_calls = 0
total_response_time = 0.0
total_prompt_tokens = 0
total_completion_tokens = 0

start_total = time.time()

for file_path in files:
    start = time.time()
    total_calls += 1

    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            elapsed = time.time() - start
            results.append({
                "file": file_path.name,
                "status": "success",
                "response_time_seconds": round(elapsed, 6),
                "output_file": file_path.name,
                "input_bytes": 0,
                "output_bytes": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })
            continue

        # 调用 DeepSeek API
        prompt = (
            "Please translate the following text to English. "
            "If it is already in English, return it as-is.\n\n"
            f"{content}"
        )

        # 记录调用前的 token
        counter.prompt_tokens = 0
        counter.completion_tokens = 0

        response = llm.invoke(prompt)
        elapsed = time.time() - start

        translated_text = response.content
        success_calls += 1
        total_response_time += elapsed
        total_prompt_tokens += counter.prompt_tokens
        total_completion_tokens += counter.completion_tokens

        output_path = output_dir / file_path.name
        with output_path.open("w", encoding="utf-8") as f:
            f.write(translated_text)

        results.append({
            "file": file_path.name,
            "status": "success",
            "response_time_seconds": round(elapsed, 6),
            "output_file": file_path.name,
            "input_bytes": len(content.encode("utf-8")),
            "output_bytes": len(translated_text.encode("utf-8")),
            "prompt_tokens": counter.prompt_tokens,
            "completion_tokens": counter.completion_tokens,
            "total_tokens": counter.total_tokens,
        })

    except Exception as exc:
        elapsed = time.time() - start
        failures.append({
            "file": file_path.name,
            "status": "failure",
            "response_time_seconds": round(elapsed, 6),
            "error": str(exc),
        })

# ===================== 生成统计报告 =====================

total_elapsed = time.time() - start_total
success_rate = (success_calls / total_calls * 100) if total_calls > 0 else 0
avg_time = (total_response_time / success_calls) if success_calls > 0 else 0
avg_tokens = (total_prompt_tokens + total_completion_tokens) / success_calls if success_calls > 0 else 0

report = {
    "config": {
        "model": DEEPSEEK_MODEL,
        "api_base": DEEPSEEK_API_BASE,
    },
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "total_files": total_calls,
    "success_count": success_calls,
    "failure_count": len(failures),
    "api_call_success_rate": f"{success_rate:.2f}%",
    "total_elapsed_seconds": round(total_elapsed, 6),
    "average_request_time_ms": round(avg_time * 1000, 4),
    "average_request_time_seconds": round(avg_time, 6),
    "total_prompt_tokens": total_prompt_tokens,
    "total_completion_tokens": total_completion_tokens,
    "total_tokens": total_prompt_tokens + total_completion_tokens,
    "average_tokens_per_request": round(avg_tokens, 2),
    "success_files": results,
    "failed_files": failures,
}

with output_json.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# ===================== 打印结果 =====================

print("=" * 60)
print("  统计报告")
print("=" * 60)
print(f"  API 调用成功率:      {success_calls}/{total_calls} = {success_rate:.2f}%")
print(f"  失败数:              {len(failures)}")
print(f"  总耗时:              {total_elapsed:.4f}s")
print(f"  平均单次请求耗时:    {avg_time * 1000:.4f} ms ({avg_time:.6f}s)")
print(f"  总 Prompt Token:     {total_prompt_tokens:,}")
print(f"  总 Completion Token: {total_completion_tokens:,}")
print(f"  总 Token 消耗:       {total_prompt_tokens + total_completion_tokens:,}")
print(f"  平均 Token/请求:     {avg_tokens:.2f}")
print(f"  报告已保存:          {output_json}")
print("=" * 60)
