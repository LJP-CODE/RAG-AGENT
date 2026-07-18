"""
AgentMonitor — AI Agent 的链路追踪与性能监控模块

为每次对话请求创建 Trace，记录步骤、耗时、工具调用、Token 用量。
支持查询历史追踪和会话级别的统计汇总。
"""

import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


class Trace:
    """单次请求的追踪记录。"""

    def __init__(self, trace_id: str, session_id: str, question: str):
        self.trace_id = trace_id
        self.session_id = session_id
        self.question = question
        self.timestamp = datetime.now().isoformat()
        self.steps: List[Dict[str, Any]] = []
        self.answer: Optional[str] = None
        self.total_tokens: int = 0
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None
        self.error: Optional[str] = None

    def add_step(self, action: str, tool: str, tool_input: str, observation: str,
                 duration_ms: float) -> None:
        """记录一次工具调用步骤。"""
        self.steps.append({
            "action": action,
            "tool": tool,
            "input": tool_input,
            "observation": observation,
            "duration_ms": round(duration_ms, 2),
        })

    def finish(self, answer: str, total_tokens: int = 0) -> None:
        """标记追踪完成。"""
        self.answer = answer
        self.total_tokens = total_tokens
        self.end_time = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        """追踪总耗时（毫秒）。"""
        end = self.end_time or time.perf_counter()
        return round((end - self.start_time) * 1000, 2)

    @property
    def tools_used(self) -> List[str]:
        """去重的工具名称列表。"""
        seen: set = set()
        result: list = []
        for step in self.steps:
            t = step["tool"]
            if t and t not in seen:
                seen.add(t)
                result.append(t)
        return result

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "steps": self.steps,
            "tools_used": self.tools_used,
            "step_count": self.step_count,
            "total_tokens": self.total_tokens,
            "error": self.error,
        }


class AgentMonitor:
    """Agent 链路追踪与性能监控器。

    用法::

        monitor = AgentMonitor()

        trace = monitor.start_trace("user-1", "显卡PCB有多少层？")
        try:
            # ... 执行 Agent ...
            trace.add_step("Thought", "RAG_Search", "显卡PCB层数", "...", 320)
            trace.finish("通常 8-12 层", total_tokens=450)
        except Exception as e:
            trace.error = str(e)
    """

    def __init__(self, max_traces: int = 1000):
        self.max_traces = max_traces
        self._traces: Dict[str, Trace] = {}
        self._session_traces: Dict[str, List[str]] = defaultdict(list)

    # ------------------------------------------------------------
    # 追踪生命周期
    # ------------------------------------------------------------

    def start_trace(self, session_id: str, question: str) -> Trace:
        """开始一次新的请求追踪。

        Args:
            session_id: 会话标识
            question:   用户提问

        Returns:
            新建的 Trace 对象
        """
        # ── 淘汰超限的旧追踪记录 ──
        if len(self._traces) >= self.max_traces:
            oldest_ids = sorted(
                self._traces.keys(),
                key=lambda tid: self._traces[tid].timestamp,
            )
            for tid in oldest_ids[:len(oldest_ids) // 2]:
                self._traces.pop(tid, None)
                for sid, tids in list(self._session_traces.items()):
                    if tid in tids:
                        tids.remove(tid)
                    if not tids:
                        self._session_traces.pop(sid, None)

        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        trace = Trace(trace_id, session_id, question)
        self._traces[trace_id] = trace
        self._session_traces[session_id].append(trace_id)
        return trace

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """按 trace_id 查询追踪。"""
        return self._traces.get(trace_id)

    def end_trace(self, trace: Trace, answer: str, total_tokens: int = 0) -> None:
        """结束一次追踪，记录最终答案。"""
        trace.finish(answer, total_tokens)

    def log_error(self, error: str, trace: Optional[Trace] = None) -> None:
        """记录错误信息。

        Args:
            error: 错误描述
            trace: 可选的 Trace 对象，关联到某次追踪
        """
        if trace is not None:
            trace.error = error
        # 这里可扩展写入日志文件 / 外部监控系统

    # ------------------------------------------------------------
    # 查询 & 统计
    # ------------------------------------------------------------

    def get_session_traces(self, session_id: str, limit: int = 10) -> List[Trace]:
        """查询某会话最近的追踪记录。"""
        trace_ids = self._session_traces.get(session_id, [])
        traces = [self._traces[tid] for tid in trace_ids if tid in self._traces]
        return traces[-limit:]

    def get_all_traces(self, limit: int = 100) -> List[Trace]:
        """查询所有追踪记录（最新在前）。"""
        sorted_traces = sorted(
            self._traces.values(),
            key=lambda t: t.timestamp,
            reverse=True,
        )
        return sorted_traces[:limit]

    def summary(self) -> Dict[str, Any]:
        """返回监控统计摘要。"""
        traces = list(self._traces.values())
        completed = [t for t in traces if t.answer is not None]
        errors = [t for t in traces if t.error is not None]

        return {
            "total_traces": len(traces),
            "completed": len(completed),
            "errors": len(errors),
            "active_sessions": len(self._session_traces),
            "avg_duration_ms": (
                round(sum(t.duration_ms for t in completed) / len(completed), 2)
                if completed else 0
            ),
            "avg_tokens": (
                round(sum(t.total_tokens for t in completed) / len(completed))
                if completed else 0
            ),
        }

    def clear_session(self, session_id: str) -> int:
        """清除某会话的全部追踪记录。返回清除的追踪数量。"""
        trace_ids = self._session_traces.pop(session_id, [])
        for tid in trace_ids:
            self._traces.pop(tid, None)
        return len(trace_ids)


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    monitor = AgentMonitor()

    # 模拟一次请求
    trace = monitor.start_trace("demo-user", "1+2×3等于多少？")
    trace.add_step("Thought", "Calculator", "1+2*3", "7", 150)
    expected_steps = 3  # 用于演示断言
    trace.add_step("Thought", "Get_Time", "", "2025-01-01", 80)
    monitor.end_trace(trace, "答案是 7", total_tokens=200)

    print("═" * 55)
    print("📊 单次追踪")
    print("═" * 55)
    d = trace.to_dict()
    print(f"  Trace ID: {d['trace_id']}")
    print(f"  会话: {d['session_id']}")
    print(f"  问题: {d['question']}")
    print(f"  答案: {d['answer']}")
    print(f"  耗时: {d['duration_ms']} ms")
    print(f"  步骤: {d['step_count']}")
    print(f"  工具: {', '.join(d['tools_used'])}")
    print(f"  Token: {d['total_tokens']}")

    # 汇总统计
    print("\n" + "═" * 55)
    print("📈 汇总统计")
    print("═" * 55)
    for k, v in monitor.summary().items():
        print(f"  {k}: {v}")
