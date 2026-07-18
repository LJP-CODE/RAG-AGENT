"""
multi_agent.py — 多智能体架构 (LangGraph StateGraph)

架构:
  START → Supervisor（拆解任务）
           ├─→ Researcher（检索: RAG + Web + News）──┐
           └─→ Calculator（计算）────────────────────┘
                                                      ↓
                                              Writer（汇总）
                                                      ↓
                                              END / Supervisor（多轮）

Agent 职责 & 权限隔离:
  1. Supervisor  — 分析意图、拆解→最多3个子任务；不能调用任何工具
  2. Researcher  — 只能访问 RAG/Web/News/Read 检索工具；不能修改数据
  3. Calculator  — 只能访问 Calculator 工具；不能联网
  4. Writer      — 汇总所有输出生成最终回答；不能调用任何外部工具

每 Agent 独立日志记录，互不越界。

依赖:
  pip install langgraph langchain-openai langchain-core

环境变量:
  DEEPSEEK_API_KEY  — LLM API Key（必填）
  NEWS_API_KEY      — NewsAPI Key（可选，用于 News_Search）
"""

from __future__ import annotations

import json
import logging
import math
import operator
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
)

# ── 路径处理 ──
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# ── 加载 .env ──
from dotenv import load_dotenv

_env_path = Path(".env")
if not _env_path.exists():
    _env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ── LangChain / LangGraph ──
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool as lc_tool
from langgraph.graph import StateGraph, END
from langgraph.types import Send

# ── 项目内部模块 ──
from rag_system import RAGSystem
from web_tools import web_search, read_webpage, news_search
from agent_guardrails import AgentGuardrails


# ═══════════════════════════════════════════════════════════
# 日志 — 按 Agent 名称隔离
# ═══════════════════════════════════════════════════════════

class _AgentLogger:
    """每个 Agent 独立的日志记录器。"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._logger = logging.getLogger(f"multi_agent.{agent_name}")
        self._logs: List[str] = []

    def info(self, msg: str) -> None:
        self._logger.info("[%s] %s", self.agent_name, msg)
        self._logs.append(f"[{self.agent_name}] {msg}")

    def error(self, msg: str) -> None:
        self._logger.error("[%s] %s", self.agent_name, msg)
        self._logs.append(f"[{self.agent_name}] ERROR: {msg}")

    def dump(self) -> str:
        return "\n".join(self._logs)

    def clear(self) -> None:
        self._logs.clear()


# ═══════════════════════════════════════════════════════════
# System Prompts（每个 Agent 独立）
# ═══════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM_PROMPT = """你是任务主管（Supervisor Agent）。

## 职责
分析用户问题，拆解为最多 3 个子任务，分配给合适的专家。

## 可用专家
- researcher: 信息检索（知识库、网络搜索、新闻）
- calculator: 数学计算和数据分析

## 规则
1. 你只能规划任务，不能直接调用工具或回答问题
2. 每个子任务描述必须明确、可执行
3. 如果用户问题不需要工具（如闲聊），直接返回空的 tasks 列表
4. 如果之前轮次的结果已足够，设置 "complete": true

## 输出格式
必须严格输出以下 JSON 格式（不要包含其他文字）：

{
  "plan": "总体计划的一句话说明",
  "tasks": [
    {"id": 1, "type": "research", "description": "具体搜索/检索任务描述"},
    {"id": 2, "type": "calculate", "description": "具体计算任务描述"}
  ],
  "complete": false
}

type 只能是 "research" 或 "calculate"。
如果是纯计算问题（如"1+2*3等于多少"），创建 calculate 任务。
如果是纯信息检索问题（如"今天有什么新闻"），创建 research 任务。
如果是混合问题（如"查询GDP并计算增长率"），同时创建两种任务。"""

RESEARCHER_SYSTEM_PROMPT = """你是研究员（Researcher Agent）。

## 职责
使用工具检索信息，输出结构化摘要。

## 可用工具
- RAG_Search: PCB/硬件/焊接/元器件等知识库问题
- Web_Search: 实时信息（天气、股价、赛事等）
- News_Search: 新闻时事
- Read_Webpage: 读取网页全文

## 规则
1. 你只能使用上述检索工具，不能修改任何数据
2. 对于每个检索任务，根据内容选择最合适的工具
3. 如果知识库没有答案，自动使用 Web_Search
4. 优先使用中文来源

## 输出格式
分点列出信息，每条标注来源工具名称。"""

CALCULATOR_SYSTEM_PROMPT = """你是计算员（Calculator Agent）。

## 职责
只做数学计算和数据分析。

## 可用工具
- Calculator: 执行数学表达式计算

## 规则
1. 你只能使用 Calculator 工具，不能联网或检索
2. 输入必须是合法的数学表达式
3. 保留 2 位小数

## 输出格式
计算结果 + 计算步骤（如有复杂运算）。"""

WRITER_SYSTEM_PROMPT = """你是总结员（Writer Agent）。

## 职责
汇总所有专家的输出，生成完整、准确的最终回答。

## 规则
1. 你只能汇总已有信息，不能调用任何外部工具
2. 如果研究员和计算员的结果冲突，优先采信更可靠的来源
3. 回答必须引用具体数据，不能凭空编造

## 输出格式
1. **核心结论**（1-2 句话概括）
2. **详细分析**（分点展开）
3. **数据来源**（列出每条信息的来源）
4. **注意事项**（如有不确定或有局限的地方）"""


# ═══════════════════════════════════════════════════════════
# State（共享状态类型）
# ═══════════════════════════════════════════════════════════

class MultiAgentState(TypedDict, total=False):
    """多智能体共享状态。

    使用 operator.add 作为 reducer 的字段会自动累积，
    不需要在每次返回时携带历史数据。
    """

    # ── 对话消息（自动累积）──
    messages: Annotated[List[BaseMessage], operator.add]

    # ── 当前轮用户输入 ──
    user_query: str

    # ── Supervisor 输出 ──
    plan: str
    tasks: List[Dict[str, Any]]
    complete: bool

    # ── 各 Agent 输出（自动累积）──
    research_results: Annotated[List[str], operator.add]
    calculation_results: Annotated[List[str], operator.add]

    # ── Writer 最终输出 ──
    final_answer: str

    # ── 多轮控制 ──
    round: int
    needs_more: bool

    # ── 日志（自动累积）──
    agent_logs: Annotated[List[str], operator.add]

    # ── 错误 ──
    error: str


# ═══════════════════════════════════════════════════════════
# Tool 定义（供 Researcher / Calculator 使用）
# ═══════════════════════════════════════════════════════════

# 这些是 LangChain Tool 定义。实际函数引用外部模块，
# 但通过 LangChain 的 @tool 装饰器包装以便 bind_tools。


def _create_researcher_tools():
    """创建研究员专用工具集。"""

    @lc_tool
    def RAG_Search(query: str) -> str:
        """搜索电子硬件/PCB/焊接/元器件知识库。输入：中文问题。"""
        return _global_rag_search(query)

    @lc_tool
    def Web_Search(query: str) -> str:
        """搜索互联网获取实时信息（天气、股价、事件等）。输入：搜索关键词。"""
        return web_search(query, max_results=5)

    @lc_tool
    def News_Search(query: str) -> str:
        """搜索全球新闻。输入：搜索关键词。"""
        return news_search(query, max_results=5)

    @lc_tool
    def Read_Webpage(url: str) -> str:
        """读取指定网页的正文内容。输入：完整的 https:// URL。"""
        return read_webpage(url, max_chars=3000)

    return [RAG_Search, Web_Search, News_Search, Read_Webpage]


def _create_calculator_tools():
    """创建计算员专用工具集。"""

    @lc_tool
    def Calculator(expression: str) -> str:
        """安全数学计算。支持的运算：+ - * / ** % () sqrt sin cos tan log abs round floor ceil pi e。
        输入：数学表达式字符串，如 'sqrt(144)' 或 '1+2*3'。"""
        return _global_calculator(expression)

    return [Calculator]


# ═══════════════════════════════════════════════════════════
# 全局工具实现（由 MultiAgentSystem 在初始化时注入）
# ═══════════════════════════════════════════════════════════

_global_rag_search = None
_global_calculator = None


# ═══════════════════════════════════════════════════════════
# MultiAgentSystem — 主类
# ═══════════════════════════════════════════════════════════

class MultiAgentSystem:
    """
    多智能体编排系统。

    用法::

        mas = MultiAgentSystem()
        mas.initialize()                       # 初始化 RAG + 构建 Graph
        result = mas.run("今天有什么科技新闻？")
        print(result["final_answer"])

    或者异步运行::

        result = await mas.arun("1+2*3等于多少？")
    """

    # ── 最大 ReAct 迭代 ──
    MAX_TOOL_ITERATIONS = 3
    # ── 多轮对话上限 ──
    MAX_ROUNDS = 3

    def __init__(self):
        # ── 日志器 ──
        self._logger = logging.getLogger("multi_agent")
        self.supervisor_log = _AgentLogger("Supervisor")
        self.researcher_log = _AgentLogger("Researcher")
        self.calculator_log = _AgentLogger("Calculator")
        self.writer_log = _AgentLogger("Writer")

        # ── 组件（initialize() 中填充）──
        self.rag: Optional[RAGSystem] = None
        self.llm: Optional[ChatOpenAI] = None
        self.researcher_llm: Optional[ChatOpenAI] = None
        self.calculator_llm: Optional[ChatOpenAI] = None
        self.supervisor_llm: Optional[ChatOpenAI] = None
        self.writer_llm: Optional[ChatOpenAI] = None
        self.researcher_tools: List = []
        self.calculator_tools: List = []
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None
        self.guardrails = AgentGuardrails()

        # ── 初始化标记 ──
        self._initialized = False

    # ----------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------

    def initialize(
        self,
        knowledge_path: str = "data/knowledge/knowledge.txt",
        chunk_size: int = 256,
        overlap: int = 128,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.1,
    ) -> None:
        """
        初始化所有组件：RAG、LLM、工具、LangGraph 状态图。

        Args:
            knowledge_path: 知识库文件路径
            chunk_size:     文本分块大小
            overlap:        分块重叠量
            model:          LLM 模型名
            base_url:       LLM API 地址
            temperature:    LLM 采样温度
        """
        if self._initialized:
            self._logger.warning("MultiAgentSystem 已初始化，跳过重复初始化")
            return

        # ── 0. 检查 API Key ──
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 DEEPSEEK_API_KEY 环境变量。\n"
                "请设置: $env:DEEPSEEK_API_KEY = 'your-api-key'"
            )

        # ── 1. 初始化 RAG ──
        self._logger.info("初始化 RAG 系统...")
        self.rag = RAGSystem(
            knowledge_path=knowledge_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        self.rag.initialize()
        self._logger.info("RAG 系统就绪")

        # ── 2. 注入全局工具实现 ──
        global _global_rag_search, _global_calculator
        _global_rag_search = self._rag_search_impl
        _global_calculator = self._calculator_impl

        # ── 3. 创建工具集 ──
        self.researcher_tools = _create_researcher_tools()
        self.calculator_tools = _create_calculator_tools()

        # ── 4. 创建 LLM 实例（每个 Agent 独立）──
        llm_kwargs = dict(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
        self.llm = ChatOpenAI(**llm_kwargs)  # type: ignore[arg-type]
        self.supervisor_llm = ChatOpenAI(**llm_kwargs)  # type: ignore[arg-type]
        self.researcher_llm = ChatOpenAI(**llm_kwargs).bind_tools(  # type: ignore[arg-type]
            self.researcher_tools
        )
        self.calculator_llm = ChatOpenAI(**llm_kwargs).bind_tools(  # type: ignore[arg-type]
            self.calculator_tools
        )
        self.writer_llm = ChatOpenAI(**llm_kwargs)  # type: ignore[arg-type]

        # ── 5. 构建 LangGraph ──
        self._build_graph()

        self._initialized = True
        self._logger.info("MultiAgentSystem 初始化完成")

    # ----------------------------------------------------------
    # 工具实现（内部）
    # ----------------------------------------------------------

    def _rag_search_impl(self, query: str) -> str:
        """RAG 知识库搜索的实际实现。"""
        try:
            result = self.rag.ask(query, temperature=0.1)
            answer = result.get("answer", "")
            return answer if answer else "未在知识库中找到相关答案。"
        except Exception as e:
            return f"知识库查询出错：{e}"

    @staticmethod
    def _calculator_impl(expression: str) -> str:
        """安全数学计算的实现。"""
        if not expression or not isinstance(expression, str):
            return "错误：请输入有效的数学表达式"

        cleaned = expression.strip()
        cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("^", "**")

        if not re.match(r'^[\d+\-*/().,%^eE\s]+$', cleaned):
            return "错误：表达式包含非法字符"

        safe_globals = {"__builtins__": {}}
        safe_locals = {
            "abs": abs, "round": round, "int": int, "float": float,
            "max": max, "min": min, "sum": sum, "pow": math.pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "log2": math.log2, "floor": math.floor, "ceil": math.ceil,
            "factorial": math.factorial,
            "pi": math.pi, "e": math.e,
        }

        try:
            result = eval(cleaned, safe_globals, safe_locals)
            if isinstance(result, float):
                if result == int(result):
                    result = int(result)
                else:
                    result = round(result, 2)
            return str(result)
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except Exception as e:
            return f"计算错误：{e}"

    # ----------------------------------------------------------
    # LangGraph 构建
    # ----------------------------------------------------------

    def _build_graph(self) -> None:
        """构建多智能体 StateGraph。"""
        builder = StateGraph(MultiAgentState)

        # ── 添加节点 ──
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("calculator", self._calculator_node)
        builder.add_node("writer", self._writer_node)

        # ── 边 ──
        builder.add_edge("__start__", "supervisor")

        # Supervisor → 根据任务类型分发（支持 Send 并行）
        builder.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "researcher": "researcher",
                "calculator": "calculator",
                "writer": "writer",
            },
        )

        # Researcher → Writer
        builder.add_edge("researcher", "writer")

        # Calculator → Writer
        builder.add_edge("calculator", "writer")

        # Writer → 判断是否需要多轮 / 结束
        builder.add_conditional_edges(
            "writer",
            self._route_after_writer,
            {
                "supervisor": "supervisor",
                END: END,
            },
        )

        # ── 编译（带内存检查点，支持多轮）──
        try:
            from langgraph.checkpoint.memory import MemorySaver
            memory = MemorySaver()
            self.compiled_graph = builder.compile(checkpointer=memory)
        except ImportError:
            self.compiled_graph = builder.compile()
        self._logger.info("LangGraph 状态图构建完成")

    # ----------------------------------------------------------
    # 路由函数
    # ----------------------------------------------------------

    @staticmethod
    def _route_after_supervisor(
        state: MultiAgentState,
    ) -> List[Send] | str:
        """Supervisor 完成后决定分发到哪些 Agent（支持并行）。

        返回:
            - List[Send]: 并行分发到多个 Agent
            - str: 路由到单个节点
        """
        if state.get("complete", False):
            return "writer"

        tasks = state.get("tasks", [])
        if not tasks:
            return "writer"

        types = set(t.get("type", "") for t in tasks)
        has_research = "research" in types
        has_calc = "calculate" in types

        if has_research and has_calc:
            # ★ 并行模式：Send 同时分发到两个 Agent
            return [
                Send("researcher", {}),
                Send("calculator", {}),
            ]
        elif has_research:
            return "researcher"
        elif has_calc:
            return "calculator"
        else:
            return "writer"

    @staticmethod
    def _route_after_writer(
        state: MultiAgentState,
    ) -> Literal["supervisor", "__end__"]:
        """Writer 完成后判断是否需要多轮。"""
        if state.get("needs_more") and state.get("round", 0) < MultiAgentSystem.MAX_ROUNDS:
            return "supervisor"
        return END

    # ----------------------------------------------------------
    # Agent 节点实现
    # ----------------------------------------------------------

    def _supervisor_node(self, state: MultiAgentState) -> dict:
        """Supervisor Agent: 分析用户意图，拆解任务。"""
        log = self.supervisor_log
        log.clear()
        log.info("Supervisor 开始分析用户问题")

        user_query = state.get("user_query", "")
        current_round = state.get("round", 0)
        log.info(f"当前轮次: {current_round + 1}/{self.MAX_ROUNDS}")

        # 构建提示
        context_parts = [f"## 用户问题\n{user_query}"]

        # 如果是多轮，附加上一轮结果
        if current_round > 0:
            prev_research = "\n".join(state.get("research_results", []))
            prev_calc = "\n".join(state.get("calculation_results", []))
            prev_answer = state.get("final_answer", "")

            if prev_research:
                context_parts.append(f"\n## 上一轮研究结果\n{prev_research}")
            if prev_calc:
                context_parts.append(f"\n## 上一轮计算结果\n{prev_calc}")
            if prev_answer:
                context_parts.append(f"\n## 上一轮最终回答\n{prev_answer}")

            context_parts.append(
                "\n请判断：以上信息是否已足够回答用户问题？"
                "如果足够，设置 complete=true 且 tasks 为空数组。"
                "如果不够，列出还需要补充的任务。"
            )

        context = "\n".join(context_parts)

        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        # LLM 调用（Supervisor 不绑定任何工具）
        response = self.supervisor_llm.invoke(messages)
        raw_output = response.content if hasattr(response, "content") else str(response)
        log.info(f"Supervisor 原始输出: {raw_output[:200]}...")

        # 解析 JSON
        try:
            parsed = self._extract_json(raw_output)
            plan = parsed.get("plan", "未提供计划")
            tasks = parsed.get("tasks", [])
            complete = parsed.get("complete", False)
        except (json.JSONDecodeError, ValueError) as e:
            log.error(f"JSON 解析失败: {e}")
            # 降级：尝试从文本中提取意图
            plan = "根据用户问题自动判断"
            tasks = self._fallback_task_extraction(user_query)
            complete = False

        # 验证 tasks 格式
        validated_tasks = []
        for t in tasks:
            if isinstance(t, dict) and "type" in t and "description" in t:
                t_type = t["type"]
                if t_type in ("research", "calculate"):
                    validated_tasks.append({
                        "id": t.get("id", len(validated_tasks) + 1),
                        "type": t_type,
                        "description": t["description"],
                    })

        log.info(f"计划: {plan}")
        log.info(f"任务数: {len(validated_tasks)}, 完成: {complete}")
        for t in validated_tasks:
            log.info(f"  - [{t['type']}] {t['description']}")

        return {
            "plan": plan,
            "tasks": validated_tasks,
            "complete": complete,
            "agent_logs": log._logs.copy(),
        }

    def _researcher_node(self, state: MultiAgentState) -> dict:
        """Researcher Agent: 使用检索工具执行研究任务。"""
        log = self.researcher_log
        log.clear()
        log.info("Researcher 开始检索")

        tasks = state.get("tasks", [])
        # 只处理 research 类型的任务
        research_tasks = [t for t in tasks if t.get("type") == "research"]

        if not research_tasks:
            log.info("没有 research 任务，跳过")
            return {"research_results": ["（无研究任务）"], "agent_logs": log._logs.copy()}

        all_results: List[str] = []

        for task in research_tasks:
            task_desc = task.get("description", "")
            log.info(f"执行研究任务: {task_desc}")

            messages = [
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"请执行以下检索任务：\n\n{task_desc}\n\n"
                    f"如果需要，可以使用多个工具。输出结构化摘要。"
                ),
            ]

            try:
                result = self._tool_loop(
                    llm=self.researcher_llm,
                    tools=self.researcher_tools,
                    messages=messages,
                    log=log,
                    max_iterations=self.MAX_TOOL_ITERATIONS,
                )
                all_results.append(f"【任务】{task_desc}\n{result}")
                log.info(f"研究任务完成，结果长度: {len(result)} 字符")
            except Exception as e:
                error_msg = f"研究任务失败: {task_desc} — {e}"
                log.error(error_msg)
                all_results.append(f"【任务】{task_desc}\n错误: {e}")

        combined = "\n\n---\n\n".join(all_results)
        return {
            "research_results": [combined],
            "agent_logs": log._logs.copy(),
        }

    def _calculator_node(self, state: MultiAgentState) -> dict:
        """Calculator Agent: 使用 Calculator 工具执行计算任务。"""
        log = self.calculator_log
        log.clear()
        log.info("Calculator 开始计算")

        tasks = state.get("tasks", [])
        calc_tasks = [t for t in tasks if t.get("type") == "calculate"]

        if not calc_tasks:
            log.info("没有 calculate 任务，跳过")
            return {"calculation_results": ["（无计算任务）"], "agent_logs": log._logs.copy()}

        all_results: List[str] = []

        for task in calc_tasks:
            task_desc = task.get("description", "")
            log.info(f"执行计算任务: {task_desc}")

            messages = [
                SystemMessage(content=CALCULATOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"请执行以下计算任务：\n\n{task_desc}\n\n"
                    f"请列出计算步骤和最终结果。"
                ),
            ]

            try:
                result = self._tool_loop(
                    llm=self.calculator_llm,
                    tools=self.calculator_tools,
                    messages=messages,
                    log=log,
                    max_iterations=self.MAX_TOOL_ITERATIONS,
                )
                all_results.append(f"【任务】{task_desc}\n{result}")
                log.info(f"计算任务完成")
            except Exception as e:
                error_msg = f"计算任务失败: {task_desc} — {e}"
                log.error(error_msg)
                all_results.append(f"【任务】{task_desc}\n错误: {e}")

        combined = "\n\n---\n\n".join(all_results)
        return {
            "calculation_results": [combined],
            "agent_logs": log._logs.copy(),
        }

    def _writer_node(self, state: MultiAgentState) -> dict:
        """Writer Agent: 汇总所有专家输出，生成最终回答。"""
        log = self.writer_log
        log.clear()
        log.info("Writer 开始汇总")

        user_query = state.get("user_query", "")
        plan = state.get("plan", "")
        research = "\n".join(state.get("research_results", []))
        calculation = "\n".join(state.get("calculation_results", []))
        current_round = state.get("round", 0)

        # 构建汇总上下文
        context_parts = [f"## 用户原始问题\n{user_query}"]

        if plan:
            context_parts.append(f"\n## 执行计划\n{plan}")
        if research:
            context_parts.append(f"\n## 研究员输出\n{research}")
        if calculation:
            context_parts.append(f"\n## 计算员输出\n{calculation}")

        context = "\n".join(context_parts)

        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"请汇总以下信息，生成完整回答：\n\n{context}\n\n"
                f"请按照指定格式输出。如果信息不足以完整回答用户问题，"
                f"请在「注意事项」中说明还需补充的信息。"
            ),
        ]

        # Writer 不绑定工具，只做纯 LLM 生成
        response = self.writer_llm.invoke(messages)
        final_answer = response.content if hasattr(response, "content") else str(response)

        log.info(f"Writer 完成，回答长度: {len(final_answer)} 字符")

        # 判断是否需要更多轮次
        needs_more = (
            "还需补充" in final_answer
            or "信息不足" in final_answer
            or "无法确定" in final_answer
        )
        if needs_more and current_round + 1 >= self.MAX_ROUNDS:
            needs_more = False  # 已达上限，不再重试
            log.info("已达最大轮次，不再请求补充")

        return {
            "final_answer": final_answer,
            "needs_more": needs_more,
            "agent_logs": log._logs.copy(),
        }

    # ----------------------------------------------------------
    # 工具调用循环（ReAct 模式）
    # ----------------------------------------------------------

    def _tool_loop(
        self,
        llm: ChatOpenAI,
        tools: List,
        messages: List[BaseMessage],
        log: _AgentLogger,
        max_iterations: int = 3,
    ) -> str:
        """
        执行工具调用循环：LLM ↔ 工具执行 → 直到 LLM 不再请求工具。

        每个 Agent 在需要时通过此循环调用其允许的工具，
        工具白名单由调用方通过 ``tools`` 参数控制。
        """
        # 构建工具名→函数映射
        tool_map: Dict[str, Any] = {}
        for t in tools:
            if hasattr(t, "name") and callable(t):
                tool_map[t.name] = t

        current_messages = list(messages)

        for iteration in range(max_iterations):
            response = llm.invoke(current_messages)
            current_messages.append(response)

            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

            if not tool_calls:
                # LLM 不再请求工具 → 返回文本内容
                return response.content if hasattr(response, "content") else str(response)

            # 执行每个工具调用
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", "")

                log.info(f"调用工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})")

                if tool_name not in tool_map:
                    result = f"错误：工具 '{tool_name}' 不在允许列表中"
                    log.error(result)
                else:
                    try:
                        # 工具参数可能是 dict，转换为位置参数或关键字参数
                        tool_func = tool_map[tool_name]
                        if isinstance(tool_args, dict):
                            # LangChain @tool 装饰的函数接受关键字参数
                            result = tool_func.invoke(tool_args) if hasattr(tool_func, "invoke") else tool_func(**tool_args)
                        else:
                            result = tool_func.invoke({"input": str(tool_args)}) if hasattr(tool_func, "invoke") else tool_func(str(tool_args))

                        result = str(result)
                        log.info(f"工具返回: {result[:150]}...")
                    except Exception as e:
                        result = f"工具执行错误：{e}"
                        log.error(result)

                current_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )

        # 达到最大迭代次数 → 让 LLM 总结
        final_response = llm.invoke(current_messages)
        return final_response.content if hasattr(final_response, "content") else str(final_response)

    # ----------------------------------------------------------
    # JSON 提取
    # ----------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从 LLM 输出中提取 JSON 对象。"""
        # 尝试直接解析
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试提取 { ... } 最外层
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从 LLM 输出中提取有效 JSON: {text[:300]}")

    @staticmethod
    def _fallback_task_extraction(user_query: str) -> List[Dict[str, Any]]:
        """JSON 解析失败时的降级任务提取。"""
        tasks = []
        query_lower = user_query.lower()

        # 简单启发式：检查是否包含计算相关词汇
        calc_keywords = ["计算", "等于", "多少", "求和", "平均值", "+", "-", "*", "/",
                         "sqrt", "sin", "cos", "tan", "log", "公式", "方程"]
        research_keywords = ["什么是", "怎么", "为什么", "新闻", "天气", "如何",
                             "介绍", "定义", "PCB", "硬件", "芯片", "元器件", "焊接"]

        has_calc = any(kw in user_query or kw in query_lower for kw in calc_keywords)
        has_research = any(kw in user_query for kw in research_keywords)

        if has_calc:
            tasks.append({
                "id": len(tasks) + 1,
                "type": "calculate",
                "description": user_query,
            })
        if has_research or not has_calc:
            tasks.append({
                "id": len(tasks) + 1,
                "type": "research",
                "description": user_query,
            })

        return tasks

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------

    def run(
        self,
        query: str,
        session_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行一次多智能体推理。

        Args:
            query:      用户问题
            session_id: 会话标识（用于多轮记忆）
            config:     LangGraph 运行时配置（可选）

        Returns:
            {
                "final_answer": str,
                "plan": str,
                "tasks": list,
                "research_results": list,
                "calculation_results": list,
                "agent_logs": list,
                "round": int,
                "total_time_ms": float,
            }
        """
        if not self._initialized:
            raise RuntimeError("MultiAgentSystem 尚未初始化，请先调用 initialize()")

        # ── 安全护栏：输入过滤 ──
        safe, msg = self.guardrails.filter_input(query)
        if not safe:
            return {
                "final_answer": f"输入被拒绝：{msg}",
                "plan": "",
                "tasks": [],
                "research_results": [],
                "calculation_results": [],
                "agent_logs": [f"[Guardrails] 输入被拒绝: {msg}"],
                "round": 0,
                "total_time_ms": 0,
                "error": msg,
            }

        # ── 构建初始状态 ──
        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "plan": "",
            "tasks": [],
            "complete": False,
            "research_results": [],
            "calculation_results": [],
            "final_answer": "",
            "round": 0,
            "needs_more": False,
            "agent_logs": [],
            "error": "",
        }

        # ── LangGraph 配置 ──
        graph_config = config or {}
        graph_config.setdefault("configurable", {})
        graph_config["configurable"].setdefault("thread_id", session_id)

        # ── 执行状态图 ──
        start_time = time.time()
        all_research = []
        all_calculation = []
        all_logs = []
        final_state = initial_state

        try:
            # 多轮循环（最多 MAX_ROUNDS 轮）
            for round_num in range(self.MAX_ROUNDS):
                final_state["round"] = round_num

                # 执行一轮
                result = self.compiled_graph.invoke(
                    final_state,
                    config=graph_config,
                )

                # 累积结果
                research = result.get("research_results", [])
                calc = result.get("calculation_results", [])
                logs = result.get("agent_logs", [])

                all_research.extend(research)
                all_calculation.extend(calc)
                all_logs.extend(logs)

                # 更新状态为下一轮准备
                final_state = result

                # 如果不需要更多轮，退出
                if not result.get("needs_more", False):
                    break

                # 清除累积列表以便下一轮新鲜接收
                final_state["research_results"] = []
                final_state["calculation_results"] = []
                final_state["agent_logs"] = []

            elapsed_ms = (time.time() - start_time) * 1000

            # ── 输出审核 ──
            raw_answer = final_state.get("final_answer", "")
            _, safe_answer = self.guardrails.filter_output(raw_answer)

            return {
                "final_answer": safe_answer,
                "plan": final_state.get("plan", ""),
                "tasks": final_state.get("tasks", []),
                "research_results": all_research,
                "calculation_results": all_calculation,
                "agent_logs": all_logs,
                "round": final_state.get("round", 0),
                "total_time_ms": round(elapsed_ms, 2),
                "error": final_state.get("error", ""),
            }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self._logger.exception("MultiAgentSystem.run 异常")
            return {
                "final_answer": f"多智能体执行出错：{e}",
                "plan": final_state.get("plan", ""),
                "tasks": final_state.get("tasks", []),
                "research_results": all_research,
                "calculation_results": all_calculation,
                "agent_logs": all_logs + [f"[System] ERROR: {e}"],
                "round": final_state.get("round", 0),
                "total_time_ms": round(elapsed_ms, 2),
                "error": str(e),
            }

    async def arun(
        self,
        query: str,
        session_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """异步执行一次多智能体推理（仅包装，图内部仍同步）。"""
        import asyncio
        return await asyncio.to_thread(self.run, query, session_id, config)

    # ----------------------------------------------------------
    # 日志导出
    # ----------------------------------------------------------

    def get_all_logs(self) -> str:
        """获取所有 Agent 的最近日志。"""
        parts = [
            "=== Supervisor ===",
            self.supervisor_log.dump(),
            "=== Researcher ===",
            self.researcher_log.dump(),
            "=== Calculator ===",
            self.calculator_log.dump(),
            "=== Writer ===",
            self.writer_log.dump(),
        ]
        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# 模块级便捷函数
# ═══════════════════════════════════════════════════════════

_instance: Optional[MultiAgentSystem] = None


def get_instance() -> MultiAgentSystem:
    """获取（或创建）模块级单例。"""
    global _instance
    if _instance is None:
        _instance = MultiAgentSystem()
        _instance.initialize()
    return _instance


def ask(query: str, session_id: str = "default") -> str:
    """快速问答：一行代码完成多智能体推理。

    用法::

        from multi_agent import ask
        answer = ask("今天的科技新闻有哪些？")
        print(answer)
    """
    mas = get_instance()
    result = mas.run(query, session_id=session_id)
    return result.get("final_answer", "未知错误")


# ═══════════════════════════════════════════════════════════
# 独立测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── 配置日志输出 ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  🤖 多智能体系统 独立测试")
    print("=" * 60)

    # ── 初始化 ──
    print("\n⏳ 初始化多智能体系统...")
    mas = MultiAgentSystem()
    mas.initialize()
    print("✅ 初始化完成\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("🧑 请输入问题: ").strip()
        if not query:
            query = "Python语言诞生于哪一年？1+2*3等于多少？"

    print(f"\n📝 用户问题: {query}")
    print("-" * 60)

    result = mas.run(query)

    print(f"\n📋 执行计划: {result.get('plan', 'N/A')}")
    tasks = result.get("tasks", [])
    if tasks:
        for t in tasks:
            print(f"   - [{t.get('type', '?')}] {t.get('description', '')}")

    print(f"\n📊 最终回答:\n{result.get('final_answer', 'N/A')}")
    print(f"\n⏱️ 耗时: {result.get('total_time_ms', 0):.0f} ms")
    print(f"🔄 轮次: {result.get('round', 0) + 1}")

    print(f"\n{'=' * 60}")
    print("📋 Agent 日志:")
    print(mas.get_all_logs())

    if result.get("error"):
        print(f"\n⚠️ 错误: {result['error']}")
