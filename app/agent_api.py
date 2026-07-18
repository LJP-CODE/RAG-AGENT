"""
agent_api.py — FastAPI 服务

提供基于 ReAct Agent 的 Web API，集成 RAG 知识库问答、安全计算、
时间查询、Web 搜索等工具，支持多会话记忆管理。

所有配置通过 config_loader 集中管理，API Key 仅存储在环境变量中。

用法:
    # 启动服务（开发模式）
    uvicorn agent_api:app --reload --host 0.0.0.0 --port 8000

    # 生产模式
    uvicorn agent_api:app --host 0.0.0.0 --port 8000

请求示例:
    # 同步问答
    curl -X POST "http://localhost:8000/ask" \\
      -H "Content-Type: application/json" \\
      -H "X-API-Key: sk-your-api-key" \\
      -d '{"question": "显卡PCB一般有多少层？", "session_id": "user1"}'

    # SSE 流式问答
    curl -X POST "http://localhost:8000/ask/stream" \\
      -H "Content-Type: application/json" \\
      -H "X-API-Key: sk-your-api-key" \\
      -d '{"question": "显卡PCB一般有多少层？", "session_id": "user1"}' \\
      --no-buffer

    curl "http://localhost:8000/health"

    curl -X DELETE "http://localhost:8000/session/user1"
"""

import json
import os
import sys
import time
import traceback
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 确保本模块所在目录在 sys.path 中，以便同目录模块和上级模块的导入
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
_project_root = str(Path(_this_dir).parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag_system import RAGSystem
from web_tools import web_search, read_webpage
from agent_guardrails import AgentGuardrails, create_guardrails_from_config
from agent_monitor import AgentMonitor
from config_loader import get_config
from session_manager import get_session_manager


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="AI Agent 问答 API", version="2.1.0")


# ============================================================
# API Key 认证
# ============================================================

# X-API-Key 请求头提取器
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_valid_api_keys() -> set:
    """
    从环境变量 API_KEYS 读取合法的 Key 列表。

    API_KEYS 格式：逗号分隔的 Key 字符串。
    示例: API_KEYS=sk-abc123,sk-def456,sk-ghi789
    """
    raw = os.getenv("API_KEYS", "")
    if not raw.strip():
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI 依赖：验证请求中的 X-API-Key 是否合法。

    逻辑：
    1. 如果未配置 API_KEYS 环境变量 → 允许所有请求（向后兼容）
    2. 如果配置了 API_KEYS → 仅允许列表中的 Key 通过
    3. Key 不在白名单中 → 返回 401 Unauthorized

    Returns:
        api_key: 验证通过的 API Key 字符串
    """
    valid_keys = _load_valid_api_keys()

    # 未配置任何 Key → 开放模式（向后兼容）
    if not valid_keys:
        return "anonymous"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="缺少 X-API-Key 请求头，请在请求中提供有效的 API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="无效的 API Key，拒绝访问",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


# ============================================================
# 请求 / 响应 模型
# ============================================================

class AskRequest(BaseModel):
    """问答请求体。"""
    question: str
    session_id: str = "default"       # 用于区分不同用户/对话
    temperature: float = 0.1          # LLM 采样温度


class AskResponse(BaseModel):
    """问答响应体。"""
    answer: str                       # Agent 最终回答
    session_id: str                   # 会话标识
    tools_used: List[str]             # 本次调用实际使用了哪些工具
    total_time_ms: float              # 总耗时（毫秒）
    steps: int                        # 推理步数（工具调用次数）


class SecurityStatus(BaseModel):
    """安全状态响应体。"""
    session_id: str
    consecutive_violations: int
    is_frozen: bool
    call_count: int


class FrozenSessionsResponse(BaseModel):
    """冻结会话列表响应。"""
    total_frozen: int
    frozen_sessions: Dict[str, float]


class SessionSummary(BaseModel):
    """会话摘要（列表用，不含消息详情）。"""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class SessionDetail(BaseModel):
    """会话详情（含完整消息列表）。"""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list


class CreateSessionResponse(BaseModel):
    """创建会话响应。"""
    id: str
    title: str
    created_at: str


# ============================================================
# 全局变量
# ============================================================

# 配置（通过 config_loader 单例获取）
config = get_config()

# RAG 知识库实例
rag: Optional[RAGSystem] = None

# 语言模型
llm: Optional[ChatOpenAI] = None

# 工具列表
tools: Optional[list] = None

# Agent 提示词模板
prompt: Optional[ChatPromptTemplate] = None

# 会话记忆存储
sessions: Dict[str, ConversationBufferMemory] = {}
sessions_lock = asyncio.Lock()

# ── 安全护栏 & 链路追踪（无状态，全局单例）──
guardrails = create_guardrails_from_config(config)
monitor = AgentMonitor()


# ============================================================
# Token 统计回调 — 从 DeepSeek 响应中真实提取 token 用量
# ============================================================

class TokenCounterCallback(BaseCallbackHandler):
    """LangChain 回调：捕获每次 LLM 调用的真实 token 用量。

    兼容 DeepSeek（OpenAI 兼容接口）的 response_metadata.token_usage 格式。
    每次 Agent 请求创建新实例，避免跨会话污染。
    """

    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def on_llm_end(self, response, **kwargs) -> None:
        """LLM 调用结束时提取 token_usage。"""
        self.call_count += 1
        usage = {}

        # 路径 1: response.llm_output["token_usage"]（ChatOpenAI 标准路径）
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})

        # 路径 2: response.generations[0][0].message.response_metadata（备选路径）
        if not usage and hasattr(response, "generations") and response.generations:
            try:
                gen = response.generations[0][0]
                msg = getattr(gen, "message", None)
                if msg and hasattr(msg, "response_metadata"):
                    usage = msg.response_metadata.get("token_usage", {})
            except (IndexError, AttributeError):
                pass

        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)


# ============================================================
# 工具函数（复用 agent_system.py 逻辑）
# ============================================================

def rag_search(query: str) -> str:
    """
    工具：知识库搜索

    调用 RAGSystem.ask() 回答电子硬件 / PCB / 焊接工艺等相关问题。
    在回答末尾附上来源文档引用，提升可信度。
    """
    try:
        result = rag.ask(query, temperature=config.agent.temperature)
        answer = result.get("answer", "")
        if not answer:
            return "未在知识库中找到相关答案。"

        # 附上 Rerank 后的来源引用
        reranked_docs = result.get("reranked_docs", [])
        if reranked_docs:
            citations = []
            for i, doc in enumerate(reranked_docs[:3], 1):
                snippet = doc.get("content", "")[:120].replace("\n", " ")
                citations.append(f"  [来源{i}] {snippet}...")
            answer += "\n\n📚 参考来源（知识库）：\n" + "\n".join(citations)

        return answer
    except Exception as e:
        return f"知识库查询出错：{e}"


def calculator(expression: str) -> str:
    """
    工具：安全数学计算

    使用 eval() 但严格限制命名空间，只暴露白名单内的函数和常量。
    """
    import re
    import math

    if not expression or not isinstance(expression, str):
        return "错误：请输入有效的数学表达式"

    cleaned = expression.strip()

    # 替换人类习惯的符号 → Python 运算符（先替换再校验）
    cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("^", "**")

    # 正则校验：只允许数学表达式合法字符
    if not re.match(r'^[\d+\-*/().,%^eE\s]+$', cleaned):
        return "错误：表达式包含非法字符"

    expr = cleaned

    # 白名单命名空间（严格限制 eval 可用内容）
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
        result = eval(expr, safe_globals, safe_locals)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误：{e}"


def get_current_time(_: str = "") -> str:
    """
    工具：获取当前日期时间
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 构建 Agent 组件（工具、LLM、提示词）
# ============================================================

def build_agent_components():
    """
    构建并返回 (tools, llm, prompt) 三元组。

    这些组件是全局共享的（无状态），在启动时构造一次即可。
    """
    # ── 注册工具 ──
    tools = [
        Tool(
            name="RAG_Search",
            func=rag_search,
            description=(
                "适用于电子硬件、PCB 设计、显卡、焊接工艺、元器件等知识库相关的问题。"
                "输入：用户的中文问题。输出：基于知识库的详细回答。"
            ),
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description=(
                "适用于数学计算任务。输入：数学表达式，如 '1+2*3'、'sqrt(144)'、'2**10'。"
                "输出：计算结果。"
            ),
        ),
        Tool(
            name="Get_Time",
            func=get_current_time,
            description=(
                "获取当前的日期和时间。当用户问「现在几点」「今天几号」「当前时间」时使用。"
                "无需特定输入，传入空字符串即可。"
            ),
        ),
        Tool(
            name="Web_Search",
            func=web_search,
            description=(
                "搜索互联网获取实时信息。当本地知识库没有答案、或用户问的是"
                "最新新闻/事件/人物/天气/股价等实时信息时使用。"
                "输入是搜索关键词，输出是搜索结果标题和摘要。"
                "如果摘要不够详细，可以再调用 Read_Webpage 读取全文。"
            ),
        ),
        Tool(
            name="Read_Webpage",
            func=read_webpage,
            description=(
                "读取指定网页的正文内容。输入是完整的 URL。"
                "当 Web_Search 返回的摘要信息不够详细时，可以使用此工具获取全文。"
                "也适用于用户直接要求查看某个网页内容时使用。"
            ),
        ),
    ]

    # ── LLM（DeepSeek 兼容 OpenAI API）──
    # API Key 仅从环境变量读取，不在配置文件中存储
    api_key = config.llm.api_key
    if not config.llm.is_configured:
        raise ValueError(
            "未找到 DEEPSEEK_API_KEY 环境变量。\n"
            "请复制 .env.example 为 .env 并填入真实密钥，\n"
            "或执行: $env:DEEPSEEK_API_KEY = 'your-api-key'"
        )

    llm = ChatOpenAI(
        model=config.llm.model,
        api_key=api_key,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
    )

    # ── Agent 提示词（Tool Calling 格式）──
    # Tool Calling 模式不需要 ReAct 的 Thought/Action 文本模板，
    # LLM 原生返回 tool_calls JSON，框架自动处理工具调度。
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助理，可以调用工具来完成任务。

可用工具使用场景：
- 硬件/PCB/焊接工艺等本地知识库问题 → RAG_Search
- 本地知识库没有答案，或需要实时信息（新闻、股价、天气、人物）→ Web_Search
- Web_Search 摘要不够详细，或用户要看具体网页 → Read_Webpage
- 数学计算 → Calculator
- 日期时间查询 → Get_Time
- 多轮对话时参考历史记录

请用中文回答用户问题，回答要准确、详细。如果使用工具，优先选择合适的工具获取信息后再回答。
如果调用了知识库或搜索工具，回答中注明信息来源。

【重要规则：搜索失败处理】
当你调用 Web_Search 工具后，必须遵循以下规则：

1. **搜索无结果或服务不可用**：
   如果 Web_Search 返回的内容中包含「未找到相关信息」「未找到有效结果」
   「搜索服务暂时不可用」等提示，你必须直接告诉用户：
   "没有找到相关信息，建议换个关键词试试。"
   禁止在这种情况下继续编造答案。

2. **搜索结果不相关**：
   如果 Web_Search 返回了结果，但内容与用户的问题明显不相关
   （例如用户问的是科技新闻，返回的却是无关网站链接），你必须告诉用户：
   "搜索结果不太相关，建议换个关键词重试。"
   禁止强行使用不相关的内容回答问题。

3. **禁止编造**：
   绝对不要基于不相关、无效或为空的搜索结果编造答案。
   如果没有可靠的信息来源，诚实地告诉用户你无法回答，
   并给出改进搜索的建议（如换关键词、加城市名、简化问题等）。

4. **RAG_Search 同样适用**：
   上述规则也适用于 RAG_Search。如果知识库返回「未在知识库中找到相关答案」，
   不要编造，建议用户尝试 Web_Search 或换个问法。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    return tools, llm, prompt


# ============================================================
# 获取或创建会话记忆
# ============================================================

async def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """
    根据 session_id 获取已有的 ConversationBufferMemory，不存在则新建。

    每个会话的记忆独立存储，不会互相干扰。
    """
    async with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )
            print(f"  📝 创建新会话: {session_id}")
        return sessions[session_id]


# ============================================================
# Lifespan 生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时，自动加载 RAG 系统并构建 Agent 组件。"""
    global rag, llm, tools, prompt, config

    print("=" * 60)
    print("  AI Agent API 服务启动中...")
    print("=" * 60)

    # 0. 配置已在模块顶部通过 get_config() 加载
    #    可通过环境变量 CONFIG_PATH 覆盖配置文件路径
    cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = get_config(cfg_path)

    # 0.5 初始化会话持久化管理器
    _ = get_session_manager()

    # 1. 加载 RAG 系统
    print("\n🔄 [1/2] 正在加载 RAG 系统...")
    rag = RAGSystem(
        knowledge_path=config.rag.knowledge_path,
        chunk_size=config.rag.chunk_size,
        overlap=config.rag.chunk_overlap,  # RAGSystem 参数名为 overlap
    )
    rag.PERSIST_DIR = config.data.chroma_db_dir
    rag.initialize()

    # 2. 构建 Agent 组件
    print("\n🔄 [2/2] 正在构建 Agent 组件...")
    tools, llm, prompt = build_agent_components()
    print(f"    工具数量: {len(tools)}")
    for t in tools:
        print(f"      - {t.name}")

    # 3. 模型预热（减少首次请求延迟）
    print("\n🔄 [3/3] 模型预热中...")
    try:
        _ = llm.invoke("ping")
        print("    ✅ LLM 预热完成")
    except Exception as e:
        print(f"    ⚠️  LLM 预热跳过: {e}")

    try:
        rag.ask("预热测试", temperature=0.0)
        print("    ✅ RAG 预热完成（含 Rerank 模型加载）")
    except Exception as e:
        print(f"    ⚠️  RAG 预热跳过: {e}")

    print("\n✅ Agent 已就绪，等待请求...")
    print(f"   当前活跃会话数: {len(sessions)}")
    print(f"   输入长度限制: {config.security.max_input_length} 字符")
    print(f"   输出长度限制: {config.security.max_output_length} 字符")
    print(f"   限流: {config.security.rate_limit_per_minute}/分钟, "
          f"{config.security.rate_limit_per_hour}/小时")
    print(f"   会话最大调用: {config.security.max_calls_per_session} 次")
    print(f"   冻结触发: 连续 {config.security.freeze_trigger_count} 次违规")
    print(f"   冻结时长: {config.security.freeze_duration_minutes} 分钟")
    print("=" * 60)

    yield  # 服务运行中...

    # ── 关闭时清理 ──
    print("\n🛑 服务关闭中...")
    deleted = guardrails.cleanup_audit_logs()
    if deleted:
        print(f"  🧹 清理了 {deleted} 个过期审计日志文件")


# 注册 lifespan
app.router.lifespan_context = lifespan


# ============================================================
# /ask — 问答接口
# ============================================================

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, api_key: str = Depends(verify_api_key)):
    """
    接收用户问题，使用 Agent 自动选择工具执行推理，返回最终答案。

    每次请求根据 session_id 绑定对应的会话记忆，实现多轮对话上下文。
    集成安全护栏（输入过滤 + 限流 + 输出审核 + 异常行为检测）与链路追踪。
    """
    global rag, llm, tools, prompt

    # 校验服务是否就绪
    if rag is None or llm is None:
        raise HTTPException(status_code=503, detail="服务尚未完全初始化，请稍后重试")

    # ════════════════════════════════════════════════════
    # 前置防护：限流 → 输入过滤 → 异常行为检测
    # ════════════════════════════════════════════════════

    # 限流检查（含冻结检查 + 会话调用次数检查）
    allowed, rate_msg = guardrails.check_rate_limit(request.session_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=rate_msg)

    # 输入安全过滤
    safe, input_msg, triggered_rules = guardrails.filter_input(request.question)
    if not safe:
        # 记录违规（可能触发冻结）
        guardrails.record_violation(
            request.session_id, triggered_rules, request.question
        )
        # 检查是否刚被冻结
        if guardrails.is_frozen(request.session_id):
            raise HTTPException(
                status_code=429,
                detail=f"该会话因连续触发安全规则已被冻结 "
                       f"{config.security.freeze_duration_minutes} 分钟",
            )
        raise HTTPException(status_code=400, detail=input_msg)

    # 本次请求无违规，重置连续违规计数
    guardrails.record_clean_request(request.session_id)

    # ════════════════════════════════════════════════════
    # 开始链路追踪
    # ════════════════════════════════════════════════════
    trace = monitor.start_trace(request.session_id, request.question)

    # 获取该会话的记忆
    memory = await get_or_create_memory(request.session_id)

    # 每次请求重新创建 AgentExecutor（绑定独立的会话记忆）
    token_counter = TokenCounterCallback()
    agent_executor = AgentExecutor(
        agent=create_tool_calling_agent(llm, tools, prompt),
        tools=tools,
        memory=memory,
        verbose=config.agent.verbose,
        max_iterations=config.agent.max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        callbacks=[token_counter],
    )

    # ════════════════════════════════════════════════════
    # 执行推理
    # ════════════════════════════════════════════════════
    start_time = time.time()
    try:
        # ── 正常路径：Agent 推理 ──
        # 使用脱敏后的参数（日志安全）
        safe_params = guardrails.sanitize_parameters({
            "input": request.question,
            "session_id": request.session_id,
            "temperature": request.temperature,
        })
        result = await agent_executor.ainvoke({"input": request.question})
        elapsed_ms = (time.time() - start_time) * 1000

        # 提取工具使用信息并记录到追踪
        intermediate_steps = result.get("intermediate_steps", [])
        tools_used = list(dict.fromkeys(
            step[0].tool for step in intermediate_steps if hasattr(step[0], "tool")
        ))

        for step in intermediate_steps:
            action = getattr(step[0], "log", "") or str(step[0])
            tool_name = getattr(step[0], "tool", "unknown")
            tool_input = getattr(step[0], "tool_input", "")
            observation = str(step[1]) if len(step) > 1 else ""
            trace.add_step(action, tool_name, tool_input, observation, duration_ms=0)

        raw_output = result.get("output", "")

    except OutputParserException as e:
        # ════════════════════════════════════════════════════
        # 异常分支 1：输出解析失败 → 降级为直接调用 LLM
        # ════════════════════════════════════════════════════
        elapsed_ms = (time.time() - start_time) * 1000
        tools_used = []
        intermediate_steps = []

        # 记录完整错误信息到日志（方便调试）
        print(f"  ⚠️  [Parser降级] OutputParserException 被捕获:")
        print(f"     错误详情: {e}")
        print(f"     堆栈跟踪:\n{traceback.format_exc()}")
        monitor.log_error(
            f"OutputParserException: {e}\n{traceback.format_exc()}",
            trace,
        )

        # 降级：绕过 Agent 框架，直接调用 LLM 回答问题
        try:
            fallback_messages = [
                SystemMessage(content=(
                    "你是一个智能助理，请用中文回答用户问题，回答要准确、详细。"
                    "如果知识不足，诚实地说明无法回答，不要编造信息。"
                )),
                HumanMessage(content=request.question),
            ]
            fallback_result = llm.invoke(fallback_messages)
            raw_output = (
                fallback_result.content
                if hasattr(fallback_result, "content")
                else str(fallback_result)
            )
            print(f"  ✅ [Parser降级] 降级 LLM 调用成功")

        except Exception as llm_err:
            # LLM 降级本身也失败了 — 返回纯友好提示
            print(f"  ❌ [Parser降级] 降级 LLM 也失败了: {llm_err}")
            raw_output = (
                "抱歉，我在处理您的请求时遇到了一些困难。\n\n"
                "这可能是由于模型输出的格式异常导致的，请尝试：\n"
                "1. 换一种方式重新提问\n"
                "2. 将问题简化后再问\n"
                "3. 等待片刻后重试\n\n"
                "如问题持续出现，请联系管理员。"
            )

    except Exception as e:
        # ════════════════════════════════════════════════════
        # 异常分支 2：其他所有异常 → 降级为直接调用 LLM
        # ════════════════════════════════════════════════════
        elapsed_ms = (time.time() - start_time) * 1000
        tools_used = []
        intermediate_steps = []

        # 记录完整错误信息到日志（方便调试）
        error_type = type(e).__name__
        error_detail = str(e)
        print(f"  ❌ [Agent异常] {error_type}: {error_detail}")
        print(f"     堆栈跟踪:\n{traceback.format_exc()}")
        monitor.log_error(
            f"{error_type}: {error_detail}\n{traceback.format_exc()}",
            trace,
        )

        # ── 降级策略：直接调用 LLM 回答（绕过 Agent 框架）──
        # 无论是 RuntimeError、ConnectionError 还是其他未预期的异常，
        # 优先尝试直接 LLM 调用，给用户一个有价值的回复
        try:
            fallback_messages = [
                SystemMessage(content=(
                    "你是一个智能助理，请用中文回答用户问题，回答要准确、详细。"
                    "如果知识不足，诚实地说明无法回答，不要编造信息。"
                )),
                HumanMessage(content=request.question),
            ]
            fallback_result = llm.invoke(fallback_messages)
            raw_output = (
                fallback_result.content
                if hasattr(fallback_result, "content")
                else str(fallback_result)
            )
            print(f"  ✅ [Agent降级] Agent 异常 ({error_type})，已降级为直接 LLM 调用")

        except Exception as llm_err:
            # LLM 降级本身也失败了 — 返回纯友好提示
            print(f"  ❌ [Agent降级] 降级 LLM 也失败了: {llm_err}")
            monitor.log_error(
                f"LLM fallback also failed: {llm_err}",
                trace,
            )

            # 根据错误类型给出不同的友好提示
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raw_output = (
                    "抱歉，处理您的请求超时了。\n\n"
                    "这可能是因为问题比较复杂或当前系统负载较高，"
                    "请尝试简化问题或稍后重试。"
                )
            elif "rate limit" in str(e).lower() or "too many" in str(e).lower():
                raw_output = (
                    "抱歉，当前请求过于频繁，请稍等片刻后再试。"
                )
            elif "connection" in str(e).lower() or "connect" in str(e).lower():
                raw_output = (
                    "抱歉，服务暂时无法连接到后端模型，请稍后重试。"
                )
            else:
                raw_output = (
                    f"抱歉，服务暂时无法处理您的请求，请稍后重试。\n\n"
                    f"[错误参考: {error_type}]"
                )

    # ════════════════════════════════════════════════════
    # 后置防护：输出审核（所有分支共用）
    # ════════════════════════════════════════════════════
    output_clean, final_answer, output_rules = guardrails.filter_output(raw_output)

    # 输出违规记录（不拒绝请求，但记录审计日志）
    if output_rules:
        guardrails.record_violation(
            request.session_id, output_rules, raw_output
        )

    # ════════════════════════════════════════════════════
    # 结束追踪（所有分支共用）
    # ════════════════════════════════════════════════════
    total_tokens = token_counter.total_tokens  # 从 DeepSeek 响应真实提取
    monitor.end_trace(trace, final_answer, total_tokens=total_tokens)

    # ════════════════════════════════════════════════════
    # 持久化保存：用户消息 + AI 回答
    # ════════════════════════════════════════════════════
    try:
        mgr = get_session_manager()
        mgr.add_message(request.session_id, "user", request.question)
        mgr.add_message(
            request.session_id, "assistant", final_answer,
            tools_used=tools_used, time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        # 保存失败不应影响正常响应
        print(f"  ⚠️  会话保存失败: {e}")

    # ════════════════════════════════════════════════════
    # 返回 AskResponse（绝不返回 500）
    # ════════════════════════════════════════════════════
    return AskResponse(
        answer=final_answer,
        session_id=request.session_id,
        tools_used=tools_used,
        total_time_ms=round(elapsed_ms, 2),
        steps=len(intermediate_steps),
    )


# ============================================================
# /ask/stream — SSE 流式问答接口
# ============================================================

@app.post("/ask/stream")
async def ask_question_stream(request: AskRequest, api_key: str = Depends(verify_api_key)):
    """
    流式输出版本的 /ask 接口。

    通过 Server-Sent Events (SSE) 实时推送 Agent 推理过程与回答内容。

    **SSE 事件类型：**

    | 事件      | 说明                                             |
    |-----------|--------------------------------------------------|
    | `status`  | Agent 状态更新（开始、工具调用等）               |
    | `token`   | 逐字输出的回答文本片段                           |
    | `done`    | 完成信息（总回答、工具、耗时、步数）             |
    | `error`   | 错误信息                                         |

    **客户端示例（JavaScript）：**

    ```js
    const es = new EventSource("/ask/stream");
    es.addEventListener("status", e => console.log("状态:", JSON.parse(e.data)));
    es.addEventListener("token",  e => outputEl.textContent += JSON.parse(e.data).token);
    es.addEventListener("done",   e => console.log("完成:", JSON.parse(e.data)));
    es.addEventListener("error",  e => console.error(JSON.parse(e.data).error));
    ```
    """
    global rag, llm, tools, prompt

    # ── 校验服务是否就绪 ──
    if rag is None or llm is None:
        raise HTTPException(status_code=503, detail="服务尚未完全初始化，请稍后重试")

    # ════════════════════════════════════════════════════
    # 前置防护（与 /ask 相同）
    # ════════════════════════════════════════════════════

    allowed, rate_msg = guardrails.check_rate_limit(request.session_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=rate_msg)

    safe, input_msg, triggered_rules = guardrails.filter_input(request.question)
    if not safe:
        guardrails.record_violation(request.session_id, triggered_rules, request.question)
        if guardrails.is_frozen(request.session_id):
            raise HTTPException(
                status_code=429,
                detail=f"该会话因连续触发安全规则已被冻结 "
                       f"{config.security.freeze_duration_minutes} 分钟",
            )
        raise HTTPException(status_code=400, detail=input_msg)

    guardrails.record_clean_request(request.session_id)

    # ════════════════════════════════════════════════════
    # SSE 异步事件生成器
    # ════════════════════════════════════════════════════

    async def event_generator():
        """生成 SSE 事件流：status → token → done / error"""
        # ── 发送开始状态 ──
        yield {
            "event": "status",
            "data": json.dumps({
                "phase": "started",
                "message": "Agent 开始分析问题...",
                "session_id": request.session_id,
                "timestamp": datetime.now().isoformat(),
            }, ensure_ascii=False),
        }

        # ── 链路追踪 ──
        trace = monitor.start_trace(request.session_id, request.question)

        # ── 获取会话记忆 ──
        memory = await get_or_create_memory(request.session_id)

        # ── 创建 AgentExecutor ──
        token_counter = TokenCounterCallback()
        agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(llm, tools, prompt),
            tools=tools,
            memory=memory,
            verbose=config.agent.verbose,
            max_iterations=config.agent.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            callbacks=[token_counter],
        )

        # ── 执行推理 ──
        start_time = time.time()
        try:
            result = await agent_executor.ainvoke({"input": request.question})
            elapsed_ms = (time.time() - start_time) * 1000
        except Exception as e:
            monitor.log_error(str(e), trace)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": f"Agent 执行失败: {str(e)}",
                    "session_id": request.session_id,
                }, ensure_ascii=False),
            }
            return

        # ── 提取工具使用信息 ──
        intermediate_steps = result.get("intermediate_steps", [])
        tools_used = list(dict.fromkeys(
            step[0].tool for step in intermediate_steps if hasattr(step[0], "tool")
        ))

        # ── 发送工具调用状态（每步一个事件） ──
        for step in intermediate_steps:
            action = getattr(step[0], "log", "") or str(step[0])
            tool_name = getattr(step[0], "tool", "unknown")
            tool_input = getattr(step[0], "tool_input", "")
            observation = str(step[1]) if len(step) > 1 else ""
            trace.add_step(action, tool_name, tool_input, observation, duration_ms=0)

            yield {
                "event": "status",
                "data": json.dumps({
                    "phase": "tool_call",
                    "tool": tool_name,
                    "input_preview": str(tool_input)[:200],
                    "observation_preview": observation[:200],
                }, ensure_ascii=False),
            }

        # ── 输出审核 ──
        raw_output = result.get("output", "")
        output_clean, final_answer, output_rules = guardrails.filter_output(raw_output)

        if output_rules:
            guardrails.record_violation(request.session_id, output_rules, raw_output)

        # ── 发送生成中的状态 ──
        yield {
            "event": "status",
            "data": json.dumps({
                "phase": "generating",
                "message": "正在生成回答...",
                "tools_used": tools_used,
            }, ensure_ascii=False),
        }

        # ════════════════════════════════════════════════════
        # 逐字流式输出回答（每次输出 3 个字符，间隔 50ms）
        # ════════════════════════════════════════════════════
        chunk_size = 3
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i:i + chunk_size]
            yield {
                "event": "token",
                "data": json.dumps({"token": chunk}, ensure_ascii=False),
            }
            await asyncio.sleep(0.05)

        # ════════════════════════════════════════════════════
        # 发送完成信息
        # ════════════════════════════════════════════════════
        total_tokens = token_counter.total_tokens
        monitor.end_trace(trace, final_answer, total_tokens=total_tokens)

        yield {
            "event": "done",
            "data": json.dumps({
                "status": "completed",
                "session_id": request.session_id,
                "answer": final_answer,
                "tools_used": tools_used,
                "total_time_ms": round(elapsed_ms, 2),
                "steps": len(intermediate_steps),
            }, ensure_ascii=False),
        }

    return EventSourceResponse(event_generator())


# ============================================================
# /sessions — 会话管理接口（持久化）
# ============================================================

@app.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    """获取所有历史会话列表（按更新时间倒序）。"""
    mgr = get_session_manager()
    return mgr.list_sessions()


@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session():
    """新建一个空白会话，返回会话 ID。"""
    mgr = get_session_manager()
    session = mgr.create_session()
    return CreateSessionResponse(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
    )


@app.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session_detail(session_id: str):
    """获取指定会话的完整数据（含所有消息）。"""
    mgr = get_session_manager()
    data = mgr.get_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"会话 '{session_id}' 不存在")
    return SessionDetail(
        id=data["id"],
        title=data.get("title", "新对话"),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        messages=data.get("messages", []),
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话的持久化数据（同时清除内存中的对话记忆和限流计数）。"""
    mgr = get_session_manager()
    # 删除持久化文件
    existed = mgr.delete_session(session_id)
    # 同时清除内存中的 ConversationBufferMemory
    async with sessions_lock:
        if session_id in sessions:
            del sessions[session_id]
            guardrails.reset_rate_limit(session_id)
            guardrails.unfreeze_session(session_id)
    if existed:
        return {"status": "ok", "message": f"会话 '{session_id}' 已删除"}
    return {"status": "not_found", "message": f"会话 '{session_id}' 不存在"}


# ============================================================
# /health — 健康检查接口
# ============================================================

@app.get("/health")
async def health_check():
    """返回服务状态，包括 RAG 和 Agent 的就绪情况及安全统计。"""
    return {
        "status": "ok",
        "rag_loaded": rag is not None,
        "agent_ready": llm is not None,
        "active_sessions": len(sessions),
        "frozen_sessions": len(guardrails.get_frozen_sessions()),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# /session/{session_id} — 清除会话记忆
# ============================================================

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清除指定会话的对话记忆和限流计数。"""
    async with sessions_lock:
        if session_id in sessions:
            del sessions[session_id]
            guardrails.reset_rate_limit(session_id)
            # 如果已冻结，也解冻
            guardrails.unfreeze_session(session_id)
            return {
                "status": "ok",
                "message": f"会话 '{session_id}' 已清除（含记忆、限流、冻结状态）",
            }
        return {
            "status": "not_found",
            "message": f"会话 '{session_id}' 不存在",
        }


# ============================================================
# /security/status/{session_id} — 查询会话安全状态
# ============================================================

@app.get("/security/status/{session_id}", response_model=SecurityStatus)
async def get_security_status(session_id: str):
    """查询指定会话的安全状态（违规计数、冻结状态、调用次数）。"""
    summary = guardrails.get_violation_summary(session_id)
    return SecurityStatus(**summary)


# ============================================================
# /security/frozen — 查询所有冻结会话
# ============================================================

@app.get("/security/frozen", response_model=FrozenSessionsResponse)
async def get_frozen_sessions():
    """查询所有当前处于冻结状态的会话。"""
    frozen = guardrails.get_frozen_sessions()
    return FrozenSessionsResponse(
        total_frozen=len(frozen),
        frozen_sessions=frozen,
    )


# ============================================================
# /security/unfreeze/{session_id} — 手动解冻会话
# ============================================================

@app.post("/security/unfreeze/{session_id}")
async def unfreeze_session(session_id: str):
    """手动解冻指定会话（管理员操作）。"""
    if guardrails.unfreeze_session(session_id):
        return {
            "status": "ok",
            "message": f"会话 '{session_id}' 已手动解冻",
        }
    return {
        "status": "not_found",
        "message": f"会话 '{session_id}' 未处于冻结状态",
    }


# ============================================================
# /config/reload — 重新加载配置（运行时热更新）
# ============================================================

@app.post("/config/reload")
async def reload_config():
    """重新加载 config.yaml 配置（热更新）。"""
    global config, guardrails
    cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = get_config(cfg_path)
    config.reload(cfg_path)
    # 重新获取配置
    config = get_config(cfg_path)
    # 更新护栏配置
    guardrails.update_config(
        max_input_length=config.security.max_input_length,
        max_output_length=config.security.max_output_length,
        rate_limit_per_minute=config.security.rate_limit_per_minute,
        rate_limit_per_hour=config.security.rate_limit_per_hour,
        freeze_duration_minutes=config.security.freeze_duration_minutes,
        max_calls_per_session=config.security.max_calls_per_session,
        freeze_trigger_count=config.security.freeze_trigger_count,
    )
    return {
        "status": "ok",
        "message": "配置已重新加载",
        "security": {
            "max_input_length": config.security.max_input_length,
            "max_output_length": config.security.max_output_length,
            "rate_limit_per_minute": config.security.rate_limit_per_minute,
            "rate_limit_per_hour": config.security.rate_limit_per_hour,
        },
    }


# ============================================================
# 直接运行入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("正在启动 AI Agent API 服务...")
    print(f"  配置文件: {os.getenv('CONFIG_PATH', 'config.yaml')}")
    print(f"  .env 加载: {_project_root}/.env")
    print("推荐使用: uvicorn agent_api:app --reload --host 0.0.0.0 --port 8000\n")

    uvicorn.run(
        "agent_api:app",
        host=config.app.host,
        port=config.app.port,
        reload=config.app.reload,
    )
