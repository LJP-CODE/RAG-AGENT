"""
AI Agent 交互式终端客户端（自包含版本）
======================================
内置 Agent 构建逻辑，不依赖外部 agent_api.py。
使用 create_tool_calling_agent 适配 DeepSeek 原生 tool calling。

功能:
  - 5 个工具: RAG_Search, Calculator, Get_Time, Web_Search, Read_Webpage
  - 长期记忆 (LongTermMemory) 持久化对话历史
  - 安全护栏 (AgentGuardrails) 输入过滤 + 输出审核
  - 多会话切换

用法:
    python chat_cli.py
"""

import os
import sys
import time
import re
import math
from datetime import datetime
from pathlib import Path

# ── 确保能找到项目模块 ──
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

# ── 离线模式（必须在 HF 相关 import 之前设置）──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── 加载 .env ──
from dotenv import load_dotenv
load_dotenv()

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.rag_system import RAGSystem
from app.memory_store import LongTermMemory
from app.agent_guardrails import AgentGuardrails
from web_tools import web_search, read_webpage, reset_read_counter


# ============================================================
# 全局状态
# ============================================================

rag: "RAGSystem | None" = None
agent_executor: "AgentExecutor | None" = None
sessions: dict = {}                          # session_id → ConversationBufferMemory
guardrails = AgentGuardrails()
long_term_mem = LongTermMemory()


# ============================================================
# Agent 构建（一次性初始化）
# ============================================================

def build_agent():
    """构建 Tool Calling Agent。"""
    global rag, agent_executor

    # ── 1. 加载 RAG ──
    print("  [RAG] 正在加载知识库...")
    rag = RAGSystem(
        knowledge_path="data/knowledge/knowledge.txt",
        chunk_size=256,
        overlap=128,
    )
    rag.initialize()

    # ── 2. 工具函数 ──
    def rag_search(query: str) -> str:
        try:
            result = rag.ask(query, temperature=0.1)
            return result.get("answer", "") or "未在知识库中找到相关答案。"
        except Exception as e:
            return f"知识库查询出错：{e}"

    def calculator(expression: str) -> str:
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
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return str(result)
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except Exception as e:
            return f"计算错误：{e}"

    def get_current_time(_: str = "") -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 3. 注册工具 ──
    tools = [
        Tool(
            name="RAG_Search",
            func=rag_search,
            description=(
                "查询电子硬件 / PCB 设计 / 显卡 / 焊接工艺 / 元器件等本地知识库。"
                "输入是用户问题，输出是基于本地知识库的详细回答。"
            ),
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description=(
                "执行数学计算。输入是数学表达式，如 '1+2*3'、'sqrt(144)'、'2**10'。"
                "支持运算符 + - * / ** %% () 及函数 sqrt sin cos tan log abs round。"
            ),
        ),
        Tool(
            name="Get_Time",
            func=get_current_time,
            description="获取当前的日期和时间。当用户问「现在几点」「今天几号」时使用。",
        ),
        Tool(
            name="Web_Search",
            func=web_search,
            description=(
                "搜索互联网获取实时信息。当本地知识库没有答案、或用户问的是"
                "最新新闻/事件/人物/天气/股价等实时信息时使用。"
                "输入是搜索关键词，输出是搜索结果标题和摘要。"
            ),
        ),
        Tool(
            name="Read_Webpage",
            func=read_webpage,
            description=(
                "读取指定网页的正文内容。输入是完整的 URL。"
                "当 Web_Search 摘要不够详细时使用。"
            ),
        ),
    ]

    # ── 4. LLM ──
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools(tools)

    # ── 5. Prompt 模板 ──
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助理，可以根据用户问题调用合适的工具。

可用工具：
1. RAG_Search    — 查询本地电子硬件 / PCB / 显卡知识库
2. Calculator    — 执行数学计算
3. Get_Time      — 获取当前日期和时间
4. Web_Search    — 搜索互联网获取实时信息
5. Read_Webpage  — 读取网页正文全文

工具选择规则：
1. 硬件/PCB/焊接工艺等本地知识库问题 → RAG_Search
2. 本地知识库没有答案，或需要实时信息 → Web_Search
3. Web_Search 摘要不够详细 → Read_Webpage
4. 数学计算 → Calculator
5. 时间日期 → Get_Time
6. 不需要工具时直接回答
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # ── 6. Agent Executor ──
    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    )
    print(f"    工具数量: {len(tools)}")


# ============================================================
# 会话记忆
# ============================================================

def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """获取或创建会话记忆，并注入长期记忆中的历史上下文。"""
    if session_id not in sessions:
        sessions[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        # 从长期记忆加载历史
        history = long_term_mem.load_memory(session_id)
        if history and history != "暂无历史记录。":
            print(f"  📝 已加载长期记忆: {session_id}")
    return sessions[session_id]


# ============================================================
# 交互式对话循环
# ============================================================

def chat_loop():
    """主交互循环。"""
    global agent_executor

    print("\n💬 AI Agent 交互终端（输入 /help 查看命令）\n")

    # 设置默认会话
    session_id = input("🔑 会话标识（直接回车使用 default）：").strip() or "default"
    memory = get_or_create_memory(session_id)

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # ── 内置命令 ──
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print("👋 再见！")
            break

        if cmd == "/clear":
            if session_id in sessions:
                del sessions[session_id]
                memory = get_or_create_memory(session_id)
            long_term_mem.clear_memory(session_id)
            os.system("cls" if os.name == "nt" else "clear")
            print("🧹 会话已清除（含长期记忆）\n")
            continue

        if cmd == "/help":
            print("""
  📋 可用命令:
    /help        显示帮助
    /clear       清除当前对话历史和长期记忆
    /tools       显示可用工具列表
    /quit, /exit 退出
    /status      显示系统状态

  直接输入问题即可开始对话 💬
  """)
            continue

        if cmd == "/tools":
            print("""
  🔧 可用工具:
    RAG_Search    - 查询 PCB / 硬件知识库
    Calculator    - 执行数学计算
    Get_Time      - 获取当前时间
    Web_Search    - 搜索互联网
    Read_Webpage  - 读取网页全文
  """)
            continue

        if cmd == "/status":
            print(f"""
  📊 系统状态:
    会话 ID:       {session_id}
    活跃会话数:    {len(sessions)}
    Agent 就绪:    {agent_executor is not None}
    RAG 就绪:      {rag is not None}
    当前时间:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  """)
            continue

        # ── 安全护栏：输入过滤 + 限流 ──
        safe, msg = guardrails.filter_input(user_input)
        if not safe:
            print(f"\n🛡️ 输入被拒绝：{msg}")
            continue

        allowed, msg = guardrails.check_rate_limit(session_id)
        if not allowed:
            print(f"\n🛡️ 请求受限：{msg}")
            continue

        # ── 调用 Agent ──
        try:
            reset_read_counter(session_id)
            agent_executor.memory = memory

            start_time = time.time()
            result = agent_executor.invoke({"input": user_input})
            elapsed = (time.time() - start_time) * 1000

            answer = result.get("output", "")
            steps = result.get("intermediate_steps", [])

            # 提取使用的工具
            tools_used = list(dict.fromkeys(
                step[0].tool for step in steps if hasattr(step[0], "tool")
            ))

            # ── 输出审核 ──
            _, answer = guardrails.filter_output(answer)

            # ── 保存长期记忆 ──
            long_term_mem.save_memory(session_id, user_input, answer)

            # 打印回复
            print(f"\n🤖 AI: {answer}")
            if tools_used:
                print(f"   ⚙️  工具: {', '.join(tools_used)}")
            print(f"   ⏱️  耗时: {elapsed:.0f}ms | 步骤: {len(steps)}")
            print()

        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  🤖 AI Agent 交互终端")
    print("=" * 50)

    try:
        print("🔄 正在加载 Agent（RAG + 工具 + 长期记忆）...")
        start = time.time()
        build_agent()
        elapsed = time.time() - start
        print(f"✅ Agent 就绪（{elapsed:.1f}s）")
        print("-" * 50)
        chat_loop()
    except KeyboardInterrupt:
        print("\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
