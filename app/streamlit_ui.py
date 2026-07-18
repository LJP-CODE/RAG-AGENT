"""
streamlit_ui.py — RAG-Agent Web 对话界面（Streamlit）

基于 Streamlit 的浏览器端对话 UI，调用后端 FastAPI（http://localhost:8000/ask）。

功能:
    - ChatGPT 风格对话气泡（用户蓝色靠右，AI 灰色靠左）
    - 侧边栏：标题、API 地址/Key 配置、新建对话、会话 ID 展示
    - 每条 AI 回答下方展示调用的工具 + 耗时
    - API 超时 / 连接失败友好提示
    - 会话自动管理（UUID），多轮对话上下文记忆

运行:
    streamlit run app/streamlit_ui.py

注意:
    app/__init__.py 有损坏，不能以 `python -m streamlit run` 方式启动；
    直接用 `streamlit run app/streamlit_ui.py` 即可。
"""

import json
import uuid
from datetime import datetime

import requests
import streamlit as st

# ============================================================
# 页面配置（必须是第一个 Streamlit 命令）
# ============================================================
st.set_page_config(
    page_title="RAG-Agent 智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 常量
# ============================================================
DEFAULT_API_URL = "http://localhost:8000/ask"
DEFAULT_BASE_URL = "http://localhost:8000"       # 用于 /sessions 等管理接口
REQUEST_TIMEOUT = 120  # Agent 可能调用多个工具，留足时间

# ============================================================
# CSS 样式（ChatGPT 风格）
# ============================================================
CUSTOM_CSS = """
<style>
/* ── 全局 ── */
html, body, [class*="stApp"] {
    font-family: 'Segoe UI', 'Microsoft YaHei', 'PingFang SC', sans-serif;
}

.stApp {
    background: #f7f8fa;
}

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e6ec;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #1f2329;
    font-size: 1.25rem;
    font-weight: 700;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #5a6470;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── 主内容区 ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 0.5rem;
    max-width: 900px;
}

/* ── 对话容器 ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 0 20px 0;
}

/* ── 消息行 ── */
.message-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 6px;
    animation: fadeIn 0.25s ease;
}
.message-row.user {
    justify-content: flex-end;
}
.message-row.assistant {
    justify-content: flex-start;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── 头像 ── */
.avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    margin: 0 10px;
}
.avatar.user-avatar {
    background: #4a90d9;
    order: 2;
}
.avatar.ai-avatar {
    background: #e8ecf1;
    order: 1;
}

/* ── 气泡 ── */
.bubble {
    max-width: 72%;
    padding: 10px 16px;
    border-radius: 16px;
    font-size: 0.95rem;
    line-height: 1.6;
    word-wrap: break-word;
    white-space: pre-wrap;
    position: relative;
}
.bubble.user-bubble {
    background: #4a90d9;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}
.bubble.ai-bubble {
    background: #ffffff;
    color: #1f2329;
    border: 1px solid #e2e6ec;
    border-bottom-left-radius: 4px;
}
.bubble.error-bubble {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    border-radius: 12px;
    font-size: 0.9rem;
}
.bubble.system-bubble {
    background: #f0f3f7;
    color: #8a93a0;
    border-radius: 12px;
    font-size: 0.85rem;
    text-align: center;
}

/* ── 元信息（工具 + 耗时） ── */
.meta-line {
    font-size: 0.78rem;
    color: #9aa3ad;
    margin-top: 6px;
    padding-left: 54px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.meta-line .tool-tag {
    display: inline-block;
    background: #e8ecf1;
    color: #5a6470;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 500;
}
.meta-line .time-tag {
    color: #b0b8c2;
    font-size: 0.75rem;
}

/* ── 输入框容器（固定在底部） ── */
.fixed-bottom {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(0deg, #f7f8fa 0%, #f7f8fa 80%, transparent 100%);
    padding: 12px 20px 16px 20px;
    z-index: 100;
}

/* ── 按钮美化 ── */
div.stButton > button {
    background: #4a90d9;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 8px 20px;
    transition: background 0.15s;
}
div.stButton > button:hover {
    background: #3a7ec1;
}
div.stButton > button:active {
    background: #2f6aa6;
}

/* ── 隐藏 Streamlit 默认元素 ── */
#MainMenu { display: none; }
footer { display: none; }
/* 不隐藏 header — 保留侧边栏展开/折叠箭头 */

/* ── 滚动条 ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: #d0d5dd;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: #b0b8c2; }
</style>
"""


# ============================================================
# 工具函数
# ============================================================

def generate_session_id() -> str:
    """生成新的会话 ID（短 UUID）。"""
    return uuid.uuid4().hex[:12]


def init_session_state():
    """初始化所有 session_state 变量。"""
    defaults = {
        "messages": [],           # [{"role": "user/assistant", "content": "...", "tools_used": [], "time_ms": 0}]
        "session_id": generate_session_id(),
        "api_url": DEFAULT_API_URL,
        "api_key": "",
        "busy": False,
        "conversation_count": 0,  # 新建对话计数器，用于触发 UI 刷新
        "_sessions_cache": None,  # 会话列表缓存（None 表示未加载）
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def new_conversation():
    """新建对话：清空消息、生成新 session_id。"""
    st.session_state.messages = []
    st.session_state.session_id = generate_session_id()
    st.session_state.busy = False
    st.session_state.conversation_count += 1


def call_api(question: str, session_id: str, api_url: str,
             api_key: str = "", temperature: float = 0.1,
             timeout: int = REQUEST_TIMEOUT) -> dict:
    """
    调用后端 /ask 接口。

    Args:
        question: 用户问题
        session_id: 会话标识
        api_url: API 完整地址
        api_key: X-API-Key（可选，后端开放模式下不需要）
        temperature: LLM 温度
        timeout: 超时秒数

    Returns:
        {"answer": str, "tools_used": list, "total_time_ms": float}

    Raises:
        requests.exceptions.ConnectionError: 无法连接后端
        requests.exceptions.Timeout: 请求超时
        requests.exceptions.HTTPError: HTTP 错误
        ValueError: 响应非 JSON
    """
    payload = {
        "question": question,
        "session_id": session_id,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()

    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    return {
        "answer": data.get("answer", "") or "（空回答）",
        "tools_used": data.get("tools_used", []) or [],
        "total_time_ms": float(data.get("total_time_ms", 0) or 0),
    }


# ============================================================
# 会话 API 辅助函数
# ============================================================

def _base_url() -> str:
    """从 api_url 推导出 base URL（去掉 /ask 后缀）。"""
    url = st.session_state.api_url
    if url.endswith("/ask"):
        return url[:-4]
    if url.endswith("/ask/"):
        return url[:-5]
    return DEFAULT_BASE_URL


def fetch_sessions() -> list:
    """GET /sessions 获取所有历史会话列表。"""
    try:
        resp = requests.get(f"{_base_url()}/sessions", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def create_session_api() -> dict | None:
    """POST /sessions 创建新会话。"""
    try:
        resp = requests.post(f"{_base_url()}/sessions", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def load_session_api(session_id: str) -> dict | None:
    """GET /sessions/{id} 加载会话详情。"""
    try:
        resp = requests.get(f"{_base_url()}/sessions/{session_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def delete_session_api(session_id: str) -> bool:
    """DELETE /sessions/{id} 删除会话。"""
    try:
        resp = requests.delete(f"{_base_url()}/sessions/{session_id}", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================
# 渲染组件
# ============================================================

def render_sidebar():
    """渲染侧边栏：ChatGPT 风格会话列表。"""
    with st.sidebar:
        # ── 顶部 ──
        st.markdown("## 🤖 RAG-Agent")
        st.markdown("*智能知识库问答助手*")

        # 新建对话按钮
        st.button(
            "✨ 新建对话",
            on_click=_on_new_conversation,
            use_container_width=True,
            type="primary",
        )

        st.divider()
        st.markdown("### 💬 历史会话")

        # 首次加载或手动刷新
        if st.session_state.get("_sessions_cache") is None:
            with st.spinner("加载会话列表…"):
                st.session_state._sessions_cache = fetch_sessions()

        sessions = st.session_state.get("_sessions_cache") or []

        # ── 刷新 & 状态 ──
        col_refresh, col_count = st.columns([1, 2])
        with col_refresh:
            if st.button("🔄", help="刷新会话列表", use_container_width=True):
                st.session_state._sessions_cache = fetch_sessions()
                st.rerun()
        with col_count:
            if sessions:
                st.caption(f"共 {len(sessions)} 个会话")
            else:
                st.caption("发送消息即可创建会话")

        # ── 会话列表 ──
        if not sessions:
            st.info("📭 暂无历史会话\n\n在下方输入问题并发送，第一条消息会自动创建会话并显示在这里。")
        else:
            for s in sessions:
                sid = s["id"]
                title = s.get("title", "新对话")
                is_active = (sid == st.session_state.session_id)

                col_main, col_del = st.columns([8, 1])
                with col_main:
                    indicator = "🔵 " if is_active else "💬 "
                    if st.button(
                        f"{indicator}{title}",
                        key=f"session_{sid}",
                        use_container_width=True,
                        help=f"{s.get('message_count', 0)} 条消息 · {s.get('created_at', '')[:10]}",
                    ):
                        _switch_to_session(sid)
                with col_del:
                    if st.button("🗑", key=f"del_{sid}", help="删除此会话"):
                        _delete_session(sid)

        # ── 底部：当前会话信息（轻量）──
        st.divider()
        st.caption(f"会话 ID: `{st.session_state.session_id[:8]}…`")


def _on_new_conversation():
    """新建对话回调（供按钮 on_click 使用）。"""
    new_session = create_session_api()
    if new_session:
        st.session_state.session_id = new_session["id"]
        st.session_state.messages = []
        st.session_state.busy = False
        st.session_state.conversation_count += 1
        st.session_state._sessions_cache = fetch_sessions()


def _switch_to_session(session_id: str):
    """切换到指定会话，加载其消息历史。"""
    if session_id == st.session_state.session_id:
        return  # 已经是当前会话

    data = load_session_api(session_id)
    if data is None:
        st.toast(f"加载会话 {session_id} 失败", icon="❌")
        return

    st.session_state.session_id = session_id
    st.session_state.messages = data.get("messages", [])
    st.session_state.busy = False
    st.rerun()


def _delete_session(session_id: str):
    """删除指定会话。"""
    ok = delete_session_api(session_id)
    if ok:
        st.session_state._sessions_cache = fetch_sessions()
        # 如果删除的是当前会话，创建新会话
        if session_id == st.session_state.session_id:
            new_session = create_session_api()
            if new_session:
                st.session_state.session_id = new_session["id"]
            st.session_state.messages = []
            st.session_state.busy = False
        st.rerun()


def render_message(role: str, content: str, tools_used: list = None,
                   time_ms: float = None):
    """
    渲染单条消息气泡。

    Args:
        role: 'user' | 'assistant' | 'error' | 'system'
        content: 消息文本
        tools_used: 使用的工具列表（仅 assistant）
        time_ms: 耗时毫秒（仅 assistant）
    """
    tools_used = tools_used or []
    is_user = (role == "user")
    is_error = (role == "error")

    if is_error:
        st.markdown(
            f'<div class="message-row assistant">'
            f'<div class="bubble error-bubble">⚠️ {content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    if role == "system":
        st.markdown(
            f'<div style="text-align:center;margin:12px 0;">'
            f'<span class="bubble system-bubble">{content}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # 用户/助手气泡
    avatar_emoji = "👤" if is_user else "🤖"
    avatar_class = "user-avatar" if is_user else "ai-avatar"
    bubble_class = "user-bubble" if is_user else "ai-bubble"

    st.markdown(
        f'<div class="message-row {"user" if is_user else "assistant"}">'
        f'<div class="avatar {avatar_class}">{avatar_emoji}</div>'
        f'<div class="bubble {bubble_class}">{content}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 助手消息额外展示元信息：工具 + 耗时
    if not is_user and (tools_used or time_ms is not None):
        meta_parts = []
        for tool in tools_used:
            meta_parts.append(f'<span class="tool-tag">🔧 {tool}</span>')
        if time_ms is not None:
            meta_parts.append(f'<span class="time-tag">⏱️ {time_ms:.0f} ms</span>')
        st.markdown(
            f'<div class="meta-line">{" ".join(meta_parts)}</div>',
            unsafe_allow_html=True,
        )


def render_all_messages():
    """渲染所有历史消息。"""
    # 用 container 包裹以便控制 CSS
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(
            role=msg["role"],
            content=msg["content"],
            tools_used=msg.get("tools_used", []),
            time_ms=msg.get("time_ms") if msg.get("time_ms") else None,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 主函数
# ============================================================

def main():
    """Streamlit 主入口。"""
    # 注入自定义 CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 初始化状态
    init_session_state()

    # 渲染侧边栏
    render_sidebar()

    # ── 主区域：标题 + 消息列表 ──
    st.markdown("## 💬 对话")

    # 消息展示区域
    messages_placeholder = st.empty()
    with messages_placeholder.container():
        if not st.session_state.messages:
            # 空状态占位
            st.markdown(
                '<div style="text-align:center;color:#b0b8c2;padding:60px 20px;">'
                '<div style="font-size:3rem;margin-bottom:16px;">🤖</div>'
                '<div style="font-size:1.1rem;font-weight:500;margin-bottom:8px;">'
                '欢迎使用 RAG-Agent 智能助手</div>'
                '<div style="font-size:0.9rem;">'
                '在下方输入问题，Enter 发送</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            render_all_messages()

    # ── 底部输入区 ──
    # st.chat_input() 天然支持 Enter 发送，同时显示发送按钮
    prompt = st.chat_input(
        placeholder="输入你的问题，Enter 发送",
        disabled=st.session_state.busy,
    )
    if prompt and prompt.strip():
        _handle_send(prompt.strip())

    # ── 自动滚动到底部（通过 JS） ──
    st.markdown(
        """
        <script>
        function scrollToBottom() {
            setTimeout(function() {
                var chatContainer = window.parent.document.querySelector(
                    '.main .block-container'
                );
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }, 100);
        }
        scrollToBottom();
        </script>
        """,
        unsafe_allow_html=True,
    )


def _handle_send(question: str):
    """处理发送消息流程。"""
    # 1. 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "tools_used": [],
        "time_ms": 0,
    })

    # 2. 添加思考中占位
    st.session_state.messages.append({
        "role": "assistant",
        "content": "⏳ 思考中…",
        "tools_used": [],
        "time_ms": 0,
    })
    st.session_state.busy = True

    # 触发 rerun 以展示用户消息 + 思考中占位
    # st.chat_input() 会自动清空，无需手动处理
    st.rerun()


def call_and_update():
    """
    在 rerun 后检测到 busy 状态，实际调用 API。

    此函数应在 main() 中、渲染消息之后被调用。
    它检测最后一条消息是否为「思考中」，若是则发起 API 请求。
    """
    if not st.session_state.busy:
        return

    messages = st.session_state.messages
    if not messages or messages[-1]["role"] != "assistant":
        return

    last_msg = messages[-1]
    if last_msg.get("content") != "⏳ 思考中…":
        return

    # 找到对应的用户问题（思考中消息的前一条）
    if len(messages) < 2 or messages[-2]["role"] != "user":
        return

    question = messages[-2]["content"]

    # 移除思考中占位
    messages.pop()

    try:
        result = call_api(
            question=question,
            session_id=st.session_state.session_id,
            api_url=st.session_state.api_url,
            api_key=st.session_state.api_key,
        )
        messages.append({
            "role": "assistant",
            "content": result["answer"],
            "tools_used": result["tools_used"],
            "time_ms": result["total_time_ms"],
        })
        # 刷新会话列表缓存（反映新消息数）
        try:
            st.session_state._sessions_cache = fetch_sessions()
        except Exception:
            pass
    except requests.exceptions.ConnectionError:
        messages.append({
            "role": "error",
            "content": (
                "无法连接到 API 服务。\n\n"
                "请确认后端已启动：\n"
                "```\nuvicorn agent_api:app --host 0.0.0.0 --port 8000\n```\n\n"
                f"当前 API 地址：`{st.session_state.api_url}`"
            ),
            "tools_used": [],
            "time_ms": 0,
        })
    except requests.exceptions.Timeout:
        messages.append({
            "role": "error",
            "content": (
                f"请求超时（{REQUEST_TIMEOUT} 秒未响应）。\n\n"
                "Agent 调用工具可能耗时较长，请稍后重试或简化问题。"
            ),
            "tools_used": [],
            "time_ms": 0,
        })
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        detail_text = f"\n\n服务器返回：{detail}" if detail else ""
        messages.append({
            "role": "error",
            "content": f"API 返回 HTTP {code} 错误。{detail_text}",
            "tools_used": [],
            "time_ms": 0,
        })
    except ValueError:
        messages.append({
            "role": "error",
            "content": "API 返回的内容格式异常，无法解析。请检查后端日志。",
            "tools_used": [],
            "time_ms": 0,
        })
    except Exception as e:
        messages.append({
            "role": "error",
            "content": f"发生未知错误：{e}",
            "tools_used": [],
            "time_ms": 0,
        })

    st.session_state.busy = False
    st.rerun()


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    main()
    # 在 main() 之后检查是否需要执行 API 调用
    # 这是 Streamlit 的标准模式：main 渲染 UI，然后检测 busy 状态
    call_and_update()
