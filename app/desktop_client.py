"""
desktop_client.py — RAG-Agent 完整对话客户端（PyQt5）

重构目标：从单聊天框升级为带侧边栏的多会话客户端，SQLite 本地持久化。

布局:
    ┌─────────────┬───────────────────────────────┐
    │ 历史会话列表  │  当前对话标题                  │
    │             │ ┌───────────────────────────┐ │
    │  • 会话 A    │ │                           │ │
    │  • 会话 B    │ │      消息气泡区域          │ │
    │  • 会话 C    │ │                           │ │
    │             │ └───────────────────────────┘ │
    │             │ ┌─────────────────┐ ┌──────┐  │
    │ [新建对话]   │ │ 输入框           │ │ 发送 │  │
    │             │ └─────────────────┘ └──────┘  │
    └─────────────┴───────────────────────────────┘
        状态栏：状态(就绪/思考中)          会话 id 前8位

会话管理:
    - 左侧列表显示历史会话标题（默认取首条消息前 20 字）
    - 点击会话 → 右侧加载完整对话记录
    - 新建对话 → 清空右侧、左侧新增"新对话"
    - 右键会话 → 删除；双击会话 → 重命名
    - 每个会话唯一 session_id（uuid），关闭重开仍可见

存储:
    SQLite，./data/conversations.db
    sessions(id, session_id, title, created_at)
    messages(id, session_id, role, content, tools_used, time_ms, created_at)

分层（单文件内模块化）:
    配置 → QSS → 数据模型 → 存储层 → 网络层 → 控件 → 主窗口 → 入口

运行:
    python app\\desktop_client.py   （直接运行脚本，勿用 -m；app/__init__.py 既有损坏）
"""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QKeyEvent
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# 配置
# ============================================================

API_URL = "http://localhost:8000/ask"
HEALTH_URL = "http://localhost:8000/health"
REQUEST_TIMEOUT = 60  # Agent 可能调用多个工具，留足时间

# 数据库路径：锚定项目根目录（app 的父目录）下的 data/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "conversations.db")

DEFAULT_TITLE = "新对话"
TITLE_PREVIEW_LEN = 20  # 默认标题取首条消息前 N 个字


# ============================================================
# QSS 样式表（集中管理）
# ============================================================

QSS = """
QMainWindow { background: #eef1f5; }

/* —— 左侧边栏 —— */
#sidebar {
    background: #ffffff;
    border-right: 1px solid #e2e6ec;
}
#sidebarTitle {
    color: #5a6470;
    font-size: 12px;
    font-weight: bold;
    padding: 4px 2px;
}
QListWidget#sessionList {
    background: #ffffff;
    border: none;
    outline: none;
    font-size: 13px;
}
QListWidget#sessionList::item {
    padding: 8px 10px;
    border-radius: 8px;
    margin: 2px 6px;
    color: #2c333d;
}
QListWidget#sessionList::item:hover {
    background: #f0f3f7;
}
QListWidget#sessionList::item:selected {
    background: #e3ecf6;
    color: #1f2329;
}
QPushButton#newChatBtn {
    background: #4a90d9;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 9px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton#newChatBtn:hover { background: #3a7ec1; }
QPushButton#newChatBtn:pressed { background: #2f6aa6; }

/* —— 右侧标题栏 —— */
#chatTitle {
    color: #1f2329;
    font-size: 15px;
    font-weight: bold;
    padding: 4px 2px;
}

/* —— 消息滚动区 —— */
QScrollArea#messageScroll {
    background: #f7f8fa;
    border: 1px solid #e2e6ec;
    border-radius: 12px;
}
QScrollArea#messageScroll > QWidget > QWidget { background: transparent; }
#messageContainer { background: transparent; }

/* —— 输入框 —— */
QTextEdit#inputEdit {
    background: #ffffff;
    border: 1px solid #d8dde4;
    border-radius: 10px;
    padding: 8px 10px;
    color: #1f2329;
    selection-background-color: #4a90d9;
}
QTextEdit#inputEdit:focus { border: 1px solid #4a90d9; }

/* —— 发送按钮 —— */
QPushButton#sendBtn {
    background: #4a90d9;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    font-size: 14px;
}
QPushButton#sendBtn:hover { background: #3a7ec1; }
QPushButton#sendBtn:pressed { background: #2f6aa6; }
QPushButton#sendBtn:disabled { background: #b9c4d4; }

/* —— 状态栏 —— */
QStatusBar {
    background: #e7eaf0;
    border-top: 1px solid #d8dde4;
    color: #5a6470;
    font-size: 12px;
}
QStatusBar QLabel { color: #5a6470; }
#statusLabel, #sessionLabel { padding: 0 6px; }
#sessionLabel { color: #8a93a0; font-family: Consolas, "Microsoft YaHei"; }

/* —— 消息气泡（QLabel + objectName，由 QSS 着色）—— */
QLabel#userBubble {
    background: #4a90d9; color: #ffffff;
    border-radius: 12px; padding: 8px 12px;
}
QLabel#aiBubble {
    background: #ffffff; color: #1f2329;
    border: 1px solid #e2e6ec; border-radius: 12px; padding: 8px 12px;
}
QLabel#systemBubble {
    background: #e9ecf2; color: #8a93a0;
    border-radius: 10px; padding: 5px 10px;
}
QLabel#errorBubble {
    background: #fdecec; color: #d94a4a;
    border-radius: 10px; padding: 6px 10px;
}
QLabel#metaLabel { color: #9aa3ad; font-size: 10px; }
QLabel#roleLabel { color: #4a90d9; font-size: 11px; font-weight: bold; padding: 0 2px; }
"""


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Session:
    """一个会话的元信息。"""
    id: int                       # 数据库自增主键
    session_id: str               # 对外唯一标识（uuid），与后端 API 对接
    title: str
    created_at: str


@dataclass
class Message:
    """一条对话消息。"""
    id: int
    session_id: str
    role: str                     # 'user' | 'ai' | 'error'
    content: str
    tools_used: list[str] = field(default_factory=list)
    time_ms: float = 0.0
    created_at: str = ""


# ============================================================
# 存储层：SQLite 持久化
# ============================================================

class ConversationStore:
    """会话与消息的 SQLite 存取。所有方法在主线程同步调用即可。"""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                tools_used  TEXT,           -- JSON 数组字符串
                time_ms     REAL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);
        """)
        self.conn.commit()

    # —— 会话 ——
    def list_sessions(self) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC"
        ).fetchall()
        return [Session(r["id"], r["session_id"], r["title"], r["created_at"])
                for r in rows]

    def create_session(self, session_id: str, title: str = DEFAULT_TITLE) -> Session:
        now = datetime.now().isoformat(timespec="seconds")
        cur = self.conn.execute(
            "INSERT INTO sessions(session_id, title, created_at) VALUES (?,?,?)",
            (session_id, title, now),
        )
        self.conn.commit()
        return Session(cur.lastrowid, session_id, title, now)

    def rename_session(self, session_id: str, title: str):
        self.conn.execute(
            "UPDATE sessions SET title=? WHERE session_id=?", (title, session_id)
        )
        self.conn.commit()

    def delete_session(self, session_id: str):
        # 外键未开 cascade，手动删消息再删会话
        self.conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        self.conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        self.conn.commit()

    # —— 消息 ——
    def add_message(self, session_id: str, role: str, content: str,
                    tools_used: list[str] | None = None, time_ms: float = 0.0) -> Message:
        now = datetime.now().isoformat(timespec="seconds")
        tools_json = json.dumps(tools_used or [], ensure_ascii=False)
        cur = self.conn.execute(
            "INSERT INTO messages(session_id, role, content, tools_used, time_ms, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (session_id, role, content, tools_json, time_ms, now),
        )
        self.conn.commit()
        return Message(cur.lastrowid, session_id, role, content,
                       tools_used or [], time_ms, now)

    def list_messages(self, session_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [
            Message(
                r["id"], r["session_id"], r["role"], r["content"],
                json.loads(r["tools_used"]) if r["tools_used"] else [],
                float(r["time_ms"] or 0.0),
                r["created_at"],
            )
            for r in rows
        ]

    def close(self):
        self.conn.close()


# ============================================================
# 网络层：QThread 异步调用 /ask
# ============================================================

class AskWorker(QThread):
    """后台线程调用后端 /ask，避免阻塞 UI。"""

    finished = pyqtSignal(dict)   # 成功：响应 JSON
    error = pyqtSignal(str)       # 失败：友好错误信息

    def __init__(self, url: str, payload: dict, timeout: int = REQUEST_TIMEOUT):
        super().__init__()
        self.url = url
        self.payload = payload
        self.timeout = timeout

    def run(self):
        try:
            resp = requests.post(self.url, json=self.payload, timeout=self.timeout)
            resp.raise_for_status()
            self.finished.emit(resp.json())
        except requests.exceptions.ConnectionError:
            self.error.emit(
                "无法连接到 API 服务（http://localhost:8000）。\n"
                "请确认后端已启动：python agent_api.py"
            )
        except requests.exceptions.Timeout:
            self.error.emit(f"请求超时（{self.timeout} 秒未响应），请稍后重试。")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            self.error.emit(f"API 返回 HTTP {code} 错误，请检查后端日志。")
        except ValueError:
            self.error.emit("API 返回的内容不是有效 JSON，请检查后端。")
        except Exception as e:
            self.error.emit(f"发生未知错误：{e}")


class HealthProbe(QThread):
    """启动时探测后端是否在线。"""
    done = pyqtSignal(bool)

    def run(self):
        try:
            r = requests.get(HEALTH_URL, timeout=3)
            self.done.emit(r.status_code == 200)
        except Exception:
            self.done.emit(False)


# ============================================================
# 工具函数
# ============================================================

def _md_lite(text: str) -> str:
    """极简 Markdown → HTML：转义 + 加粗 + 行内代码 + 换行。"""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = text.replace("\n", "<br>")
    return text


def _preview_title(text: str) -> str:
    """取首条消息前 N 个字作为会话标题，去除多余空白。"""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:TITLE_PREVIEW_LEN] + ("…" if len(text) > TITLE_PREVIEW_LEN else "")


# ============================================================
# 控件：输入框（Enter 发送 / Shift+回车换行）
# ============================================================

class InputEdit(QTextEdit):
    """输入框：Enter 发送，Shift+Enter 换行。"""
    sendPressed = pyqtSignal()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if e.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(e)   # Shift+回车 → 换行
            else:
                self.sendPressed.emit()    # 回车 → 发送
        else:
            super().keyPressEvent(e)


# ============================================================
# 控件：消息气泡
# ============================================================

class MessageBubble(QWidget):
    """单条消息气泡。role 决定对齐与配色（配色由 QSS 按 objectName 控制）。"""

    _OBJ_MAP = {
        "user": "userBubble",
        "ai": "aiBubble",
        "system": "systemBubble",
        "error": "errorBubble",
    }

    def __init__(self, role: str, text: str,
                 tools: list[str] | None = None, time_ms: float | None = None,
                 max_width: int = 640):
        super().__init__()
        self.role = role
        self._build(text, tools, time_ms, max_width)

    def _build(self, text, tools, time_ms, max_width):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bubble = QWidget()
        bubble.setMaximumWidth(max_width)
        v = QVBoxLayout(bubble)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)

        if self.role == "ai":
            role_lbl = QLabel("🤖 AI")
            role_lbl.setObjectName("roleLabel")
            v.addWidget(role_lbl)

        body = QLabel(_md_lite(text))
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setObjectName(self._OBJ_MAP.get(self.role, "aiBubble"))
        v.addWidget(body)

        # meta 行：工具 + 耗时（用户/系统/错误消息不显示）
        if self.role in ("user", "ai") and (tools or time_ms is not None):
            parts = []
            if tools:
                parts.append("🔧 " + ", ".join(tools))
            if time_ms is not None:
                parts.append(f"⏱ {time_ms:.0f} ms")
            meta = QLabel("  ·  ".join(parts))
            meta.setObjectName("metaLabel")
            v.addWidget(meta)

        # 对齐：user 右，其余左
        if self.role == "user":
            outer.addStretch(1)
            outer.addWidget(bubble, 0, Qt.AlignRight)
        else:
            outer.addWidget(bubble, 0, Qt.AlignLeft)
            outer.addStretch(1)


# ============================================================
# 主窗口（UI + 控制逻辑）
# ============================================================

class ChatWindow(QMainWindow):
    def __init__(self, store: ConversationStore):
        super().__init__()
        self.store = store
        self.worker: Optional[AskWorker] = None
        self.pending_bubble: Optional[MessageBubble] = None
        self.pending_session_id: Optional[str] = None  # 当前请求所属会话
        self.current_session: Optional[Session] = None
        self._loading_list = False   # 防止填充列表时触发 itemChanged

        self.setWindowTitle("🤖 RAG-Agent 智能助手")
        self.resize(960, 720)
        self.setMinimumSize(700, 520)

        self._build_ui()
        self._reload_session_list()

        # 启动时若有历史会话，默认选中最新一个
        if self.session_list.count() > 0:
            self.session_list.setCurrentRow(0)
        else:
            self._new_conversation()

        # 窗口显示后把焦点放到输入框（延时到事件循环，确保生效）
        QTimer.singleShot(0, self.input_edit.setFocus)

        self._set_status("连接中")
        self._probe = HealthProbe(self)
        self._probe.done.connect(self._on_probe)
        self._probe.start()

    # ─────────────────────────── UI 构建 ───────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        h.addWidget(self._build_sidebar())
        h.addWidget(self._build_chat_area(), 1)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")
        self.session_label = QLabel("会话 —")
        self.session_label.setObjectName("sessionLabel")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.session_label)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(230)
        v = QVBoxLayout(sidebar)
        v.setContentsMargins(10, 12, 10, 12)
        v.setSpacing(8)

        title = QLabel("历史会话")
        title.setObjectName("sidebarTitle")
        v.addWidget(title)

        self.session_list = QListWidget()
        self.session_list.setObjectName("sessionList")
        # 双击重命名 / 选中切换
        self.session_list.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        self.session_list.itemChanged.connect(self._on_session_renamed)
        self.session_list.customContextMenuRequested.connect(self._on_context_menu)
        v.addWidget(self.session_list, 1)

        new_btn = QPushButton("+ 新建对话")
        new_btn.setObjectName("newChatBtn")
        new_btn.setAutoDefault(False)   # 防止回车误触发按钮
        new_btn.setCursor(Qt.PointingHandCursor)
        new_btn.clicked.connect(self._new_conversation)
        v.addWidget(new_btn)
        return sidebar

    def _build_chat_area(self) -> QWidget:
        area = QWidget()
        v = QVBoxLayout(area)
        v.setContentsMargins(14, 12, 14, 10)
        v.setSpacing(10)

        # 顶部标题
        self.chat_title = QLabel(DEFAULT_TITLE)
        self.chat_title.setObjectName("chatTitle")
        v.addWidget(self.chat_title)

        # 消息滚动区
        self.scroll = QScrollArea()
        self.scroll.setObjectName("messageScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)

        self.message_container = QWidget()
        self.message_container.setObjectName("messageContainer")
        self.message_layout = QVBoxLayout(self.message_container)
        self.message_layout.setContentsMargins(10, 10, 10, 10)
        self.message_layout.setSpacing(10)
        self.message_layout.addStretch(1)
        self.scroll.setWidget(self.message_container)
        v.addWidget(self.scroll, 1)

        # 输入区
        row = QHBoxLayout()
        row.setSpacing(10)
        self.input_edit = InputEdit()
        self.input_edit.setObjectName("inputEdit")
        self.input_edit.setPlaceholderText("输入问题，Enter 发送，Shift+回车换行…")
        self.input_edit.setFont(QFont("Microsoft YaHei", 10))
        self.input_edit.setFixedHeight(90)
        self.input_edit.sendPressed.connect(self._on_send)
        row.addWidget(self.input_edit, 1)

        self.send_btn = QPushButton("发送\n↵")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setFixedWidth(90)
        self.send_btn.setAutoDefault(False)   # 防止回车误触发按钮
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._on_send)
        row.addWidget(self.send_btn)
        v.addLayout(row)
        return area

    # ─────────────────────────── 会话列表 ───────────────────────────

    def _reload_session_list(self):
        """从数据库重新加载左侧会话列表。"""
        self._loading_list = True
        self.session_list.clear()
        for s in self.store.list_sessions():
            item = QListWidgetItem(s.title)
            item.setData(Qt.UserRole, s.session_id)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.session_list.addItem(item)
        self._loading_list = False

    def _current_session_id(self) -> Optional[str]:
        item = self.session_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _select_by_session_id(self, session_id: str):
        for i in range(self.session_list.count()):
            if self.session_list.item(i).data(Qt.UserRole) == session_id:
                self.session_list.setCurrentRow(i)
                return

    def _on_session_selected(self, current: QListWidgetItem, _previous):
        if current is None or self._loading_list:
            return
        session_id = current.data(Qt.UserRole)
        self._load_conversation(session_id)

    def _on_session_renamed(self, item: QListWidgetItem):
        """双击/编辑后保存新标题。"""
        if self._loading_list:
            return
        session_id = item.data(Qt.UserRole)
        title = item.text().strip() or DEFAULT_TITLE
        self.store.rename_session(session_id, title)
        if self.current_session and self.current_session.session_id == session_id:
            self.current_session.title = title
            self.chat_title.setText(title)

    def _on_context_menu(self, pos):
        item = self.session_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act_rename = menu.addAction("✏️ 重命名")
        act_delete = menu.addAction("🗑️ 删除会话")
        action = menu.exec_(self.session_list.mapToGlobal(pos))
        if action == act_rename:
            self.session_list.editItem(item)
        elif action == act_delete:
            self._delete_session(item)

    def _delete_session(self, item: QListWidgetItem):
        session_id = item.data(Qt.UserRole)
        title = item.text()
        if QMessageBox.question(
            self, "删除会话",
            f"确定删除会话「{title}」？该操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.store.delete_session(session_id)
        # 重新加载列表；若删的是当前会话，切换到下一个或新建
        self._reload_session_list()
        if self.session_list.count() > 0:
            self.session_list.setCurrentRow(0)
        else:
            self.current_session = None
            self._clear_messages()
            self._new_conversation()

    def _new_conversation(self):
        """新建对话：生成 session_id，写库，加入列表并选中。"""
        # 若当前已是空的新对话，避免重复创建
        if (self.current_session and self.current_session.title == DEFAULT_TITLE):
            self.input_edit.setFocus()
            return
        session_id = uuid.uuid4().hex[:12]
        self.current_session = self.store.create_session(session_id, DEFAULT_TITLE)
        self._loading_list = True
        item = QListWidgetItem(DEFAULT_TITLE)
        item.setData(Qt.UserRole, session_id)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.session_list.insertItem(0, item)   # 最新在最上
        self._loading_list = False
        self.session_list.setCurrentRow(0)
        self._clear_messages()
        self.input_edit.setFocus()

    # ─────────────────────────── 对话区 ───────────────────────────

    def _load_conversation(self, session_id: str):
        """加载某个会话的完整消息记录到右侧。"""
        sessions = [s for s in self.store.list_sessions()
                    if s.session_id == session_id]
        self.current_session = sessions[0] if sessions else None
        if self.current_session is None:
            return
        self.chat_title.setText(self.current_session.title)
        self._refresh_session_label()
        self._clear_messages()
        for m in self.store.list_messages(session_id):
            self._append_bubble(m.role, m.content,
                                tools=m.tools_used,
                                time_ms=m.time_ms if m.time_ms else None)
        self.input_edit.setFocus()

    def _clear_messages(self):
        while self.message_layout.count():
            child = self.message_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.deleteLater()
        self.message_layout.addStretch(1)
        self.pending_bubble = None

    def _append_bubble(self, role, text, tools=None, time_ms=None) -> MessageBubble:
        bubble = MessageBubble(role, text, tools, time_ms)
        self.message_layout.insertWidget(self.message_layout.count() - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _scroll_to_bottom(self):
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    # ─────────────────────────── 状态栏 ───────────────────────────

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _refresh_session_label(self):
        sid = self.current_session.session_id if self.current_session else None
        self.session_label.setText(f"会话 {sid[:8]}" if sid else "会话 —")

    def _set_busy(self, busy: bool):
        self.send_btn.setEnabled(not busy)
        self.input_edit.setEnabled(not busy)
        if not busy:
            self.input_edit.setFocus()

    # ─────────────────────────── 发送逻辑 ───────────────────────────

    def _on_send(self):
        try:
            if self.worker is not None and self.worker.isRunning():
                return
        except RuntimeError:
            self.worker = None
        if self.current_session is None:
            self._new_conversation()
            if self.current_session is None:
                return

        question = self.input_edit.toPlainText().strip()
        if not question:
            return

        self.input_edit.clear()

        # 入库用户消息
        self.store.add_message(self.current_session.session_id, "user", question)
        self._append_bubble("user", question)

        # 首条消息：用预览更新标题
        if self.current_session.title == DEFAULT_TITLE:
            new_title = _preview_title(question) or DEFAULT_TITLE
            self.current_session.title = new_title
            self.store.rename_session(self.current_session.session_id, new_title)
            self._set_list_item_title(self.current_session.session_id, new_title)
            self.chat_title.setText(new_title)

        # 思考中占位
        self.pending_bubble = self._append_bubble("ai", "🤔 思考中…")
        self.pending_session_id = self.current_session.session_id
        self._set_busy(True)
        self._set_status("思考中...")

        payload = {
            "question": question,
            "session_id": self.current_session.session_id,
            "temperature": 0.1,
        }
        self.worker = AskWorker(API_URL, payload)
        self.worker.finished.connect(self._on_success)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker.start()

    def _on_success(self, data: dict):
        self.worker = None
        answer = data.get("answer", "") or "（空回答）"
        tools = data.get("tools_used", []) or []
        time_ms = float(data.get("total_time_ms", 0) or 0)
        # 若请求期间已切换会话，只入库到原会话，不在当前界面展示
        same_session = (self.current_session is not None
                        and self.current_session.session_id == self.pending_session_id)
        self.store.add_message(self.pending_session_id, "ai", answer, tools, time_ms)
        if same_session:
            self._remove_bubble(self.pending_bubble)
            self._append_bubble("ai", answer, tools=tools, time_ms=time_ms)
            self._set_busy(False)
            self._set_status(f"就绪 · 上次 {time_ms:.0f} ms")
        else:
            self._set_busy(False)
            self._set_status("就绪")

    def _on_error(self, message: str):
        self.worker = None
        # 错误不入库，仅展示（若仍在原会话）
        same_session = (self.current_session is not None
                        and self.current_session.session_id == self.pending_session_id)
        if same_session:
            self._remove_bubble(self.pending_bubble)
            self._append_bubble("error", "⚠ " + message)
        self._set_busy(False)
        self._set_status("出错，请重试")

    def _remove_bubble(self, bubble: Optional[MessageBubble]):
        if bubble is None:
            return
        self.message_layout.removeWidget(bubble)
        bubble.deleteLater()
        self.pending_bubble = None

    def _set_list_item_title(self, session_id: str, title: str):
        for i in range(self.session_list.count()):
            item = self.session_list.item(i)
            if item.data(Qt.UserRole) == session_id:
                self._loading_list = True
                item.setText(title)
                self._loading_list = False
                return

    # ─────────────────────────── 后端探测 ───────────────────────────

    def _on_probe(self, ok: bool):
        if ok:
            self._set_status("就绪")
        else:
            self._set_status("未连接后端")
            QMessageBox.warning(
                self, "后端未启动",
                "无法连接到 http://localhost:8000。\n"
                "请先启动后端：python agent_api.py\n\n"
                "客户端仍可使用，发送时若后端未启动会再次提示。",
            )


# ============================================================
# 入口
# ============================================================

def _install_excepthook():
    """安装全局异常钩子：槽函数里未捕获的异常会弹框显示，而非静默崩溃退出。"""
    def hook(exc_type, exc_value, tb):
        import traceback
        text = "".join(traceback.format_exception(exc_type, exc_value, tb))
        sys.stderr.write(text)
        try:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, "发生异常", text[-3000:])
        except Exception:
            pass
    sys.excepthook = hook


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setStyleSheet(QSS)
    _install_excepthook()

    store = ConversationStore()
    window = ChatWindow(store)
    window.show()

    code = app.exec_()
    store.close()
    # 用 os._exit 跳过 Python/Qt 析构：Python 3.14 + PyQt5 在解释器关闭时
    # 会触发 STATUS_STACK_BUFFER_OVERRUN（0xC0000409）退出崩溃，功能不受影响，
    # 这里规避以获得干净退出。
    os._exit(code)


if __name__ == "__main__":
    main()
