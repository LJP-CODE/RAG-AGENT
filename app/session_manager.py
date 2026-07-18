"""
session_manager.py — 会话持久化管理

提供会话的 CRUD 操作，数据以 JSON 文件存储在 data/sessions/ 目录下。
每个会话独立一个 JSON 文件，包含消息历史、标题和元信息。

用法:
    from session_manager import SessionManager

    mgr = SessionManager()
    session = mgr.create_session()                    # 新建空白会话
    mgr.add_message(sid, "user", "问题内容")           # 添加用户消息
    mgr.add_message(sid, "assistant", "回答", ...)     # 添加助手消息
    sessions = mgr.list_sessions()                    # 列出所有会话
    mgr.delete_session(sid)                           # 删除会话
"""

import json
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

_logger = logging.getLogger("session_manager")


# ============================================================
# 常量
# ============================================================

DEFAULT_SESSIONS_DIR = str(
    Path(__file__).resolve().parent.parent / "data" / "sessions"
)


# ============================================================
# SessionManager
# ============================================================

class SessionManager:
    """会话管理器：JSON 文件持久化，CRUD + 消息管理。"""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Args:
            storage_dir: 会话文件存储目录。默认 data/sessions/
        """
        self.storage_dir = storage_dir or DEFAULT_SESSIONS_DIR
        os.makedirs(self.storage_dir, exist_ok=True)

    # ── 路径工具 ──────────────────────────────────────────

    def _session_path(self, session_id: str) -> str:
        """返回会话 JSON 文件的完整路径。"""
        # 防止路径穿越：只允许字母数字和短横线
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    def _read(self, session_id: str) -> Optional[dict]:
        """读取单个会话数据，不存在返回 None。"""
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            _logger.warning("读取会话 %s 失败: %s", session_id, e)
            return None

    def _write(self, session_id: str, data: dict) -> bool:
        """写入会话数据到文件。"""
        path = self._session_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except OSError as e:
            _logger.error("写入会话 %s 失败: %s", session_id, e)
            return False

    # ── CRUD ──────────────────────────────────────────────

    def create_session(self, title: str = "新对话") -> dict:
        """
        创建一个空白新会话。

        Returns:
            dict: 包含 id, title, messages, created_at 等字段的会话对象
        """
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        session = {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._write(session_id, session)
        _logger.info("创建会话: %s", session_id)
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        """获取单个会话的完整数据。"""
        return self._read(session_id)

    def list_sessions(self) -> list:
        """
        列出所有会话（按更新时间倒序）。

        Returns:
            list[dict]: 每个会话的摘要信息（不含 messages 完整内容）
        """
        sessions = []
        try:
            for fname in os.listdir(self.storage_dir):
                if not fname.endswith(".json"):
                    continue
                sid = fname[:-5]  # 去掉 .json
                data = self._read(sid)
                if data is None:
                    continue
                # 返回摘要（不包含完整 messages，减少传输量）
                sessions.append({
                    "id": data["id"],
                    "title": data.get("title", "新对话"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                })
        except OSError as e:
            _logger.error("列出会话失败: %s", e)

        # 按更新时间倒序
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话的持久化文件。"""
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
            _logger.info("删除会话: %s", session_id)
            return True
        except OSError as e:
            _logger.error("删除会话 %s 失败: %s", session_id, e)
            return False

    def update_title(self, session_id: str, title: str) -> bool:
        """更新会话标题。"""
        data = self._read(session_id)
        if data is None:
            return False
        data["title"] = title
        data["updated_at"] = datetime.now().isoformat()
        return self._write(session_id, data)

    def add_message(self, session_id: str, role: str, content: str,
                    tools_used: list = None, time_ms: float = 0) -> bool:
        """
        向会话中添加一条消息。

        Args:
            session_id: 会话 ID
            role: 'user' | 'assistant' | 'error'
            content: 消息文本
            tools_used: 使用的工具列表（仅 assistant 有值）
            time_ms: 响应耗时（仅 assistant 有值）

        Returns:
            bool: 是否添加成功
        """
        data = self._read(session_id)
        if data is None:
            _logger.warning("add_message: 会话 %s 不存在，自动创建", session_id)
            data = {
                "id": session_id,
                "title": "新对话",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "messages": [],
            }

        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if role == "assistant" and tools_used:
            msg["tools_used"] = tools_used
        if role == "assistant" and time_ms:
            msg["time_ms"] = time_ms

        data["messages"].append(msg)
        data["updated_at"] = datetime.now().isoformat()

        # 自动根据第一条用户消息设置标题
        if data["title"] == "新对话" and role == "user":
            # 用前 30 个字符作为标题
            title_text = content.replace("\n", " ").strip()
            data["title"] = title_text[:30] + ("…" if len(title_text) > 30 else "")

        return self._write(session_id, data)

    def get_messages(self, session_id: str) -> list:
        """获取会话的所有消息列表。"""
        data = self._read(session_id)
        if data is None:
            return []
        return data.get("messages", [])

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在。"""
        return os.path.exists(self._session_path(session_id))


# ============================================================
# 模块级单例（供 agent_api 直接使用）
# ============================================================

_session_manager: Optional[SessionManager] = None


def get_session_manager(storage_dir: Optional[str] = None) -> SessionManager:
    """获取 SessionManager 单例。"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(storage_dir)
    return _session_manager
