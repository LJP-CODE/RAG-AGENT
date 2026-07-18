"""
Agent API — 透明转发到 app/agent_api.py（唯一 Agent 入口）

启动方式:
    uvicorn app.agent_api:app --reload --host 0.0.0.0 --port 8000
    或
    uvicorn agent_api:app --reload --host 0.0.0.0 --port 8000  （兼容旧用法）

本地 CLI:
    python chat_cli.py
"""

import sys
import os
from pathlib import Path

# 确保 app/ 在 sys.path 中
_app_dir = str(Path(__file__).resolve().parent / "app")
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# 离线模式
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 透明转发：从 app/agent_api.py 重新导出所有符号
from app.agent_api import (        # noqa: E402, F401
    app,
    AskRequest,
    AskResponse,
    SecurityStatus,
    FrozenSessionsResponse,
)
