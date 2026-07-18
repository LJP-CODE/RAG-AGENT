"""
app — AI Agent 核心模块

子模块:
    - rag_system:      RAG 知识库检索
    - web_tools:       网络搜索 & 网页读取
    - vector_memory:   长期向量记忆（Chroma + BGE Embedding）
    - memory_store:    长期记忆（JSON 文件存储，旧方案）
    - agent_guardrails: Agent 安全护栏
    - agent_monitor:   Agent 运行监控
    - agent_system:    多 Agent 编排系统
    - tool_registry:   工具注册中心
"""

from .vector_memory import VectorMemory
from .web_tools import web_search, read_webpage

__all__ = [
    "VectorMemory",
    "web_search",
    "read_webpage",
]
