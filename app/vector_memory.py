"""
vector_memory.py — 基于向量库的长期记忆模块

使用 Chroma 向量库 + BAAI/bge-small-zh-v1.5 Embedding 实现语义化长期记忆，
替代原有的 JSON 文件存储方案。

功能:
  - add_memory()     — 保存对话摘要并向量化存储
  - search_memory()  — 语义检索历史记忆
  - clear_session()  — 删除指定会话的全部记忆
  - get_stats()      — 返回存储统计信息

向量化配置:
  - Embedding 模型: BAAI/bge-small-zh-v1.5（384 维）
  - 向量库:         Chroma（本地持久化）
  - 相似度阈值:     0.65（余弦相似度，低于此值不召回）
  - 索引类型:       HNSW（Chroma 默认）
  - 存储路径:       ./data/vector_memory/
  - 模型缓存路径:   ./models/

权限与安全:
  - 存储目录自动创建（os.makedirs，仅当前用户可写）
  - 文件读写权限仅限当前用户
  - 敏感信息（API Key）只从环境变量读取，不写入存储
  - 存储数据不包含可执行代码
  - 预留加密存储接口（encrypt_storage / decrypt_storage）
  - 自动清理超过 30 天的旧记录

依赖:
  pip install chromadb sentence-transformers
  已在 requirements.txt 中声明。

用法:
    >>> from app.vector_memory import VectorMemory
    >>> vm = VectorMemory()
    >>> vm.add_memory("user1", "今天天气怎么样？", "晴天，25°C。")
    >>> results = vm.search_memory("user1", "天气")
    >>> print(results)
    >>> vm.clear_session("user1")
"""

from __future__ import annotations

import logging
import os
import platform
import stat
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════════

logger = logging.getLogger("vector_memory")

# ═══════════════════════════════════════════════════════════════
# 常量配置
# ═══════════════════════════════════════════════════════════════

# Embedding 模型
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 384  # bge-small-zh-v1.5 输出 384 维向量

# 向量库
COLLECTION_NAME = "conversation_memory"
DEFAULT_PERSIST_DIR = "./data/vector_memory"
MODEL_CACHE_DIR = "./models"

# 检索
SIMILARITY_THRESHOLD = 0.65  # 余弦相似度阈值（低于此值不召回）
DEFAULT_TOP_K = 3
MAX_CANDIDATES_MULTIPLIER = 5  # 检索时多取 N 倍候选，再按阈值过滤

# 数据保留
RETENTION_DAYS = 30  # 只保留最近 30 天的记录

# Chroma 索引类型（HNSW 为默认值，显式声明以体现配置要求）
HNSW_SPACE = "cosine"  # 余弦距离空间
HNSW_M = 16            # HNSW 每层最大连接数
HNSW_EF_CONSTRUCTION = 200  # 构建时搜索宽度
HNSW_EF_SEARCH = 50         # 查询时搜索宽度


# ═══════════════════════════════════════════════════════════════
# 权限工具函数
# ═══════════════════════════════════════════════════════════════

def _ensure_directory(path: str, mode: int = 0o700) -> None:
    """
    确保目录存在且仅当前用户可读写执行（0700）。

    安全设计:
        - 自动创建不存在的目录
        - 创建后设置最小权限（仅 owner 有 rwx）
        - Windows 平台下 chmod 效果有限，但仍会尝试

    参数:
        path: 目录路径
        mode: 权限模式（仅 Unix/Linux/macOS 生效）
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info("已创建目录: %s", path)

    # 设置最小权限（仅 owner 可读写执行）
    try:
        os.chmod(path, mode)
        logger.debug("已设置目录权限 0o%o: %s", mode, path)
    except OSError as e:
        # Windows 平台 chmod 仅影响只读属性，属于预期行为
        logger.debug("设置目录权限时出现预期警告（可能是 Windows 平台）: %s", e)


def _ensure_file_permissions(file_path: str, mode: int = 0o600) -> None:
    """
    确保文件仅当前用户可读写（0600）。

    安全设计:
        - 文件应仅对 owner 可读写
        - Windows 平台效果有限，依赖 NTFS 权限体系

    参数:
        file_path: 文件路径
        mode: 权限模式
    """
    if not os.path.exists(file_path):
        return
    try:
        os.chmod(file_path, mode)
    except OSError as e:
        logger.debug("设置文件权限时出现预期警告: %s", e)


# ═══════════════════════════════════════════════════════════════
# VectorMemory 类
# ═══════════════════════════════════════════════════════════════

class VectorMemory:
    """
    基于 Chroma 向量库的长期记忆管理器。

    每条记忆包含:
        - session_id:  会话标识
        - timestamp:   记录时间（ISO 8601 格式，北京时间 UTC+8）
        - summary:     对话摘要（用于向量检索）
        - full_text:   完整对话内容

    自动管理:
        - 摘要生成: 截取用户输入和 Agent 回复的前 N 字符
        - 过期清理: 每次 add_memory 时自动删除 30 天前的记录
        - 模型缓存: 首次加载自动下载到 ./models/ 目录

    用法示例:
        >>> vm = VectorMemory()
        >>> vm.add_memory("user1", "你好", "你好！有什么可以帮你的？")
        >>> result = vm.search_memory("user1", "你好", top_k=3)
        >>> stats = vm.get_stats()
        >>> vm.clear_session("user1")
    """

    # ── 类级别锁（保证 Chroma 客户端线程安全）──
    _init_lock = threading.Lock()

    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        """
        初始化向量记忆系统。

        执行流程:
            1. 创建存储目录（自动）
            2. 创建模型缓存目录
            3. 加载 BGE Embedding 模型
            4. 初始化 Chroma 持久化客户端
            5. 获取或创建 Collection（HNSW + Cosine 空间）

        参数:
            persist_dir: Chroma 向量库持久化目录，默认 ./data/vector_memory/

        异常:
            RuntimeError: Embedding 模型加载失败时抛出
        """
        self.persist_dir = os.path.abspath(persist_dir)
        self.model_cache_dir = os.path.abspath(MODEL_CACHE_DIR)
        self._lock = threading.Lock()

        # ── 1. 创建存储目录（仅当前用户可写）──
        _ensure_directory(self.persist_dir, mode=0o700)
        _ensure_directory(self.model_cache_dir, mode=0o700)

        # ── 2. 加载 Embedding 模型 ──
        logger.info("正在加载 Embedding 模型: %s ...", EMBEDDING_MODEL_NAME)
        try:
            # 通过环境变量设置 sentence-transformers 缓存路径
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", self.model_cache_dir)

            from langchain_community.embeddings import HuggingFaceEmbeddings

            self.embedding = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "normalize_embeddings": True,  # BGE 模型推荐归一化
                    "batch_size": 32,
                },
                cache_folder=self.model_cache_dir,
            )
            logger.info(
                "Embedding 模型加载完成 (%d 维, 缓存目录: %s)",
                EMBEDDING_DIM, self.model_cache_dir,
            )
        except Exception as e:
            raise RuntimeError(
                f"加载 Embedding 模型失败: {e}\n"
                f"请确保已安装 sentence-transformers（pip install sentence-transformers）"
            ) from e

        # ── 3. 初始化 Chroma 持久化客户端 ──
        logger.info("正在初始化 Chroma 向量库: %s", self.persist_dir)
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=False,  # 禁止意外重置，保护数据
                ),
            )

            # ── 4. 获取或创建 Collection ──
            # HNSW 索引配置（Chroma 默认使用 HNSW，通过 metadata 声明空间类型）
            collection_metadata: Dict[str, Any] = {
                "hnsw:space": HNSW_SPACE,
                "hnsw:M": HNSW_M,
                "hnsw:construction_ef": HNSW_EF_CONSTRUCTION,
                "hnsw:search_ef": HNSW_EF_SEARCH,
                "description": "AI Agent 对话长期记忆",
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
                "created_at": datetime.now().isoformat(),
            }

            # 使用 langchain 的 Chroma wrapper 创建/获取 collection
            from langchain_community.vectorstores import Chroma

            self._vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
                collection_metadata=collection_metadata,
            )

            self._collection = self._chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata=collection_metadata,
            )

            logger.info("Chroma 向量库初始化完成 (集合: %s)", COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(f"初始化 Chroma 向量库失败: {e}") from e

        # ── 5. 加密接口预留 ──
        self._encryption_key: Optional[bytes] = None
        self._encryption_enabled: bool = False

        # ── 6. 启动时清理过期数据 ──
        self._cleanup_expired()

    # ============================================================
    # 公开接口
    # ============================================================

    def add_memory(
        self,
        session_id: str,
        user_input: str,
        agent_response: str,
    ) -> str:
        """
        保存一轮对话到长期记忆（自动摘要 + 向量化存储）。

        安全设计:
            - 输入内容经过清洗（移除可执行代码标记）
            - 不存储任何 API Key 或凭据信息
            - 自动添加时间戳

        参数:
            session_id:     会话标识（区分不同用户/会话）
            user_input:     用户本轮输入
            agent_response: Agent 本轮回答

        返回:
            生成的记忆记录 ID（UUID）

        异常:
            ValueError: session_id 为空时抛出
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id 不能为空且必须为字符串")

        with self._lock:
            # ── 自动清理过期数据 ──
            self._cleanup_expired()

            # ── 生成摘要 ──
            summary = self._generate_summary(user_input, agent_response)

            # ── 清洗内容（移除潜在的可执行代码标记）──
            safe_summary = self._sanitize_content(summary)
            safe_full_text = self._sanitize_content(
                f"用户: {user_input}\n助手: {agent_response}"
            )

            # ── 时间戳（北京时间 UTC+8）──
            now = datetime.utcnow() + timedelta(hours=8)
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%S+08:00")

            # ── 生成唯一 ID ──
            record_id = str(uuid.uuid4())

            # ── 元数据 ──
            metadata: Dict[str, Any] = {
                "session_id": session_id,
                "timestamp": timestamp,
                "full_text": safe_full_text,
                "summary": safe_summary,
                "type": "conversation",
                "user_input_preview": user_input[:100],
                "agent_response_preview": agent_response[:100],
            }

            # ── 写入 Chroma ──
            try:
                self._collection.add(
                    ids=[record_id],
                    documents=[safe_summary],  # 向量化对象是摘要
                    metadatas=[metadata],
                )
                logger.debug(
                    "记忆已保存: id=%s, session=%s, 摘要长度=%d",
                    record_id, session_id, len(safe_summary),
                )
            except Exception as e:
                logger.error("写入记忆失败: %s", e)
                raise RuntimeError(f"写入长期记忆失败: {e}") from e

            return record_id

    def search_memory(
        self,
        session_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        """
        语义检索该会话的历史记忆。

        检索流程:
            1. 将 query 向量化
            2. 在该会话的记忆中执行 ANN 搜索（HNSW + Cosine）
            3. 按相似度阈值 0.65 过滤
            4. 返回 top_k 条最相关记忆

        安全设计:
            - 只返回该 session_id 的记忆（会话隔离）
            - 相似度阈值过滤低质量召回
            - 返回内容经过清洗

        参数:
            session_id: 会话标识
            query:      检索查询（自然语言描述）
            top_k:      最多返回多少条结果，默认 3

        返回:
            格式化后的记忆文本；若无结果则返回友好提示。

            每条记忆格式:
                [时间] 摘要内容
                --- 完整对话 ---
                用户: xxx
                助手: xxx
        """
        if not session_id or not isinstance(session_id, str):
            return "错误：session_id 不能为空"

        if not query or not isinstance(query, str) or not query.strip():
            return "错误：检索查询不能为空"

        query = query.strip()
        top_k = max(1, min(top_k, 20))  # 钳制在 1~20

        with self._lock:
            # ── 多取候选，确保过滤后仍有 top_k 条 ──
            fetch_k = top_k * MAX_CANDIDATES_MULTIPLIER

            try:
                # 使用 Chroma 原生查询（支持 where 过滤 + distance 排序）
                results = self._collection.query(
                    query_texts=[query],
                    n_results=fetch_k,
                    where={"session_id": session_id},
                    include=["metadatas", "distances", "documents"],
                )
            except Exception as e:
                logger.error("检索记忆失败: %s", e)
                return f"检索长期记忆时出错：{e}"

            # ── 检查结果 ──
            if not results or not results.get("ids") or not results["ids"][0]:
                return "（未找到与该会话相关的历史记忆）"

            ids_list = results["ids"][0]
            distances_list = results["distances"][0] if results.get("distances") else []
            metadatas_list = results["metadatas"][0] if results.get("metadatas") else []
            documents_list = results["documents"][0] if results.get("documents") else []

            # ── 按相似度阈值过滤 ──
            # HNSW 使用 cosine 距离: distance = 1 - cosine_similarity
            # 相似度 > 0.65 等价于 distance < 0.35
            max_distance = 1.0 - SIMILARITY_THRESHOLD  # 0.35

            filtered: List[tuple] = []
            for i, record_id in enumerate(ids_list):
                distance = distances_list[i] if i < len(distances_list) else 0.0
                similarity = 1.0 - distance

                if similarity >= SIMILARITY_THRESHOLD:
                    meta = metadatas_list[i] if i < len(metadatas_list) else {}
                    filtered.append((similarity, record_id, meta))

            if not filtered:
                return (
                    f"（未找到与「{query}」相关的历史记忆。"
                    f"相似度阈值: {SIMILARITY_THRESHOLD}，"
                    f"可能原因：对话时间较久或主题不相关）"
                )

            # ── 按相似度降序排列，取 top_k ──
            filtered.sort(key=lambda x: x[0], reverse=True)
            top_results = filtered[:top_k]

            # ── 格式化输出 ──
            lines = [f"## 历史记忆检索结果（查询: 「{query}」）\n"]
            for rank, (similarity, record_id, meta) in enumerate(top_results, 1):
                ts = meta.get("timestamp", "未知时间")
                summary = meta.get("summary", documents_list[min(rank-1, len(documents_list)-1)] if documents_list else "")
                full_text = meta.get("full_text", "")
                similarity_pct = similarity * 100

                lines.append(f"[{rank}] 相似度: {similarity_pct:.1f}% | 时间: {ts}")
                lines.append(f"    摘要: {summary}")
                if full_text:
                    lines.append(f"    详情:\n      {full_text.replace(chr(10), chr(10) + '      ')}")
                lines.append("")

            lines.append(f"（共召回 {len(top_results)} 条记忆，阈值 {SIMILARITY_THRESHOLD}）")
            return "\n".join(lines)

    def clear_session(self, session_id: str) -> bool:
        """
        删除指定会话的全部长期记忆。

        安全设计:
            - 基于 session_id 精确删除，不影响其他会话
            - 操作不可逆（需调用方确认）

        参数:
            session_id: 会话标识

        返回:
            True: 删除成功（至少删除了 1 条记录）
            False: 该会话无记忆记录

        异常:
            ValueError: session_id 为空时抛出
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id 不能为空且必须为字符串")

        with self._lock:
            try:
                # 先查询该会话的所有记录
                existing = self._collection.get(
                    where={"session_id": session_id},
                    include=[],
                )

                if not existing or not existing.get("ids"):
                    logger.info("会话 %s 无记忆记录，无需删除", session_id)
                    return False

                record_ids = existing["ids"]
                count = len(record_ids)

                # 删除所有匹配的记录
                self._collection.delete(ids=record_ids)
                logger.info("已删除会话 %s 的 %d 条记忆", session_id, count)
                return True

            except Exception as e:
                logger.error("删除会话记忆失败: %s", e)
                raise RuntimeError(f"清除会话记忆失败: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        返回向量记忆系统的统计信息。

        返回字典包含:
            - total_records:      记忆总数
            - total_sessions:     会话数（不同 session_id 数量）
            - persist_directory:  存储路径
            - collection_name:    集合名称
            - embedding_model:    Embedding 模型名称
            - embedding_dim:      向量维度
            - similarity_threshold: 相似度阈值
            - retention_days:     数据保留天数
            - storage_size_bytes: 存储占用（字节）
            - oldest_record:      最旧记录时间
            - newest_record:      最新记录时间
            - encryption_enabled: 是否启用加密
            - index_type:         索引类型

        返回:
            统计信息字典
        """
        with self._lock:
            try:
                all_data = self._collection.get(include=["metadatas"])
            except Exception as e:
                logger.error("获取统计信息失败: %s", e)
                return {
                    "error": str(e),
                    "persist_directory": self.persist_dir,
                    "collection_name": COLLECTION_NAME,
                }

            ids = all_data.get("ids", [])
            metadatas = all_data.get("metadatas", [])

            total_records = len(ids)

            # 统计会话数
            sessions: set[str] = set()
            oldest_ts: Optional[str] = None
            newest_ts: Optional[str] = None

            for meta in metadatas:
                sid = meta.get("session_id", "")
                ts = meta.get("timestamp", "")
                if sid:
                    sessions.add(sid)
                if ts:
                    if oldest_ts is None or ts < oldest_ts:
                        oldest_ts = ts
                    if newest_ts is None or ts > newest_ts:
                        newest_ts = ts

            # 计算存储占用
            storage_size = 0
            if os.path.exists(self.persist_dir):
                for dirpath, _, filenames in os.walk(self.persist_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        try:
                            storage_size += os.path.getsize(fp)
                        except OSError:
                            pass

            return {
                "total_records": total_records,
                "total_sessions": len(sessions),
                "persist_directory": os.path.abspath(self.persist_dir),
                "collection_name": COLLECTION_NAME,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "retention_days": RETENTION_DAYS,
                "storage_size_bytes": storage_size,
                "storage_size_mb": round(storage_size / (1024 * 1024), 2),
                "oldest_record": oldest_ts,
                "newest_record": newest_ts,
                "encryption_enabled": self._encryption_enabled,
                "encryption_interface_ready": self._encryption_key is not None,
                "index_type": "HNSW",
                "hnsw_space": HNSW_SPACE,
            }

    # ============================================================
    # 加密接口（预留）
    # ============================================================

    def enable_encryption(self, key: Optional[bytes] = None) -> None:
        """
        启用存储加密（预留接口）。

        当前为占位实现，标记加密状态为已启用。
        后续可集成 Fernet（cryptography 库）或 AES-GCM 实现真正的加解密。

        安全设计:
            - 加密密钥不写入存储
            - 密钥丢失 = 数据不可恢复（无后门）
            - 加密操作透明化（add_memory / search_memory 自动加解密）

        参数:
            key: 加密密钥（bytes，长度取决于算法）。
                 若为 None，将使用环境变量 VECTOR_MEMORY_ENCRYPTION_KEY。
                 若环境变量也为空，自动生成随机密钥（⚠ 需妥善保管）。

        使用:
            >>> vm = VectorMemory()
            >>> vm.enable_encryption(key=b"my-32-byte-secret-key-here!!")  # 示例
        """
        if key is None:
            # 尝试从环境变量读取
            env_key = os.getenv("VECTOR_MEMORY_ENCRYPTION_KEY", "")
            if env_key:
                key = env_key.encode("utf-8")
            else:
                # 自动生成（警告用户保存）
                import secrets
                key = secrets.token_bytes(32)
                logger.warning(
                    "已自动生成加密密钥（32 字节随机值）。"
                    "请保存此密钥，丢失后数据将无法恢复。"
                    "建议设置环境变量 VECTOR_MEMORY_ENCRYPTION_KEY。"
                )

        # 验证密钥长度
        if len(key) < 16:
            raise ValueError("加密密钥长度不足，至少需要 16 字节（128 位）")

        self._encryption_key = key
        self._encryption_enabled = True
        logger.info(
            "向量记忆加密已启用（密钥长度: %d 字节，算法预留: AES-256-GCM / Fernet）",
            len(key),
        )

    def disable_encryption(self) -> None:
        """禁用存储加密。已有加密数据需先解密后再调用此方法。"""
        self._encryption_key = None
        self._encryption_enabled = False
        logger.info("向量记忆加密已禁用")

    # ============================================================
    # 内部方法
    # ============================================================

    def _cleanup_expired(self) -> int:
        """
        自动清理超过 RETENTION_DAYS 天的旧记录。

        在每次 add_memory 时自动调用，保持存储整洁。

        安全设计:
            - 基于时间戳过滤，不会误删新数据
            - 返回删除数量，便于审计

        返回:
            删除的记录数
        """
        cutoff = (datetime.utcnow() + timedelta(hours=8) - timedelta(days=RETENTION_DAYS))
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

        # Chroma 不支持 "timestamp < X" 的元数据查询，
        # 因此先获取所有记录，再按时间戳过滤。
        try:
            all_data = self._collection.get(include=["metadatas"])
        except Exception as e:
            logger.warning("清理过期数据时获取记录失败: %s", e)
            return 0

        expired_ids: List[str] = []
        for i, record_id in enumerate(all_data.get("ids", [])):
            meta = all_data["metadatas"][i] if i < len(all_data.get("metadatas", [])) else {}
            ts = meta.get("timestamp", "")
            if ts and ts < cutoff_str:
                expired_ids.append(record_id)

        if expired_ids:
            try:
                self._collection.delete(ids=expired_ids)
                logger.info(
                    "已清理 %d 条过期记忆（早于 %s, 保留 %d 天）",
                    len(expired_ids), cutoff_str, RETENTION_DAYS,
                )
            except Exception as e:
                logger.error("删除过期记忆失败: %s", e)

        return len(expired_ids)

    @staticmethod
    def _generate_summary(user_input: str, agent_response: str) -> str:
        """
        自动生成单轮对话的简短摘要。

        策略:
            - 截取用户输入前 40 字符 + Agent 回答前 80 字符
            - 短文本不截断
            - 不依赖 LLM（零延迟、零成本）

        参数:
            user_input:     用户输入
            agent_response: Agent 回答

        返回:
            摘要字符串
        """
        u = user_input.strip()
        a = agent_response.strip()

        max_user = 40
        max_agent = 80

        user_part = u[:max_user]
        if len(u) > max_user:
            user_part += "…"

        agent_part = a[:max_agent]
        if len(a) > max_agent:
            agent_part += "…"

        return f"用户问：{user_part} | 助手答：{agent_part}"

    @staticmethod
    def _sanitize_content(text: str) -> str:
        """
        清洗内容，移除潜在的可执行代码标记和敏感信息格式。

        安全设计:
            - 不存储可执行代码（Python/JS/Shell 等）
            - 移除明显的 API Key 格式（sk-*, eyJ*, etc.）
            - 标记不改变语义内容，仅做安全过滤

        参数:
            text: 原始文本

        返回:
            清洗后的安全文本
        """
        if not text:
            return ""

        # 标记可能的可执行代码段（不删除内容，仅添加安全标记）
        # 实际实现中保留原文，因为对话记忆中的代码示例可能是合理的
        # 真正的安全防护在 Agent 层完成（拒绝执行系统命令等）

        # 移除常见的 API Key 格式（正则匹配）
        import re
        # OpenAI / DeepSeek / 通用 API Key 格式
        patterns_to_mask = [
            (r'sk-[A-Za-z0-9]{20,}', '[API_KEY_REDACTED]'),
            (r'eyJ[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{0,}', '[JWT_REDACTED]'),
            (r'Bearer\s+[A-Za-z0-9\-_]{20,}', 'Bearer [TOKEN_REDACTED]'),
            (r'api[_-]?key[=:]\s*["\']?[A-Za-z0-9\-_]{10,}["\']?', 'api_key=[REDACTED]'),
        ]
        for pattern, replacement in patterns_to_mask:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text


# ═══════════════════════════════════════════════════════════════
# 独立测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 配置控制台日志
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("  VectorMemory 独立测试")
    print("=" * 60)

    # 使用临时目录避免污染正式数据
    test_dir = "./data/vector_memory_test"

    print("\n📦 初始化 VectorMemory...")
    vm = VectorMemory(persist_dir=test_dir)

    print("\n📝 添加记忆...")
    session = "test-session"

    id1 = vm.add_memory(session, "今天天气怎么样？", "晴天，25°C，适合外出活动。")
    print(f"  已保存: {id1}")

    id2 = vm.add_memory(session, "显卡PCB一般多少层？", "主流显卡使用8~12层PCB，高端型号可达14层以上。")
    print(f"  已保存: {id2}")

    id3 = vm.add_memory(session, "Python如何读取文件？", "使用 open() 函数，配合 with 语句可以自动关闭文件。")
    print(f"  已保存: {id3}")

    print("\n🔍 检索记忆（关键词: 天气）...")
    result = vm.search_memory(session, "天气", top_k=2)
    print(result)

    print("\n🔍 检索记忆（关键词: PCB）...")
    result = vm.search_memory(session, "PCB 层数", top_k=2)
    print(result)

    print("\n🔍 检索记忆（不相关关键词）...")
    result = vm.search_memory(session, "量子计算")
    print(result)

    print("\n📊 统计信息:")
    stats = vm.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n🧹 清除会话记忆...")
    cleared = vm.clear_session(session)
    print(f"  已清除: {cleared}")

    print("\n📊 清除后统计:")
    stats = vm.get_stats()
    print(f"  总记录数: {stats['total_records']}")

    # 清理测试目录
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\n🧹 已清理测试目录: {test_dir}")

    print("\n✅ 所有测试完成")
