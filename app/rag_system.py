"""
RAGSystem — RAG 核心类

封装完整流程：文档加载 → 分块 → Embedding → Chroma 向量库 →
向量检索 → Cross-Encoder Rerank → DeepSeek LLM 生成

API Key 通过环境变量读取，不硬编码。
"""

import os
import time
from typing import Optional

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Token 统计回调
# ============================================================

class TokenCounter:
    """轻量 token 计数器（不依赖 LangChain callback）。"""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


# ============================================================
# RAGSystem 类
# ============================================================

class RAGSystem:
    """
    完整的 RAG 系统。

    用法:
        rag = RAGSystem("data/knowledge.txt")
        rag.initialize()                          # 一键初始化
        result = rag.ask("你的问题")              # 执行问答
        print(result["answer"])
    """

    # ---------- 默认配置 ----------
    PERSIST_DIR = "./chroma_db"
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    LLM_MODEL = "deepseek-chat"
    DEEPSEEK_API_BASE = "https://api.deepseek.com"
    TOP_K = 10                                # 向量初筛数量
    RERANK_TOP_K = 5                          # Rerank 后保留数量
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_DEVICE = "cpu"

    def __init__(self, knowledge_path: str, chunk_size: int = 256, overlap: int = 128):
        """
        初始化 RAG 系统。

        参数:
            knowledge_path: 知识库 TXT 文件路径
            chunk_size:      每块最大字符数
            overlap:         相邻块重叠字符数
        """
        if not os.path.exists(knowledge_path):
            raise FileNotFoundError(f"知识库文件不存在: {knowledge_path}")

        self.knowledge_path = knowledge_path
        self.chunk_size = chunk_size
        self.overlap = overlap

        # 运行时状态（initialize() 后填充）
        self.chunks: Optional[list] = None
        self.vector_store: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.retriever = None
        self.llm = None
        self.doc_count = 0
        self._rerank_model = None

        # Token 统计
        self.token_counter = TokenCounter()

    # ----------------------------------------------------------
    # 1. 加载文档并分块
    # ----------------------------------------------------------

    def load_and_split(self) -> list:
        """
        加载 TXT 文档并递归分块。

        流程:
            1. TextLoader 读取文件
            2. RecursiveCharacterTextSplitter 按段落、句子分割

        返回:
            文档块列表（每个元素是 LangChain Document 对象）
        """
        print(f"  [加载] 文件: {os.path.basename(self.knowledge_path)}")

        loader = TextLoader(self.knowledge_path, encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
            length_function=len,
        )

        self.chunks = splitter.split_documents(docs)
        self.doc_count = len(self.chunks)

        raw_size = os.path.getsize(self.knowledge_path)
        print(f"  [加载] 大小: {raw_size:,} 字节")
        print(f"  [分块] 共生成 {self.doc_count:,} 个文档块")
        print(f"  [分块] chunk_size={self.chunk_size}, overlap={self.overlap}")

        return self.chunks

    # ----------------------------------------------------------
    # 2. 构建 Chroma 向量库
    # ----------------------------------------------------------

    def build_vector_store(self, chunks: list) -> Chroma:
        """
        将文档块向量化并存入 Chroma 持久化向量库。

        参数:
            chunks: 文档块列表（load_and_split 的返回值）

        返回:
            Chroma 向量库实例
        """
        if not chunks:
            raise ValueError("chunks 为空，请先调用 load_and_split()")

        print(f"  [嵌入] 使用模型: {self.EMBEDDING_MODEL}")
        print(f"  [嵌入] 首次运行会自动下载模型（约 30-50MB）...")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print(f"  [向量] 持久化目录: {os.path.abspath(self.PERSIST_DIR)}")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.PERSIST_DIR,
        )

        print(f"  [向量] 向量库构建完成（{len(chunks)} 个文档块已索引）")
        return self.vector_store

    # ----------------------------------------------------------
    # 3. 构建检索问答链
    # ----------------------------------------------------------

    def build_qa_chain(self, vector_store: Chroma):
        """
        构建检索问答链（向量检索 → LLM 生成）。

        参数:
            vector_store: Chroma 向量库实例
        """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 DEEPSEEK_API_KEY 环境变量。\n"
                "请设置: export DEEPSEEK_API_KEY='your-api-key'  "
                "(Windows: $env:DEEPSEEK_API_KEY='your-api-key')"
            )

        api_base = os.getenv("DEEPSEEK_API_BASE") or self.DEEPSEEK_API_BASE

        # 构建 LLM（ChatOpenAI + DeepSeek 兼容接口）
        self.llm = ChatOpenAI(
            model=self.LLM_MODEL,
            api_key=api_key,
            base_url=api_base,
            temperature=0.1,
        )

        print(f"  [LLM] 提供商: DeepSeek | 模型: {self.LLM_MODEL}")

        # 构建检索器（初筛取 TOP_K 个）
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.TOP_K},
        )

        print(f"  [检索] 向量初筛 Top-K: {self.TOP_K}")
        print(f"  [检索] Rerank 后保留 Top-K: {self.RERANK_TOP_K}")

        # 构建问答链（仅用于兼容旧接口，生成阶段由 ask() 手动控制以插入 Rerank）
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            verbose=False,
        )

    # ----------------------------------------------------------
    # 4. Rerank 模型（懒加载）
    # ----------------------------------------------------------

    def _get_rerank_model(self):
        """
        获取 Cross-Encoder 重排序模型（单例，首次调用时加载）。

        模型: cross-encoder/ms-marco-MiniLM-L-6-v2
        设备: CPU

        如果下载慢，可使用 HuggingFace 镜像:
            $env:HF_ENDPOINT = "https://hf-mirror.com"
        """
        if self._rerank_model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "缺少 sentence-transformers 包，请安装:\n"
                    "    pip install sentence-transformers"
                )

            print(f"  [Rerank] 加载重排序模型: {self.RERANK_MODEL}")
            print(f"  [Rerank] 设备: {self.RERANK_DEVICE}")
            print(f"  [Rerank] 首次加载会自动下载模型（约 80MB）...")
            print(f"  [Rerank] 提示: 如果下载慢，设置环境变量 "
                  f"HF_ENDPOINT=https://hf-mirror.com")

            self._rerank_model = CrossEncoder(
                self.RERANK_MODEL,
                device=self.RERANK_DEVICE,
            )
            print(f"  [Rerank] 模型加载完成")

        return self._rerank_model

    # ----------------------------------------------------------
    # 5. 执行问答
    # ----------------------------------------------------------

    def ask(
        self,
        question: str,
        temperature: float = 0.3,
        top_k: Optional[int] = None,
        return_scores: bool = False,
    ) -> dict:
        """
        执行一次完整的 RAG 问答：向量检索 → Rerank → LLM 生成。

        所有异常均在内部兜底，绝不向上抛出。无论何种失败，返回值始终
        包含 ``answer`` 字段。

        参数:
            question:      用户问题
            temperature:   LLM 采样温度（0.0-1.0）
            top_k:         Rerank 后保留的文档数（默认使用 RERANK_TOP_K=5）
            return_scores: 是否在结果中返回每条文档的 Rerank 分数

        返回:
            {
                "answer":            str,    # 模型回答（失败时为友好提示）
                "question":          str,    # 原始问题
                "source_docs":       list,   # 初筛文档块
                "reranked_docs":     list,   # Rerank 后文档块（送入 LLM 的）
                "rerank_scores":     list,   # [可选] 每条文档的 Rerank 分数
                "retrieval_time":    float,  # 向量检索耗时（秒）
                "rerank_time":       float,  # Rerank 耗时（秒）
                "generation_time":   float,  # LLM 生成耗时（秒）
                "total_time":        float,  # 总计（秒）
                "prompt_tokens":     int,
                "completion_tokens": int,
                "total_tokens":      int,
            }
        """
        # ── 空结果模板（各阶段返回统一格式）──
        def _empty_result(answer_text: str) -> dict:
            return {
                "answer": answer_text,
                "question": question,
                "source_docs": [],
                "reranked_docs": [],
                "retrieval_time": 0.0,
                "rerank_time": 0.0,
                "generation_time": 0.0,
                "total_time": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        # ── 前置检查：问答链是否就绪 ──
        if self.qa_chain is None or self.retriever is None or self.llm is None:
            return _empty_result("系统尚未初始化，请稍后重试。")

        rerank_top_k = top_k or self.RERANK_TOP_K

        # ════════════════════════════════════════════════════
        # 阶段1: 向量检索（失败 → 返回友好提示，不抛异常）
        # ════════════════════════════════════════════════════
        t_retrieval_start = time.time()
        try:
            retrieved_docs = self.retriever.invoke(question)
        except Exception as e:
            retrieval_time = time.time() - t_retrieval_start
            print(f"  ❌ [RAG] 向量检索失败: {e}")
            result = _empty_result("知识库检索失败，请稍后重试。")
            result["retrieval_time"] = round(retrieval_time, 4)
            result["total_time"] = round(retrieval_time, 4)
            return result

        retrieval_time = time.time() - t_retrieval_start

        # 检索结果为空
        if not retrieved_docs:
            return _empty_result("未找到相关文档，无法回答问题。")

        # ════════════════════════════════════════════════════
        # 阶段2: Rerank 重排序（失败 → 跳过 Rerank，用原始结果）
        # ════════════════════════════════════════════════════
        t_rerank_start = time.time()
        reranked_docs = retrieved_docs  # 默认：Rerank 失败时直接用原始结果
        rerank_time = 0.0
        scored = []                     # 仅在 Rerank 成功时填充

        try:
            rerank_model = self._get_rerank_model()
            pairs = [[question, doc.page_content] for doc in retrieved_docs]
            scores = rerank_model.predict(pairs)
            rerank_time = time.time() - t_rerank_start

            # 按分数降序排列，取 top_k
            scored = list(zip(retrieved_docs, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, _ in scored[:rerank_top_k]]

        except Exception as e:
            # Rerank 失败：记录日志，直接用原始检索结果
            rerank_time = time.time() - t_rerank_start
            print(f"  ⚠️  [RAG] Rerank 失败，跳过重排序，使用原始检索结果: {e}")
            # reranked_docs 保持为 retrieved_docs 的副本，截取 top_k 条
            reranked_docs = retrieved_docs[:rerank_top_k]

        # ════════════════════════════════════════════════════
        # 阶段3: LLM 生成（失败 → 返回友好提示，不抛异常）
        # ════════════════════════════════════════════════════
        self.token_counter.reset()
        self.llm.temperature = temperature

        # 手动构造 prompt（用 Rerank 后的文档）
        context = "\n\n".join(doc.page_content for doc in reranked_docs)
        prompt = (
            "你是一个专业的 AI 助手。请根据以下上下文内容回答用户的问题。\n"
            "如果上下文中没有足够的信息，请如实说不知道，不要编造。\n\n"
            "上下文:\n"
            "---\n"
            f"{context}\n"
            "---\n\n"
            f"用户问题: {question}\n\n"
            "请给出详细、准确的回答："
        )

        t_generation_start = time.time()
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            generation_time = time.time() - t_generation_start
            total_time = retrieval_time + rerank_time + generation_time
            print(f"  ❌ [RAG] LLM 生成失败: {e}")
            result = _empty_result("生成回答失败，请稍后重试。")
            result["source_docs"] = [
                {"content": doc.page_content, "char_count": len(doc.page_content)}
                for doc in retrieved_docs
            ]
            result["reranked_docs"] = [
                {"content": doc.page_content, "char_count": len(doc.page_content)}
                for doc in reranked_docs
            ]
            result["retrieval_time"] = round(retrieval_time, 4)
            result["rerank_time"] = round(rerank_time, 4)
            result["generation_time"] = round(generation_time, 4)
            result["total_time"] = round(total_time, 4)
            return result

        generation_time = time.time() - t_generation_start

        # ── 提取回答文本与 token 用量 ──
        answer_text = response.content if hasattr(response, "content") else str(response)
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                self.token_counter.prompt_tokens = usage.get("prompt_tokens", 0)
                self.token_counter.completion_tokens = usage.get("completion_tokens", 0)

        # ════════════════════════════════════════════════════
        # 汇总结果
        # ════════════════════════════════════════════════════
        total_time = retrieval_time + rerank_time + generation_time

        result = {
            "answer": answer_text,
            "question": question,
            "source_docs": [
                {
                    "content": doc.page_content,
                    "char_count": len(doc.page_content),
                }
                for doc in retrieved_docs
            ],
            "reranked_docs": [
                {
                    "content": doc.page_content,
                    "char_count": len(doc.page_content),
                }
                for doc in reranked_docs
            ],
            "retrieval_time": round(retrieval_time, 4),
            "rerank_time": round(rerank_time, 4),
            "generation_time": round(generation_time, 4),
            "total_time": round(total_time, 4),
            "prompt_tokens": self.token_counter.prompt_tokens,
            "completion_tokens": self.token_counter.completion_tokens,
            "total_tokens": self.token_counter.total_tokens,
        }

        if return_scores and scored:
            result["rerank_scores"] = [
                {"index": i + 1, "score": round(float(s), 6)}
                for i, (_, s) in enumerate(scored)
            ]

        return result

    # ----------------------------------------------------------
    # 6. 一键初始化
    # ----------------------------------------------------------

    def initialize(self) -> dict:
        """
        一键完成所有初始化：加载文档 → 分块 → 向量库 → 问答链。

        返回:
            {"doc_count": int, "chunk_size": int, "overlap": int}
        """
        print("\n>>> 第一步：加载文档并分块")
        chunks = self.load_and_split()

        print("\n>>> 第二步：构建向量库")
        vector_store = self.build_vector_store(chunks)

        print("\n>>> 第三步：构建检索问答链")
        self.build_qa_chain(vector_store)

        print(f"\n  ✅ 初始化完成 | 文档块数: {self.doc_count}")
        print()

        return {
            "doc_count": self.doc_count,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }


# ============================================================
# CLI 入口（可直接运行测试）
# ============================================================

def main():
    """CLI 快速测试：初始化 → 回答内置示例问题。"""
    import sys

    knowledge_path = "data/knowledge/knowledge.txt"
    questions = sys.argv[1:] if len(sys.argv) > 1 else [
        "显卡PCB一般有多少层？各层结构是怎样的？",
        "什么是底部填充胶？它起什么作用？",
        "显存和数据线之间为什么要等长绕线？",
    ]

    print("\n" + "=" * 60)
    print("  RAG 系统 CLI 测试（含 Rerank）")
    print("=" * 60 + "\n")

    # 初始化
    rag = RAGSystem(knowledge_path)
    rag.initialize()

    # 问答
    print("=" * 60)
    print("  开始问答")
    print("=" * 60 + "\n")

    for i, question in enumerate(questions, 1):
        print(f"--- 问题 {i}/{len(questions)} ---")
        print(f"  Q: {question}\n")

        result = rag.ask(question, temperature=0.3)

        print(f"  A: {result['answer'][:300]}...\n")
        print(f"  检索耗时:    {result['retrieval_time']:.4f}s")
        print(f"  重排耗时:    {result['rerank_time']:.4f}s")
        print(f"  生成耗时:    {result['generation_time']:.4f}s")
        print(f"  总耗时:      {result['total_time']:.4f}s")
        print(f"  初筛块数:    {len(result['source_docs'])}")
        print(f"  重排后块数:  {len(result['reranked_docs'])}")
        print(f"  Token:       {result['total_tokens']}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
