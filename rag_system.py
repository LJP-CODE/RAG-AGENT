import os
import time
from typing import Optional

# 嵌入模型（BAAI/bge-small-zh-v1.5）已本地缓存，强制离线模式，
# 避免 sentence_transformers/transformers 每次启动都向 huggingface.co
# 发 HEAD 请求检查更新（国内网络访问 HF 会被拒/超时，导致启动卡死）。
# 必须在任何 HF 相关 import 之前设置。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TokenCounter:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


class RAGSystem:
    PERSIST_DIR = './chroma_db'
    EMBEDDING_MODEL = 'BAAI/bge-small-zh-v1.5'
    LLM_MODEL = 'deepseek-chat'
    DEEPSEEK_API_BASE = 'https://api.deepseek.com'
    TOP_K = 10
    RERANK_TOP_K = 5
    RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    RERANK_DEVICE = 'cpu'

    def __init__(self, knowledge_path: str, chunk_size: int = 256, overlap: int = 128):
        if not os.path.exists(knowledge_path):
            raise FileNotFoundError(f'知识库文件不存在: {knowledge_path}')
        self.knowledge_path = knowledge_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = None
        self.vector_store = None
        self.qa_chain = None
        self.retriever = None
        self.llm = None
        self.doc_count = 0
        self._rerank_model = None
        self.token_counter = TokenCounter()

    def load_and_split(self):
        loader = TextLoader(self.knowledge_path, encoding='utf-8')
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=['\n\n', '\n', '。', '！', '？', '；', '，', '、', ' ', ''],
        )
        self.chunks = splitter.split_documents(docs)
        self.doc_count = len(self.chunks)
        return self.chunks

    def build_vector_store(self, chunks):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.PERSIST_DIR,
        )
        return self.vector_store

    def build_qa_chain(self, vector_store):
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError('请设置 DEEPSEEK_API_KEY 环境变量')
        api_base = os.getenv('DEEPSEEK_API_BASE') or self.DEEPSEEK_API_BASE
        self.llm = ChatOpenAI(
            model=self.LLM_MODEL,
            api_key=api_key,
            base_url=api_base,
            temperature=0.1,
        )
        self.retriever = vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': self.TOP_K},
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            return_source_documents=True,
        )

    def _get_rerank_model(self):
        if self._rerank_model is None:
            from sentence_transformers import CrossEncoder
            self._rerank_model = CrossEncoder(self.RERANK_MODEL, device=self.RERANK_DEVICE)
        return self._rerank_model

    def ask(self, question: str, temperature: float = 0.3, top_k=None, return_scores=False):
        """执行一次完整的 RAG 问答：向量检索 → Rerank → LLM 生成。

        返回:
            {
                "answer":            str,    # 模型回答
                "question":          str,    # 原始问题
                "source_docs":       list,   # 向量初筛的全部文档块 [{content, char_count}, ...]
                "reranked_docs":     list,   # Rerank 后送入 LLM 的文档块 [{content, char_count}, ...]
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
        if self.qa_chain is None or self.retriever is None or self.llm is None:
            raise RuntimeError('问答链未构建，请先调用 initialize()')
        rerank_top_k = top_k or self.RERANK_TOP_K

        # ---- 阶段1: 向量检索 ----
        t0 = time.time()
        retrieved_docs = self.retriever.invoke(question)
        retrieval_time = time.time() - t0
        if not retrieved_docs:
            return {
                'answer': '未找到相关文档，无法回答问题。',
                'question': question,
                'source_docs': [],
                'reranked_docs': [],
                'retrieval_time': round(retrieval_time, 4),
                'rerank_time': 0.0,
                'generation_time': 0.0,
                'total_time': round(retrieval_time, 4),
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
            }

        # ---- 阶段2: Rerank 重排序 ----
        rerank_model = self._get_rerank_model()
        t0 = time.time()
        pairs = [[question, doc.page_content] for doc in retrieved_docs]
        scores = rerank_model.predict(pairs)
        rerank_time = time.time() - t0
        scored = list(zip(retrieved_docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored[:rerank_top_k]]

        # ---- 阶段3: LLM 生成 ----
        self.token_counter.reset()
        self.llm.temperature = temperature
        context = '\n\n'.join(doc.page_content for doc in reranked_docs)
        prompt = (
            '你是一个专业的 AI 助手。请根据以下上下文内容回答用户的问题。\n'
            '如果上下文中没有足够的信息，请如实说不知道，不要编造。\n\n'
            '上下文:\n---\n'
            f'{context}\n---\n\n'
            f'用户问题: {question}\n\n'
            '请给出详细、准确的回答：'
        )
        t0 = time.time()
        response = self.llm.invoke(prompt)
        generation_time = time.time() - t0
        answer_text = response.content if hasattr(response, 'content') else str(response)
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            if usage:
                self.token_counter.prompt_tokens = usage.get('prompt_tokens', 0)
                self.token_counter.completion_tokens = usage.get('completion_tokens', 0)
        total_time = retrieval_time + rerank_time + generation_time

        # ---- 汇总 ----
        result = {
            'answer': answer_text,
            'question': question,
            'source_docs': [
                {'content': doc.page_content, 'char_count': len(doc.page_content)}
                for doc in retrieved_docs
            ],
            'reranked_docs': [
                {'content': doc.page_content, 'char_count': len(doc.page_content)}
                for doc in reranked_docs
            ],
            'retrieval_time': round(retrieval_time, 4),
            'rerank_time': round(rerank_time, 4),
            'generation_time': round(generation_time, 4),
            'total_time': round(total_time, 4),
            'prompt_tokens': self.token_counter.prompt_tokens,
            'completion_tokens': self.token_counter.completion_tokens,
            'total_tokens': self.token_counter.total_tokens,
        }
        if return_scores:
            result['rerank_scores'] = [
                {'index': i + 1, 'score': round(float(s), 6)}
                for i, (_, s) in enumerate(scored)
            ]
        return result

    def initialize(self):
        chunks = self.load_and_split()
        vector_store = self.build_vector_store(chunks)
        self.build_qa_chain(vector_store)
        return {'doc_count': self.doc_count}
