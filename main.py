"""
RAG 问答 API — FastAPI 服务

提供 HTTP 接口调用 RAGSystem 的问答能力。

启动方式:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

或使用 Python 直接运行:
    python main.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_system import RAGSystem

# ============================================================
# 全局 RAG 实例
# ============================================================

rag: Optional[RAGSystem] = None


# ============================================================
# Lifespan 生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动时自动执行：
      1. 创建 RAGSystem 实例
      2. 加载知识库文档并分块
      3. 构建 Chroma 向量库
      4. 初始化 LLM
    （Rerank 模型在首次问答时懒加载）
    """
    global rag

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  未设置 DEEPSEEK_API_KEY 环境变量，服务启动但问答接口将返回 500")

    print("=" * 60)
    print("  RAG 服务启动中...")
    print("=" * 60)

    rag = RAGSystem(
        knowledge_path="data/knowledge/knowledge.txt",
        chunk_size=256,
        overlap=128,
    )

    rag.initialize()

    print("\n" + "=" * 60)
    print("  ✅ RAG 系统已就绪，等待请求...")
    print("=" * 60)

    yield  # 服务运行中...


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="RAG 问答 API",
    version="2.0.0",
    description="基于向量检索 + Cross-Encoder Rerank + DeepSeek LLM 的 RAG 问答服务",
    lifespan=lifespan,
)


# ============================================================
# Pydantic 模型
# ============================================================

class AskRequest(BaseModel):
    """问答请求体。"""
    question: str = Field(..., min_length=1, description="用户问题")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="LLM 采样温度")
    top_k: int = Field(default=5, ge=1, le=20, description="Rerank 后保留的文档块数")


class SourceItem(BaseModel):
    """来源文档块信息。"""
    index: int
    content_preview: str
    char_count: int


class AskResponse(BaseModel):
    """问答响应体。"""
    answer: str
    question: str
    source_count: int
    reranked_source_count: int
    sources: list[SourceItem]
    reranked_sources: list[SourceItem]
    retrieval_time_ms: float
    rerank_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class HealthResponse(BaseModel):
    """健康检查响应体。"""
    status: str
    chunks: int
    knowledge_file: str
    llm_model: str
    embedding_model: str
    rerank_model: str


# ============================================================
# 接口路由
# ============================================================

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    执行一次 RAG 问答。

    流程: 向量检索 → Cross-Encoder Rerank → LLM 生成
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG 系统未初始化，请稍后重试")

    try:
        result = rag.ask(
            question=request.question,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")

    # ask() 返回的 key 是 source_docs / reranked_docs（与新版 rag_system 一致）
    sources_raw = result.get("source_docs", [])
    reranked_raw = result.get("reranked_docs", [])

    return AskResponse(
        answer=result["answer"],
        question=result["question"],
        source_count=len(sources_raw),
        reranked_source_count=len(reranked_raw),
        sources=[
            SourceItem(
                index=i + 1,
                content_preview=doc["content"][:150] + "...",
                char_count=doc["char_count"],
            )
            for i, doc in enumerate(sources_raw)
        ],
        reranked_sources=[
            SourceItem(
                index=i + 1,
                content_preview=doc["content"][:150] + "...",
                char_count=doc["char_count"],
            )
            for i, doc in enumerate(reranked_raw)
        ],
        retrieval_time_ms=round(result["retrieval_time"] * 1000, 2),
        rerank_time_ms=round(result["rerank_time"] * 1000, 2),
        generation_time_ms=round(result["generation_time"] * 1000, 2),
        total_time_ms=round(result["total_time"] * 1000, 2),
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口。"""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG 系统未初始化")

    return HealthResponse(
        status="ok",
        chunks=rag.doc_count,
        knowledge_file=rag.knowledge_path,
        llm_model=rag.LLM_MODEL,
        embedding_model=rag.EMBEDDING_MODEL,
        rerank_model=rag.RERANK_MODEL,
    )


# ============================================================
# 直接运行入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
