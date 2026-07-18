# ============================================================
# AI Agent — Dockerfile
# 基于 Python 3.11-slim，支持 Chroma 向量库
# ============================================================

FROM python:3.11-slim

# ── 设置工作目录 ──
WORKDIR /app

# ── 环境变量 ──
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# 配置文件路径（可通过 docker-compose 覆盖）
ENV CONFIG_PATH=config.yaml

# ── 安装系统依赖（Chroma 需要 gcc/g++ 编译）──
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 复制依赖文件并安装 Python 包 ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 复制应用代码 ──
COPY app/ ./app/
COPY data/ ./data/

# ── 复制配置文件（非敏感，可挂载覆盖）──
COPY config.yaml .

# ── 创建数据目录（持久化挂载点）──
RUN mkdir -p /app/data/chroma_db /app/data/long_term_memory /app/data/agent_logs

# ── 暴露端口 ──
EXPOSE 8000

# ── 健康检查 ──
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── 启动命令 ──
CMD ["uvicorn", "app.agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
