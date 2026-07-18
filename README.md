# RAG-Agent 智能问答系统

基于 RAG + LangChain + DeepSeek 的企业级智能问答系统，支持本地知识库检索、联网搜索、数学计算、时间查询等多种工具调用，提供 Web API、Web 界面、桌面客户端三种使用方式。

## ✨ 核心特性

- 🔍 **RAG 知识库检索**：基于私有文档精准问答，Token 优化 58%，检索耗时 16ms
- 🤖 **AI Agent 工具调度**：5 种工具自主调用（RAG 检索、联网搜索、数学计算、时间查询、网页阅读）
- 🌐 **多种交互方式**：Web API + Streamlit 界面 + PyQt5 桌面客户端
- 🔒 **安全护栏**：5 层安全防护 + 可观测性 + 限流控制
- 🐳 **一键部署**：Docker + docker-compose 容器化部署

## 📦 环境要求

- Python 3.11+
- DeepSeek API Key（从 https://platform.deepseek.com 获取）
- （可选）SerpApi Key（从 https://serpapi.com 获取，用于联网搜索）

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/LJP-CODE/RAG-AGENT.git
cd RAG-AGENT
2. 创建虚拟环境
bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. 安装依赖
bash
pip install -r requirements.txt
如果下载慢，可使用国内镜像源：

bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
4. 配置环境变量
bash
cp .env.example .env
编辑 .env 文件，填入你的 DeepSeek API Key：

bash
DEEPSEEK_API_KEY=你的DeepSeek密钥
SERPAPI_API_KEY=你的SerpApi密钥（可选）
5. 启动后端服务
bash
uvicorn main:app --reload
启动成功后，访问 http://localhost:8000/docs 可以看到 Swagger API 文档。

6. 启动 Web 界面（可选，新开一个终端）
bash
streamlit run app/streamlit_ui.py
启动成功后，访问 http://localhost:8501 可以看到 Web 对话界面。

7. 启动桌面客户端（可选，新开一个终端）
bash
python app/desktop_client.py
8. 测试
bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "显卡PCB一般多少层？"}'


## 项目结构
```
rag-agent/
├── app/                    # 核心代码
│   ├── agent_api.py        # Agent API 服务
│   ├── agent_system.py     # Agent 核心编排
│   ├── agent_guardrails.py # 安全护栏
│   ├── agent_monitor.py    # 监控日志
│   ├── memory_store.py     # 记忆存储
│   ├── vector_memory.py    # 向量记忆
│   ├── multi_agent.py      # 多智能体协作
│   ├── rag_system.py       # RAG 系统
│   ├── tool_registry.py    # 工具注册中心
│   ├── web_tools.py        # Web 工具集
│   └── desktop_client.py   # 桌面客户端
├── data/                   # 数据目录
│   ├── knowledge/          # 知识库文件
│   ├── chroma_db/          # Chroma 向量数据库（运行时生成）
│   ├── agent_logs/         # Agent 运行日志
│   └── long_term_memory/   # 长期记忆持久化
├── tests/                  # 测试
├── scripts/                # 工具脚本
├── .env.example            # 环境变量模板
├── config.yaml             # 应用配置文件
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 镜像
└── docker-compose.yml      # 一键部署
```

性能数据
指标	数据
平均响应时间	16ms
Token 优化	降低 58%（4,778 → 2,015）
知识库大小	1,890 字符
文档块数	13
