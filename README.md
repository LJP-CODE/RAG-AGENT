# RAG 知识库问答 API

## 功能
- 基于 RAG 的企业知识库问答
- 支持通过 HTTP 接口调用

## 启动
```bash
uvicorn main:app --reload
```

## 调用示例
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "显卡PCB一般多少层？"}'
```

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
│   ├── rag_pipeline.py     # RAG 流水线 v1
│   ├── rag_pipeline_v2.py  # RAG 流水线 v2
│   ├── tool_registry.py    # 工具注册中心
│   ├── web_tools.py        # Web 工具集
│   └── desktop_client.py   # 桌面客户端
├── data/                   # 数据目录
│   ├── knowledge/          # 知识库文件 + 向量库
│   ├── chroma_db/          # Chroma 向量数据库
│   ├── conversations.db    # 对话存储
│   ├── agent_logs/         # Agent 运行日志
│   ├── long_term_memory/   # 长期记忆持久化
│   ├── audit_logs/         # 审计日志
│   └── day1_data/          # Day1 测试数据
├── tests/                  # 测试
│   ├── test_all_features.py
│   ├── test_api.py
│   ├── test_questions.json
│   └── test_write.txt
├── scripts/                # 工具脚本
│   ├── chat_cli.py         # 交互式对话客户端
│   ├── chat.bat            # Windows 启动脚本
│   ├── batch_read.py       # 批量文件读取
│   ├── generate_test_files.py
│   ├── read_txt_files.py
│   └── load_test.py
├── reports/                # 生成的报告/日志
├── .env.example            # 环境变量模板
├── config.yaml             # 应用配置文件
├── config_loader.py        # 配置加载器
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 镜像
└── docker-compose.yml      # 一键部署
```

## 性能
平均响应时间: XXXXms

知识库大小: 1,890 字符

文档块数: 13
