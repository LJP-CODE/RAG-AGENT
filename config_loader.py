"""
config_loader.py — 集中式配置管理模块

职责：
1. 加载 config.yaml 配置文件（支持深度合并默认值）
2. 加载 .env 环境变量
3. 启动时检查必要 Key 是否存在，不存在则警告
4. 提供类型安全的配置访问接口
5. 所有 API Key 仅在环境变量中存储，代码中通过 os.getenv() 读取
6. 日志中自动脱敏 Key

用法:
    from config_loader import get_config

    cfg = get_config()
    print(cfg.security.max_input_length)   # 2000
    print(cfg.llm.api_key)                 # 从 DEEPSEEK_API_KEY 环境变量读取
"""

import os
import sys
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


# ============================================================
# 环境变量加载（模块导入时自动执行）
# ============================================================

def _find_and_load_dotenv() -> Path:
    """查找并加载 .env 文件，返回实际加载的路径。"""
    candidates = [
        Path(".env"),                          # 当前工作目录
        Path(__file__).parent / ".env",        # config_loader 所在目录
        Path(__file__).parent.parent / ".env", # 项目根目录
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=False)
            return p.resolve()
    return Path(".")


_ENV_PATH = _find_and_load_dotenv()


# ============================================================
# 脱敏工具
# ============================================================

def mask_key(value: str) -> str:
    """将 API Key 脱敏显示，仅保留首尾少量字符。

    >>> mask_key("sk-abc123def456")
    'sk-a***456'
    >>> mask_key("short")
    '***'
    """
    if not value or not isinstance(value, str):
        return "***"
    if len(value) <= 6:
        return "***"
    return value[:4] + "***" + value[-3:]


def mask_value_for_log(value: str) -> str:
    """对任意字符串中的敏感信息进行脱敏，用于日志输出。"""
    if not value:
        return value

    patterns = [
        # API Key 格式
        (r'(sk-[a-zA-Z0-9]{4})[a-zA-Z0-9]+([a-zA-Z0-9]{3})', r'\1***\2'),
        # JWT Token
        (r'(eyJ[a-zA-Z0-9_-]{4})[a-zA-Z0-9_-]+([a-zA-Z0-9_-]{3})', r'\1***\2'),
        # 手机号（中国）
        (r'(\d{3})\d{4}(\d{4})', r'\1****\2'),
        # 身份证号
        (r'(\d{3})\d{13}(\d{2})', r'\1*************\2'),
    ]

    result = value
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)

    return result


# ============================================================
# 配置数据类
# ============================================================

class AppConfig:
    """应用服务器配置。"""
    def __init__(self, data: dict):
        self.host: str = data.get("host", "0.0.0.0")
        self.port: int = data.get("port", 8000)
        self.reload: bool = data.get("reload", False)


class LLMConfig:
    """LLM 配置。API Key 仅从环境变量读取。"""
    def __init__(self, data: dict):
        self.model: str = data.get("model", "deepseek-chat")
        self.base_url: str = data.get("base_url", "https://api.deepseek.com")
        self.temperature: float = float(data.get("temperature", 0.1))
        # API Key 仅从环境变量读取，不在配置文件中存储
        self.api_key: str = os.getenv("DEEPSEEK_API_KEY", "")

    @property
    def is_configured(self) -> bool:
        """检查 API Key 是否已配置。"""
        return bool(self.api_key and self.api_key not in (
            "", "your_deepseek_api_key_here", "你的DeepSeek密钥"
        ))


class RAGConfig:
    """RAG 知识库配置。"""
    def __init__(self, data: dict):
        self.knowledge_path: str = data.get("knowledge_path", "data/knowledge/knowledge.txt")
        self.chunk_size: int = data.get("chunk_size", 256)
        self.chunk_overlap: int = data.get("chunk_overlap", data.get("overlap", 128))
        self.embedding_model: str = data.get("embedding_model", "BAAI/bge-small-zh-v1.5")
        self.top_k: int = data.get("top_k", 5)


class DataConfig:
    """数据目录配置。"""
    def __init__(self, data: dict):
        raw = dict(data)
        # Windows 本地开发：自动将 Docker 绝对路径替换为本地相对路径
        if os.name == "nt":
            for key, default_path in [
                ("chroma_db_dir", "./chroma_db"),
                ("long_term_memory_dir", "./long_term_memory"),
                ("agent_logs_dir", "./agent_logs"),
                ("audit_log_dir", "./audit_logs"),
            ]:
                val = raw.get(key, default_path)
                if isinstance(val, str) and (val.startswith("/") or
                   (len(val) > 1 and val[1] == ":")):
                    raw[key] = default_path

        self.chroma_db_dir: str = raw.get("chroma_db_dir", "./data/chroma_db")
        self.long_term_memory_dir: str = raw.get("long_term_memory_dir", "./data/long_term_memory")
        self.agent_logs_dir: str = raw.get("agent_logs_dir", "./data/agent_logs")
        self.audit_log_dir: str = raw.get("audit_log_dir", "./data/audit_logs")


class AgentConfig:
    """Agent 行为配置。"""
    def __init__(self, data: dict):
        self.max_iterations: int = data.get("max_iterations", 5)
        self.temperature: float = float(data.get("temperature", 0.1))
        self.verbose: bool = data.get("verbose", True)


class SecurityConfig:
    """安全护栏配置。"""
    def __init__(self, data: dict):
        self.max_input_length: int = data.get("max_input_length", 2000)
        self.max_output_length: int = data.get("max_output_length", 8000)
        self.rate_limit_per_minute: int = data.get("rate_limit_per_minute", 10)
        self.rate_limit_per_hour: int = data.get("rate_limit_per_hour", 100)
        self.freeze_duration_minutes: int = data.get("freeze_duration_minutes", 5)
        self.max_calls_per_session: int = data.get("max_calls_per_session", 50)
        self.audit_log_retention_days: int = data.get("audit_log_retention_days", 30)
        self.sensitive_words: List[str] = data.get("sensitive_words", [])
        # 冻结触发阈值：连续 N 次触发安全规则
        self.freeze_trigger_count: int = data.get("freeze_trigger_count", 3)


class MemoryConfig:
    """向量记忆配置。"""
    def __init__(self, data: dict):
        self.persist_dir: str = data.get("persist_dir", "./data/vector_memory")
        self.max_history_days: int = data.get("max_history_days", 30)
        self.similarity_threshold: float = float(data.get("similarity_threshold", 0.65))


class WebSearchConfig:
    """Web 搜索工具配置。"""
    def __init__(self, data: dict):
        self.max_results: int = data.get("max_results", 5)
        self.timeout_seconds: int = data.get("timeout_seconds", 10)
        self.api_key: str = os.getenv("NEWS_API_KEY", "")


class ReadWebpageConfig:
    """网页读取工具配置。"""
    def __init__(self, data: dict):
        self.max_chars: int = data.get("max_chars", 3000)
        self.timeout_seconds: int = data.get("timeout_seconds", 15)


class ToolsConfig:
    """工具配置集合。"""
    def __init__(self, data: dict):
        self.web_search = WebSearchConfig(data.get("web_search", {}))
        self.read_webpage = ReadWebpageConfig(data.get("read_webpage", {}))
        # 天气 API Key
        self.weather_api_key: str = os.getenv("WEATHER_API_KEY", "")


# ============================================================
# 配置加载器（单例）
# ============================================================

class ConfigLoader:
    """集中式配置加载器（单例模式）。

    在首次调用 get_config() 时自动初始化，加载 config.yaml 和 .env。

    用法::

        from config_loader import get_config

        cfg = get_config()
        print(cfg.security.max_input_length)
        print(cfg.llm.api_key)            # 来自环境变量 DEEPSEEK_API_KEY
        print(cfg.tools.web_search.max_results)
    """

    _instance: Optional["ConfigLoader"] = None
    _config_path: Optional[str] = None

    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is not None:
            return cls._instance
        instance = super().__new__(cls)
        cls._instance = instance
        cls._config_path = config_path
        return instance

    def __init__(self, config_path: str = "config.yaml"):
        # __init__ 可能被多次调用（单例模式下），用标志位防止重复初始化
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # 1. 加载 YAML 配置
        self._raw = self._load_yaml(config_path)

        # 2. 构建类型化配置对象
        self.app = AppConfig(self._raw.get("app", {}))
        self.llm = LLMConfig(self._raw.get("llm", {}))
        self.rag = RAGConfig(self._raw.get("rag", {}))
        self.data = DataConfig(self._raw.get("data", {}))
        self.agent = AgentConfig(self._raw.get("agent", {}))
        self.security = SecurityConfig(self._raw.get("security", {}))
        self.memory = MemoryConfig(self._raw.get("memory", {}))
        self.tools = ToolsConfig(self._raw.get("tools", {}))

        # 3. 启动时检查必要配置
        self._check_required_keys()

    # ----------------------------------------------------------
    # YAML 加载
    # ----------------------------------------------------------

    @staticmethod
    def _defaults() -> dict:
        """返回所有配置项的硬编码默认值（config.yaml 不存在时使用）。"""
        return {
            "app": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False,
            },
            "rag": {
                "knowledge_path": "data/knowledge/knowledge.txt",
                "chunk_size": 256,
                "chunk_overlap": 128,
                "embedding_model": "BAAI/bge-small-zh-v1.5",
                "top_k": 5,
            },
            "data": {
                "chroma_db_dir": "/app/data/chroma_db",
                "long_term_memory_dir": "/app/data/long_term_memory",
                "agent_logs_dir": "/app/data/agent_logs",
                "audit_log_dir": "/app/data/audit_logs",
                "knowledge_dir": "/app/data/knowledge",
            },
            "llm": {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
                "temperature": 0.1,
            },
            "agent": {
                "max_iterations": 5,
                "temperature": 0.1,
                "verbose": True,
            },
            "security": {
                "max_input_length": 2000,
                "max_output_length": 8000,
                "rate_limit_per_minute": 10,
                "rate_limit_per_hour": 100,
                "freeze_duration_minutes": 5,
                "max_calls_per_session": 50,
                "audit_log_retention_days": 30,
                "freeze_trigger_count": 3,
                "sensitive_words": [],
            },
            "memory": {
                "persist_dir": "./data/vector_memory",
                "max_history_days": 30,
                "similarity_threshold": 0.65,
            },
            "tools": {
                "web_search": {
                    "max_results": 5,
                    "timeout_seconds": 10,
                },
                "read_webpage": {
                    "max_chars": 3000,
                    "timeout_seconds": 15,
                },
            },
        }

    def _load_yaml(self, config_path: str) -> dict:
        """加载 YAML 配置文件，深度合并默认值。"""
        defaults = self._defaults()
        path = Path(config_path)

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                # 深度合并：loaded 覆盖 defaults
                for section, values in loaded.items():
                    if section in defaults and isinstance(values, dict):
                        defaults[section].update(values)
                    else:
                        defaults[section] = values

        return defaults

    # ----------------------------------------------------------
    # 启动检查
    # ----------------------------------------------------------

    def _check_required_keys(self) -> None:
        """启动时检查必要的 API Key 是否已配置，缺失则发出警告。"""
        required = {
            "DEEPSEEK_API_KEY": "DeepSeek 大模型 API Key（从 https://platform.deepseek.com 获取）",
        }
        optional = {
            "BING_API_KEY": "Bing Web Search Key — 网络搜索（从 https://portal.azure.com 获取）",
            "NEWS_API_KEY": "NewsAPI Key — 新闻搜索（从 https://newsapi.org 获取）",
            "WEATHER_API_KEY": "和风天气 Key — 天气查询（从 https://dev.qweather.com 获取）",
        }

        for key, desc in required.items():
            val = os.getenv(key, "")
            if not val or val.startswith("your_") or val.startswith("你的"):
                warnings.warn(
                    f"⚠️  缺少必要的环境变量 {key}: {desc}\n"
                    f"   请设置: $env:{key} = 'your-{key.lower()}-here'",
                    RuntimeWarning,
                )

        for key, desc in optional.items():
            val = os.getenv(key, "")
            if not val or val.startswith("your_") or val.startswith("你的"):
                print(f"  ℹ️  可选环境变量 {key} 未设置: {desc}")

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    def to_dict(self) -> dict:
        """返回原始配置字典（用于向后兼容）。"""
        return self._raw

    def reload(self, config_path: Optional[str] = None) -> "ConfigLoader":
        """重新加载配置（用于运行时更新）。"""
        path = config_path or self._config_path or "config.yaml"
        ConfigLoader._instance = None
        return ConfigLoader(path)


# ============================================================
# 模块级单例访问
# ============================================================

def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """获取全局配置单例。

    首次调用时自动加载 config.yaml 和 .env 文件。
    后续调用返回同一实例。

    Args:
        config_path: YAML 配置文件路径（仅首次调用时生效）。
                     可通过环境变量 CONFIG_PATH 覆盖。

    Returns:
        ConfigLoader 实例
    """
    _path = os.getenv("CONFIG_PATH", config_path)
    return ConfigLoader(_path)


# ============================================================
# 模块级快捷访问（可选）
# ============================================================

def mask_api_key_for_log(key_value: str) -> str:
    """对 API Key 值进行日志脱敏。"""
    return mask_key(key_value)


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("═" * 60)
    print("  配置加载器 测试")
    print("═" * 60)

    cfg = get_config()

    print(f"\n📁 .env 路径: {_ENV_PATH}")
    print(f"\n📋 配置节: {list(cfg._raw.keys())}")

    print(f"\n🔧 App 配置:")
    print(f"   host={cfg.app.host}, port={cfg.app.port}")

    print(f"\n🤖 LLM 配置:")
    print(f"   model={cfg.llm.model}, base_url={cfg.llm.base_url}")
    print(f"   temperature={cfg.llm.temperature}")
    print(f"   api_key={'✅ 已配置' if cfg.llm.is_configured else '❌ 未配置'}")

    print(f"\n📚 RAG 配置:")
    print(f"   chunk_size={cfg.rag.chunk_size}, chunk_overlap={cfg.rag.chunk_overlap}")
    print(f"   embedding_model={cfg.rag.embedding_model}, top_k={cfg.rag.top_k}")

    print(f"\n🛡️  安全配置:")
    print(f"   max_input_length={cfg.security.max_input_length}")
    print(f"   max_output_length={cfg.security.max_output_length}")
    print(f"   rate_limit={cfg.security.rate_limit_per_minute}/min, "
          f"{cfg.security.rate_limit_per_hour}/hr")
    print(f"   freeze_duration={cfg.security.freeze_duration_minutes}min")
    print(f"   max_calls_per_session={cfg.security.max_calls_per_session}")
    print(f"   audit_log_retention={cfg.security.audit_log_retention_days}days")
    print(f"   freeze_trigger_count={cfg.security.freeze_trigger_count}")

    print(f"\n🧠 Memory 配置:")
    print(f"   persist_dir={cfg.memory.persist_dir}")
    print(f"   max_history_days={cfg.memory.max_history_days}")
    print(f"   similarity_threshold={cfg.memory.similarity_threshold}")

    print(f"\n🔨 Tools 配置:")
    print(f"   web_search.max_results={cfg.tools.web_search.max_results}")
    print(f"   web_search.timeout={cfg.tools.web_search.timeout_seconds}s")
    print(f"   read_webpage.max_chars={cfg.tools.read_webpage.max_chars}")
    print(f"   read_webpage.timeout={cfg.tools.read_webpage.timeout_seconds}s")
    print(f"   weather_api_key={'✅' if cfg.tools.weather_api_key else '❌'}")
    print(f"   news_api_key={'✅' if cfg.tools.web_search.api_key else '❌'}")

    print(f"\n🔑 脱敏测试:")
    print(f"   key='sk-abc123def456xyz789' → {mask_key('sk-abc123def456xyz789')}")
    print(f"   key='a1b2c3' → {mask_key('a1b2c3')}")
    print(f"   key='' → {mask_key('')}")

    print("\n" + "═" * 60)
    print("  测试完成")
    print("═" * 60)
