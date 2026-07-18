"""
ToolRegistry — AI Agent 的工具注册中心

支持动态注册、配置文件加载/保存、批量获取工具列表。
设计为与 LangChain / langchain_classic 的 Tool 对象兼容。
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

from langchain_classic.agents import Tool


class ToolRegistry:
    """工具注册中心，统一管理 Agent 可用的所有工具。

    用法::

        registry = ToolRegistry()
        registry.register_function("Calculator", calc_func, "数学计算工具")
        tools = registry.get_all_tools()
    """

    def __init__(self, config_path: str = "./tools_config.json"):
        """
        Args:
            config_path: JSON 配置文件路径，用于持久化工具列表
        """
        self.config_path = config_path

        # name -> Callable：保存原始函数引用，不依赖 Tool 包装
        self._functions: Dict[str, Callable] = {}

        # name -> Tool：缓存已包装好的 LangChain Tool 对象
        self._tools: Dict[str, Tool] = {}

        # 从配置文件自动加载（如果有）
        if os.path.exists(config_path):
            self.load_from_config()

    # ------------------------------------------------------------
    # 注册工具
    # ------------------------------------------------------------

    def register_function(
        self,
        name: str,
        func: Callable,
        description: str,
        **tool_kwargs: Any,
    ) -> Tool:
        """注册一个 Python 函数为 Agent 可调用的工具。

        Args:
            name:        工具名称（Agent 通过此名称调用工具）
            func:        实际执行的 Python 函数
            description: 工具描述（Agent 据此决定何时使用该工具）
            **tool_kwargs: 透传给 Tool() 构造函数的额外参数

        Returns:
            注册后的 Tool 对象（已缓存，可重复调用）
        """
        if not name or not callable(func):
            raise ValueError("name 必须为非空字符串，func 必须为可调用对象")

        tool = Tool(
            name=name,
            func=func,
            description=description,
            **tool_kwargs,
        )

        self._functions[name] = func
        self._tools[name] = tool
        return tool

    def register_api(
        self,
        name: str,
        endpoint: str,
        description: str,
        **http_kwargs: Any,
    ) -> Tool:
        """【预留接口】从 API 端点动态注册工具。

        通过 HTTP 请求调用外部 API 作为工具。当前为桩代码框架，
        后续可对接 requests / aiohttp 实现真实的 API 调用工具。

        Args:
            name:        工具名称
            endpoint:    API URL
            description: 工具描述
            **http_kwargs: 请求参数（method、headers、timeout 等）

        Returns:
            注册后的 Tool 对象
        """
        def _api_stub(input_str: str) -> str:
            # ── 预留：替换为真实的 HTTP 调用 ──
            return (
                f"[API Stub] name={name}, endpoint={endpoint}, "
                f"input={input_str}, 尚未对接真实服务"
            )

        return self.register_function(name, _api_stub, f"{description} (API 预留)")

    # ------------------------------------------------------------
    # 查询 / 获取
    # ------------------------------------------------------------

    def get_tool(self, name: str) -> Optional[Tool]:
        """按名称获取单个 Tool 对象。"""
        return self._tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """获取所有已注册的工具列表，供 Agent 初始化使用。

        返回的列表可直接传给 ``initialize_agent(tools=...)``。
        """
        return list(self._tools.values())

    def list_registered(self) -> List[Dict[str, str]]:
        """返回当前注册的工具清单（仅元信息，不含函数）。"""
        return [
            {"name": name, "description": tool.description}
            for name, tool in self._tools.items()
        ]

    @property
    def count(self) -> int:
        """当前注册的工具数量。"""
        return len(self._tools)

    # ------------------------------------------------------------
    # 配置文件持久化
    # ------------------------------------------------------------

    def save_config(self, config_path: Optional[str] = None) -> None:
        """将当前工具配置保存到 JSON 文件。

        Args:
            config_path: 输出路径，默认使用 ``self.config_path``
        """
        path = config_path or self.config_path
        tool_list = self.list_registered()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tools": tool_list}, f, ensure_ascii=False, indent=2)

    def load_from_config(self, config_path: Optional[str] = None) -> None:
        """从 JSON 配置文件加载工具清单并注册。

        配置文件中只包含 name / description，实际的函数实现
        需要调用 ``register_function()`` 手动注册。本方法仅为
        辅助查看 / 校验配置内容，不自动绑定函数。

        Args:
            config_path: 输入路径，默认使用 ``self.config_path``
        """
        path = config_path or self.config_path
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        for tool_def in config.get("tools", []):
            name = tool_def.get("name")
            desc = tool_def.get("description", "")

            if name in self._functions:
                # 函数已注册 → 重新包装 Tool（避免遗漏）
                self.register_function(name, self._functions[name], desc)
            elif name in self._tools:
                # 已有 Tool 但无原始函数 → 更新描述
                self._tools[name].description = desc
            else:
                # 创建桩函数工具，后续可被 register_function 同名覆盖
                def _placeholder(input_str: str = "", _n=name) -> str:
                    return f"工具 '{_n}' 尚未实现，请注册对应函数。"

                self.register_function(name, _placeholder, desc)
                print(f"  🔧 从配置创建桩工具: {name}（可用 register_function 替换）")

    # ------------------------------------------------------------
    # 批处理 / 撤销
    # ------------------------------------------------------------

    def register_batch(self, tool_defs: List[Dict]) -> List[Tool]:
        """批量注册多个函数工具。每个字典需包含 'name', 'func', 'description'。"""
        results = []
        for defn in tool_defs:
            results.append(
                self.register_function(
                    name=defn["name"],
                    func=defn["func"],
                    description=defn.get("description", ""),
                )
            )
        return results

    def unregister(self, name: str) -> bool:
        """移除一个已注册的工具。

        Returns:
            True 表示成功移除，False 表示工具不存在
        """
        if name in self._tools:
            del self._tools[name]
            self._functions.pop(name, None)
            return True
        return False

    def clear(self) -> None:
        """清空所有已注册的工具。"""
        self._tools.clear()
        self._functions.clear()


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    import math

    def calculator(expr: str) -> str:
        try:
            return str(eval(expr, {"__builtins__": {}}, {"math": math}))
        except Exception as e:
            return f"计算错误: {e}"

    def get_time(_: str = "") -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    DEMO_CONFIG = "./_demo_tools_config.json"

    # ════════════════════════════════════════════════════════════
    # 1. 程序化注册
    # ════════════════════════════════════════════════════════════
    print("═" * 50)
    print("1️⃣  程序化注册工具")
    registry = ToolRegistry(DEMO_CONFIG)
    registry.register_function("Calculator", calculator, "数学计算工具")
    registry.register_function("Get_Time", get_time, "获取当前时间")

    print(f"\n注册了 {registry.count} 个工具：")
    for t in registry.list_registered():
        print(f"   • {t['name']}: {t['description']}")

    # ════════════════════════════════════════════════════════════
    # 2. 持久化到 JSON
    # ════════════════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("2️⃣  保存配置 → JSON")
    registry.save_config()
    with open(DEMO_CONFIG, "r", encoding="utf-8") as f:
        print(f.read())

    # ════════════════════════════════════════════════════════════
    # 3. 从配置重建
    # ════════════════════════════════════════════════════════════
    print("═" * 50)
    print("3️⃣  从配置文件重建（桩工具占位）")
    registry2 = ToolRegistry(DEMO_CONFIG)
    for tool in registry2.get_all_tools():
        print(f"   {tool.name} → {tool.run('test')}")

    # ════════════════════════════════════════════════════════════
    # 4. 桩工具替换为真实函数
    # ════════════════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("4️⃣  同名注册覆盖桩工具 → 替换为真实函数")
    registry2.register_function("Calculator", calculator, "数学计算工具")
    registry2.register_function("Get_Time", get_time, "获取当前时间")
    for tool in registry2.get_all_tools():
        print(f"   {tool.name} → {tool.run('1+2*3') if tool.name == 'Calculator' else tool.run('')}")

    # ════════════════════════════════════════════════════════════
    # 5. API 预留接口
    # ════════════════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("5️⃣  预留 API 工具注册")
    api_tool = registry2.register_api("Weather", "https://api.weather.com", "查询天气")
    print(f"   {api_tool.name} → {api_tool.run('上海')}")

    # 清理演示文件
    if os.path.exists(DEMO_CONFIG):
        os.remove(DEMO_CONFIG)

    print("\n" + "═" * 50)
    print("✅ 演示完成")
