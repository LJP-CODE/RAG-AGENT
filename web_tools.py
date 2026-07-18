"""
web_tools.py — 网络搜索与网页读取工具

提供两个函数供 LangChain Agent 调用：
  1. web_search(query, max_results=5)  — 通过 Bing 搜索互联网（国内可直连）
  2. read_webpage(url, max_chars=3000) — 获取网页正文内容（纯文本）

安全限制:
  - 请求超时:  web_search 10 秒, read_webpage 15 秒
  - 网页大小限制: 最大读取 1 MB，超过则截断
  - URL 白名单:  仅允许 http:// 和 https:// 协议
  - 禁止读取本地文件: file:// 协议直接拒绝
  - User-Agent:  合法浏览器标识（Chrome 131 / Windows 10）
  - 调用频率:  单次会话中 read_webpage 最多调用 3 次
  - 遵守 robots.txt 规范

权限:
  - 无需 API Key（Bing 免费网页搜索）
  - 无需特殊文件系统权限
  - 网络访问仅限外网 HTTP / HTTPS（走系统代理）
  - 不读取或写入任何本地文件（除 Python logging 日志外）
  - robots.txt 检查使用标准库 urllib.robotparser

依赖:
  pip install requests beautifulsoup4 lxml
  以上依赖已在 requirements.txt 中声明。

用法示例:
    >>> from web_tools import web_search, read_webpage
    >>> print(web_search("Python 编程", max_results=3))
    >>> print(read_webpage("https://example.com", max_chars=1000))
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ─── 日志（仅此一处文件写入，符合"除日志外不读写本地文件"的要求）───
logger = logging.getLogger("web_tools")

# ═══════════════════════════════════════════════════════════════
# 全局 HTTP 会话 & 常量
# ═══════════════════════════════════════════════════════════════

# 复用连接池，提升多次请求性能
_session = requests.Session()

# 合法浏览器 User-Agent（Chrome 131 on Windows 10）
# 避免被目标网站当作爬虫拦截
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",  # Do Not Track
})

# ─── 超时限制 ───
SEARCH_TIMEOUT = 10             # web_search 请求超时（秒）
READ_TIMEOUT = 15               # read_webpage 请求超时（秒）

# ─── 网页大小限制 ───
MAX_RESPONSE_SIZE = 1 * 1024 * 1024  # 最大读取 1 MB，超过则截断

# ─── 调用频率限制 ───
MAX_READ_CALLS_PER_SESSION = 3  # 单次会话最多调用 read_webpage 次数

# ─── Bing 搜索接口（国内可直连，无需 Key）──
# 原先使用 DuckDuckGo（html.duckduckgo.com/html/），但国内网络无法访问，
# 改用 Bing（www.bing.com）作为搜索后端。
BING_SEARCH_URL = "https://www.bing.com/search"

# ═══════════════════════════════════════════════════════════════
# read_webpage 调用计数器（线程安全，每次 API 请求开始时重置）
# ═══════════════════════════════════════════════════════════════

# Agent 在一次 /ask 请求中是同步执行的，因此使用简单全局计数器即可。
# agent_api.py 在每次请求开头调用 reset_read_counter() 归零。
_read_call_count: int = 0
_counter_lock = threading.Lock()

# robots.txt 缓存（避免重复请求同一站点的 robots.txt）
_robots_cache: dict[str, Optional[RobotFileParser]] = {}
_robots_cache_lock = threading.Lock()


def _get_robots_parser(domain: str) -> Optional[RobotFileParser]:
    """
    获取指定域名的 robots.txt 解析器（带缓存）。

    参数:
        domain: 域名（如 "example.com"）

    返回:
        RobotFileParser 实例，若获取失败则返回 None（允许继续访问）
    """
    with _robots_cache_lock:
        if domain in _robots_cache:
            return _robots_cache[domain]

    rp = RobotFileParser()
    rp.set_url(f"https://{domain}/robots.txt")
    try:
        rp.read()
        logger.debug("已获取 %s 的 robots.txt", domain)
    except Exception:
        # robots.txt 获取失败时不阻断请求（多数网站允许未声明时抓取）
        logger.debug("无法获取 %s 的 robots.txt，允许继续", domain)
        rp = None

    with _robots_cache_lock:
        _robots_cache[domain] = rp
    return rp


def _check_robots(url: str) -> Optional[str]:
    """
    检查目标 URL 是否被 robots.txt 禁止访问。

    参数:
        url: 目标网页地址

    返回:
        若被禁止，返回错误信息字符串；允许访问则返回 None
    """
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.hostname or ""
    if not domain:
        return None  # 无法解析域名时不阻断

    rp = _get_robots_parser(domain)
    if rp is None:
        return None  # robots.txt 不可用时允许访问

    # 使用 "*" 作为通用爬虫 User-Agent
    if not rp.can_fetch("*", url):
        logger.info("robots.txt 禁止访问: %s", url)
        return f"访问被拒绝：{domain} 的 robots.txt 禁止抓取该页面"

    return None


def reset_read_counter(session_id: str = "") -> None:
    """
    重置 read_webpage 的全局调用计数。
    由 agent_api.py 在每次 /ask 请求开始时调用，确保每次对话独立计数。

    参数:
        session_id: 会话标识符（保留参数，用于日志 & 兼容性；实际使用全局计数器）
    """
    global _read_call_count
    with _counter_lock:
        _read_call_count = 0
        logger.debug(
            "已重置 read_webpage 计数器 (session=%s)", session_id or "__default__"
        )


def _increment_read_counter() -> Optional[str]:
    """
    增加 read_webpage 全局调用计数，若超限返回错误信息。

    返回:
        未超限返回 None，超限返回错误信息字符串
    """
    global _read_call_count
    with _counter_lock:
        if _read_call_count >= MAX_READ_CALLS_PER_SESSION:
            return (
                f"调用次数已达上限：单次请求中最多允许调用 Read_Webpage "
                f"{MAX_READ_CALLS_PER_SESSION} 次。"
                f"当前已调用 {_read_call_count} 次，请基于已有信息作答。"
            )
        _read_call_count += 1
        logger.debug(
            "read_webpage 调用: %d/%d",
            _read_call_count, MAX_READ_CALLS_PER_SESSION,
        )
        return None


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _validate_url(url: str) -> Optional[str]:
    """
    校验 URL 合法性（白名单 & 黑名单）。

    白名单: http:// 和 https://
    黑名单: file://（及其他本地文件协议）

    参数:
        url: 待校验的 URL 字符串

    返回:
        合法返回 None，不合法返回错误信息字符串
    """
    if not url or not isinstance(url, str):
        return "错误：请输入有效的 URL"

    url = url.strip()

    # ── 黑名单：禁止本地文件协议 ──
    if re.match(r'^file://', url, re.IGNORECASE):
        return "错误：禁止读取本地文件（file:// 协议不被允许）"

    # ── 白名单：仅允许 http:// 和 https:// ──
    if not re.match(r'^https?://', url, re.IGNORECASE):
        return (
            "错误：URL 必须以 http:// 或 https:// 开头。"
            "仅支持 HTTP/HTTPS 协议，不支持 ftp://、file:// 等协议。"
        )

    # ── 基本格式检查 ──
    parsed = urlparse(url)
    if not parsed.netloc:
        return f"错误：URL 格式不正确 — 缺少有效域名: {url}"

    return None


def _clean_html_and_extract_text(html: str) -> str:
    """
    解析 HTML，移除无关节點并提取正文纯文本。

    自动移除标签: script, style, nav, footer, header, aside,
                   noscript, iframe, svg, form, input, button,
                   select, meta, link, source, img, video, audio

    提取策略:
        1. 优先使用 <article>
        2. 其次 <main> 或 role="main"
        3. 再尝试常见内容容器 (.content, .post, .article 等)
        4. 最后回退到 <body>

    参数:
        html: 原始 HTML 字符串

    返回:
        清理后的纯文本字符串
    """
    soup = BeautifulSoup(html, "lxml")

    # ── 移除不需要的标签 ──
    tags_to_remove = [
        "script", "style", "nav", "footer", "header", "aside",
        "noscript", "iframe", "svg",
        "form", "input", "button", "select", "textarea",
        "meta", "link", "source",
        "img", "video", "audio", "canvas",
    ]
    for tag in soup(tags_to_remove):
        tag.decompose()

    # ── 按优先级提取正文容器 ──
    body = None
    for selector in [
        "article",
        "main",
        '[role="main"]',
        ".content",
        ".post",
        ".article",
        ".post-content",
        ".entry-content",
        "#content",
        "body",
    ]:
        candidate = soup.select_one(selector)
        if candidate and candidate.get_text(strip=True):
            body = candidate
            break

    if body is None:
        body = soup  # 最终回退

    # ── 提取文本并清理 ──
    text = body.get_text(separator="\n", strip=True)

    # 合并连续空行（3 个以上 → 2 个）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 按行去首尾空白
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = "\n".join(lines)

    return cleaned


# ═══════════════════════════════════════════════════════════════
# 1. web_search — DuckDuckGo 互联网搜索
# ═══════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 5) -> str:
    """
    使用 Bing 搜索互联网，返回摘要格式结果。

    接口说明:
        - 后端: Bing 搜索（www.bing.com/search）
        - 无需 API Key，无需注册，国内可直连
        - 基于 HTML 页面解析提取搜索结果（li.b_algo 结构）

    安全限制:
        - 请求超时: 10 秒

    参数:
        query:       搜索关键词（纯文本）
        max_results: 最多返回多少条结果，默认 5，范围 1~10

    返回:
        格式化后的搜索结果纯文本，包含序号、标题、摘要、URL。
        若无结果或出错，返回友好提示。

    使用场景:
        - 用户询问最新新闻、事件、人物
        - 知识库（RAG_Search）中没有答案的问题
        - 需要实时数据的问题（天气、股价、赛事等）
    """
    # ── 参数校验 ──
    if not query or not isinstance(query, str) or not query.strip():
        return "错误：请输入有效的搜索关键词"

    query = query.strip()
    max_results = max(1, min(max_results, 10))  # 钳制在 1~10

    try:
        # ── 发起搜索请求（超时 10 秒）──
        # Bing 使用 GET，关键词放在 q 参数；cc=CN 走中国版，setlang=zh-CN 中文界面
        resp = _session.get(
            BING_SEARCH_URL,
            params={"q": query, "cc": "CN", "setlang": "zh-CN"},
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()

        # ── 解析 HTML 搜索结果 ──
        # Bing 自然结果位于 <li class="b_algo"> 内：
        #   标题+链接: h2 a（a.href 即真实目标 URL）
        #   摘要:      .b_caption p
        soup = BeautifulSoup(resp.text, "lxml")
        result_items = soup.select("li.b_algo")

        if not result_items:
            return (
                f"未找到关于「{query}」的搜索结果。\n"
                f"建议：尝试更换搜索关键词或使用更简单的表达。"
            )

        # ── 提取前 N 条结果 ──
        lines = [f"## 网络搜索结果：「{query}」"]
        count = 0

        for item in result_items:
            if count >= max_results:
                break

            # 标题 + URL
            title_tag = item.select_one("h2 a")
            title = title_tag.get_text(strip=True) if title_tag else "（无标题）"
            href = title_tag.get("href", "") if title_tag else ""
            # Bing 偶尔返回相对链接，补全协议
            if href.startswith("//"):
                href = "https:" + href

            # 摘要（多种可能的容器，依次尝试）
            snippet = ""
            for sel in (".b_caption p", ".b_caption .b_paractl", "p"):
                snippet_tag = item.select_one(sel)
                if snippet_tag:
                    snippet = snippet_tag.get_text(strip=True)
                    if snippet:
                        break

            # 跳过完全空白的结果
            if not title and not snippet:
                continue

            count += 1
            lines.append(f"\n[{count}] {title}")
            if href:
                lines.append(f"    URL: {href}")
            if snippet:
                lines.append(f"    摘要: {snippet}")

        if count == 0:
            return (
                f"未找到关于「{query}」的搜索结果。\n"
                f"建议：尝试更换搜索关键词。"
            )

        lines.append(f"\n（共 {count} 条结果，来源: Bing 搜索）")
        return "\n".join(lines)

    # ── 具体异常处理 ──
    except requests.Timeout:
        return (
            f"搜索超时：请求「{query}」超过 {SEARCH_TIMEOUT} 秒未响应。\n"
            f"建议：稍后重试或尝试更精确的搜索关键词。"
        )
    except requests.ConnectionError:
        return (
            "网络连接失败：无法访问 Bing 搜索服务。\n"
            "建议：检查网络连接或系统代理设置。"
        )
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "?"
        return f"搜索服务返回 HTTP {status_code} 错误，请稍后重试。"
    except requests.RequestException as e:
        return f"搜索请求异常：{e}"
    except ImportError:
        return "缺少依赖：请安装 beautifulsoup4 和 lxml（pip install beautifulsoup4 lxml）"
    except Exception as e:
        logger.exception("web_search 未预期的错误")
        return f"搜索过程发生未知错误：{e}"


# ═══════════════════════════════════════════════════════════════
# 2. read_webpage — 读取网页正文
# ═══════════════════════════════════════════════════════════════

def read_webpage(url: str, max_chars: int = 3000) -> str:
    """
    获取指定 URL 的网页内容，提取正文纯文本。

    安全限制:
        - 请求超时: 15 秒
        - 最大响应体: 1 MB（超过截断并报错）
        - URL 白名单: 仅 http:// 和 https://
        - 禁止 file:// 协议
        - 单次会话最多调用 3 次
        - 遵守目标站点 robots.txt 规范
        - 使用合法浏览器 User-Agent

    HTML 解析:
        - 自动移除 script, style, nav, footer, header, aside 等非正文标签
        - 优先提取 <article> → <main> → .content → <body> 中的内容

    参数:
        url:       网页 URL（必须以 http:// 或 https:// 开头）
        max_chars: 最大返回字符数，默认 3000，范围 500~50000

    返回:
        网页正文纯文本（含标题和来源标注）。
        若失败返回友好错误提示。

    使用场景:
        - Web_Search 返回的摘要信息不够详细时
        - 用户要求查看某个网页的具体内容
        - Agent 需要从链接中提取更多上下文信息
    """
    # ── 1. URL 白名单 & 黑名单校验 ──
    validation_error = _validate_url(url)
    if validation_error:
        return validation_error

    url = url.strip()

    # ── 2. robots.txt 规范检查 ──
    robots_error = _check_robots(url)
    if robots_error:
        return robots_error

    # ── 3. 调用频率限制（单次请求最多 3 次）──
    # agent_api.py 在每次 /ask 请求开头调用 reset_read_counter() 归零
    limit_error = _increment_read_counter()
    if limit_error:
        return limit_error

    # ── 4. 参数钳制 ──
    max_chars = max(500, min(max_chars, 50000))  # 限制在 500~50000

    try:
        # ── 5. 发起 HTTP GET 请求（超时 15 秒）──
        # 使用 stream=True 以便精确控制读取大小（1 MB 上限）
        resp = _session.get(
            url,
            timeout=READ_TIMEOUT,
            stream=True,
            allow_redirects=True,
        )
        resp.raise_for_status()

        # ── 6. 网页大小限制：最大读取 1 MB ──
        # 逐块读取，超过 MAX_RESPONSE_SIZE 则截断并返回错误
        content_chunks: list[bytes] = []
        total_bytes = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                total_bytes += len(chunk)
                if total_bytes > MAX_RESPONSE_SIZE:
                    resp.close()
                    return (
                        f"网页内容过大：响应体超过 {MAX_RESPONSE_SIZE // (1024 * 1024)} MB 限制"
                        f"（实际大小 > {total_bytes / (1024 * 1024):.1f} MB）。\n"
                        f"建议：使用 Web_Search 获取摘要信息代替直接读取。"
                    )
                content_chunks.append(chunk)

        # 合并读取的字节
        content = b"".join(content_chunks)

        # 检测编码
        resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
        try:
            html_text = content.decode(resp.encoding)
        except (UnicodeDecodeError, LookupError):
            html_text = content.decode("utf-8", errors="replace")

    except requests.Timeout:
        return (
            f"读取超时：访问 {url} 超过 {READ_TIMEOUT} 秒未响应。\n"
            f"建议：稍后重试或尝试其他来源。"
        )
    except requests.ConnectionError:
        return (
            f"网络连接失败：无法访问 {url}。\n"
            f"建议：检查 URL 是否正确，或确认网络连接/代理设置。"
        )
    except requests.TooManyRedirects:
        return f"请求失败：{url} 重定向次数过多，无法获取内容。"
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        status_messages = {
            403: f"访问被拒绝（403 Forbidden）：{url} 可能设有反爬虫保护。",
            404: f"页面不存在（404 Not Found）：{url}。",
            410: f"页面已被永久删除（410 Gone）：{url}。",
            429: f"请求过于频繁（429 Too Many Requests）：{url}，请稍后再试。",
            500: f"目标服务器内部错误（500）：{url}。",
            502: f"网关错误（502 Bad Gateway）：{url}。",
            503: f"服务暂不可用（503）：{url}。",
        }
        return status_messages.get(
            status,
            f"HTTP 错误 {status}：访问 {url} 失败。",
        )
    except requests.RequestException as e:
        return f"网络请求异常：{e}"
    except ImportError:
        return "缺少依赖：请安装 beautifulsoup4 和 lxml（pip install beautifulsoup4 lxml）"

    # ── 7. HTML 解析 & 正文提取 ──
    try:
        soup = BeautifulSoup(html_text, "lxml")

        # 提取标题
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 清理并提取正文
        cleaned = _clean_html_and_extract_text(html_text)

        if not cleaned:
            return (
                f"未能从 {url} 提取到有效的正文内容。\n"
                f"可能原因：页面为纯 JavaScript 渲染、内容为空、或被反爬虫机制拦截。\n"
                f"建议：尝试使用 Web_Search 搜索相关关键词。"
            )

        # ── 8. 组装结果 ──
        result_parts: list[str] = []
        if title:
            result_parts.append(f"标题: {title}")
        result_parts.append(f"来源: {url}")
        result_parts.append("")

        # ── 9. 截断处理 ──
        if len(cleaned) > max_chars:
            result_parts.append(cleaned[:max_chars])
            result_parts.append(
                f"\n\n...（内容过长，原始全文共 {len(cleaned):,} 字符，"
                f"以上仅显示前 {max_chars:,} 字符。"
                f"如需更多内容，可增加 max_chars 参数再次请求。）"
            )
        else:
            result_parts.append(cleaned)
            if len(cleaned) > 0:
                result_parts.append(
                    f"\n\n（全文共 {len(cleaned):,} 字符，已完整显示）"
                )

        return "\n".join(result_parts)

    except Exception as e:
        logger.exception("read_webpage HTML 解析出错")
        return f"网页解析出错：{e}"


# ═══════════════════════════════════════════════════════════════
# 独立测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 配置控制台日志（测试用）
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("  web_tools 独立测试")
    print("=" * 60)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        arg = sys.argv[2] if len(sys.argv) > 2 else ""

        if mode == "search":
            print(web_search(arg))
        elif mode == "read":
            print(read_webpage(arg))
        elif mode == "reset":
            reset_read_counter("__default__")
            print("已重置 read_webpage 计数器")
        else:
            print("用法:")
            print("  python web_tools.py search <关键词>")
            print("  python web_tools.py read   <URL>")
            print("  python web_tools.py reset          # 重置计数器")
    else:
        # ── 演示模式 ──
        print("\n📝 搜索演示：")
        result = web_search("Python 编程语言", max_results=3)
        print(result)

        print("\n" + "=" * 60)

        print("\n📝 URL 校验演示：")
        print("  https://example.com   →", _validate_url("https://example.com") or "✓ 合法")
        print("  file:///etc/passwd    →", _validate_url("file:///etc/passwd") or "✓ 合法")
        print("  ftp://files.example   →", _validate_url("ftp://files.example") or "✓ 合法")

        print("\n📝 网页读取演示（需要网络）：")
        print(read_webpage("https://httpbin.org/get", max_chars=500))
