"""
web_tools.py — 网络搜索、网页读取与外部 API 工具

提供五个函数供 Agent 调用：
  1. web_search(query, max_results=5)     — 通过 Bing Web Search API 搜索互联网
  2. read_webpage(url, max_chars=3000)    — 获取网页正文内容
  3. news_search(query, max_results=5)    — 通过 NewsAPI 搜索全球新闻
  4. weather_query(city)                  — 通过和风天气 API 查询实时天气
  5. academic_search(query, max_results=5)— 通过 arXiv API 搜索学术论文

依赖:
  pip install requests beautifulsoup4 lxml

环境变量:
  SERPAPI_API_KEY — SerpAPI 密钥 (https://serpapi.com)，优先使用
  BING_API_KEY    — Bing Web Search API 密钥 (https://portal.azure.com)，备选
  NEWS_API_KEY    — NewsAPI 密钥 (https://newsapi.org)
  WEATHER_API_KEY — 和风天气密钥 (https://dev.qweather.com)
"""

import os
import re
import time
import logging
import requests
from typing import Optional

# ─── 全局 HTTP 会话（复用连接池，提升性能）───
_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
})

# ─── 常量 ───
REQUEST_TIMEOUT = 15          # 单次请求超时（秒）
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
BING_API_KEY = os.getenv("BING_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "")
SERPAPI_URL = "https://serpapi.com/search"

# ─── 各 API 独立超时（秒）───
NEWS_TIMEOUT = 10
WEATHER_TIMEOUT = 10
ARXIV_TIMEOUT = 10

# ─── 同类 API 调用最小间隔（秒）───
RATE_LIMIT_INTERVAL = 2.0

# ═══════════════════════════════════════════════════════
# 日志 — API Key 自动脱敏
# ═══════════════════════════════════════════════════════

_logger = logging.getLogger("web_tools")


class _APIKeySanitizer(logging.Filter):
    """日志过滤器：自动将 API Key 脱敏为 ***。

    匹配规则：长度 >= 20 的连续字母数字字符串 → 替换为 `***`。
    涵盖 NewsAPI (32 chars)、和风天气 (32 chars) 等常见 Key 格式。
    """

    _KEY_PATTERN = re.compile(r'[A-Za-z0-9\-_]{20,}')

    @classmethod
    def sanitize(cls, text: str) -> str:
        """对任意字符串中的 API Key 进行脱敏。"""
        if not isinstance(text, str):
            return text
        return cls._KEY_PATTERN.sub("***", text)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg and isinstance(record.msg, str):
            record.msg = self.sanitize(record.msg)
        if record.args:
            record.args = tuple(
                self.sanitize(a) if isinstance(a, str) else a
                for a in record.args
            )
        return True


_logger.addFilter(_APIKeySanitizer())


def _sanitize_error(msg: str) -> str:
    """对错误/异常消息脱敏，确保不泄露 API Key。"""
    return _APIKeySanitizer.sanitize(msg)


# ═══════════════════════════════════════════════════════
# 限流器
# ═══════════════════════════════════════════════════════

class RateLimiter:
    """简单的时间间隔限流器。

    确保同类 API 调用之间的间隔不少于 ``min_interval`` 秒。

    - NewsAPI  免费版: 100 次 / 天
    - 和风天气 免费版: 10 次 / 分钟
    - arXiv    官方规定: 1 次 / 秒

    本限流器统一使用 2 秒最小间隔作为安全底线。
    """

    def __init__(self, min_interval: float = RATE_LIMIT_INTERVAL):
        self.min_interval = min_interval
        self._last_call: float = 0.0

    def acquire(self) -> None:
        """阻塞当前线程，直到满足最小调用间隔。"""
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()


# 每个外部 API 使用独立的限流器实例
_news_limiter = RateLimiter()
_weather_limiter = RateLimiter()
_arxiv_limiter = RateLimiter()


# ============================================================
# 辅助函数
# ============================================================

def _safe_extract(soup, selector: str, default: str = "") -> str:
    """安全提取 HTML 元素的文本，不存在时返回默认值。"""
    tag = soup.select_one(selector)
    return tag.get_text(strip=True) if tag else default


# ============================================================
# 1. web_search — Bing Web Search API
# ============================================================

# ─── 搜索垃圾结果过滤关键词（不区分大小写）───
_SPAM_KEYWORDS = [
    "quote", "zitat", "reddit", "edmentum", "calculator",
    "google", "youtube", "obsproject", "online-calculator",
    "citations", "proverbes",
]


def _is_spam_result(title: str, snippet: str, url: str = "") -> bool:
    """检查搜索结果是否为垃圾/无意义结果。

    对标题、摘要进行关键词匹配，匹配到任一垃圾关键词则跳过。
    """
    combined = f"{title} {snippet} {url}".lower()
    for kw in _SPAM_KEYWORDS:
        if kw in combined:
            return True
    return False


def web_search(query: str, max_results: int = 5) -> str:
    """
    搜索互联网获取实时信息（优先 SerpAPI，备选 Bing）。

    参数:
        query:       搜索关键词（支持中文）
        max_results: 最多返回多少条结果（默认 5，范围 1-10）

    返回:
        格式化后的搜索结果文本；失败时返回友好错误说明（绝不抛出异常）。
    """
    if not query or not isinstance(query, str):
        return "错误：请输入有效的搜索关键词"

    max_results = max(1, min(max_results, 10))

    # ── 优先 SerpAPI，其次 Bing ──
    if SERPAPI_KEY:
        return _search_serpapi(query, max_results)
    elif BING_API_KEY:
        return _search_bing(query, max_results)
    else:
        _logger.warning("web_search 被调用但 SERPAPI_API_KEY 和 BING_API_KEY 均未配置")
        return (
            "错误：未配置搜索 API Key，无法使用网络搜索功能。\n"
            "请在 .env 文件中添加: SERPAPI_API_KEY=your_key\n"
            "获取方式: https://serpapi.com 注册即可（免费 100 次/月）"
        )


def _search_serpapi(query: str, max_results: int) -> str:
    """通过 SerpAPI (Google Search) 搜索。"""
    try:
        resp = _session.get(
            SERPAPI_URL,
            params={
                "api_key": SERPAPI_KEY,
                "q": query,
                "engine": "google",
                "hl": "zh-cn",
                "gl": "cn",
                "num": max_results,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        organic = data.get("organic_results", [])
        if not organic:
            return f"未找到与「{query}」相关的信息，建议换个关键词重试。"

        # ── 过滤垃圾结果 ──
        filtered = []
        skipped = 0
        for r in organic:
            title = r.get("title", "") or ""
            snippet = r.get("snippet", "") or ""
            link = r.get("link", "") or ""
            if _is_spam_result(title, snippet, link):
                skipped += 1
                continue
            filtered.append(r)

        if not filtered:
            return f"搜索「{query}」未找到有效结果，建议换关键词重试。"

        if skipped > 0:
            _logger.info(
                "serpapi 过滤了 %d 条垃圾结果 (query=%r, total=%d, kept=%d)",
                skipped, query, len(organic), len(filtered),
            )

        lines = [f"## 搜索结果：「{query}」"]
        for i, r in enumerate(filtered[:max_results], 1):
            title = r.get("title", "（无标题）") or "（无标题）"
            link = r.get("link", "")
            snippet = r.get("snippet", "")

            lines.append(f"\n[{i}] {title}")
            if link:
                lines.append(f"    链接: {link}")
            if snippet:
                lines.append(f"    摘要: {snippet}")

        lines.append(f"\n共 {min(len(filtered), max_results)} 条结果（来源: Google）")
        return "\n".join(lines)

    except requests.Timeout:
        _logger.warning("serpapi 超时: query=%r", query)
        return f"搜索服务暂时不可用：请求超时（{REQUEST_TIMEOUT} 秒）。请稍后重试。"
    except requests.ConnectionError as e:
        _logger.warning("serpapi 连接失败: %s", _sanitize_error(str(e)))
        return "搜索服务暂时不可用：网络连接失败。请检查网络后稍后重试。"
    except requests.HTTPError as e:
        status = e.response.status_code
        _logger.warning("serpapi HTTP 错误: status=%d", status)
        if status == 401:
            return "搜索服务暂时不可用：API Key 无效或已过期（401）。请稍后重试。"
        elif status == 403:
            return "搜索服务暂时不可用：API Key 无权限或配额已用完（403）。请稍后重试。"
        else:
            return f"搜索服务暂时不可用：HTTP {status}。请稍后重试。"
    except requests.RequestException as e:
        _logger.warning("serpapi 请求异常: %s", _sanitize_error(str(e)))
        return f"搜索服务暂时不可用：{_sanitize_error(str(e))}。请稍后重试。"
    except Exception as e:
        _logger.exception("serpapi 未预期异常")
        return f"搜索服务暂时不可用：{_sanitize_error(str(e))}。请稍后重试。"


def _search_bing(query: str, max_results: int) -> str:
    """通过 Bing Web Search API v7 搜索（备选后端）。"""
    try:
        resp = _session.get(
            BING_SEARCH_URL,
            params={
                "q": query,
                "count": max_results,
                "mkt": "zh-CN",
                "textFormat": "Raw",
            },
            headers={"Ocp-Apim-Subscription-Key": BING_API_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        web_pages = (data.get("webPages") or {}).get("value", [])
        if not web_pages:
            return "未找到相关信息，建议换个关键词重试。"

        # ── 过滤垃圾结果 ──
        filtered_pages = []
        skipped_count = 0
        for page in web_pages:
            name = page.get("name", "") or ""
            snippet = page.get("snippet", "") or ""
            url = page.get("url", "") or ""
            if _is_spam_result(name, snippet, url):
                skipped_count += 1
                continue
            filtered_pages.append(page)

        if not filtered_pages:
            _logger.info(
                "bing 全部 %d 条结果被垃圾过滤跳过 (query=%r, skipped=%d)",
                len(web_pages), query, skipped_count,
            )
            return f"搜索 '{query}' 未找到有效结果，建议换关键词或加城市名重试。"

        if skipped_count > 0:
            _logger.info(
                "bing 过滤了 %d 条垃圾结果 (query=%r, total=%d, kept=%d)",
                skipped_count, query, len(web_pages), len(filtered_pages),
            )

        lines = [f"## 搜索结果：「{query}」"]
        for i, page in enumerate(filtered_pages[:max_results], 1):
            name = page.get("name", "（无标题）") or "（无标题）"
            url = page.get("url", "")
            snippet = page.get("snippet", "")

            lines.append(f"\n[{i}] {name}")
            if url:
                lines.append(f"    链接: {url}")
            if snippet:
                lines.append(f"    摘要: {snippet}")

        lines.append(f"\n共 {min(len(filtered_pages), max_results)} 条结果（来源: Bing）")
        return "\n".join(lines)

    except requests.Timeout:
        _logger.warning("bing 超时: query=%r", query)
        return f"搜索服务暂时不可用：请求超时（{REQUEST_TIMEOUT} 秒）。请稍后重试。"
    except requests.ConnectionError as e:
        _logger.warning("bing 连接失败: %s", _sanitize_error(str(e)))
        return "搜索服务暂时不可用：网络连接失败。请检查网络后稍后重试。"
    except requests.HTTPError as e:
        status = e.response.status_code
        _logger.warning("bing HTTP 错误: status=%d", status)
        if status == 401:
            return "搜索服务暂时不可用：API Key 无效或已过期（401）。请稍后重试。"
        elif status == 403:
            return "搜索服务暂时不可用：API Key 无权限或配额已用完（403）。请稍后重试。"
        elif status == 429:
            return "搜索服务暂时不可用：API 请求次数已达每月限额（429）。请稍后重试。"
        else:
            return f"搜索服务暂时不可用：HTTP {status}。请稍后重试。"
    except requests.RequestException as e:
        _logger.warning("bing 请求异常: %s", _sanitize_error(str(e)))
        return f"搜索服务暂时不可用：{_sanitize_error(str(e))}。请稍后重试。"
    except Exception as e:
        _logger.exception("bing 未预期异常")
        return f"搜索服务暂时不可用：{_sanitize_error(str(e))}。请稍后重试。"


# ============================================================
# 2. read_webpage — 读取网页正文
# ============================================================

def read_webpage(url: str, max_chars: int = 3000) -> str:
    """
    获取指定 URL 的网页内容并提取正文。

    使用 BeautifulSoup 解析 HTML，自动移除 script、style、nav、
    footer、header、aside 等非正文元素，提取纯文本。

    参数:
        url:       网页 URL（必须以 http:// 或 https:// 开头）
        max_chars: 最多返回多少字符（默认 3000）

    返回:
        网页正文文本；失败时返回错误说明。

    使用场景:
        - web_search 返回的摘要信息不够详细
        - 用户要求查看某个网页的具体内容
        - 需要从链接中提取更多上下文
    """
    if not url or not isinstance(url, str):
        return "错误：请输入有效的 URL"

    # 基本 URL 校验
    url = url.strip()
    if not re.match(r'^https?://', url, re.IGNORECASE):
        return "错误：URL 必须以 http:// 或 https:// 开头"

    max_chars = max(500, min(max_chars, 50000))  # 限制 500~50000 字符

    try:
        # ── 发起请求 ──
        resp = _session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        # 检测编码（优先使用响应头中的编码，否则用 Content-Type 检测）
        resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"

        # ── HTML 解析 ──
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")

        # ── 移除不需要的标签 ──
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe", "svg",
                         "form", "input", "button", "select",
                         "meta", "link", "source"]):
            tag.decompose()

        # ── 提取正文 ──
        # 策略：优先使用 <article>，否则用 <main>，最后用 <body>
        body = None
        for selector in ["article", "main", '[role="main"]', ".content", ".post", ".article", "body"]:
            candidate = soup.select_one(selector)
            if candidate and candidate.get_text(strip=True):
                body = candidate
                break
        if body is None:
            body = soup

        # ── 清理文本 ──
        text = body.get_text(separator="\n", strip=True)

        # 合并多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除过长的无空格行（可能是代码或乱码）
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned = "\n".join(lines)

        # ── 截断 ──
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        result_parts = []
        if title:
            result_parts.append(f"标题: {title}")
        result_parts.append(f"来源: {url}")
        result_parts.append("")
        result_parts.append(cleaned[:max_chars])

        if len(cleaned) > max_chars:
            result_parts.append(f"\n\n...（内容过长，仅显示前 {max_chars} 字符）")

        return "\n".join(result_parts)

    except requests.Timeout:
        return f"读取超时：访问 {url} 超过 {REQUEST_TIMEOUT} 秒未响应"
    except requests.ConnectionError:
        return f"网络连接失败：无法访问 {url}，请检查 URL 或网络连接"
    except requests.HTTPError as e:
        status = e.response.status_code
        if status == 403:
            return f"访问被拒绝（403）：{url} 可能反爬虫拦截"
        elif status == 404:
            return f"页面不存在（404）：{url}"
        else:
            return f"HTTP 错误 {status}：{url}"
    except requests.RequestException as e:
        return f"请求异常：{e}"
    except ImportError:
        return "缺少依赖：请安装 beautifulsoup4 和 lxml（pip install beautifulsoup4 lxml）"
    except ValueError as e:
        return f"URL 格式错误：{e}"
    except Exception as e:
        return f"网页解析出错：{e}"


# ============================================================
# 3. news_search — NewsAPI 新闻搜索
# ============================================================

def news_search(query: str, max_results: int = 5) -> str:
    """
    使用 NewsAPI (https://newsapi.org) 搜索全球新闻。

    API Key 从环境变量 ``NEWS_API_KEY`` 读取。
    免费版限制：每天 100 次请求。

    参数:
        query:       搜索关键词（支持中文）
        max_results: 最多返回多少条结果（默认 5，范围 1-20）

    返回:
        格式化后的新闻结果文本，每条含标题、描述、来源名称、链接；
        失败时返回友好错误说明。

    使用场景:
        - 用户询问最新新闻、时事热点
        - 需要获取特定主题的媒体报道
    """
    # ── 检查 API Key ──
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        _logger.warning("news_search 被调用但 NEWS_API_KEY 未配置")
        return (
            "错误：未配置 NEWS_API_KEY 环境变量，无法使用新闻搜索功能。"
            "请在 .env 文件中添加: NEWS_API_KEY=your_key"
        )

    if not query or not isinstance(query, str):
        return "错误：请输入有效的搜索关键词"

    max_results = max(1, min(max_results, 20))
    _news_limiter.acquire()

    try:
        resp = _session.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "pageSize": max_results,
                "language": "zh",
                "sortBy": "publishedAt",
                "apiKey": api_key,
            },
            timeout=NEWS_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            msg = _sanitize_error(data.get("message", "未知错误"))
            return f"新闻搜索失败：{msg}"

        articles = data.get("articles", [])
        if not articles:
            return f"未找到关于「{query}」的新闻。"

        lines = [f"## 新闻搜索：「{query}」"]
        for i, article in enumerate(articles[:max_results], 1):
            title = article.get("title", "（无标题）") or "（无标题）"
            description = article.get("description", "") or ""
            source = (article.get("source") or {}).get("name", "未知来源")
            url = article.get("url", "")

            lines.append(f"\n[{i}] {title}")
            lines.append(f"    来源: {source}")
            if description:
                lines.append(f"    摘要: {description}")
            if url:
                lines.append(f"    链接: {url}")

        lines.append(f"\n共 {min(len(articles), max_results)} 条新闻（来源: NewsAPI）")
        return "\n".join(lines)

    except requests.Timeout:
        return f"新闻搜索超时：请求「{query}」超过 {NEWS_TIMEOUT} 秒未响应"
    except requests.HTTPError as e:
        status = e.response.status_code
        if status == 401:
            return "新闻搜索失败：API Key 无效或已过期，请检查 NEWS_API_KEY"
        elif status == 429:
            return "新闻搜索失败：请求次数已达每日限额（免费版 100 次/天），请稍后再试"
        else:
            return f"新闻搜索失败：HTTP {status}"
    except requests.ConnectionError:
        return "网络连接失败：无法访问 NewsAPI 服务，请检查网络连接"
    except requests.RequestException as e:
        return f"新闻搜索请求异常：{_sanitize_error(str(e))}"
    except Exception as e:
        _logger.exception("news_search 未预期异常")
        return f"新闻搜索出错：{_sanitize_error(str(e))}"


# ============================================================
# 4. weather_query — 和风天气实时查询
# ============================================================

def weather_query(city: str) -> str:
    """
    使用和风天气 API (https://dev.qweather.com) 查询城市实时天气。

    两步流程:
        1. 城市搜索 → 获取 Location ID
        2. 实时天气 → 获取天气详情

    API Key 从环境变量 ``WEATHER_API_KEY`` 读取。
    免费版限制：每分钟 10 次请求（本函数一次调用会发 2 个请求）。

    参数:
        city: 城市名称，支持中文（如 "北京"、"上海"、"东京"）

    返回:
        格式化后的天气信息（温度、体感温度、天气状况、湿度、风力等）；
        失败时返回友好错误说明。

    使用场景:
        - 用户询问某地天气
        - 出差/旅行前的天气查询
    """
    # ── 检查 API Key ──
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        _logger.warning("weather_query 被调用但 WEATHER_API_KEY 未配置")
        return (
            "错误：未配置 WEATHER_API_KEY 环境变量，无法使用天气查询功能。"
            "请在 .env 文件中添加: WEATHER_API_KEY=your_key"
        )

    if not city or not isinstance(city, str):
        return "错误：请输入有效的城市名称"

    city = city.strip()
    _weather_limiter.acquire()

    try:
        # ── 步骤 1: 城市搜索 → 获取 Location ID ──
        geo_resp = _session.get(
            "https://geoapi.qweather.com/v2/city/lookup",
            params={"location": city, "key": api_key},
            timeout=WEATHER_TIMEOUT,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if geo_data.get("code") != "200":
            _logger.error("和风天气城市搜索失败: %s", _sanitize_error(str(geo_data)))
            return f"城市查询失败，请确认城市名称是否正确"

        locations = geo_data.get("location", [])
        if not locations:
            return f"未找到城市「{city}」，请确认城市名称是否正确"

        location_id = locations[0]["id"]
        city_name = locations[0].get("name", city)

        # ── 步骤 2: 获取实时天气 ──
        weather_resp = _session.get(
            "https://devapi.qweather.com/v7/weather/now",
            params={"location": location_id, "key": api_key},
            timeout=WEATHER_TIMEOUT,
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

        if weather_data.get("code") != "200":
            _logger.error("和风天气查询失败: %s", _sanitize_error(str(weather_data)))
            return f"天气查询失败，请稍后重试"

        now = weather_data.get("now", {})
        if not now:
            return f"未获取到「{city_name}」的天气数据"

        lines = [
            f"## 实时天气：{city_name}",
            f"    温度: {now.get('temp', 'N/A')}°C",
            f"    体感温度: {now.get('feelsLike', 'N/A')}°C",
            f"    天气状况: {now.get('text', 'N/A')}",
            f"    相对湿度: {now.get('humidity', 'N/A')}%",
            f"    风向: {now.get('windDir', 'N/A')}",
            f"    风力等级: {now.get('windScale', 'N/A')} 级",
            f"    风速: {now.get('windSpeed', 'N/A')} km/h",
            f"    能见度: {now.get('vis', 'N/A')} km",
            f"    气压: {now.get('pressure', 'N/A')} hPa",
            f"",
            f"数据来源: 和风天气",
        ]
        return "\n".join(lines)

    except requests.Timeout:
        return f"天气查询超时：请求「{city}」超过 {WEATHER_TIMEOUT} 秒未响应"
    except requests.HTTPError as e:
        status = e.response.status_code
        if status in (401, 403):
            return "天气查询失败：API Key 无效或已过期，请检查 WEATHER_API_KEY"
        elif status == 429:
            return "天气查询失败：请求次数已达限额（免费版 10 次/分钟），请稍后再试"
        else:
            return f"天气查询失败：HTTP {status}"
    except requests.ConnectionError:
        return "网络连接失败：无法访问和风天气服务，请检查网络连接"
    except requests.RequestException as e:
        return f"天气查询请求异常：{_sanitize_error(str(e))}"
    except Exception as e:
        _logger.exception("weather_query 未预期异常")
        return f"天气查询出错：{_sanitize_error(str(e))}"


# ============================================================
# 5. academic_search — arXiv 学术论文搜索
# ============================================================

def academic_search(query: str, max_results: int = 5) -> str:
    """
    使用 arXiv API (https://arxiv.org) 搜索学术论文。

    arXiv API 免费开放，无需 API Key。
    官方限制：每秒 1 次请求（本限流器使用 2 秒间隔）。

    参数:
        query:       搜索关键词（支持英文，如 "large language model"）
        max_results: 最多返回多少条结果（默认 5，范围 1-20）

    返回:
        格式化后的论文信息（标题、作者、摘要、arXiv ID、发布时间）；
        失败时返回友好错误说明。

    使用场景:
        - 用户需要查找学术文献
        - 了解某个研究方向的最新进展
        - 查找特定论文的详细信息
    """
    if not query or not isinstance(query, str):
        return "错误：请输入有效的搜索关键词"

    max_results = max(1, min(max_results, 20))
    _arxiv_limiter.acquire()

    try:
        resp = _session.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=ARXIV_TIMEOUT,
        )
        resp.raise_for_status()

        # ── 解析 Atom XML ──
        import xml.etree.ElementTree as ET

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)

        if not entries:
            return f"未找到关于「{query}」的学术论文。"

        lines = [f"## 学术搜索：「{query}」"]
        for i, entry in enumerate(entries[:max_results], 1):
            # 标题
            title = (entry.findtext("atom:title", default="（无标题）", namespaces=ns) or "").strip()
            title = " ".join(title.split())

            # 作者
            authors = []
            for author_elem in entry.findall("atom:author", ns):
                name = (author_elem.findtext("atom:name", default="", namespaces=ns) or "").strip()
                if name:
                    authors.append(name)
            author_str = ", ".join(authors) if authors else "未知作者"
            if len(author_str) > 120:
                author_str = author_str[:117] + "..."

            # 摘要
            abstract = (entry.findtext("atom:summary", default="（无摘要）", namespaces=ns) or "").strip()
            abstract = " ".join(abstract.split())
            if len(abstract) > 400:
                abstract = abstract[:400] + "..."

            # arXiv ID
            arxiv_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else arxiv_url

            # 发布时间
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

            lines.append(f"\n[{i}] {title}")
            lines.append(f"    作者: {author_str}")
            lines.append(f"    发布时间: {published[:10] if published else 'N/A'}")
            lines.append(f"    arXiv ID: {arxiv_id}")
            if arxiv_url:
                lines.append(f"    链接: {arxiv_url}")
            if abstract:
                lines.append(f"    摘要: {abstract}")

        lines.append(f"\n共 {min(len(entries), max_results)} 篇论文（来源: arXiv）")
        return "\n".join(lines)

    except requests.Timeout:
        return f"学术搜索超时：请求「{query}」超过 {ARXIV_TIMEOUT} 秒未响应"
    except requests.ConnectionError:
        return "网络连接失败：无法访问 arXiv API，请检查网络连接"
    except requests.RequestException as e:
        return f"学术搜索请求异常：{_sanitize_error(str(e))}"
    except ET.ParseError:
        return "学术搜索出错：无法解析 arXiv 返回的 XML 数据"
    except Exception as e:
        _logger.exception("academic_search 未预期异常")
        return f"学术搜索出错：{_sanitize_error(str(e))}"


# ============================================================
# 独立测试入口
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  web_tools 独立测试")
    print("=" * 60)

    if len(sys.argv) > 1:
        # 命令行模式:
        #   python web_tools.py search   "关键词"
        #   python web_tools.py read     "https://..."
        #   python web_tools.py news     "关键词"
        #   python web_tools.py weather  "城市名"
        #   python web_tools.py academic "关键词"
        mode = sys.argv[1]
        arg = sys.argv[2] if len(sys.argv) > 2 else ""

        if mode == "search":
            print(web_search(arg))
        elif mode == "read":
            print(read_webpage(arg))
        elif mode == "news":
            print(news_search(arg))
        elif mode == "weather":
            print(weather_query(arg))
        elif mode == "academic":
            print(academic_search(arg))
        else:
            print("用法:")
            print("  python web_tools.py search   <关键词>")
            print("  python web_tools.py read     <URL>")
            print("  python web_tools.py news     <关键词>")
            print("  python web_tools.py weather  <城市名>")
            print("  python web_tools.py academic <关键词>")
    else:
        # 演示模式
        print("\n📝 搜索演示：")
        print(web_search("Python 编程语言", max_results=3))

        print("\n" + "=" * 60)
        print("\n📝 网页读取演示（需要网络）：")
        print(read_webpage("https://httpbin.org/get", max_chars=500))

        print("\n" + "=" * 60)
        print("\n📝 新闻搜索演示（需要 NEWS_API_KEY）：")
        print(news_search("人工智能", max_results=2))

        print("\n" + "=" * 60)
        print("\n📝 天气查询演示（需要 WEATHER_API_KEY）：")
        print(weather_query("北京"))

        print("\n" + "=" * 60)
        print("\n📝 学术搜索演示：")
        print(academic_search("large language model", max_results=2))
