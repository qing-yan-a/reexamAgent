from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .file_tools import WORKSPACE_ROOT
from .registry import register_tool


TVLY_EXE = Path.home() / ".local" / "bin" / "tvly.exe"
MAX_WEB_RESULTS = 10
MAX_SNIPPET_CHARS = 500
MAX_URLS_PER_CALL = 2
MAX_CONTENT_CHARS = 3000
COMMAND_TIMEOUT_SECONDS = 30


def get_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower()


def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_pdf_url(url: str) -> bool:
    lower_url = url.lower()
    return lower_url.endswith(".pdf") or ".pdf" in lower_url


class WebSearchInput(BaseModel):
    query: str = Field(description="简短搜索 query，不要传长篇任务说明。")
    max_results: int = Field(default=5, description="最多返回多少条结果，最大 10。")


@tool(args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """搜索公开网页资料，返回候选来源列表。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError("max_results 必须是正整数")
    max_results = min(max_results, MAX_WEB_RESULTS)
    if not TVLY_EXE.exists():
        raise FileNotFoundError(f"找不到 Tavily CLI：{TVLY_EXE}")

    command = [str(TVLY_EXE), "search", query.strip(), "--max-results", str(max_results), "--json"]
    result = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=COMMAND_TIMEOUT_SECONDS,
        shell=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"web_search 调用 Tavily CLI 失败：exit_code={result.returncode}, stderr={result.stderr[:500]}")

    data = json.loads(result.stdout)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    raw_results = data.get("results", [])
    return {
        "query": query.strip(),
        "result_count": len(raw_results),
        "results": [
            {
                "source_index": index,
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "snippet": str(item.get("content") or "")[:MAX_SNIPPET_CHARS],
                "source": get_domain(str(item.get("url") or "")),
                "score": item.get("score"),
                "retrieved_at": retrieved_at,
            }
            for index, item in enumerate(raw_results[:max_results])
        ],
    }


class ExtractSelectedSourcesInput(BaseModel):
    urls: list[str] = Field(description="用户已经确认要抽取的 URL 列表，单次最多 2 个。")
    allow_pdf: bool = Field(default=False, description="是否允许抽取 PDF。")


@tool(args_schema=ExtractSelectedSourcesInput)
def extract_selected_sources(urls: list[str], allow_pdf: bool = False) -> dict[str, Any]:
    """只对用户已经确认过的 URL 做网页正文抽取。"""
    if not isinstance(urls, list) or not urls:
        raise ValueError("urls 必须是非空列表")
    if len(urls) > MAX_URLS_PER_CALL:
        raise ValueError(f"单次最多只能抽取 {MAX_URLS_PER_CALL} 个 URL")

    normalized_urls = []
    for url in urls:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("urls 里的每一项都必须是非空字符串")
        clean_url = url.strip()
        if not is_http_url(clean_url):
            raise ValueError(f"只允许 http/https URL：{clean_url}")
        if is_pdf_url(clean_url) and not allow_pdf:
            raise ValueError(f"PDF 默认不抽取，需要用户明确允许 allow_pdf=true：{clean_url}")
        normalized_urls.append(clean_url)

    if not TVLY_EXE.exists():
        raise FileNotFoundError(f"找不到 Tavily CLI：{TVLY_EXE}")
    command = [str(TVLY_EXE), "extract", *normalized_urls, "--format", "markdown", "--json"]
    result = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=COMMAND_TIMEOUT_SECONDS,
        shell=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"extract_selected_sources 调用 Tavily CLI 失败：exit_code={result.returncode}, stderr={result.stderr[:500]}"
        )

    data = json.loads(result.stdout)
    extracted_at = datetime.now(timezone.utc).isoformat()
    raw_results = data.get("results", [])
    return {
        "url_count": len(normalized_urls),
        "extracted_count": len(raw_results),
        "results": [
            {
                "url": str(item.get("url") or ""),
                "title": str(item.get("title") or ""),
                "content_preview": str(item.get("raw_content") or "")[:MAX_CONTENT_CHARS],
                "content_chars": len(str(item.get("raw_content") or "")),
                "truncated": len(str(item.get("raw_content") or "")) > MAX_CONTENT_CHARS,
                "extracted_at": extracted_at,
            }
            for item in raw_results
        ],
        "failed_results": data.get("failed_results", []),
    }


class SourceReviewInput(BaseModel):
    research_goal: str = Field(description="用户本轮想研究的目标。")
    sources: list[dict[str, Any]] = Field(description="web_search 返回的候选来源列表。")


OFFICIAL_HINTS = {"edu.cn", "yz.chsi.com.cn", "gov.cn", "研究生院", "学院", "大学"}
COMMUNITY_HINTS = {"zhihu.com", "csdn.net", "bilibili.com", "tieba.baidu.com", "xiaohongshu.com"}
QUESTION_HINTS = {"真题", "回忆", "机试", "面试题", "题型", "专业课"}
EXPERIENCE_HINTS = {"复试经验", "上岸", "流程", "面试流程", "备考"}
OFFICIAL_TEXT_HINTS = {"复试方案", "招生简章", "专业目录", "调剂", "录取办法"}


def contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


@tool(args_schema=SourceReviewInput)
def source_review(research_goal: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """对 web_search 返回的复试资料候选来源做初筛。"""
    if not isinstance(research_goal, str) or not research_goal.strip():
        raise ValueError("research_goal 必须是非空字符串")
    if not isinstance(sources, list):
        raise ValueError("sources 必须是列表")

    reviews = []
    for index, item in enumerate(sources):
        title = str(item.get("title", ""))
        url = str(item.get("url", ""))
        snippet = str(item.get("snippet", ""))
        source = str(item.get("source", "")) or get_domain(url)
        text = f"{title}\n{snippet}\n{source}".lower()

        relevance_hits = []
        if contains_any(text, QUESTION_HINTS):
            relevance_hits.append("past_questions")
        if contains_any(text, EXPERIENCE_HINTS):
            relevance_hits.append("experience")
        if contains_any(text, OFFICIAL_TEXT_HINTS):
            relevance_hits.append("official_verification")

        relevance = "high" if len(relevance_hits) >= 2 else "medium" if relevance_hits else "low"
        credibility_hint = "high" if contains_any(source, OFFICIAL_HINTS) else "unknown" if contains_any(source, COMMUNITY_HINTS) else "medium"
        risk_flags = []
        if credibility_hint == "unknown":
            risk_flags.append("社区/经验内容，需要人工核验")
        if is_pdf_url(url):
            risk_flags.append("PDF 来源，抽取前需要用户确认")
        if not risk_flags:
            risk_flags.append("需要人工核验发布时间和原文语境")

        next_action = "discard" if relevance == "low" else "keep" if credibility_hint == "high" else "needs_user_check"
        reviews.append(
            {
                "source_index": item.get("source_index", index),
                "title": title,
                "url": url,
                "source": source,
                "relevance": relevance,
                "evidence_types": relevance_hits,
                "credibility_hint": credibility_hint,
                "risk_flags": risk_flags,
                "next_action": next_action,
                "review_note": "仅基于标题、URL、摘要和来源域名判断，尚未读取全文。",
            }
        )

    return {"research_goal": research_goal, "review_count": len(reviews), "reviews": reviews}


register_tool(web_search, "low")
register_tool(source_review, "low")
register_tool(extract_selected_sources, "low")
