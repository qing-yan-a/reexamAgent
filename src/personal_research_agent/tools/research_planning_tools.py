from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .registry import register_tool
from .research_session_tools import (
    ALLOWED_QUERY_STATUS,
    ALLOWED_QUERY_TYPES,
    load_session,
    save_session,
    utc_now,
    validate_search_query_item,
)


QUERY_TYPE_ORDER = ["past_questions", "experience", "official_verification"]


def make_query_item(query_id: str, query: str, query_type: str, status: str = "pending", notes: str = "") -> dict[str, str]:
    item = {"query_id": query_id.strip(), "query": query.strip(), "query_type": query_type.strip(), "status": status.strip(), "notes": notes}
    validate_search_query_item(item)
    return item


def build_query_plan(school: str, major: str, year: str) -> list[dict[str, str]]:
    prefix = f"{school.strip()} {major.strip()}".strip()
    year_text = year.strip()
    dated_prefix = f"{prefix} {year_text}".strip() if year_text and year_text != "latest" else prefix
    return [
        make_query_item("past_questions_1", f"{dated_prefix} 复试 真题 回忆", "past_questions"),
        make_query_item("past_questions_2", f"{dated_prefix} 复试 机试题", "past_questions"),
        make_query_item("past_questions_3", f"{dated_prefix} 复试 面试题", "past_questions"),
        make_query_item("experience_1", f"{dated_prefix} 复试经验", "experience"),
        make_query_item("experience_2", f"{dated_prefix} 上岸经验 复试", "experience"),
        make_query_item("experience_3", f"{dated_prefix} 复试 流程", "experience"),
        make_query_item("official_verification_1", f"{dated_prefix} 复试方案", "official_verification"),
        make_query_item("official_verification_2", f"{school.strip()} 研究生院 招生简章 {major.strip()}", "official_verification"),
        make_query_item("official_verification_3", f"{dated_prefix} 专业目录", "official_verification"),
    ]


class PlanSearchQueriesInput(BaseModel):
    school: str = Field(description="目标学校。")
    major: str = Field(description="目标专业。")
    year: str = Field(default="latest", description="目标年份。")
    overwrite: bool = Field(default=False, description="是否覆盖已有 search_queries。")


@tool(args_schema=PlanSearchQueriesInput)
def plan_search_queries(school: str, major: str, year: str = "latest", overwrite: bool = False) -> dict[str, Any]:
    """为复试资料整理任务生成多轮搜索计划，并写入当前 research_session。"""
    if not school.strip() or not major.strip():
        raise ValueError("school 和 major 必须是非空字符串")
    session = load_session()
    if session.get("search_queries") and not overwrite:
        raise ValueError("当前 research_session 已存在 search_queries；如需重建，请明确传 overwrite=true")
    query_plan = build_query_plan(school, major, year or "latest")
    session["search_queries"] = query_plan
    session["updated_at"] = utc_now()
    save_session(session)
    return {"query_count": len(query_plan), "search_queries": query_plan, "message": "已生成多轮搜索计划。"}


def query_type_rank(query_type: str) -> int:
    try:
        return QUERY_TYPE_ORDER.index(query_type)
    except ValueError:
        return len(QUERY_TYPE_ORDER)


class GetPendingSearchQueriesInput(BaseModel):
    limit: int = Field(default=3, description="最多返回多少条 pending query。")
    query_type: str | None = Field(default=None, description="可选 query_type。")


@tool(args_schema=GetPendingSearchQueriesInput)
def get_pending_search_queries(limit: int = 3, query_type: str | None = None) -> dict[str, Any]:
    """读取当前 research_session 中尚未执行的 search_queries。"""
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit 必须是正整数")
    if query_type is not None and query_type not in ALLOWED_QUERY_TYPES:
        raise ValueError(f"不支持的 query_type: {query_type}")
    search_queries = load_session().get("search_queries", [])
    if not isinstance(search_queries, list):
        raise ValueError("当前 research_session.search_queries 结构无效")
    pending_queries = []
    for item in search_queries:
        validate_search_query_item(item)
        if item["status"] == "pending" and (query_type is None or item["query_type"] == query_type):
            pending_queries.append(item)
    pending_queries.sort(key=lambda item: (query_type_rank(item["query_type"]), item["query_id"]))
    return {"pending_count": len(pending_queries), "results": pending_queries[:limit]}


class UpdateSearchQueryStatusInput(BaseModel):
    query_id: str = Field(description="要更新的 query_id。")
    status: str = Field(description="新的状态：pending/done/failed。")
    notes: str = Field(default="", description="备注。")


@tool(args_schema=UpdateSearchQueryStatusInput)
def update_search_query_status(query_id: str, status: str, notes: str = "") -> dict[str, Any]:
    """更新当前 research_session 中某条 search query 的状态。"""
    if not query_id.strip():
        raise ValueError("query_id 必须是非空字符串")
    if status not in ALLOWED_QUERY_STATUS:
        raise ValueError(f"不支持的 status: {status}")
    session = load_session()
    search_queries = session.get("search_queries", [])
    if not isinstance(search_queries, list):
        raise ValueError("当前 research_session.search_queries 结构无效")
    updated = False
    for item in search_queries:
        validate_search_query_item(item)
        if item["query_id"] == query_id.strip():
            item["status"] = status
            item["notes"] = notes
            updated = True
            break
    if not updated:
        raise ValueError(f"未找到 query_id={query_id.strip()} 对应的 search query")
    session["search_queries"] = search_queries
    session["updated_at"] = utc_now()
    save_session(session)
    return {"query_id": query_id.strip(), "status": status, "notes": notes, "message": "search query 状态已更新。"}


register_tool(plan_search_queries, "low")
register_tool(get_pending_search_queries, "low")
register_tool(update_search_query_status, "low")
