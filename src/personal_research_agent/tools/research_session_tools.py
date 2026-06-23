from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personal_research_agent.session_manager import (
    default_research_session,
    read_research_session,
    require_active_session_id,
    utc_now,
    write_research_session,
)
from personal_research_agent.source_selection import select_sources_for_session

from .registry import register_tool


LIST_FIELDS = {
    "search_queries",
    "candidate_sources",
    "reviewed_sources",
    "selected_sources",
    "extracted_sources",
    "failed_sources",
    "open_gaps",
    "notes",
}
STRING_FIELDS = {"research_goal", "vertical", "school", "major", "year", "output_dir"}
BOOL_FIELDS = {"draft_ready"}
READINESS_MANAGED_FIELDS = {"draft_ready"}
ALLOWED_UPDATE_FIELDS = (LIST_FIELDS | STRING_FIELDS | BOOL_FIELDS) - READINESS_MANAGED_FIELDS
ALLOWED_QUERY_TYPES = {"past_questions", "experience", "official_verification"}
ALLOWED_QUERY_STATUS = {"pending", "done", "failed"}


def load_session() -> dict[str, Any]:
    return read_research_session(require_active_session_id())


def save_session(session: dict[str, Any]) -> None:
    write_research_session(require_active_session_id(), session)


def validate_search_query_item(item: dict[str, Any]) -> None:
    if not isinstance(item, dict):
        raise ValueError("search_queries 的每一项都必须是对象")
    if not str(item.get("query_id", "")).strip():
        raise ValueError("search_queries[].query_id 不能为空")
    if not str(item.get("query", "")).strip():
        raise ValueError("search_queries[].query 不能为空")
    if item.get("query_type") not in ALLOWED_QUERY_TYPES:
        raise ValueError(f"不支持的 query_type: {item.get('query_type')}")
    if item.get("status") not in ALLOWED_QUERY_STATUS:
        raise ValueError(f"不支持的 status: {item.get('status')}")
    if not isinstance(item.get("notes", ""), str):
        raise ValueError("search_queries[].notes 必须是字符串")


def validate_updates(updates: dict[str, Any]) -> None:
    if not isinstance(updates, dict) or not updates:
        raise ValueError("updates 必须是非空对象")
    readonly_fields = set(updates) & READINESS_MANAGED_FIELDS
    if readonly_fields:
        raise ValueError("draft_ready 只能由 evaluate_research_readiness 计算并写回")
    unknown_fields = set(updates) - ALLOWED_UPDATE_FIELDS
    if unknown_fields:
        raise ValueError(f"不允许更新未知字段：{sorted(unknown_fields)}")
    for field in STRING_FIELDS & updates.keys():
        if not isinstance(updates[field], str):
            raise ValueError(f"{field} 必须是字符串")
    for field in LIST_FIELDS & updates.keys():
        if not isinstance(updates[field], list):
            raise ValueError(f"{field} 必须是列表")
        if field == "search_queries":
            for item in updates[field]:
                validate_search_query_item(item)
    for field in BOOL_FIELDS & updates.keys():
        if not isinstance(updates[field], bool):
            raise ValueError(f"{field} 必须是布尔值")


NEGATIVE_EVIDENCE_KEYWORDS = {
    "不足",
    "稀缺",
    "未发现",
    "没发现",
    "没有",
    "无真题",
    "无法",
    "缺失",
    "需登录",
    "需要登录",
    "需手动",
    "丢弃",
    "discard",
    "failed",
    "error",
}


def split_evidence_fragments(text: str) -> list[str]:
    fragments = [text]
    for separator in ["。", "；", ";", "\n", "\r"]:
        next_fragments = []
        for fragment in fragments:
            next_fragments.extend(fragment.split(separator))
        fragments = next_fragments
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def add_positive_evidence_fragments(evidence_parts: list[str], value: Any) -> None:
    for fragment in split_evidence_fragments(str(value)):
        lowered = fragment.lower()
        if not any(keyword in lowered for keyword in NEGATIVE_EVIDENCE_KEYWORDS):
            evidence_parts.append(fragment)


def analyze_research_readiness(session: dict[str, Any]) -> dict[str, Any]:
    evidence_parts: list[str] = []
    for field in ["selected_sources", "extracted_sources", "reviewed_sources"]:
        for item in session.get(field, []) if isinstance(session.get(field, []), list) else []:
            if isinstance(item, dict):
                for value in item.values():
                    add_positive_evidence_fragments(evidence_parts, value)
            else:
                add_positive_evidence_fragments(evidence_parts, item)

    for item in session.get("search_queries", []) if isinstance(session.get("search_queries", []), list) else []:
        if isinstance(item, dict) and item.get("status") == "done":
            add_positive_evidence_fragments(evidence_parts, item.get("notes", ""))

    combined_text = "\n".join(evidence_parts).lower()
    has_question_clues = any(keyword in combined_text for keyword in ["真题", "回忆", "机试", "面试题", "专业课", "题型"])
    has_process_or_experience = any(keyword in combined_text for keyword in ["复试经验", "流程", "上岸", "面试流程"])
    has_official_verification = any(keyword in combined_text for keyword in ["复试方案", "招生简章", "专业目录", "研究生院", "学院官网"])

    open_gaps = []
    if not has_question_clues:
        open_gaps.append("历年复试真题或题型线索不足")
    if not has_process_or_experience:
        open_gaps.append("复试流程或经验线索不足")
    if not has_official_verification:
        open_gaps.append("官方政策或复试方案待核验")

    draft_ready = has_question_clues and has_process_or_experience
    note = (
        "已有真题/题型线索和流程经验线索，可以进入不完整草稿。"
        if draft_ready
        else "当前核心资料不足，建议优先补搜历年复试真题、机试题、面试题和复试经验。"
    )
    return {"open_gaps": open_gaps, "draft_ready": draft_ready, "readiness_note": note}


@tool
def get_research_session() -> dict[str, Any]:
    """读取当前复试资料 research_session 状态。"""
    session_id = require_active_session_id()
    return {"path": f"memory/sessions/{session_id}/research_session.json", "session": load_session()}


class CreateResearchSessionInput(BaseModel):
    research_goal: str = Field(description="本次研究任务目标。")
    school: str = Field(description="目标学校。")
    major: str = Field(description="目标专业或方向。")
    year: str = Field(default="latest", description="目标年份。")
    vertical: str = Field(default="postgraduate_reexam", description="垂直场景。")
    overwrite: bool = Field(default=False, description="是否覆盖已有 session。")


@tool(args_schema=CreateResearchSessionInput)
def create_research_session(
    research_goal: str,
    school: str,
    major: str,
    year: str = "latest",
    vertical: str = "postgraduate_reexam",
    overwrite: bool = False,
) -> dict[str, Any]:
    """创建或重置当前 active session 的 research_session。"""
    session_id = require_active_session_id()
    if not research_goal.strip() or not school.strip() or not major.strip():
        raise ValueError("research_goal、school、major 必须是非空字符串")
    existing = load_session()
    has_existing_goal = bool(str(existing.get("research_goal", "")).strip())
    same_identity = (
        str(existing.get("vertical", "")).strip() == (vertical.strip() or "postgraduate_reexam")
        and str(existing.get("school", "")).strip() == school.strip()
        and str(existing.get("major", "")).strip() == major.strip()
        and str(existing.get("year", "")).strip() == (year.strip() or "latest")
    )
    if has_existing_goal and not overwrite and not same_identity:
        raise ValueError("已存在 research_session；如需重置，请明确传 overwrite=true")

    session = existing if same_identity and not overwrite else default_research_session(session_id=session_id)
    now = utc_now()
    session.update(
        {
            "session_id": session_id,
            "research_goal": research_goal.strip(),
            "vertical": vertical.strip() or "postgraduate_reexam",
            "school": school.strip(),
            "major": major.strip(),
            "year": year.strip() if year.strip() else "latest",
            "updated_at": now,
        }
    )
    session.setdefault("created_at", now)
    save_session(session)
    return {"path": f"memory/sessions/{session_id}/research_session.json", "session": session}


class UpdateResearchSessionInput(BaseModel):
    updates: dict[str, Any] = Field(description="要更新的字段。列表字段会整体替换，应传合并后的完整列表。")


@tool(args_schema=UpdateResearchSessionInput)
def update_research_session(updates: dict[str, Any]) -> dict[str, Any]:
    """更新当前 research_session 的一个或多个字段。"""
    validate_updates(updates)
    session = load_session()
    session.update(updates)
    session["updated_at"] = utc_now()
    save_session(session)
    return {"path": f"memory/sessions/{require_active_session_id()}/research_session.json", "session": session}


class EvaluateResearchReadinessInput(BaseModel):
    persist: bool = Field(default=True, description="是否把评估结果写回当前 research_session。")


@tool(args_schema=EvaluateResearchReadinessInput)
def evaluate_research_readiness(persist: bool = True) -> dict[str, Any]:
    """评估当前复试资料 research_session 的资料缺口和草稿就绪状态。"""
    if not isinstance(persist, bool):
        raise ValueError("persist 必须是布尔值")
    session = load_session()
    result = analyze_research_readiness(session)
    if persist:
        session["open_gaps"] = result["open_gaps"]
        session["draft_ready"] = result["draft_ready"]
        session["updated_at"] = utc_now()
        save_session(session)
    return {
        "path": f"memory/sessions/{require_active_session_id()}/research_session.json",
        "persisted": persist,
        "open_gaps": result["open_gaps"],
        "draft_ready": result["draft_ready"],
        "readiness_note": result["readiness_note"],
        "session": session,
    }


class SelectSourcesInput(BaseModel):
    source_indexes: list[int] = Field(default_factory=list, description="要保留的候选来源 source_index。")
    urls: list[str] = Field(default_factory=list, description="要保留的候选来源 URL。")
    source_keys: list[str] = Field(default_factory=list, description="前端或系统生成的稳定 source_key。")


@tool(args_schema=SelectSourcesInput)
def select_sources(
    source_indexes: list[int] | None = None,
    urls: list[str] | None = None,
    source_keys: list[str] | None = None,
) -> dict[str, Any]:
    """把用户确认保留的候选来源写入当前 research_session.selected_sources。"""
    if not (source_indexes or urls or source_keys):
        raise ValueError("source_indexes、urls、source_keys 至少传一个")
    return select_sources_for_session(
        require_active_session_id(),
        source_indexes=source_indexes or [],
        urls=urls or [],
        source_keys=source_keys or [],
        selection_method="agent_tool",
    )


register_tool(get_research_session, "low")
register_tool(create_research_session, "low")
register_tool(update_research_session, "low")
register_tool(evaluate_research_readiness, "low")
register_tool(select_sources, "low")
