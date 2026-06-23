from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .research_outputs import research_output_dir, to_project_relative
from .session_manager import read_research_session, require_active_session_id, utc_now, write_research_session
from .tools.research_session_tools import analyze_research_readiness
from .tools.web_tools import source_review, web_search


REEXAM_INTENT_KEYWORDS = {"复试", "考研复试", "研究生复试", "复试资料", "复试真题", "复试经验", "复试方案"}
SEARCH_ACTION_KEYWORDS = {"搜索", "搜", "找", "查", "整理", "收集", "补搜", "继续"}
COMPUTER_MAJOR_KEYWORDS = {"计算机", "软件", "电子信息", "人工智能", "网络空间安全", "网安", "大数据"}
QUERY_TYPE_ORDER = ["past_questions", "experience", "official_verification"]
POST_SEARCH_ACTION_KEYWORDS = {
    "抽取",
    "提取",
    "读取正文",
    "正文",
    "生成草稿",
    "写草稿",
    "保存草稿",
    "生成文档",
    "整理成文档",
}
POST_SEARCH_TARGET_KEYWORDS = {"已选", "选中", "保留", "来源", "selected_sources", "候选来源", "草稿"}


def last_human_text(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


def is_reexam_search_intent(text: str) -> bool:
    if not text.strip():
        return False
    has_reexam = any(keyword in text for keyword in REEXAM_INTENT_KEYWORDS)
    has_action = any(keyword in text for keyword in SEARCH_ACTION_KEYWORDS)
    has_school = bool(re.search(r"[\u4e00-\u9fa5]{2,20}(大学|学院|研究院)", text))
    return has_reexam and (has_action or has_school)


def is_reexam_post_search_action(text: str) -> bool:
    """临时规避：抽取正文/生成草稿先交给普通 agent 工具路径，不进入搜索子流程。"""
    if not text.strip():
        return False
    has_post_action = any(keyword in text for keyword in POST_SEARCH_ACTION_KEYWORDS)
    has_post_target = any(keyword in text for keyword in POST_SEARCH_TARGET_KEYWORDS)
    return has_post_action and has_post_target


def parse_reexam_goal_text(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^(帮我|请|麻烦|我想|想)?(搜索|搜|找|查|整理|收集)?", "", text.strip())
    school_match = re.search(r"([\u4e00-\u9fa5]{2,20}(?:大学|学院|研究院))", cleaned)
    year_match = re.search(r"(20\d{2})", text)
    major = ""
    for keyword in sorted(COMPUTER_MAJOR_KEYWORDS, key=len, reverse=True):
        if keyword in text:
            major = keyword
            break
    if not major:
        major = "计算机"
    school = school_match.group(1) if school_match else ""
    year = year_match.group(1) if year_match else "latest"
    research_goal = f"整理{school}{major}{year}研究生复试资料" if school else text.strip()
    return {
        "is_reexam_search": is_reexam_search_intent(text),
        "school": school,
        "major": major,
        "year": year,
        "research_goal": research_goal,
    }


def active_reexam_session_snapshot() -> dict[str, Any]:
    """读取当前活动 session 的轻量信息，供 LLM Router 判断“继续搜索吧”这类省略表达。"""
    try:
        session_id = require_active_session_id()
        session = read_research_session(session_id)
    except Exception:
        return {}
    return {
        "session_id": session.get("session_id") or session_id,
        "research_goal": session.get("research_goal", ""),
        "vertical": session.get("vertical", ""),
        "school": session.get("school", ""),
        "major": session.get("major", ""),
        "year": session.get("year", ""),
        "search_query_count": len(session.get("search_queries") or []),
        "candidate_source_count": len(session.get("candidate_sources") or []),
        "open_gaps": session.get("open_gaps") or [],
    }


def build_reexam_router_messages(user_text: str, active_session: dict[str, Any]) -> list[BaseMessage]:
    """构造专门给 Router 模型看的短提示词，只让它输出 JSON 决策。"""
    system = """
你是 reexamAgent 的问题路由器，只负责判断用户这句话是否要进入“研究生复试资料搜索专用流程”。

你必须只输出一个 JSON 对象，不要输出 Markdown，不要解释。
JSON 字段如下：
{
  "intent": "reexam_search" | "normal_chat",
  "action": "start_search" | "continue_search" | "ask_clarification" | "normal",
  "confidence": 0.0,
  "school": "",
  "major": "",
  "year": "latest",
  "research_goal": "",
  "missing_fields": [],
  "reason": ""
}

判断规则：
- 用户明确要求搜索、整理、收集、继续查找某学校某专业的研究生复试资料、复试真题、复试经验、复试方案时，intent=reexam_search。
- 用户说“搜索资料吧”“继续补搜”“开始吧”这类省略表达时，如果当前活动 session 已经是 postgraduate_reexam 任务，可以视为 continue_search，并沿用 session 的 school/major/year。
- 如果是普通聊天、解释概念、代码问题、FastAPI/Docker/数据库问题，intent=normal_chat。
- 如果用户要求抽取已选来源正文、生成草稿、保存草稿，这是搜索后的资料处理，不要进入复试搜索流程，intent=normal_chat。
- 如果用户明显想搜复试资料但缺少学校，action=ask_clarification，并在 missing_fields 写入缺失字段。
- year 不明确时填 "latest"；major 不明确但当前 session 有 major 时沿用；否则可以填 "计算机" 作为默认专业。
""".strip()
    payload = {
        "user_text": user_text,
        "active_session": active_session,
    }
    return [
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]


def parse_reexam_router_output(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON；模型偶尔带多余文本时做一次宽松兜底。"""
    raw = str(text or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return {
                "intent": "normal_chat",
                "action": "normal",
                "confidence": 0.0,
                "reason": "router_output_not_json",
            }
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "intent": "normal_chat",
                "action": "normal",
                "confidence": 0.0,
                "reason": "router_json_parse_failed",
            }
    if not isinstance(data, dict):
        return {"intent": "normal_chat", "action": "normal", "confidence": 0.0}
    return data


def reexam_goal_from_router(user_text: str, router: dict[str, Any], active_session: dict[str, Any]) -> dict[str, Any]:
    """把 Router 的结构化判断转成搜索流程需要的 reexam_goal。"""
    is_reexam = str(router.get("intent") or "").strip() == "reexam_search"
    session_is_reexam = str(active_session.get("vertical") or "").strip() in {"", "postgraduate_reexam"}
    school = str(router.get("school") or "").strip()
    major = str(router.get("major") or "").strip()
    year = str(router.get("year") or "").strip()

    # 用户说“继续搜索吧”时通常不会重复学校专业，这里从当前 session 补齐上下文。
    if is_reexam and session_is_reexam:
        school = school or str(active_session.get("school") or "").strip()
        major = major or str(active_session.get("major") or "").strip()
        if not year or year == "latest":
            year = str(active_session.get("year") or year or "latest").strip()

    major = major or "计算机"
    year = year or "latest"
    research_goal = str(router.get("research_goal") or "").strip()
    if is_reexam and not research_goal:
        research_goal = f"整理{school}{major}{year}研究生复试资料" if school else user_text.strip()
    return {
        "is_reexam_search": is_reexam,
        "school": school,
        "major": major,
        "year": year,
        "research_goal": research_goal,
    }


def route_confirmation_payload(goal: dict[str, Any], router: dict[str, Any]) -> dict[str, Any]:
    """生成进入专用流程前给用户确认的 interrupt payload。"""
    school = str(goal.get("school") or "").strip() or "未识别"
    major = str(goal.get("major") or "").strip() or "未识别"
    year = str(goal.get("year") or "").strip() or "latest"
    return {
        "type": "reexam_route_confirmation",
        "message": "我判断你想进入复试资料搜索专用流程。是否同意进入？",
        "goal": {
            "school": school,
            "major": major,
            "year": year,
            "research_goal": goal.get("research_goal", ""),
        },
        "reason": router.get("reason", ""),
        "options": [
            {"value": "yes", "label": "同意进入"},
            {"value": "no", "label": "取消"},
        ],
    }


def session_identity_matches(session: dict[str, Any], goal: dict[str, Any]) -> bool:
    return (
        str(session.get("vertical", "")).strip() in {"", "postgraduate_reexam"}
        and str(session.get("school", "")).strip() in {"", str(goal.get("school", "")).strip()}
        and str(session.get("major", "")).strip() in {"", str(goal.get("major", "")).strip()}
        and str(session.get("year", "")).strip() in {"", str(goal.get("year", "")).strip()}
    )


def ensure_reexam_session(goal: dict[str, Any]) -> dict[str, Any]:
    session_id = require_active_session_id()
    session = read_research_session(session_id)
    if session.get("research_goal") and not session_identity_matches(session, goal):
        return {
            "ok": False,
            "path": f"memory/sessions/{session_id}/research_session.json",
            "message": (
                "当前 session 已有不同的复试研究任务。请使用 /session new <标题> 新建会话，"
                "或手动重置 research_session 后再搜索。"
            ),
            "session": session,
        }

    now = utc_now()
    output_dir = research_output_dir(str(goal["school"]), str(goal["major"]), create=True)
    output_dir_relative = to_project_relative(output_dir)
    session.update(
        {
            "session_id": session_id,
            "research_goal": goal["research_goal"],
            "vertical": "postgraduate_reexam",
            "school": goal["school"],
            "major": goal["major"],
            "year": goal["year"],
            "output_dir": output_dir_relative,
            "updated_at": now,
        }
    )
    session.setdefault("created_at", now)
    for field in [
        "search_queries",
        "candidate_sources",
        "reviewed_sources",
        "selected_sources",
        "extracted_sources",
        "failed_sources",
        "open_gaps",
        "notes",
    ]:
        session.setdefault(field, [])
    if output_dir_relative not in session["notes"]:
        session["notes"].append(f"资料输出目录：{output_dir_relative}")
    session.setdefault("draft_ready", False)
    write_research_session(session_id, session)
    return {"ok": True, "path": f"memory/sessions/{session_id}/research_session.json", "session": session}


def evaluate_reexam_gaps(persist: bool = True) -> dict[str, Any]:
    session_id = require_active_session_id()
    session = read_research_session(session_id)
    result = analyze_research_readiness(session)
    if persist:
        session["open_gaps"] = result["open_gaps"]
        session["draft_ready"] = result["draft_ready"]
        write_research_session(session_id, session)
    result["session"] = session
    result["path"] = f"memory/sessions/{session_id}/research_session.json"
    return result


def gap_to_query_type(open_gaps: list[str]) -> str:
    joined = "\n".join(open_gaps)
    if "真题" in joined or "题型" in joined:
        return "past_questions"
    if "流程" in joined or "经验" in joined:
        return "experience"
    if "官方" in joined or "方案" in joined:
        return "official_verification"
    return "official_verification"


def build_gap_query(session: dict[str, Any], query_type: str) -> str:
    school = str(session.get("school") or "").strip()
    major = str(session.get("major") or "计算机").strip()
    year = str(session.get("year") or "latest").strip()
    prefix = f"{school} {major}".strip()
    dated_prefix = f"{prefix} {year}".strip() if year and year != "latest" else prefix
    if query_type == "past_questions":
        return f"{dated_prefix} 研究生复试 真题 回忆 机试 面试题"
    if query_type == "experience":
        return f"{dated_prefix} 研究生复试经验 复试流程 上岸经验"
    return f"{dated_prefix} 研究生复试方案 招生简章 专业目录"


def next_gap_query(session: dict[str, Any], open_gaps: list[str]) -> dict[str, str]:
    query_type = gap_to_query_type(open_gaps)
    query = build_gap_query(session, query_type)
    existing = [item for item in session.get("search_queries", []) if isinstance(item, dict)]
    same_type_count = sum(1 for item in existing if item.get("query_type") == query_type)
    query_id = f"{query_type}_{same_type_count + 1}"
    while any(item.get("query_id") == query_id for item in existing):
        same_type_count += 1
        query_id = f"{query_type}_{same_type_count + 1}"
    return {"query_id": query_id, "query": query, "query_type": query_type, "status": "pending", "notes": ""}


def append_pending_query(query_item: dict[str, str]) -> None:
    session_id = require_active_session_id()
    session = read_research_session(session_id)
    search_queries = session.get("search_queries", [])
    if not isinstance(search_queries, list):
        search_queries = []
    if not any(isinstance(item, dict) and item.get("query_id") == query_item["query_id"] for item in search_queries):
        search_queries.append(query_item)
    session["search_queries"] = search_queries
    write_research_session(session_id, session)


def run_web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    return web_search.invoke({"query": query, "max_results": max_results})


def review_web_sources(research_goal: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    if not sources:
        return {"research_goal": research_goal, "review_count": 0, "reviews": []}
    return source_review.invoke({"research_goal": research_goal, "sources": sources})


def record_search_iteration(
    query_item: dict[str, str],
    search_result: dict[str, Any] | None,
    review_result: dict[str, Any] | None,
    error: str = "",
) -> dict[str, Any]:
    session_id = require_active_session_id()
    session = read_research_session(session_id)
    now = utc_now()
    status = "failed" if error else "done"
    notes = error or f"找到 {len((search_result or {}).get('results', []))} 条候选来源，初筛 {len((review_result or {}).get('reviews', []))} 条。"

    search_queries = session.get("search_queries", [])
    if not isinstance(search_queries, list):
        search_queries = []
    found = False
    for item in search_queries:
        if isinstance(item, dict) and item.get("query_id") == query_item["query_id"]:
            item["status"] = status
            item["notes"] = notes
            found = True
            break
    if not found:
        saved_query = dict(query_item)
        saved_query["status"] = status
        saved_query["notes"] = notes
        search_queries.append(saved_query)

    candidate_sources = session.get("candidate_sources", [])
    if not isinstance(candidate_sources, list):
        candidate_sources = []
    for source in (search_result or {}).get("results", []):
        item = dict(source)
        item["query_id"] = query_item["query_id"]
        item["query_type"] = query_item["query_type"]
        item["recorded_at"] = now
        candidate_sources.append(item)

    reviewed_sources = session.get("reviewed_sources", [])
    if not isinstance(reviewed_sources, list):
        reviewed_sources = []
    for review in (review_result or {}).get("reviews", []):
        item = dict(review)
        item["query_id"] = query_item["query_id"]
        item["query_type"] = query_item["query_type"]
        item["recorded_at"] = now
        reviewed_sources.append(item)

    notes_list = session.get("notes", [])
    if not isinstance(notes_list, list):
        notes_list = []
    notes_list.append(f"{now} | {query_item['query_id']} | {notes}")

    session["search_queries"] = search_queries
    session["candidate_sources"] = candidate_sources
    session["reviewed_sources"] = reviewed_sources
    session["notes"] = notes_list
    write_research_session(session_id, session)
    return {
        "path": f"memory/sessions/{session_id}/research_session.json",
        "status": status,
        "notes": notes,
        "candidate_count": len(candidate_sources),
        "reviewed_count": len(reviewed_sources),
        "session": session,
    }


def source_confirmation_message(session: dict[str, Any], path: str) -> str:
    reviewed = [item for item in session.get("reviewed_sources", []) if isinstance(item, dict)]
    keepable = [item for item in reviewed if item.get("next_action") in {"keep", "needs_user_check"}]
    latest = keepable[-8:] if keepable else reviewed[-8:]
    lines = [
        "已跳出复试资料搜索循环，进入来源确认阶段。",
        f"session: {path}",
        f"资料输出目录: {session.get('output_dir') or 'test/<学校><专业>'}",
        "",
        "请从下面候选来源里选择要保留并抽取正文的 source_index 或 URL：",
    ]
    if not latest:
        lines.append("- 当前还没有可确认的候选来源。")
    for item in latest:
        lines.append(
            f"- [{item.get('source_index')}] {item.get('title')} | {item.get('source')} | "
            f"相关性={item.get('relevance')} | 建议={item.get('next_action')}\n  {item.get('url')}"
        )
    lines.append("")
    lines.append("下一步你可以回复：保留 source_index=...，或继续补搜某个缺口。")
    return "\n".join(lines)


def stop_message(path: str) -> str:
    return f"已停止复试资料搜索循环，当前进度已保留在 {path}。"


def format_search_iteration_summary(state: dict[str, Any]) -> dict[str, Any]:
    gaps = state.get("reexam_open_gaps") or []
    query = state.get("reexam_query") or {}
    record = state.get("reexam_record") or {}
    return {
        "type": "reexam_search_decision",
        "message": "本轮复试资料搜索已完成，请判断下一步。",
        "current_gaps": gaps,
        "recommended_query": query,
        "candidate_count": record.get("candidate_count", 0),
        "reviewed_count": record.get("reviewed_count", 0),
        "session_path": record.get("path", ""),
        "options": [
            {"value": "continue", "label": "继续补搜"},
            {"value": "next", "label": "可以下一步：来源确认"},
            {"value": "stop", "label": "停止"},
        ],
    }


def normalize_search_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"next", "下一步", "可以下一步", "来源确认", "确认", "ok", "可以"}:
        return "next"
    if text in {"stop", "停止", "退出", "结束", "no", "否"}:
        return "stop"
    return "continue"


def error_message(text: str) -> AIMessage:
    return AIMessage(content=text)
