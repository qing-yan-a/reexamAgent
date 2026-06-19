import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ACTIVE_SESSION_FILE, ARCHIVED_SESSIONS_DIR, SESSIONS_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def research_session_path(session_id: str) -> Path:
    return session_dir(session_id) / "research_session.json"


def working_memory_path(session_id: str) -> Path:
    return session_dir(session_id) / "working-memory.md"


def working_summary_path(session_id: str) -> Path:
    return session_dir(session_id) / "working-summary.md"


def ensure_session_storage() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVED_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def default_research_session(
    session_id: str = "",
    title: str = "",
    research_goal: str = "",
    vertical: str = "postgraduate_reexam",
    school: str = "",
    major: str = "",
    year: str = "",
    status: str = "active",
) -> dict[str, Any]:
    now = utc_now()
    return {
        "session_id": session_id,
        "title": title,
        "status": status,
        "research_goal": research_goal,
        "vertical": vertical,
        "school": school,
        "major": major,
        "year": year,
        "search_queries": [],
        "candidate_sources": [],
        "reviewed_sources": [],
        "selected_sources": [],
        "extracted_sources": [],
        "failed_sources": [],
        "open_gaps": [],
        "draft_ready": False,
        "notes": [],
        "created_at": now,
        "updated_at": now,
    }


def load_json_file(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_active_session(session_id: str) -> None:
    save_json_file(ACTIVE_SESSION_FILE, {"session_id": session_id, "selected_at": utc_now()})


def get_active_session_id() -> str | None:
    if not ACTIVE_SESSION_FILE.exists():
        return None
    value = str(load_json_file(ACTIVE_SESSION_FILE).get("session_id", "")).strip()
    return value or None


def require_active_session_id() -> str:
    session_id = get_active_session_id()
    if not session_id:
        raise RuntimeError("当前没有活动 research session。请先选择或新建会话。")
    return session_id


def session_title(school: str, major: str, year: str) -> str:
    parts = [part.strip() for part in [school, major, year] if part and part.strip()]
    return "-".join(parts) if parts else "未命名研究会话"


def session_goal(school: str, major: str, year: str) -> str:
    if school and major and year:
        return f"整理{school}{major}{year}研究生复试资料"
    if school and major:
        return f"整理{school}{major}研究生复试资料"
    return "整理研究资料"


def generate_session_id() -> str:
    return f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def create_session(
    school: str = "",
    major: str = "",
    year: str = "",
    research_goal: str = "",
    vertical: str = "postgraduate_reexam",
    title: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    ensure_session_storage()
    final_session_id = session_id or generate_session_id()
    final_title = title or session_title(school, major, year)
    final_goal = research_goal.strip() or session_goal(school.strip(), major.strip(), year.strip())
    directory = session_dir(final_session_id)
    directory.mkdir(parents=True, exist_ok=False)

    data = default_research_session(
        session_id=final_session_id,
        title=final_title,
        research_goal=final_goal,
        vertical=vertical,
        school=school.strip(),
        major=major.strip(),
        year=year.strip(),
    )
    write_research_session(final_session_id, data)
    working_memory_path(final_session_id).write_text("", encoding="utf-8")
    working_summary_path(final_session_id).write_text("", encoding="utf-8")
    return data


def list_sessions() -> list[dict[str, Any]]:
    ensure_session_storage()
    sessions = []
    for directory in sorted(SESSIONS_DIR.iterdir()):
        if directory.is_dir():
            data = read_research_session(directory.name)
            data.setdefault("session_id", directory.name)
            data.setdefault("title", directory.name)
            sessions.append(data)
    sessions.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return sessions


def read_research_session(session_id: str) -> dict[str, Any]:
    path = research_session_path(session_id)
    return load_json_file(path, default_research_session(session_id=session_id))


def write_research_session(session_id: str, data: dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    save_json_file(research_session_path(session_id), data)


def archive_session(session_id: str) -> Path:
    source = session_dir(session_id)
    if not source.exists():
        raise FileNotFoundError(f"会话不存在：{session_id}")
    ensure_session_storage()
    target = ARCHIVED_SESSIONS_DIR / session_id
    if target.exists():
        target = ARCHIVED_SESSIONS_DIR / f"{session_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.move(str(source), str(target))
    return target


def get_active_research_session_path() -> Path:
    return research_session_path(require_active_session_id())


def get_active_working_memory_path() -> Path:
    return working_memory_path(require_active_session_id())


def get_active_working_summary_path() -> Path:
    return working_summary_path(require_active_session_id())
