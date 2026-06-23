import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ACTIVE_SESSION_FILE, ARCHIVED_SESSIONS_DIR, SESSIONS_DIR


def utc_now() -> str:
    """返回 UTC ISO 时间字符串，用作 session 的 created_at/updated_at。"""
    return datetime.now(timezone.utc).isoformat()


def session_dir(session_id: str) -> Path:
    """返回某个 session 的本地状态目录。"""
    return SESSIONS_DIR / session_id


def research_session_path(session_id: str) -> Path:
    """返回某个 session 的核心业务状态文件路径。"""
    return session_dir(session_id) / "research_session.json"


def working_memory_path(session_id: str) -> Path:
    """返回某个 session 的短期工作记忆文件路径。"""
    return session_dir(session_id) / "working-memory.md"


def working_summary_path(session_id: str) -> Path:
    """返回某个 session 的 checkpoint 压缩摘要文件路径。"""
    return session_dir(session_id) / "working-summary.md"


def ensure_session_storage() -> None:
    """确保 session 根目录和归档目录存在。"""
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
    """构造 research_session.json 的默认结构。"""
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
        "output_dir": "",
        "notes": [],
        "created_at": now,
        "updated_at": now,
    }


def load_json_file(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取 JSON 文件；文件不存在时返回 default 或空 dict。"""
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_file(path: Path, data: dict[str, Any]) -> None:
    """以 UTF-8 和漂亮缩进保存 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_active_session(session_id: str) -> None:
    """记录当前活动 session，供 CLI 和文件态 prompt 加载使用。"""
    save_json_file(ACTIVE_SESSION_FILE, {"session_id": session_id, "selected_at": utc_now()})


def get_active_session_id() -> str | None:
    """读取当前活动 session_id；没有设置时返回 None。"""
    if not ACTIVE_SESSION_FILE.exists():
        return None
    value = str(load_json_file(ACTIVE_SESSION_FILE).get("session_id", "")).strip()
    return value or None


def require_active_session_id() -> str:
    """读取活动 session_id；没有活动 session 时抛出可读错误。"""
    session_id = get_active_session_id()
    if not session_id:
        raise RuntimeError("当前没有活动 research session。请先选择或新建会话。")
    return session_id


def session_title(school: str, major: str, year: str) -> str:
    """根据学校、专业、年份生成默认 session 标题。"""
    parts = [part.strip() for part in [school, major, year] if part and part.strip()]
    return "-".join(parts) if parts else "未命名研究会话"


def session_goal(school: str, major: str, year: str) -> str:
    """根据学校、专业、年份生成默认研究目标。"""
    if school and major and year:
        return f"整理{school}{major}{year}研究生复试资料"
    if school and major:
        return f"整理{school}{major}研究生复试资料"
    return "整理研究资料"


def generate_session_id() -> str:
    """生成基于本地时间的 session id。"""
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
    """创建一个新的 session 目录和三份基础文件。"""
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
    """列出所有未归档 session，并按 updated_at 倒序排列。"""
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
    """读取 research_session.json；缺失时返回默认结构，便于容错显示。"""
    path = research_session_path(session_id)
    return load_json_file(path, default_research_session(session_id=session_id))


def write_research_session(session_id: str, data: dict[str, Any]) -> None:
    """写入 research_session.json，并自动刷新 updated_at。"""
    data["updated_at"] = utc_now()
    save_json_file(research_session_path(session_id), data)


def archive_session(session_id: str) -> Path:
    """把 session 从活动目录移动到归档目录。"""
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
    """返回当前活动 session 的 research_session.json 路径。"""
    return research_session_path(require_active_session_id())


def get_active_working_memory_path() -> Path:
    """返回当前活动 session 的 working-memory.md 路径。"""
    return working_memory_path(require_active_session_id())


def get_active_working_summary_path() -> Path:
    """返回当前活动 session 的 working-summary.md 路径。"""
    return working_summary_path(require_active_session_id())
