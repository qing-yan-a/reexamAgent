from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personal_research_agent.config import PROJECT_ROOT
from personal_research_agent.session_manager import read_research_session
from personal_research_agent.source_selection import source_key

UNASSIGNED_OUTPUT_DIR = "__unassigned__"


def _as_list(value: Any) -> list[Any]:
    """把 JSON 字段安全转成 list；字段损坏或为空时返回空列表。"""
    return value if isinstance(value, list) else []


def _draft_dir(session: dict[str, Any]) -> Path | None:
    """解析 session 的草稿目录，只允许指向 test/ 下的安全路径。"""
    output_dir = str(session.get("output_dir") or "").strip()
    if not output_dir:
        return None
    directory = (PROJECT_ROOT / output_dir).resolve()
    test_root = (PROJECT_ROOT / "test").resolve()
    if not directory.is_relative_to(test_root):
        return None
    return directory


def _folder_mtime(path: Path) -> str:
    """读取资料文件夹的修改时间，用于工作区树排序。"""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _normalize_output_dir(value: Any) -> str:
    """把 session.output_dir 规范成项目内 test/ 相对路径。"""
    raw = str(value or "").strip()
    if not raw or raw == "未归档":
        return UNASSIGNED_OUTPUT_DIR

    project_root = PROJECT_ROOT.resolve()
    test_root = (PROJECT_ROOT / "test").resolve()
    try:
        path = Path(raw)
        if path.is_absolute():
            resolved = path.resolve()
            if resolved.is_relative_to(project_root):
                raw = resolved.relative_to(project_root).as_posix()
    except OSError:
        return UNASSIGNED_OUTPUT_DIR

    raw = raw.replace("\\", "/").strip("/")
    if not raw:
        return UNASSIGNED_OUTPUT_DIR
    if raw.startswith("test/"):
        return raw

    candidate = (test_root / raw).resolve()
    if candidate.is_relative_to(test_root):
        return candidate.relative_to(project_root).as_posix()
    return UNASSIGNED_OUTPUT_DIR


def _folder_name(output_dir: str) -> str:
    """把 output_dir 转成前端显示用的文件夹名称。"""
    if output_dir == UNASSIGNED_OUTPUT_DIR:
        return "未归档 Session"
    return Path(output_dir).name or output_dir


def _session_summary(session: dict[str, Any]) -> dict[str, Any]:
    """把 session 压缩成工作区树下每个子会话需要的字段。"""
    output_dir = _normalize_output_dir(session.get("output_dir"))
    return {
        "session_id": session.get("session_id", ""),
        "title": session.get("title") or session.get("research_goal") or session.get("session_id", ""),
        "status": session.get("status", "active"),
        "research_goal": session.get("research_goal", ""),
        "school": session.get("school", ""),
        "major": session.get("major", ""),
        "year": session.get("year", ""),
        "output_dir": "" if output_dir == UNASSIGNED_OUTPUT_DIR else output_dir,
        "updated_at": session.get("updated_at", ""),
    }


def _empty_folder(output_dir: str, exists: bool = False, updated_at: str = "") -> dict[str, Any]:
    """创建一个工作区树文件夹节点，后续再把 sessions 挂进去。"""
    return {
        "output_dir": output_dir,
        "name": _folder_name(output_dir),
        "session_count": 0,
        "updated_at": updated_at,
        "school": "",
        "major": "",
        "exists": exists,
        "sessions": [],
    }


def _is_workspace_folder(path: Path) -> bool:
    """过滤掉隐藏目录和 __pycache__，只保留用户资料文件夹。"""
    return path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"


def list_draft_files(session: dict[str, Any]) -> list[dict[str, Any]]:
    """列出当前 session 输出目录下的 Markdown 草稿文件。"""
    directory = _draft_dir(session)
    if directory is None or not directory.exists():
        return []
    files = []
    for path in sorted(directory.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True):
        files.append(
            {
                "filename": path.name,
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size": path.stat().st_size,
                "updated_at": path.stat().st_mtime,
            }
        )
    return files


def build_research_panel(session_id: str) -> dict[str, Any]:
    """读取 research_session.json，并整理成右侧研究面板的数据结构。"""
    session = read_research_session(session_id)
    search_queries = [item for item in _as_list(session.get("search_queries")) if isinstance(item, dict)]
    reviewed_sources = [item for item in _as_list(session.get("reviewed_sources")) if isinstance(item, dict)]
    candidate_sources = [item for item in _as_list(session.get("candidate_sources")) if isinstance(item, dict)]
    selected_sources = [item for item in _as_list(session.get("selected_sources")) if isinstance(item, dict)]
    extracted_sources = [item for item in _as_list(session.get("extracted_sources")) if isinstance(item, dict)]
    draft_files = list_draft_files(session)
    latest_query = search_queries[-1] if search_queries else None

    reviewed_by_key = {
        (item.get("query_id"), item.get("source_index"), item.get("url")): item
        for item in reviewed_sources
    }
    merged_sources = []
    for source in candidate_sources[-50:]:
        # 候选来源来自搜索结果，reviewed_sources 来自初筛；这里合并成前端一行来源卡片。
        key = (source.get("query_id"), source.get("source_index"), source.get("url"))
        reviewed = reviewed_by_key.get(key, {})
        merged_sources.append(
            {
                "source_key": source_key(source),
                "source_index": source.get("source_index"),
                "title": reviewed.get("title") or source.get("title", ""),
                "url": reviewed.get("url") or source.get("url", ""),
                "source": reviewed.get("source") or source.get("source", ""),
                "relevance": reviewed.get("relevance", "unknown"),
                "credibility_hint": reviewed.get("credibility_hint", "unknown"),
                "risk_flags": reviewed.get("risk_flags", []),
                "next_action": reviewed.get("next_action", "needs_user_check"),
                "query_type": source.get("query_type", ""),
            }
        )

    return {
        "session_id": session_id,
        "current_task": {
            "title": session.get("title") or session.get("research_goal") or session_id,
            "research_goal": session.get("research_goal", ""),
            "school": session.get("school", ""),
            "major": session.get("major", ""),
            "year": session.get("year", ""),
            "status": session.get("status", "active"),
        },
        "open_gaps": _as_list(session.get("open_gaps")),
        "search_progress": {
            "total_queries": len(search_queries),
            "done_queries": sum(1 for item in search_queries if item.get("status") == "done"),
            "failed_queries": sum(1 for item in search_queries if item.get("status") == "failed"),
            "latest_query": latest_query,
        },
        "search_history": search_queries[-20:],
        "candidate_sources": merged_sources,
        "selected_sources": selected_sources,
        "extracted_sources": extracted_sources,
        "draft_status": {
            "draft_ready": bool(session.get("draft_ready")),
            "draft_files": draft_files,
            "latest_draft": draft_files[0] if draft_files else None,
        },
        "output_dir": session.get("output_dir", ""),
        "updated_at": session.get("updated_at", ""),
    }


def folder_summary_from_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建左侧工作区树：先扫描 test/ 目录，再把 sessions 挂到对应 output_dir。"""
    folders: dict[str, dict[str, Any]] = {}
    test_root = PROJECT_ROOT / "test"
    if test_root.exists():
        # 真实资料文件夹即使没有 session，也应该显示在工作区树里。
        for directory in sorted((item for item in test_root.iterdir() if _is_workspace_folder(item)), key=lambda item: item.name.casefold()):
            output_dir = directory.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            folders[output_dir] = _empty_folder(output_dir, exists=True, updated_at=_folder_mtime(directory))

    for session in sessions:
        # 没有 output_dir 的旧 session 会进入“未归档 Session”分组，避免刷新后丢失入口。
        output_dir = _normalize_output_dir(session.get("output_dir"))
        folder = folders.setdefault(output_dir, _empty_folder(output_dir))
        folder["sessions"].append(_session_summary(session))
        folder["session_count"] = len(folder["sessions"])
        folder["updated_at"] = max(str(folder.get("updated_at") or ""), str(session.get("updated_at") or ""))
        if not folder.get("school") and session.get("school"):
            folder["school"] = session.get("school", "")
        if not folder.get("major") and session.get("major"):
            folder["major"] = session.get("major", "")

    for folder in folders.values():
        # 每个文件夹内部按 session 更新时间倒序排列。
        folder["sessions"].sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        folder["session_count"] = len(folder["sessions"])

    return sorted(folders.values(), key=lambda item: str(item.get("updated_at", "")), reverse=True)
