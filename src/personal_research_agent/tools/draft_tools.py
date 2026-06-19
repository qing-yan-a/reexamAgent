from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personal_research_agent.session_manager import require_active_session_id, session_dir

from .registry import register_tool
from .research_session_tools import analyze_research_readiness, load_session


class DraftMarkdownInput(BaseModel):
    title: str = Field(description="草稿标题。")
    content: str = Field(description="模型基于已抽取资料生成的 Markdown 草稿。")
    filename: str = Field(default="draft.md", description="输出文件名，只允许 .md。")


@tool(args_schema=DraftMarkdownInput)
def draft_markdown(title: str, content: str, filename: str = "draft.md") -> dict[str, Any]:
    """在当前 research session 中保存复试资料 Markdown 草稿。"""
    if not title.strip():
        raise ValueError("title 必须是非空字符串")
    if not content.strip():
        raise ValueError("content 必须是非空字符串")
    if "/" in filename or "\\" in filename or not filename.endswith(".md"):
        raise ValueError("filename 只能是当前 session 目录下的 .md 文件名")

    session = load_session()
    readiness = analyze_research_readiness(session)
    if not readiness["draft_ready"]:
        raise ValueError(f"当前资料尚未达到草稿条件：{readiness['open_gaps']}")

    session_id = require_active_session_id()
    path = session_dir(session_id) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"# {title.strip()}\n\n"
        f"> 生成时间：{datetime.now().isoformat(timespec='seconds')}\n"
        f"> 资料缺口：{', '.join(readiness['open_gaps']) or '暂无'}\n"
        f"> 人工核验点：来源发布时间、学校官网政策、经验帖真实性。\n\n"
        f"{content.strip()}\n"
    )
    path.write_text(body, encoding="utf-8")
    return {
        "path": f"memory/sessions/{session_id}/{filename}",
        "chars": len(body),
        "open_gaps": readiness["open_gaps"],
        "message": "草稿已保存；发布或售卖前仍需人工核验来源。",
    }


register_tool(draft_markdown, "high")
