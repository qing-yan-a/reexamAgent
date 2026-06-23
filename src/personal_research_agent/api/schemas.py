from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """前端创建 session 时提交的表单数据。"""

    school: str = ""
    major: str = ""
    year: str = "latest"
    title: str = ""
    research_goal: str = ""
    session_id: str = ""


class ChatEvent(BaseModel):
    """WebSocket 消息的基础形状；主要用于记录前端可能发送的事件字段。"""

    type: Literal["user_message", "approval_response", "reexam_decision", "refresh_panel"]
    message: str = ""
    value: str = ""
    user_id: str = "default"
    profile_name: str = "research_to_product"


class DraftUpdate(BaseModel):
    """保存 Markdown 草稿时的请求体。"""

    content: str = Field(default="")


class SelectSourcesRequest(BaseModel):
    """前端勾选候选来源后提交的 source_key 列表。"""

    source_keys: list[str] = Field(default_factory=list)


class ApiResponse(BaseModel):
    """通用响应模型预留；当前接口多直接返回 dict。"""

    ok: bool = True
    data: Any = None
