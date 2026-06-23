from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from personal_research_agent.api.panel import build_research_panel, folder_summary_from_sessions, list_draft_files
from personal_research_agent.api.schemas import CreateSessionRequest, DraftUpdate, SelectSourcesRequest
from personal_research_agent.cli import interrupt_payload, latest_ai_text, open_runtime
from personal_research_agent.config import DEFAULT_PROFILE, PROJECT_ROOT, get_env
from personal_research_agent.graph import build_graph
from personal_research_agent.research_outputs import research_output_dir, to_project_relative
from personal_research_agent.session_log import append_session_log
from personal_research_agent.session_manager import (
    create_session,
    ensure_session_storage,
    list_sessions,
    read_research_session,
    research_session_path,
    set_active_session,
    session_dir,
    write_research_session,
)
from personal_research_agent.source_selection import select_sources_for_session
from personal_research_agent.storage import delete_thread_checkpoints


STATIC_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="reexamAgent Web", version="0.1.0")
# 把 web/ 目录挂成静态资源目录，浏览器才能访问 /static/app.js 和 /static/styles.css。
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    """返回单页前端入口 index.html。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    """健康检查接口，用于确认后端服务和项目根目录。"""
    return {"ok": True, "service": "reexamAgent", "project_root": str(PROJECT_ROOT)}


@app.get("/sessions")
def api_sessions() -> dict[str, Any]:
    """列出所有本地 research session，供左侧 session/工作区树使用。"""
    sessions = list_sessions()
    return {"sessions": [session_summary(item) for item in sessions]}


@app.post("/sessions")
def api_create_session(payload: CreateSessionRequest) -> dict[str, Any]:
    """创建一个新的 research session，并在有学校和专业时绑定 test/ 下的输出目录。"""
    try:
        session = create_session(
            school=payload.school,
            major=payload.major,
            year=payload.year,
            title=payload.title,
            research_goal=payload.research_goal,
            session_id=payload.session_id,
        )
        if payload.school and payload.major:
            output_dir = research_output_dir(payload.school, payload.major, create=True)
            session["output_dir"] = to_project_relative(output_dir)
            write_research_session(str(session["session_id"]), session)
        set_active_session(str(session["session_id"]))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"session": session_summary(session), "panel": build_research_panel(str(session["session_id"]))}


@app.get("/sessions/{session_id}")
def api_get_session(session_id: str) -> dict[str, Any]:
    """读取单个 session 的基础摘要。"""
    ensure_session_exists(session_id)
    return {"session": session_summary(read_research_session(session_id))}


@app.delete("/sessions/{session_id}")
def api_delete_session(session_id: str) -> dict[str, Any]:
    """删除 session 目录和同名 checkpoint；不会删除 test/ 下已经保存的资料和草稿。"""
    directory = session_dir(session_id)
    if not directory.exists():
        raise HTTPException(status_code=404, detail="session 不存在")
    try:
        delete_thread_checkpoints(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"checkpoint 清理失败：{exc}") from exc
    shutil.rmtree(directory)
    return {"ok": True, "deleted": session_id, "checkpoint_deleted": True}


@app.get("/folders")
def api_folders() -> dict[str, Any]:
    """返回左侧工作区树：test/ 资料文件夹 + 归属到文件夹下的 sessions。"""
    return {"folders": folder_summary_from_sessions(list_sessions())}


@app.get("/sessions/{session_id}/research-panel")
def api_research_panel(session_id: str) -> dict[str, Any]:
    """返回右侧研究面板需要展示的结构化状态。"""
    ensure_session_exists(session_id)
    return {"panel": build_research_panel(session_id)}

#payload不是普通字符串列表，而是一个 Pydantic对象
# 可以理解成：payload = SelectSourcesRequest(  source_keys=["source_001", "source_002"])
@app.post("/sessions/{session_id}/selected-sources")
def api_select_sources(session_id: str, payload: SelectSourcesRequest) -> dict[str, Any]:
    """保存用户在前端勾选保留的候选来源。"""
    ensure_session_exists(session_id)
    result = select_sources_for_session(session_id, source_keys=payload.source_keys, selection_method="web_checkbox")
    return {"selected_count": result["selected_count"], "missing": result["missing"], "panel": build_research_panel(session_id)}


@app.get("/sessions/{session_id}/drafts")
def api_drafts(session_id: str) -> dict[str, Any]:
    """列出当前 session 输出目录下的 Markdown 草稿。"""
    ensure_session_exists(session_id)
    return {"drafts": list_draft_files(read_research_session(session_id))}


@app.get("/sessions/{session_id}/drafts/{filename}")
def api_get_draft(session_id: str, filename: str) -> dict[str, Any]:
    """读取一个 Markdown 草稿内容。"""
    path = resolve_draft_path(session_id, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="草稿不存在")
    return {"filename": filename, "path": path.relative_to(PROJECT_ROOT).as_posix(), "content": path.read_text(encoding="utf-8")}


@app.put("/sessions/{session_id}/drafts/{filename}")
def api_put_draft(session_id: str, filename: str, payload: DraftUpdate) -> dict[str, Any]:
    """保存一个 Markdown 草稿；路径会被限制在当前 session 的 output_dir 内。"""
    path = resolve_draft_path(session_id, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.content, encoding="utf-8")
    return {"ok": True, "path": path.relative_to(PROJECT_ROOT).as_posix(), "chars": len(payload.content)}


@app.websocket("/ws/sessions/{session_id}")
async def websocket_session(websocket: WebSocket, session_id: str) -> None:
    """聊天 WebSocket：连接指定 session/thread，并用 LangGraph stream 驱动前端流式输出。"""
    await websocket.accept()
    ensure_session_storage()
    set_active_session(session_id)
    user_id = get_env("USER_ID", "default")
    profile_name = DEFAULT_PROFILE
    config = {"configurable": {"thread_id": session_id, "user_id": user_id, "profile_name": profile_name}}
    append_session_log(
        "websocket_connected",
        {"thread_id": session_id, "user_id": user_id, "profile_name": profile_name},
    )

    try:
        with open_runtime(use_postgres=True) as (checkpointer, store, runtime_name):
            # Web 入口和 CLI 入口共用同一张图；差别只是这里用 stream 消费结果。
            graph = build_graph(checkpointer=checkpointer, store=store)
            await websocket.send_json({"type": "connected", "session_id": session_id, "runtime": runtime_name})
            # 刷新页面后，前端内存会清空，所以连接时先从 checkpoint 回放历史消息。
            await send_history(websocket, graph, config)
            await send_panel(websocket, session_id)
            while True:
                raw_event = await websocket.receive_text()
                try:
                    event = json.loads(raw_event)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "WebSocket 消息必须是 JSON"})
                    continue

                event_type = str(event.get("type") or "")
                user_id = str(event.get("user_id") or user_id)
                profile_name = str(event.get("profile_name") or profile_name)
                config["configurable"].update({"user_id": user_id, "profile_name": profile_name})

                if event_type == "refresh_panel":
                    # 前端手动刷新右侧研究面板，不触发模型调用。
                    await send_panel(websocket, session_id)
                    continue
                if event_type == "user_message":
                    message = str(event.get("message") or "").strip()
                    if not message:
                        await websocket.send_json({"type": "error", "message": "message 不能为空"})
                        continue
                    created_at = utc_now_iso()
                    append_session_log(
                        "user_message",
                        {
                            "thread_id": session_id,
                            "user_id": user_id,
                            "profile_name": profile_name,
                            "content": message,
                            "entrypoint": "web",
                        },
                    )
                    await websocket.send_json({"type": "message_start", "role": "user", "message_id": f"user_{uuid.uuid4().hex}", "created_at": created_at})
                    await websocket.send_json({"type": "message_delta", "delta": message})
                    state_input = {
                        # 每轮只提交本次 HumanMessage；历史由 checkpointer 根据 thread_id 自动恢复。
                        "messages": [HumanMessage(content=message, additional_kwargs={"created_at": created_at})],
                        "user_id": user_id,
                        "thread_id": session_id,
                        "profile_name": profile_name,
                        "tool_call_counts": {},
                    }
                    await stream_graph_result(websocket, session_id, graph, state_input, config)
                    continue
                if event_type in {"approval_response", "reexam_decision"}:
                    value = str(event.get("value") or "").strip()
                    append_session_log(
                        "interrupt_resume",
                        {
                            "thread_id": session_id,
                            "user_id": user_id,
                            "profile_name": profile_name,
                            "event_type": event_type,
                            "value": value,
                            "entrypoint": "web",
                        },
                    )
                    # interrupt 暂停后，Command(resume=...) 会把用户决策送回暂停点继续执行。
                    await stream_graph_result(websocket, session_id, graph, Command(resume=value), config)
                    continue
                await websocket.send_json({"type": "error", "message": f"未知事件类型：{event_type}"})
    except WebSocketDisconnect:
        append_session_log("websocket_disconnected", {"thread_id": session_id, "user_id": user_id, "profile_name": profile_name})
        return
    except Exception as exc:
        append_session_log(
            "error",
            {"thread_id": session_id, "user_id": user_id, "profile_name": profile_name, "error": str(exc), "entrypoint": "websocket"},
        )
        await websocket.send_json({"type": "error", "message": str(exc)})


def session_summary(session: dict[str, Any]) -> dict[str, Any]:
    """把完整 research_session.json 压成前端列表需要的轻量摘要。"""
    return {
        "session_id": session.get("session_id", ""),
        "title": session.get("title") or session.get("research_goal") or session.get("session_id", ""),
        "status": session.get("status", "active"),
        "research_goal": session.get("research_goal", ""),
        "school": session.get("school", ""),
        "major": session.get("major", ""),
        "year": session.get("year", ""),
        "output_dir": session.get("output_dir", ""),
        "updated_at": session.get("updated_at", ""),
    }


def ensure_session_exists(session_id: str) -> None:
    """统一检查 session 是否存在，避免每个接口重复写 404 判断。"""
    if not research_session_path(session_id).exists():
        raise HTTPException(status_code=404, detail="session 不存在")


def resolve_draft_path(session_id: str, filename: str) -> Path:
    """解析草稿文件路径，并确保它不能逃出当前 session 的 output_dir。"""
    ensure_session_exists(session_id)
    if "/" in filename or "\\" in filename or not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="filename 只能是 output_dir 下的 .md 文件名")
    session = read_research_session(session_id)
    output_dir = str(session.get("output_dir") or "").strip()
    if not output_dir:
        raise HTTPException(status_code=400, detail="当前 session 没有 output_dir")
    directory = (PROJECT_ROOT / output_dir).resolve()
    test_root = (PROJECT_ROOT / "test").resolve()
    if not directory.is_relative_to(test_root):
        raise HTTPException(status_code=400, detail="output_dir 必须位于 test/ 目录下")
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory):
        raise HTTPException(status_code=400, detail="草稿路径越界")
    return path


async def send_history(websocket: WebSocket, graph: Any, config: dict[str, Any]) -> None:
    """从 LangGraph checkpoint 读取历史消息，并发送给前端恢复聊天窗口。"""
    try:
        state = graph.get_state(config)
        values = getattr(state, "values", {}) or {}
        messages = values.get("messages", [])
        await websocket.send_json({"type": "history_reset", "messages": serialize_chat_messages(messages)})
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"历史消息读取失败：{exc}"})


def serialize_chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """把 LangChain 消息列表转换成前端能直接渲染的消息数组。"""
    serialized = []
    for index, message in enumerate(messages):
        item = serialize_chat_message(message, index)
        if item is not None:
            serialized.append(item)
    return serialized


def serialize_chat_message(message: Any, index: int) -> dict[str, Any] | None:
    """序列化单条消息；只保留 Human/AI/Tool 三类用户可理解的消息。"""
    if isinstance(message, HumanMessage):
        role = "user"
        content = message_text(message)
    elif isinstance(message, AIMessage):
        role = "assistant"
        content = message_text(message)
        if not content and message.tool_calls:
            calls = [f"{call.get('name')}({json.dumps(call.get('args', {}), ensure_ascii=False)})" for call in message.tool_calls]
            content = "模型请求工具调用：\n" + "\n".join(calls)
    elif isinstance(message, ToolMessage):
        role = "tool"
        tool_name = getattr(message, "name", "") or "tool"
        content = f"{tool_name}\n{message_text(message)}"
    else:
        return None

    if not content:
        return None
    return {
        "id": getattr(message, "id", None) or f"history_{index}",
        "role": role,
        "content": content,
        "created_at": message_created_at(message),
    }


async def send_graph_result(websocket: WebSocket, session_id: str, result: dict[str, Any]) -> None:
    """旧的一次性 invoke 结果发送函数，保留给兼容路径和调试使用。"""
    payload = interrupt_payload(result)
    if payload is not None:
        await send_interrupt(websocket, payload)
        await send_panel(websocket, session_id)
        return
    text = latest_ai_text(result)
    if text:
        message_id = f"msg_{uuid.uuid4().hex}"
        await websocket.send_json({"type": "message_start", "role": "assistant", "message_id": message_id, "created_at": utc_now_iso()})
        await websocket.send_json({"type": "message_delta", "message_id": message_id, "delta": text})
        await websocket.send_json({"type": "message_end", "message_id": message_id})
    await send_panel(websocket, session_id)


async def stream_graph_result(websocket: WebSocket, session_id: str, graph: Any, graph_input: Any, config: dict[str, Any]) -> None:
    """消费 graph.stream 事件，并转换成前端 WebSocket 事件。"""
    message_id = ""
    streamed_text = False
    fallback_text = ""

    def ensure_message_id() -> str:
        nonlocal message_id
        if not message_id:
            message_id = f"msg_{uuid.uuid4().hex}"
        return message_id

    async def send_delta(delta: str) -> None:
        """确保先发 message_start，再连续发送 message_delta。"""
        nonlocal streamed_text
        if not delta:
            return
        current_message_id = ensure_message_id()
        if not streamed_text:
            await websocket.send_json({"type": "message_start", "role": "assistant", "message_id": current_message_id, "created_at": utc_now_iso()})
            streamed_text = True
        await websocket.send_json({"type": "message_delta", "message_id": current_message_id, "delta": delta})

    async def send_static_chat_message(item: dict[str, Any]) -> None:
        """发送非 token 流消息，比如工具调用摘要和 ToolMessage 结果。"""
        text = str(item.get("content") or "")
        if not text:
            return
        current_message_id = str(item.get("id") or f"msg_{uuid.uuid4().hex}")
        await websocket.send_json(
            {
                "type": "message_start",
                "role": item.get("role", "assistant"),
                "message_id": current_message_id,
                "created_at": item.get("created_at") or utc_now_iso(),
            }
        )
        await websocket.send_json({"type": "message_delta", "message_id": current_message_id, "delta": text})
        await websocket.send_json({"type": "message_end", "message_id": current_message_id})

    for event in graph.stream(graph_input, config=config, stream_mode=["messages", "updates"]):
        # messages: LLM token 流；updates: 节点完成后的状态更新和 interrupt。
        if not (isinstance(event, tuple) and len(event) == 2):
            continue
        mode, data = event
        if mode == "messages":
            message, metadata = data
            # 只展示 agent 节点的 token，过滤摘要/长期记忆判断等内部 LLM 输出。
            if not is_user_visible_stream(metadata):
                continue
            if isinstance(message, AIMessage):
                await send_delta(message_text(message))
            continue
        if mode != "updates" or not isinstance(data, dict):
            continue

        if data.get("__interrupt__"):
            # 图暂停时把 interrupt 转成前端可交互事件，比如审批按钮或复试搜索决策。
            if streamed_text:
                await websocket.send_json({"type": "message_end", "message_id": message_id})
            payload = interrupt_payload({"__interrupt__": data["__interrupt__"]})
            append_session_log(
                "interrupt",
                {
                    "thread_id": str(config.get("configurable", {}).get("thread_id", "")),
                    "user_id": str(config.get("configurable", {}).get("user_id", "")),
                    "profile_name": str(config.get("configurable", {}).get("profile_name", "")),
                    "payload": payload,
                    "entrypoint": "web",
                },
            )
            await send_interrupt(websocket, payload)
            await send_panel(websocket, session_id)
            return

        if not streamed_text:
            # 有些节点直接返回完整 AIMessage，不走 token 流；这里做兜底展示。
            for message in iter_ai_messages_from_update(data):
                text = message_text(message)
                if text:
                    fallback_text = text
        # 工具调用和工具结果不属于自然语言 token 流，需要从 updates 里单独补发给前端。
        for item in iter_tool_display_messages_from_update(data):
            await send_static_chat_message(item)

    if fallback_text and not streamed_text:
        await send_delta(fallback_text)
    if streamed_text:
        await websocket.send_json({"type": "message_end", "message_id": message_id})
    await send_panel(websocket, session_id)


def message_text(message: BaseMessage) -> str:
    """兼容字符串内容和多块内容，把 LangChain message content 统一转成文本。"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def message_created_at(message: BaseMessage) -> str:
    """Return a persisted message timestamp when LangChain metadata carries one."""
    for source in (getattr(message, "additional_kwargs", None), getattr(message, "response_metadata", None)):
        if isinstance(source, dict):
            value = source.get("created_at")
            if value:
                return str(value)
    return ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_user_visible_stream(metadata: Any) -> bool:
    """判断当前 token 是否来自用户可见的 agent 节点。"""
    return isinstance(metadata, dict) and metadata.get("langgraph_node") == "agent"


def iter_ai_messages_from_update(update: dict[str, Any]):
    """从 updates 事件中提取节点直接返回的 AIMessage。"""
    for value in update.values():
        if not isinstance(value, dict):
            continue
        messages = value.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if isinstance(message, AIMessage):
                yield message


def iter_tool_display_messages_from_update(update: dict[str, Any]):
    """从 updates 事件中提取需要在聊天区展示的工具调用和工具结果。"""
    for value in update.values():
        if not isinstance(value, dict):
            continue
        messages = value.get("messages")
        if not isinstance(messages, list):
            continue
        for index, message in enumerate(messages):
            item = serialize_chat_message(message, index)
            if item is None:
                continue
            content = str(item.get("content") or "")
            if item.get("role") == "tool" or content.startswith("模型请求工具调用："):
                yield item


async def send_interrupt(websocket: WebSocket, payload: Any) -> None:
    """把 LangGraph interrupt payload 转成前端认识的 WebSocket 事件。"""
    if not isinstance(payload, dict):
        await websocket.send_json({"type": "interrupt_required", "interrupt_id": f"interrupt_{uuid.uuid4().hex}", "payload": payload})
        return
    if payload.get("type") == "tool_approval":
        await websocket.send_json(
            {
                "type": "approval_required",
                "approval_id": f"approval_{uuid.uuid4().hex}",
                "message": payload.get("message", ""),
                "tool_calls": payload.get("tool_calls", []),
            }
        )
        return
    if payload.get("type") == "reexam_search_decision":
        await websocket.send_json(
            {
                "type": "reexam_decision_required",
                "interrupt_id": f"interrupt_{uuid.uuid4().hex}",
                "message": payload.get("message", ""),
                "current_gaps": payload.get("current_gaps", []),
                "recommended_query": payload.get("recommended_query", {}),
                "candidate_count": payload.get("candidate_count", 0),
                "reviewed_count": payload.get("reviewed_count", 0),
                "options": [item.get("value") for item in payload.get("options", []) if isinstance(item, dict)],
            }
        )
        return
    if payload.get("type") == "reexam_route_confirmation":
        goal = payload.get("goal") or {}
        await websocket.send_json(
            {
                "type": "reexam_route_confirmation_required",
                "interrupt_id": f"interrupt_{uuid.uuid4().hex}",
                "message": payload.get("message", ""),
                "goal": goal,
                "reason": payload.get("reason", ""),
                "options": [item.get("value") for item in payload.get("options", []) if isinstance(item, dict)],
            }
        )
        return
    await websocket.send_json({"type": "interrupt_required", "interrupt_id": f"interrupt_{uuid.uuid4().hex}", "payload": payload})


async def send_panel(websocket: WebSocket, session_id: str) -> None:
    """读取并发送右侧研究面板；失败时只提示错误，不断开 WebSocket。"""
    try:
        panel = build_research_panel(session_id)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"研究面板读取失败：{exc}"})
        return
    await websocket.send_json({"type": "research_panel_update", "panel": panel})
