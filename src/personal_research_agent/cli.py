from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from .config import DEFAULT_PROFILE, get_env
from .graph import approval_payload_to_text, build_graph
from .session_manager import (
    create_session,
    get_active_session_id,
    list_sessions,
    set_active_session,
)
from .storage import open_postgres_runtime


@contextmanager
def open_runtime(use_postgres: bool = True) -> Iterator[tuple[Any, Any, str]]:
    # 生产默认走 PostgresSaver/PostgresStore。
    # 本地调试时如果 Postgres、pgvector 或环境变量没准备好，就降级到内存模式，不阻塞 CLI 试跑。
    if use_postgres:
        try:
            with open_postgres_runtime() as (checkpointer, store):
                yield checkpointer, store, "postgres"
                return
        except Exception as exc:
            print(f"[warn] Postgres runtime unavailable, fallback to in-memory: {exc}", file=sys.stderr)

    yield InMemorySaver(), InMemoryStore(index=None), "memory"


def latest_ai_text(state: dict[str, Any]) -> str:
    # graph.invoke 返回的是完整 state；CLI 只把最后一条 AIMessage 展示给用户。
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            if message.content:
                return str(message.content)
            if message.tool_calls:
                return f"模型请求工具调用：{message.tool_calls}"
    return ""


def interrupt_payload(result: dict[str, Any]) -> Any | None:
    # LangGraph 遇到 interrupt 时，会把暂停信息放到 __interrupt__。
    # CLI 负责把它打印出来并收集用户审批结果。
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def print_sessions() -> None:
    sessions = list_sessions()
    if not sessions:
        print("暂无 research session。")
        return
    for item in sessions:
        print(f"- {item['session_id']} | {item.get('title') or '未命名'} | {item.get('research_goal') or '未设置目标'}")


def ensure_session(session_id: str | None, title: str = "", goal: str = "") -> str:
    # session_id 同时承担 CLI 任务目录 id 和 LangGraph thread_id。
    # 这样 research_session 文件、working-summary 和 checkpoint 能自然对齐。
    if session_id:
        set_active_session(session_id)
        return session_id
    active = get_active_session_id()
    if active:
        set_active_session(active)
        return active
    created = create_session(title=title or "research-session", research_goal=goal)
    set_active_session(created["session_id"])
    return str(created["session_id"])


def run_chat(args: argparse.Namespace) -> None:
    user_id = args.user_id or get_env("USER_ID", "default")
    profile_name = args.profile or DEFAULT_PROFILE
    session_id = ensure_session(args.session, title=args.title or "", goal=args.goal or "")

    with open_runtime(use_postgres=not args.memory_only) as (checkpointer, store, runtime_name):
        graph = build_graph(checkpointer=checkpointer, store=store, model_name=args.model)
        # configurable 是 LangGraph 推荐的运行时配置入口。
        # thread_id 决定 checkpoint 归属；user_id 决定长期记忆 namespace。
        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
                "profile_name": profile_name,
            }
        }

        print(f"【research-agent】runtime={runtime_name} session={session_id} profile={profile_name}")
        print("输入 /exit 退出，/sessions 查看会话，/session new <标题> 创建新会话，/thread <id> 切换 thread/session。")

        while True:
            try:
                user_input = input("\n请输入：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not user_input:
                continue
            if user_input == "/exit":
                return
            if user_input == "/sessions":
                print_sessions()
                continue
            if user_input.startswith("/session new"):
                title = user_input.removeprefix("/session new").strip() or "research-session"
                created = create_session(title=title, research_goal=args.goal or "")
                session_id = str(created["session_id"])
                set_active_session(session_id)
                config["configurable"]["thread_id"] = session_id
                print(f"已创建并切换 session：{session_id}")
                continue
            if user_input.startswith("/thread "):
                session_id = user_input.removeprefix("/thread ").strip()
                set_active_session(session_id)
                config["configurable"]["thread_id"] = session_id
                print(f"已切换 thread/session：{session_id}")
                continue

            state_input = {
                # 每轮只输入本次 HumanMessage；历史消息由 checkpointer 按 thread_id 自动恢复。
                "messages": [HumanMessage(content=user_input)],
                "user_id": user_id,
                "thread_id": session_id,
                "profile_name": profile_name,
                "tool_call_counts": {},
            }
            result = graph.invoke(state_input, config=config)
            while True:
                payload = interrupt_payload(result)
                if payload is None:
                    break
                interrupt_type = payload.get("type") if isinstance(payload, dict) else "interrupt"
                print(f"\n===== {interrupt_type or 'interrupt'} =====")
                print(approval_payload_to_text(payload))
                prompt = "审批结果 yes/no：" if interrupt_type == "tool_approval" else "请输入选择："
                approval = input(prompt).strip()
                # Command(resume=...) 会让图从 interrupt 暂停点继续执行。
                result = graph.invoke(Command(resume=approval), config=config)

            answer = latest_ai_text(result)
            if answer:
                print(answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent")
    parser.add_argument("--session", help="使用指定 research session/thread id。")
    parser.add_argument("--title", help="自动创建 session 时使用的标题。")
    parser.add_argument("--goal", help="自动创建 session 时使用的研究目标。")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="profile 名称。")
    parser.add_argument("--user-id", default=None, help="长期记忆 user_id。")
    parser.add_argument("--model", default="mimo", help="模型配置名：mimo/deepseek。")
    parser.add_argument("--memory-only", action="store_true", help="仅使用内存 checkpointer/store，便于本地调试。")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_chat(args)


if __name__ == "__main__":
    main()
