from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from .memory import (
    build_summary_updates,
    compact_long_term_memory,
    long_term_memory_payload,
    retrieve_long_term_memories,
    should_save_long_term_memory,
)
from .models import build_model
from .prompts import CORE_PROMPT, load_profile_prompt, load_working_memory, load_working_summary
from .rag_db import retrieve_hybrid_rag
from .reexam_search_flow import (
    append_pending_query,
    ensure_reexam_session,
    error_message,
    evaluate_reexam_gaps,
    format_search_iteration_summary,
    last_human_text,
    next_gap_query,
    normalize_search_decision,
    parse_reexam_goal_text,
    record_search_iteration,
    review_web_sources,
    run_web_search,
    source_confirmation_message,
    stop_message,
)
from .state import AgentState
from .tools import get_registered_tools, get_tool_risk, requires_approval


# 下面几个 helper 只负责从 LangGraph 的消息列表里取“最近一次关键消息”。
# 节点代码尽量不直接写下标逻辑，后面改状态结构时更稳。
def _last_message(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> BaseMessage | None:
    return messages[-1] if messages else None


def _last_ai_message(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _last_human_message(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> HumanMessage | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def _tool_calls(ai_message: AIMessage | None) -> list[dict[str, Any]]:
    if not ai_message:
        return []
    return list(ai_message.tool_calls or [])


def _is_yes(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"y", "yes", "是", "同意", "批准", "approve", "approved"}


def load_context_node(store: Any | None = None):
    def load_context(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        # user_id/thread_id/profile_name 既可以从 state 进来，也可以从 config.configurable 进来。
        # CLI 每轮都会带 config；checkpoint 恢复时 LangGraph 也依赖 thread_id。
        user_id = state.get("user_id") or (config or {}).get("configurable", {}).get("user_id", "default")
        thread_id = state.get("thread_id") or (config or {}).get("configurable", {}).get("thread_id", "default")
        profile_name = state.get("profile_name") or (config or {}).get("configurable", {}).get("profile_name", "research_to_product")
        last_human = _last_human_message(list(state.get("messages", [])))
        retrieved_memories: list[dict[str, Any]] = []
        retrieved_rag: list[dict[str, Any]] = []

        # 长期记忆只检索，不写入 messages。
        # 真正注入模型的是 agent_node 里临时拼出来的 SystemMessage，避免污染 checkpoint。
        if store is not None and last_human is not None:
            try:
                retrieved_memories = retrieve_long_term_memories(store, str(user_id), str(last_human.content), limit=3)
            except Exception:
                retrieved_memories = []
        if last_human is not None:
            try:
                retrieved_rag = retrieve_hybrid_rag(str(last_human.content), top_k=5)
            except Exception:
                retrieved_rag = []

        return {
            "user_id": str(user_id),
            "thread_id": str(thread_id),
            "profile_name": str(profile_name),
            "retrieved_memories": retrieved_memories,
            "retrieved_rag": retrieved_rag,
            "working_summary": load_working_summary(),
            "pending_approval": [],
            "tool_approved": False,
        }

    return load_context


def agent_node(model_name: str = "mimo"):
    tools = get_registered_tools()
    model = build_model(model_name).bind_tools(tools)

    def agent(state: AgentState) -> dict[str, Any]:
        # 每次模型调用都重新组装系统上下文：核心规则 + profile + 文件态 working memory/summary
        # + 本轮按语义检索到的长期记忆。只有模型响应会写入 checkpoint。
        profile_name = state.get("profile_name", "research_to_product")
        system_parts = [
            CORE_PROMPT,
            load_profile_prompt(profile_name),
            load_working_memory(),
            state.get("working_summary") or load_working_summary(),
        ]
        retrieved_memories = state.get("retrieved_memories") or []
        if retrieved_memories:
            memory_lines = []
            for item in retrieved_memories:
                if isinstance(item, dict):
                    source = item.get("source_thread") or item.get("type") or "memory"
                    memory_lines.append(f"- [{source}] {item.get('text', item)}")
                else:
                    memory_lines.append(f"- {item}")
            system_parts.append("以下是跨线程长期记忆检索结果，只作为参考，不要当作已核验事实：\n" + "\n".join(memory_lines))

        retrieved_rag = state.get("retrieved_rag") or []
        if retrieved_rag:
            rag_lines = []
            for index, item in enumerate(retrieved_rag, start=1):
                heading = f" / {item.get('heading')}" if item.get("heading") else ""
                score = item.get("final_score", "")
                rag_lines.append(
                    f"[{index}] {item.get('path')}{heading} score={score}\n"
                    f"{str(item.get('content', '')).strip()[:1200]}"
                )
            system_parts.append(
                "以下是本地 RAG 资料检索结果，优先用于回答与资料、学校、日期、项目记录有关的问题；"
                "引用时说明来自哪个 path：\n" + "\n\n".join(rag_lines)
            )

        runtime_messages = [SystemMessage(content="\n\n".join(part for part in system_parts if part.strip()))]
        runtime_messages.extend(list(state.get("messages", [])))
        response = model.invoke(runtime_messages)
        # tool_approved 每次模型输出后重置，防止上一轮审批状态误放行下一轮工具调用。
        return {"messages": [response], "tool_approved": False}

    return agent


def parse_reexam_goal_node(state: AgentState) -> dict[str, Any]:
    text = last_human_text(list(state.get("messages", [])))
    goal = parse_reexam_goal_text(text)
    return {
        "is_reexam_search": bool(goal.get("is_reexam_search")),
        "reexam_goal": goal,
        "reexam_error": "" if goal.get("school") or not goal.get("is_reexam_search") else "没有解析到目标学校。",
        "reexam_iteration_complete": False,
        "reexam_next_action": "",
    }


def ensure_reexam_session_node(state: AgentState) -> dict[str, Any]:
    goal = dict(state.get("reexam_goal") or {})
    if not goal.get("school"):
        return {"reexam_error": "没有解析到目标学校，请用“某某大学 + 专业 + 复试资料”描述搜索目标。"}
    result = ensure_reexam_session(goal)
    if not result.get("ok"):
        return {
            "messages": [error_message(str(result.get("message", "无法初始化 research_session。")))],
            "reexam_error": str(result.get("message", "")),
            "reexam_session": result.get("session", {}),
            "reexam_session_path": str(result.get("path", "")),
        }
    return {
        "reexam_session": result.get("session", {}),
        "reexam_session_path": str(result.get("path", "")),
        "reexam_error": "",
    }


def evaluate_reexam_gaps_node(state: AgentState) -> dict[str, Any]:
    result = evaluate_reexam_gaps(persist=True)
    return {
        "reexam_readiness": result,
        "reexam_open_gaps": list(result.get("open_gaps") or []),
        "reexam_session": result.get("session", {}),
        "reexam_session_path": str(result.get("path", "")),
    }


def generate_gap_queries_node(state: AgentState) -> dict[str, Any]:
    session = dict(state.get("reexam_session") or {})
    gaps = list(state.get("reexam_open_gaps") or [])
    query_item = next_gap_query(session, gaps)
    append_pending_query(query_item)
    return {"reexam_query": query_item, "reexam_iteration_complete": False, "reexam_next_action": ""}


def run_one_web_search_node(state: AgentState) -> dict[str, Any]:
    query_item = dict(state.get("reexam_query") or {})
    query = str(query_item.get("query", "")).strip()
    if not query:
        return {"reexam_search_result": {}, "reexam_error": "没有可执行的搜索 query。"}
    try:
        result = run_web_search(query, max_results=5)
        return {"reexam_search_result": result, "reexam_error": ""}
    except Exception as exc:
        return {"reexam_search_result": {}, "reexam_error": f"web_search 失败：{exc}"}


def review_sources_node(state: AgentState) -> dict[str, Any]:
    if state.get("reexam_error"):
        return {"reexam_review_result": {}}
    search_result = dict(state.get("reexam_search_result") or {})
    sources = list(search_result.get("results") or [])
    goal = dict(state.get("reexam_goal") or {})
    try:
        review = review_web_sources(str(goal.get("research_goal") or ""), sources)
        return {"reexam_review_result": review}
    except Exception as exc:
        return {"reexam_review_result": {}, "reexam_error": f"source_review 失败：{exc}"}


def record_search_iteration_node(state: AgentState) -> dict[str, Any]:
    record = record_search_iteration(
        dict(state.get("reexam_query") or {}),
        dict(state.get("reexam_search_result") or {}),
        dict(state.get("reexam_review_result") or {}),
        str(state.get("reexam_error") or ""),
    )
    return {
        "reexam_record": record,
        "reexam_session": record.get("session", {}),
        "reexam_session_path": str(record.get("path", "")),
        "reexam_iteration_complete": True,
    }


def ask_user_next_step_node(state: AgentState) -> dict[str, Any]:
    decision = interrupt(format_search_iteration_summary(dict(state)))
    return {"reexam_next_action": normalize_search_decision(decision), "reexam_iteration_complete": False}


def source_confirmation_node(state: AgentState) -> dict[str, Any]:
    session = dict(state.get("reexam_session") or {})
    path = str(state.get("reexam_session_path") or "")
    return {"messages": [AIMessage(content=source_confirmation_message(session, path))]}


def stop_reexam_search_node(state: AgentState) -> dict[str, Any]:
    return {"messages": [AIMessage(content=stop_message(str(state.get("reexam_session_path") or "")))]}


def fake_tool_call_guard(state: AgentState) -> dict[str, Any]:
    # 旧手搓版里模型可能输出“看起来像工具调用的文本”。
    # LangGraph 只认 AIMessage.tool_calls，这个节点用于把假调用拦下来。
    return {
        "messages": [
            AIMessage(
                content=(
                    "我刚才生成了类似工具调用的文本，但那不是合法的 LangGraph tool call。"
                    "我会重新用已绑定工具或直接用自然语言回答。"
                )
            )
        ]
    }


def human_approval_node(state: AgentState) -> dict[str, Any]:
    # interrupt 会让图暂停；CLI 收到 __interrupt__ 后用 Command(resume=...) 把审批结果送回来。
    # 恢复后本函数会从 interrupt 这一行继续往下执行。
    ai_message = _last_ai_message(list(state.get("messages", [])))
    tool_calls = _tool_calls(ai_message)
    approval_items = [
        {
            "id": call.get("id"),
            "name": call.get("name"),
            "args": call.get("args", {}),
            "risk": get_tool_risk(str(call.get("name", ""))),
        }
        for call in tool_calls
        if requires_approval(str(call.get("name", "")))
    ]

    approval = interrupt(
        {
            "type": "tool_approval",
            "message": "是否批准执行这些 medium/high risk 工具调用？",
            "tool_calls": approval_items,
        }
    )
    if _is_yes(approval):
        return {"pending_approval": approval_items, "tool_approved": True}

    tool_messages = [
        ToolMessage(
            content="用户拒绝执行该工具调用，工具未执行。",
            tool_call_id=call.get("id", ""),
            name=call.get("name"),
        )
        for call in tool_calls
    ]
    return {"messages": tool_messages, "pending_approval": [], "tool_approved": False}


def record_tool_usage_node(state: AgentState) -> dict[str, Any]:
    # ToolNode 执行完后统一记账。当前主要用于 web_search 的“每个用户请求一次”预算。
    counts = dict(state.get("tool_call_counts") or {})
    ai_message = _last_ai_message(list(state.get("messages", [])))
    for call in _tool_calls(ai_message):
        name = str(call.get("name", ""))
        counts[name] = counts.get(name, 0) + 1
    return {"tool_call_counts": counts, "pending_approval": [], "tool_approved": False}


def summarize_if_needed_node(model_name: str = "mimo"):
    def summarize_if_needed(state: AgentState) -> dict[str, Any]:
        # 返回 RemoveMessage 会影响 checkpoint 中的历史消息。
        # 文件态 working-summary.md 用于保留被删中间消息的压缩信息。
        return build_summary_updates(list(state.get("messages", [])), model_name=model_name)

    return summarize_if_needed


def save_long_term_memory_node(store: Any | None = None, model_name: str = "mimo"):
    def save_long_term_memory(state: AgentState) -> dict[str, Any]:
        # 长期记忆是跨线程 store，不等于线程内 checkpoint。
        # 这里只保存筛过的用户事实/偏好，避免把每句闲聊都向量化进数据库。
        if store is None:
            return {}
        thread_id = str(state.get("thread_id") or "default")
        user_id = str(state.get("user_id") or "default")
        last = _last_human_message(list(state.get("messages", [])))
        if last is None or not should_save_long_term_memory(last, model_name=model_name):
            return {}
        namespace = ("memories", user_id)
        payload = long_term_memory_payload(last, thread_id)
        payload["profile_name"] = state.get("profile_name", "research_to_product")
        try:
            store.put(namespace, uuid.uuid4().hex, payload)
            compact_long_term_memory(store, user_id, model_name=model_name)
        except Exception:
            return {}
        return {}

    return save_long_term_memory


def route_after_agent(state: AgentState) -> str:
    # 这是图的核心路由：模型说完后，决定是结束、走工具、先审批，还是处理假工具调用。
    messages = list(state.get("messages", []))
    last = _last_message(messages)
    if not isinstance(last, AIMessage):
        return "summarize_if_needed"

    tool_calls = _tool_calls(last)
    if not tool_calls:
        content = str(last.content or "")
        if "<tool_call" in content or '"tool_calls"' in content or "function_call" in content:
            return "fake_tool_call_guard"
        return "summarize_if_needed"

    if any(requires_approval(str(call.get("name", ""))) for call in tool_calls) and not state.get("tool_approved"):
        return "human_approval"
    return "tools"


def route_after_approval(state: AgentState) -> str:
    return "tools" if state.get("tool_approved") else "agent"


def route_after_reexam_parse(state: AgentState) -> str:
    return "ensure_reexam_session" if state.get("is_reexam_search") else "agent"


def route_after_reexam_session(state: AgentState) -> str:
    return "summarize_if_needed" if state.get("reexam_error") else "evaluate_reexam_gaps"


def route_after_reexam_gaps(state: AgentState) -> str:
    if state.get("reexam_iteration_complete"):
        return "ask_user_next_step"
    session = dict(state.get("reexam_session") or {})
    has_search_history = bool(session.get("candidate_sources") or session.get("reviewed_sources") or session.get("search_queries"))
    if not has_search_history:
        return "generate_gap_queries"
    return "ask_user_next_step"


def route_after_reexam_decision(state: AgentState) -> str:
    action = str(state.get("reexam_next_action") or "continue")
    if action == "next":
        return "source_confirmation"
    if action == "stop":
        return "stop_reexam_search"
    return "generate_gap_queries"


def build_graph(checkpointer: Any | None = None, store: Any | None = None, model_name: str = "mimo"):
    # 生产图只在这里拼装。GraphTest 目录仍然是学习沙盒，不参与这个入口。
    tools = get_registered_tools()
    builder = StateGraph(AgentState)
    builder.add_node("load_context", load_context_node(store))
    builder.add_node("parse_reexam_goal", parse_reexam_goal_node)
    builder.add_node("ensure_reexam_session", ensure_reexam_session_node)
    builder.add_node("evaluate_reexam_gaps", evaluate_reexam_gaps_node)
    builder.add_node("generate_gap_queries", generate_gap_queries_node)
    builder.add_node("run_one_web_search", run_one_web_search_node)
    builder.add_node("review_sources", review_sources_node)
    builder.add_node("record_search_iteration", record_search_iteration_node)
    builder.add_node("ask_user_next_step", ask_user_next_step_node)
    builder.add_node("source_confirmation", source_confirmation_node)
    builder.add_node("stop_reexam_search", stop_reexam_search_node)
    builder.add_node("agent", agent_node(model_name))
    builder.add_node("fake_tool_call_guard", fake_tool_call_guard)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("record_tool_usage", record_tool_usage_node)
    builder.add_node("summarize_if_needed", summarize_if_needed_node(model_name))
    builder.add_node("save_long_term_memory", save_long_term_memory_node(store, model_name))

    # START -> load_context 后先判断是否进入复试资料搜索专用循环，否则走普通 agent。
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "parse_reexam_goal")
    builder.add_conditional_edges(
        "parse_reexam_goal",
        route_after_reexam_parse,
        {"ensure_reexam_session": "ensure_reexam_session", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "ensure_reexam_session",
        route_after_reexam_session,
        {"evaluate_reexam_gaps": "evaluate_reexam_gaps", "summarize_if_needed": "summarize_if_needed"},
    )
    builder.add_conditional_edges(
        "evaluate_reexam_gaps",
        route_after_reexam_gaps,
        {"generate_gap_queries": "generate_gap_queries", "ask_user_next_step": "ask_user_next_step"},
    )
    builder.add_edge("generate_gap_queries", "run_one_web_search")
    builder.add_edge("run_one_web_search", "review_sources")
    builder.add_edge("review_sources", "record_search_iteration")
    builder.add_edge("record_search_iteration", "evaluate_reexam_gaps")
    builder.add_conditional_edges(
        "ask_user_next_step",
        route_after_reexam_decision,
        {
            "generate_gap_queries": "generate_gap_queries",
            "source_confirmation": "source_confirmation",
            "stop_reexam_search": "stop_reexam_search",
        },
    )
    builder.add_edge("source_confirmation", "summarize_if_needed")
    builder.add_edge("stop_reexam_search", "summarize_if_needed")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "fake_tool_call_guard": "fake_tool_call_guard",
            "human_approval": "human_approval",
            "tools": "tools",
            "summarize_if_needed": "summarize_if_needed",
        },
    )
    builder.add_edge("fake_tool_call_guard", "summarize_if_needed")
    builder.add_conditional_edges("human_approval", route_after_approval, {"tools": "tools", "agent": "agent"})
    # ToolNode 只负责执行工具；执行后的预算记账放到单独节点，职责更清楚。
    builder.add_edge("tools", "record_tool_usage")
    builder.add_edge("record_tool_usage", "agent")
    # 没有工具要执行时，先尝试压缩 checkpoint，再把值得长期保留的信息写入 store。
    builder.add_edge("summarize_if_needed", "save_long_term_memory")
    builder.add_edge("save_long_term_memory", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if store is not None:
        compile_kwargs["store"] = store
    return builder.compile(**compile_kwargs)


def approval_payload_to_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    if payload.get("type") == "reexam_search_decision":
        lines = [str(payload.get("message", "复试资料搜索需要你判断下一步："))]
        session_path = payload.get("session_path")
        if session_path:
            lines.append(f"session: {session_path}")
        lines.append(f"候选来源累计：{payload.get('candidate_count', 0)}")
        lines.append(f"已初筛来源累计：{payload.get('reviewed_count', 0)}")
        gaps = payload.get("current_gaps") or []
        if gaps:
            lines.append("当前资料缺口：")
            lines.extend(f"- {gap}" for gap in gaps)
        query = payload.get("recommended_query") or {}
        if query:
            lines.append(f"推荐下一条 query：[{query.get('query_type')}] {query.get('query')}")
        options = payload.get("options") or []
        if options:
            lines.append("可输入：")
            lines.extend(f"- {item.get('value')}: {item.get('label')}" for item in options)
        return "\n".join(lines)
    lines = [str(payload.get("message", "需要人工审批："))]
    for index, call in enumerate(payload.get("tool_calls", []), start=1):
        args = json.dumps(call.get("args", {}), ensure_ascii=False)
        lines.append(f"[{index}] {call.get('name')} risk={call.get('risk')} args={args}")
    return "\n".join(lines)
