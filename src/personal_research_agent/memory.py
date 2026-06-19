import json
from datetime import UTC, datetime
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage

from .models import build_model
from .session_manager import get_active_working_summary_path


KEEP_FIRST_MESSAGES = 6
KEEP_RECENT_MESSAGES = 20
# 超过这个阈值才压缩。前 6 条用于保留开场身份/任务锚点，后 20 条保留最近工作上下文。
MAX_MESSAGES_BEFORE_SUMMARY = KEEP_FIRST_MESSAGES + KEEP_RECENT_MESSAGES + 8
MAX_SUMMARY_CHARS = 3000
LONG_TERM_MEMORY_COMPACT_THRESHOLD = 30
LONG_TERM_MEMORY_KEEP_RECENT = 8
MEMORY_SUMMARY_KEY = "memory_summary"


def message_to_text(message: BaseMessage) -> str:
    role = message.type
    content = message.content
    return f"{role}: {content}"


def summarize_messages(old_summary: str, pruned_messages: list[BaseMessage], model_name: str = "mimo") -> str:
    if not pruned_messages:
        return old_summary

    # 摘要模型只看“旧摘要 + 本次要删除的中间消息”，不重写最近 20 条原始对话。
    model = build_model(model_name)
    payload = "\n\n".join(message_to_text(message) for message in pruned_messages)
    messages = [
        SystemMessage(
            content=(
                "你是上下文压缩器。把旧摘要和即将被裁剪的对话压缩成新的 working-summary。"
                "只保留对后续任务有用的信息：用户目标、关键约束、文件路径、已完成事项、重要决策、未解决问题。"
                "不要保留完整工具输出和无关寒暄。中文 Markdown 输出，控制在 2000 字以内。"
            )
        ),
        HumanMessage(content=f"旧摘要：\n{old_summary}\n\n待压缩消息：\n{payload}"),
    ]
    response = model.invoke(messages)
    return str(response.content or "")[:MAX_SUMMARY_CHARS]


def load_current_summary() -> str:
    try:
        path = get_active_working_summary_path()
    except RuntimeError:
        return ""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:MAX_SUMMARY_CHARS]


def save_current_summary(content: str) -> None:
    path = get_active_working_summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content[:MAX_SUMMARY_CHARS], encoding="utf-8")


def build_summary_updates(messages: list[BaseMessage], model_name: str = "mimo") -> dict[str, Any]:
    """Return RemoveMessage updates for middle-history compression."""
    if len(messages) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return {"messages": [], "working_summary": load_current_summary()}

    first = messages[:KEEP_FIRST_MESSAGES]
    recent = list(messages[-KEEP_RECENT_MESSAGES:])
    middle = list(messages[KEEP_FIRST_MESSAGES:-KEEP_RECENT_MESSAGES])

    # 不能让 recent 以 ToolMessage 开头，否则 checkpoint 里会留下“没有对应 AI tool_call 的工具结果”。
    # 遇到这种情况，把开头 ToolMessage 一并压进摘要。
    while recent and recent[0].type == "tool":
        middle.append(recent.pop(0))

    if not middle:
        return {"messages": [], "working_summary": load_current_summary()}

    old_summary = load_current_summary()
    new_summary = summarize_messages(old_summary, middle, model_name=model_name)
    save_current_summary(new_summary)

    # add_messages reducer 识别 RemoveMessage 后，会从 checkpoint 历史中删除对应消息。
    kept_ids = {getattr(message, "id", None) for message in first + recent}
    removals = [
        RemoveMessage(id=message.id)
        for message in messages
        if getattr(message, "id", None) and message.id not in kept_ids
    ]
    return {"messages": removals, "working_summary": new_summary}


def should_save_long_term_memory(message: BaseMessage, model_name: str = "mimo") -> bool:
    # 只做硬性安全兜底；真正的“是否值得长期保存”交给模型判断。
    if not isinstance(message, HumanMessage):
        return False
    text = str(message.content or "").strip()
    if not text or text.startswith("/"):
        return False

    model = build_model(model_name)
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是长期记忆筛选器。判断用户这句话是否应该保存到跨线程长期记忆 store。"
                    "只保存对未来有长期价值的信息：稳定偏好、身份信息、项目事实、重要决策、常用路径、长期目标。"
                    "不要保存寒暄、一次性指令、普通问题、临时状态、纯测试、情绪性反馈。"
                    "只输出 JSON，不要解释，格式：{\"save\": true/false, \"reason\": \"...\"}"
                )
            ),
            HumanMessage(content=f"用户消息：{text}"),
        ]
    )
    content = str(response.content or "").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        lowered = content.lower()
        return lowered.startswith("true") or lowered.startswith("yes") or '"save": true' in lowered
    return bool(data.get("save"))


def long_term_memory_payload(message: BaseMessage, thread_id: str) -> dict[str, Any]:
    return {
        "type": "conversation_fact",
        "text": str(message.content),
        "source_thread": thread_id,
        "created_at": datetime.now(UTC).isoformat(),
    }


def summarize_long_term_memory(old_summary: str, facts: list[str], model_name: str = "mimo") -> str:
    if not facts:
        return old_summary
    model = build_model(model_name)
    payload = "\n".join(f"- {fact}" for fact in facts)
    messages = [
        SystemMessage(
            content=(
                "你是长期记忆压缩器。把零散 conversation_fact 合并成稳定的 memory_summary。"
                "只保留长期偏好、身份信息、项目事实、重要决策、常用路径、长期目标。"
                "删除寒暄、重复表达和一次性细节。中文 Markdown 输出，控制在 1200 字以内。"
            )
        ),
        HumanMessage(content=f"旧 memory_summary：\n{old_summary}\n\n待合并 facts：\n{payload}"),
    ]
    response = model.invoke(messages)
    return str(response.content or "")[:MAX_SUMMARY_CHARS]


def _item_text(item: Any) -> str:
    value = getattr(item, "value", item)
    if isinstance(value, dict):
        return str(value.get("text", value))
    return str(value)


def retrieve_long_term_memories(store: Any, user_id: str, query: str, limit: int = 3) -> list[dict[str, Any]]:
    namespace = ("memories", str(user_id))
    memories: list[dict[str, Any]] = []

    try:
        summary_item = store.get(namespace, MEMORY_SUMMARY_KEY)
    except Exception:
        summary_item = None
    if summary_item is not None:
        text = _item_text(summary_item)
        if text:
            memories.append(
                {
                    "type": "memory_summary",
                    "text": text,
                    "key": getattr(summary_item, "key", MEMORY_SUMMARY_KEY),
                    "score": getattr(summary_item, "score", None),
                }
            )

    try:
        results = store.search(namespace, query=query, filter={"type": "conversation_fact"}, limit=max(1, limit * 4))
    except Exception:
        results = []

    for item in results:
        text = _item_text(item)
        if not text:
            continue
        memories.append(
            {
                "type": "conversation_fact",
                "text": text,
                "key": getattr(item, "key", ""),
                "score": getattr(item, "score", None),
                "source_thread": getattr(item, "value", {}).get("source_thread") if isinstance(getattr(item, "value", None), dict) else None,
            }
        )
        if len(memories) >= limit:
            break
    return memories[:limit]


def compact_long_term_memory(
    store: Any,
    user_id: str,
    *,
    model_name: str = "mimo",
    threshold: int = LONG_TERM_MEMORY_COMPACT_THRESHOLD,
    keep_recent: int = LONG_TERM_MEMORY_KEEP_RECENT,
    summarizer: Callable[[str, list[str], str], str] = summarize_long_term_memory,
) -> dict[str, Any]:
    namespace = ("memories", str(user_id))
    try:
        facts = store.search(namespace, query=None, filter={"type": "conversation_fact"}, limit=threshold + keep_recent + 200)
    except Exception as exc:
        return {"compacted": False, "reason": str(exc)}

    facts = sorted(facts, key=lambda item: getattr(item, "created_at", datetime.min.replace(tzinfo=UTC)))
    if len(facts) <= threshold:
        return {"compacted": False, "fact_count": len(facts), "threshold": threshold}

    old_facts = facts[:-keep_recent] if keep_recent > 0 else facts
    if not old_facts:
        return {"compacted": False, "fact_count": len(facts), "threshold": threshold}

    try:
        summary_item = store.get(namespace, MEMORY_SUMMARY_KEY)
        old_summary = _item_text(summary_item) if summary_item is not None else ""
        new_summary = summarizer(old_summary, [_item_text(item) for item in old_facts], model_name)
        store.put(
            namespace,
            MEMORY_SUMMARY_KEY,
            {
                "type": "memory_summary",
                "text": new_summary,
                "updated_at": datetime.now(UTC).isoformat(),
                "source": "conversation_fact_compaction",
                "compacted_count": len(old_facts),
            },
        )
        for item in old_facts:
            store.delete(namespace, item.key)
    except Exception as exc:
        return {"compacted": False, "fact_count": len(facts), "reason": str(exc)}

    return {
        "compacted": True,
        "fact_count_before": len(facts),
        "deleted_facts": len(old_facts),
        "kept_recent": len(facts) - len(old_facts),
    }
