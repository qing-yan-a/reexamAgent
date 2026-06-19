import operator
from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    profile_name: str
    tool_call_counts: dict[str, int]
    pending_approval: dict[str, Any] | None
    working_summary: str
    retrieved_memories: list[dict[str, Any]]
    retrieved_rag: list[dict[str, Any]]
    is_reexam_search: bool
    reexam_goal: dict[str, Any]
    reexam_session: dict[str, Any]
    reexam_session_path: str
    reexam_open_gaps: list[str]
    reexam_readiness: dict[str, Any]
    reexam_query: dict[str, Any]
    reexam_search_result: dict[str, Any]
    reexam_review_result: dict[str, Any]
    reexam_record: dict[str, Any]
    reexam_iteration_complete: bool
    reexam_next_action: str
    reexam_error: str
    tool_approved: bool
    final_answer: str
    # Used only by tests and diagnostic helpers.
    debug_events: Annotated[list[str], operator.add]
