"""
Audit.py — LangGraph human approval demo

这个 demo 演示：

用户消息
  -> agent 调模型
  -> 如果模型请求 tool_calls，先进入 human_approval 节点
  -> 人类通过 interrupt() 审批
  -> yes: 执行工具
  -> no: 把拒绝结果作为 ToolMessage 回传给模型

运行方式：

    cd E:\study\LangGraph
    .\.venv\Scripts\python.exe .\GraphTest\Audit.py
"""

import operator
from pathlib import Path
from typing import Annotated, Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from testmain import models


AUDIT_OUTPUT = Path(__file__).resolve().parent / "audit-output.txt"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tool_approved: bool


@tool
def write_audit_note(content: str) -> str:
    """把用户确认过的审计备注写入 audit-output.txt。"""
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_OUTPUT.open("a", encoding="utf-8") as file:
        file.write(content + "\n")

    return f"已写入 {AUDIT_OUTPUT}"


@tool
def read_audit_note() -> str:
    """读取 audit-output.txt 的内容。"""
    if not AUDIT_OUTPUT.exists():
        return "audit-output.txt 还不存在。"

    return AUDIT_OUTPUT.read_text(encoding="utf-8", errors="replace")


tools = [write_audit_note, read_audit_note]
tool_node = ToolNode(tools)


def get_last_ai_message(state: AgentState) -> AIMessage:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message

    raise ValueError("没有找到 AIMessage，无法审批工具调用。")


def call_model(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """调用模型。"""
    model_name = config.get("configurable", {}).get("model", "mimo")
    model = models[model_name].bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                "你是一个 LangGraph 审批 demo 助手。"
                "当用户要求写入、记录或保存审计备注时，调用 write_audit_note。"
                "当用户要求查看审计备注时，调用 read_audit_note。"
                "工具调用前会由 human_approval 节点进行人工审批。"
            )
        ),
        *state["messages"],
    ]

    response = model.invoke(messages)
    response.pretty_print()
    return {"messages": [response], "tool_approved": False}


def should_continue(state: AgentState) -> str:
    """模型没有请求工具就结束；请求工具则进入审批节点。"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "human_approval"

    return END


def human_approval(state: AgentState) -> dict[str, Any]:
    """人工审批节点：通过 interrupt 暂停图，等待外部 resume。"""
    last_ai_message = get_last_ai_message(state)
    tool_calls = last_ai_message.tool_calls

    approval = interrupt(
        {
            "question": "是否批准执行这些工具调用？输入 yes/no。",
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                }
                for tool_call in tool_calls
            ],
        }
    )

    approved = str(approval).strip().lower() in {"y", "yes", "approve", "accept"}
    if approved:
        return {"tool_approved": True}

    rejected_messages = [
        ToolMessage(
            content="人工审批拒绝：本次工具调用未执行。",
            tool_call_id=tool_call["id"],
        )
        for tool_call in tool_calls
    ]
    return {"messages": rejected_messages, "tool_approved": False}


def route_after_approval(state: AgentState) -> str:
    """审批通过则执行工具；拒绝则回到模型，让模型向用户说明。"""
    if state.get("tool_approved"):
        return "action"

    return "agent"


graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", call_model)
graph_builder.add_node("human_approval", human_approval)
graph_builder.add_node("action", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "human_approval": "human_approval",
        END: END,
    },
)
graph_builder.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "action": "action",
        "agent": "agent",
    },
)
graph_builder.add_edge("action", "agent")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def invoke_with_human_approval(input_value: Any, config: dict[str, Any]) -> dict[str, Any]:
    """运行图；如果遇到 interrupt，把 interrupt 信息返回给外层调用者(result)，在终端收集 yes/no 后继续。"""
    result = graph.invoke(input_value, config=config)

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        interrupt_value = interrupts[0].value

        print("\n===== human_approval =====")
        print(interrupt_value["question"])#approval = interrupt里定义了question和tool_calls
        for index, tool_call in enumerate(interrupt_value["tool_calls"], start=1):
            print(f"[{index}] {tool_call['name']}({tool_call['args']})")

        answer = input("审批结果 yes/no：").strip()
        #Command把人工输入交还给之前 interrupt() 暂停的节点，然后从那个节点恢复执行
        result = graph.invoke(Command(resume=answer), config=config)

    return result


if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "audit-demo",
            "model": "mimo",
        }
    }

    print("【Audit demo】输入内容开始对话。")
    print("示例：帮我写入一条审计备注：今天测试人工审批节点")
    print("输入 /exit 退出。")

    while True:
        user_text = input("\n请输入：").strip()
        if not user_text:
            continue
        if user_text == "/exit":
            break

        final_state = invoke_with_human_approval(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

        print("\n===== 当前消息 =====")
        for message in final_state["messages"]:
            message.pretty_print()
