from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from testmain import mimo_v2_5

#注意:使用内存存储来存储记忆
memory = MemorySaver()
@tool
def search(query: str):
    """调用此函数可以浏览网络。"""
    #模拟一个网络搜索返回
    return "北京天气晴朗 大约22度 湿度30%"

#tools = [search] → 这是工具定义列表，告诉模型"你有哪些工具可以用"
#ToolNode(tools) → 这是执行器，负责拿到 tool_calls 后实际调用函数，相当于手搓版的execute_tool_call函数
tools=[search]
tool_node=ToolNode(tools)
#bind_tools() 的作用是把工具的 schema 传给模型，让模型知道有哪些工具、参数是什么格式。
bound_model=mimo_v2_5.bind_tools(tools)

def should_continue(state:MessagesState):
    """返回下一个要执行的节点。"""
    last_message = state["messages"][-1]
    #如果没有函数调用，则结束
    if not last_message.tool_calls:
        return END
    #否则如果有，我们继续
    return "run"

#定义调用模型的函数
def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    #我们返回一个列表，因为这会被添加到现有列表中
    return {"messages": response}


# ============================================================
# 构建图
#
# 流程：
#   START
#     ↓
#   agent（调用模型）
#     ↓
#   should_continue（条件判断）
#     ├─ 有 tool_calls → "action"（执行工具）→ 回到 agent
#     └─ 没有 tool_calls → END
# ============================================================

graph_builder = StateGraph(MessagesState)

# 添加两个节点：agent = 调模型，action = 执行工具
graph_builder.add_node("agent", call_model)
graph_builder.add_node("action", tool_node)

# START → agent
graph_builder.add_edge(START, "agent")

# agent 走完后，判断是执行工具还是结束
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {#字典做映射，将should_continue返回的信息映射到实际节点
        "run": "action",  # 有 tool_calls → 去 action 节点执行工具
        END: END,             # 没有 tool_calls → 结束
    },
)

# action（工具执行完）→ 回到 agent（让模型看工具结果再回复）
graph_builder.add_edge("action", "agent")

# 编译图
graph = graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "tool-test"}}

    print("请输入（输入 exit 退出）：")
    while True:
        question = input().strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(question)]},
            config=config,
        )

        for message in result["messages"]:
            message.pretty_print()