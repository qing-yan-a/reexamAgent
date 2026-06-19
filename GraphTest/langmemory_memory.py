"""
langmemory_memory.py — 全内存版（进程内，脚本关闭就丢失）

- Checkpointer: MemorySaver（线程内对话历史，内存）
- Store: InMemoryStore（跨线程记忆，内存）

适合：学习、调试、快速验证流程
"""

import json
import operator
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Sequence

import voyageai
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from testmain import PROJECT_ROOT, models

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# Voyage Embedding（供 InMemoryStore 向量搜索用）
# ============================================================

def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量嵌入文本，返回向量列表。"""
    vo = voyageai.Client()
    result = vo.embed(texts, model="voyage-4-large")
    return result.embeddings


# ============================================================
# 存储层
# ============================================================

# Store：跨线程记忆（内存，重启丢失）
store = InMemoryStore(
    index={
        "dims": 1024,
        "embed": embed_texts,
        "fields": ["text"],
    }
)

# Checkpointer：线程内对话历史（内存，重启丢失）
memory = MemorySaver()


# ============================================================
# 状态定义
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ============================================================
# 节点函数
# ============================================================

def retrieve_memories(state: AgentState, config: RunnableConfig) -> dict:
    """从 Store 中检索与当前用户消息相关的跨线程记忆，注入上下文。"""
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = ("memories", user_id)

    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"messages": []}

    results = store.search(namespace, query=last_user_msg, limit=3)

    if not results:
        return {"messages": []}

    memory_texts = []
    for item in results:
        data = item.value
        memory_texts.append(f"- {data.get('text', '')}（来自 {data.get('source_thread', '未知')}）")

    memory_context = "以下是关于该用户的跨线程记忆，可供参考：\n" + "\n".join(memory_texts)
    return {"messages": [SystemMessage(content=memory_context)]}


def call_model(state: AgentState, config: RunnableConfig) -> dict:
    """调用模型生成回复。"""
    model_name = config.get("configurable", {}).get("model")
    model = models[model_name]
    response = model.invoke(state["messages"])
    response.pretty_print()
    return {"messages": [response]}


def save_memory(state: AgentState, config: RunnableConfig) -> dict:
    """把本轮用户消息存入 Store（最小实现）。"""
    user_id = config.get("configurable", {}).get("user_id", "default")
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    namespace = ("memories", user_id)

    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if user_msg and user_msg.strip() != "/exit":
        memory_id = str(uuid.uuid4())[:8]
        store.put(
            namespace,
            memory_id,
            {
                "text": user_msg,
                "source_thread": thread_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    return {}


# ============================================================
# 构建图（单轮图，外层 while 循环负责多轮交互）
# ============================================================

graph_builder = StateGraph(AgentState)

graph_builder.add_node("retrieve_memories", retrieve_memories)
graph_builder.add_node("model", call_model)
graph_builder.add_node("save_memory", save_memory)

graph_builder.add_edge(START, "retrieve_memories")
graph_builder.add_edge("retrieve_memories", "model")
graph_builder.add_edge("model", "save_memory")
graph_builder.add_edge("save_memory", END)

graph = graph_builder.compile(checkpointer=memory, store=store)


# ============================================================
# 入口
# ============================================================

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


def save_config(config: dict) -> None:
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def switch_thread(config: dict, thread_id: str) -> dict:
    config = dict(config)
    configurable = dict(config.get("configurable", {}))
    configurable["thread_id"] = thread_id
    config["configurable"] = configurable
    save_config(config)
    return config


if __name__ == "__main__":
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    print("【内存版】输入内容开始对话。")
    print("输入 /thread 线程名 切换 thread_id，例如：/thread test2")
    print("输入 /exit 退出。")

    while True:
        current_thread = config.get("configurable", {}).get("thread_id", "unknown")
        print(f"\n当前 thread_id = {current_thread}")
        user_text = input("请输入：").strip()

        if not user_text:
            continue

        if user_text == "/exit":
            break

        if user_text.startswith("/thread "):
            new_thread_id = user_text.removeprefix("/thread ").strip()
            if not new_thread_id:
                print("请指定 thread_id，例如：/thread test2")
                continue
            config = switch_thread(config, new_thread_id)
            print(f"已切换到 thread_id = {new_thread_id}")
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(user_text)]},
            config=config,
        )
        for message in result["messages"]:
            message.pretty_print()
