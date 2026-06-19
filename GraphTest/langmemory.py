import operator
import os
import uuid
from datetime import datetime
from typing import Annotated, Sequence
from pathlib import Path
import json
import voyageai
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from testmain import PROJECT_ROOT, models

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 跨线程记忆（Store）：用 user_id 做 namespace，不同 thread 共享
# 线程内对话历史（Checkpointer）：用 thread_id 做隔离
# 两者配合，实现"跨线程记住用户偏好 + 单线程内保持对话上下文"
# ============================================================

# ---------- Voyage Embedding ----------
# InMemoryStore 的 index.embed 需要一个 callable：接收 list[str]，返回 list[list[float]]
# 直接用 voyageai SDK 封装成函数即可

def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量嵌入文本，返回向量列表。"""
    vo = voyageai.Client()
    result = vo.embed(texts, model="voyage-4-large")
    return result.embeddings


# ---------- InMemoryStore（跨线程记忆仓库） ----------
# index 配置开启向量搜索：
#   dims = embedding 维度（voyage-4-large = 1024）
#   embed = 嵌入函数（callable）
#   fields = 对 value 中哪些字段做 embedding（"$" 表示整个 value）
store = InMemoryStore(#跨线程记忆仓库
    index={
        "dims": 1024,
        "embed": embed_texts,
        "fields": ["text"],  # 只对 value 中的 "text" 字段做向量索引
    }
)

# ---------- Checkpointer（线程内对话历史） ----------

def get_postgres_uri() -> str:
    postgres_uri = os.getenv("POSTGRES_URI")
    if not postgres_uri:
        raise RuntimeError(
            "缺少 POSTGRES_URI。请在 .env 中配置，例如："
            "POSTGRES_URI=postgresql://postgres:你的密码@localhost:5432/postgres?sslmode=disable"
        )
    return postgres_uri


# ========== 内存版（进程内，脚本关闭就丢失） ==========
# memory = MemorySaver()

# ========== 数据库版（持久化，重启后同一 thread_id 可恢复历史） ==========
_postgres_saver = PostgresSaver.from_conn_string(get_postgres_uri())
memory = _postgres_saver.__enter__()
memory.setup()


# ============================================================
# 状态定义
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ============================================================
# 节点函数
# ============================================================

def retrieve_memories(state: AgentState, config: RunnableConfig) -> dict:
    """节点 2：从 Store 中检索与当前用户消息相关的跨线程记忆。

    工作流程：
    1. 从 config 中读取 user_id，确定 namespace
    2. 取用户最新一条消息作为查询
    3. 在 Store 中做语义搜索，找到相关记忆
    4. 把记忆拼成 SystemMessage 插入到消息列表最前面
    """
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = ("memories", user_id)

    # 取用户最新消息作为搜索 query
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"messages": []}

    # 语义搜索：找与当前消息最相关的记忆
    results = store.search(namespace, query=last_user_msg, limit=3)

    if not results:
        return {"messages": []}

    # 把搜到的记忆拼成一段上下文，作为 SystemMessage 注入
    memory_texts = []
    for item in results:
        data = item.value
        memory_texts.append(f"- {data.get('text', '')}（来自 {data.get('source_thread', '未知')}）")

    memory_context = "以下是关于该用户的跨线程记忆，可供参考：\n" + "\n".join(memory_texts)

    return {"messages": [SystemMessage(content=memory_context)]}


def call_model(state: AgentState, config: RunnableConfig) -> dict:
    """节点 3：调用模型生成回复。"""
    model_name = config.get("configurable", {}).get("model")
    model = models[model_name]
    response = model.invoke(state["messages"])

    response.pretty_print()

    return {"messages": [response]}


def save_memory(state: AgentState, config: RunnableConfig) -> dict:
    """节点 4：把本轮对话中有价值的信息存入 Store。

    存储格式：value = {"text": "记忆内容", "source_thread": "thread_id", "timestamp": "..."}
    namespace = ("memories", user_id)
    key = 随机生成唯一 ID

    这里做最小实现：把用户最新一条消息存为记忆。
    实际项目中可以让模型判断哪些信息值得记忆，或用摘要压缩。
    """
    user_id = config.get("configurable", {}).get("user_id", "default")
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    namespace = ("memories", user_id)

    # 取用户最新消息
    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if user_msg and user_msg.strip() != "/exit":
        memory_id = str(uuid.uuid4())[:8]
        #存原始文本
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
# 构建图
#
# 流程：
#   START
#     ↓
#   retrieve_memories（从 Store 检索跨线程记忆，注入上下文）
#     ↓
#   model（模型生成回复）
#     ↓
#   save_memory（存记忆）
#     ↓
#   END
#
# 注意：
#   这里把图改成“单轮图”。
#   外层 while 循环负责读取用户输入和切换 thread_id。
#   这样每一轮 graph.invoke 都能传入新的 config，从而真正切换 checkpoint 线程。
# ============================================================

graph_builder = StateGraph(AgentState)

graph_builder.add_node("retrieve_memories", retrieve_memories)
graph_builder.add_node("model", call_model)
graph_builder.add_node("save_memory", save_memory)

graph_builder.add_edge(START, "retrieve_memories")
graph_builder.add_edge("retrieve_memories", "model")
graph_builder.add_edge("model", "save_memory")
graph_builder.add_edge("save_memory", END)

# compile 时同时传入 checkpointer（线程历史）和 store（跨线程记忆）
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

    print("输入内容开始对话。")
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
        for message in result["messages"]:  # ← 缩进到 while 里面
            message.pretty_print()