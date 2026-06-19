import operator
import os
import tempfile
from pathlib import Path
from typing import Annotated, Sequence

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


mimo_v2_5 = ChatOpenAI(
    model=os.getenv("MIMO_MODEL", "mimo-v2.5"),
    api_key=os.getenv("MIMO_API_KEY"),
    base_url=os.getenv("MIMO_BASE_URL"),
    temperature=0,
    streaming=True,
    max_tokens=6000,
    timeout=300,
    max_retries=3,
)

deepseek_v4_flash = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    streaming=True,
    max_tokens=6000,
    timeout=300,
    max_retries=3,
)

models = {
    "mimo": mimo_v2_5,
    "deepseek": deepseek_v4_flash,
}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def call_model(state: AgentState, config: RunnableConfig) -> dict:
    model_name = config.get("configurable", {}).get("model")
    model = models[model_name]
    response = model.invoke(state["messages"])

    return {
        "messages": [response],
    }


graph_builder = StateGraph(AgentState)

graph_builder.add_node("model", call_model)
graph_builder.add_edge(START, "model")
graph_builder.add_edge("model", END)

graph = graph_builder.compile()


def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def display_graph() -> None:
    png_bytes = graph.get_graph().draw_mermaid_png()

    if is_notebook():
        display(Image(data=png_bytes))
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as file:
        file.write(png_bytes)
        image_path = file.name

    os.startfile(image_path)




if __name__ == "__main__":
    config = {
        "configurable": {
            "model": "mimo",
        }
    }
    display_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="hi，你是谁？")],
        },
        config=config,
    )

    for message in result["messages"]:
        message.pretty_print()
