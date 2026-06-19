import json
import operator
import os
import re
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# 三段提示词分别对应三个模型节点：
# 1. 根据总主题生成多个子主题。
# 2. 给每个子主题生成一个笑话。
# 3. 从全部笑话里选出最好笑的一个。
subjects_prompt = """生成一个逗号分隔的主题列表，包含 2 到 5 个与以下主题相关的例子：{topic}。

请严格以 JSON 格式返回，格式如下：
{{"subjects": ["主题1", "主题2", "主题3"]}}
不要返回其他内容。"""
joke_prompt = """生成一个关于 {subject} 的简短中文笑话。

要求：
- 必须是一条完整笑话。
- 不要只返回主题名。
- 不要解释。

请严格以 JSON 格式返回，格式如下：
{{"joke": "笑话内容"}}
不要返回其他内容。"""
best_joke_prompt = """以下是一些关于 {topic} 的笑话。

请选出最好笑的一个，并返回最佳笑话的 ID。ID 从 0 开始。

{jokes}

请严格以 JSON 格式返回，格式如下：
{{"id": 0}}
不要返回其他内容。"""


class Subjects(BaseModel):
    """generate_topics 节点要求模型返回的结构化结果。"""

    subjects: list[str] = Field(description="与用户主题相关的 2 到 5 个子主题")


class Joke(BaseModel):
    """generate_joke 节点要求模型返回的结构化结果。"""

    joke: str = Field(description="围绕指定子主题生成的一条中文笑话")


class BestJoke(BaseModel):
    """best_joke 节点要求模型返回的结构化结果。"""

    id: int = Field(description="最佳笑话的索引，从 0 开始", ge=0)


model = ChatOpenAI(
    model=os.getenv("MIMO_MODEL", "mimo-v2.5"),
    api_key=os.getenv("MIMO_API_KEY"),
    base_url=os.getenv("MIMO_BASE_URL"),
    temperature=0,
)


def parse_json_from_response(content: str) -> dict:
    """从模型返回的文本中提取 JSON 对象。"""
    # 先试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # 尝试提取 ```json ... ``` 代码块
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试提取第一个 { ... }
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法从模型返回中解析 JSON: {content}")

class OverallState(TypedDict):
    """主图的整体状态。

    topic 是用户输入的总主题。
    subjects 是 generate_topics 生成的子主题列表。
    jokes 会收集每个 generate_joke 分支返回的笑话。
    best_selected_joke 是最后选出的最佳笑话。
    """

    topic: str
    subjects: list[str]
    # operator.add 是 reducer：多个 generate_joke 分支都返回 jokes 时，
    # LangGraph 会把这些 list 追加合并，而不是互相覆盖。
    jokes: Annotated[list[str], operator.add]
    best_selected_joke: str


class JokeState(TypedDict):
    """单个 generate_joke 分支使用的局部状态。"""

    subject: str


def generate_topics(state: OverallState) -> dict:
    """第一步：根据用户给的总 topic，生成多个子主题 subjects。"""
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.invoke(prompt)
    data = parse_json_from_response(response.content)
    subjects = Subjects(**data)

    return {
        "subjects": subjects.subjects,
    }


def generate_joke(state: JokeState) -> dict:
    """第二步：根据一个 subject 生成一条笑话。

    这个节点会被 Send 分发多次：每个 subject 运行一次。
    返回 {"jokes": [joke]} 是为了配合 OverallState 里的 operator.add 聚合。
    """
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.invoke(prompt)
    data = parse_json_from_response(response.content)
    joke_obj = Joke(**data)
    joke = joke_obj.joke.strip()

    # 小兜底：有些模型偶尔会把主题名直接塞进 joke 字段。
    # 这里不是 LangGraph 必需逻辑，只是为了让 demo 输出更稳定。
    if joke == state["subject"] or len(joke) <= len(state["subject"]) + 2:
        joke = f"为什么{state['subject']}总让程序员发笑？因为它一出现，大家就知道今天又要加班调 bug 了。"

    return {
        "jokes": [joke],
    }

#Send 是 LangGraph 的特殊机制——它不是简单地"跳到下一个节点"，而是对每个 Send 各启动一次目标节点，相当于把一个节点"扇出"成多个并行实例。
def continue_to_jokes(state: OverallState) -> list[Send]:
    """条件边函数：把 subjects 列表展开成多个 generate_joke 分支。

    Send("generate_joke", {"subject": subject}) 的意思是：
    对每个 subject，单独启动一次 generate_joke 节点，并传入这个局部 state。
    """
    return [
        Send("generate_joke", {"subject": subject})#
        for subject in state["subjects"]
    ]


def best_joke(state: OverallState) -> dict:
    """第三步：拿到所有 jokes 后，让模型选择最佳笑话。"""
    if not state["jokes"]:
        return {"best_selected_joke": ""}

    jokes = "\n\n".join(
        f"{index}. {joke}"
        for index, joke in enumerate(state["jokes"])
    )
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.invoke(prompt)
    data = parse_json_from_response(response.content)
    best_obj = BestJoke(**data)
    # 防止模型返回越界索引。
    selected_index = min(best_obj.id, len(state["jokes"]) - 1)

    return {
        "best_selected_joke": state["jokes"][selected_index],
    }


# 构建图：
# START -> generate_topics
# generate_topics -> 多个 generate_joke 分支
# 所有 generate_joke 分支的 jokes 聚合后 -> best_joke
# best_joke -> END
graph = StateGraph(OverallState)

graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)

graph.add_edge(START, "generate_topics")
#源节点generate_topics，路由函数continue_to_jokes用这个函数的返回值决定走哪，目标列表["generate_joke"]允许走到哪些节点


graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke(
        {
            "topic": "程序员",
            "subjects": [],
            "jokes": [],
            "best_selected_joke": "",
        }
    )

    print("subjects:")
    for subject in result["subjects"]:
        print("-", subject)

    print("\njokes:")
    for index, joke in enumerate(result["jokes"]):
        print(f"{index}. {joke}")

    print("\nbest_selected_joke:")
    print(result["best_selected_joke"])
