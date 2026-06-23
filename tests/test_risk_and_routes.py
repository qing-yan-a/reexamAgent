import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from personal_research_agent.graph import route_after_agent, tool_error_message
from personal_research_agent.tools import get_registered_tools, get_tool_call_risk, get_tool_risk, requires_approval


class RiskAndRouteTests(unittest.TestCase):
    def test_low_risk_tool_routes_to_tools(self):
        message = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "pyproject.toml"}, "id": "call_1"}])
        self.assertEqual(route_after_agent({"messages": [message], "tool_call_counts": {}}), "tools")

    def test_high_risk_tool_routes_to_human_approval(self):
        message = AIMessage(content="", tool_calls=[{"name": "write_text_file", "args": {"path": "x.md", "content": "x"}, "id": "call_1"}])
        self.assertEqual(route_after_agent({"messages": [message], "tool_call_counts": {}}), "human_approval")

    def test_web_search_is_not_budget_limited(self):
        message = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "call_1"}])
        self.assertEqual(route_after_agent({"messages": [message], "tool_call_counts": {"web_search": 99}}), "tools")

    def test_safe_run_command_routes_to_tools(self):
        message = AIMessage(
            content="",
            tool_calls=[{"name": "run_command", "args": {"command": ["python", "-m", "unittest", "tests.test_safety"]}, "id": "call_1"}],
        )

        self.assertEqual(route_after_agent({"messages": [message], "tool_call_counts": {}}), "tools")

    def test_risky_run_command_routes_to_human_approval(self):
        message = AIMessage(
            content="",
            tool_calls=[{"name": "run_command", "args": {"command": ["python", "check_pg.py"]}, "id": "call_1"}],
        )

        self.assertEqual(route_after_agent({"messages": [message], "tool_call_counts": {}}), "human_approval")

    def test_risk_registry(self):
        self.assertEqual(get_tool_risk("read_file"), "low")
        self.assertEqual(get_tool_risk("run_command"), "low")
        self.assertEqual(get_tool_call_risk("run_command", {"command": ["python", "check_pg.py"]}), "high")
        self.assertTrue(requires_approval("write_text_file"))

    def test_tool_node_returns_tool_message_on_file_error(self):
        read_tool = next(tool for tool in get_registered_tools() if tool.name == "read_file")
        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode([read_tool], handle_tool_errors=tool_error_message))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        message = AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "IDENTITY.md"}, "id": "call_missing"}],
        )

        result = graph.invoke({"messages": [message]})

        self.assertIn("工具执行失败：FileNotFoundError", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
