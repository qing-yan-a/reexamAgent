import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage

from personal_research_agent.graph import route_after_agent
from personal_research_agent.tools import get_tool_risk, requires_approval


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

    def test_risk_registry(self):
        self.assertEqual(get_tool_risk("read_file"), "low")
        self.assertTrue(requires_approval("write_text_file"))


if __name__ == "__main__":
    unittest.main()
