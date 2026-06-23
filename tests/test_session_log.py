import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage, ToolMessage

from personal_research_agent import session_log
from personal_research_agent.graph import record_tool_usage_node


class SessionLogTests(unittest.TestCase):
    def test_append_session_log_writes_jsonl(self):
        with TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "sessions" / "session-test.jsonl"

            with patch.object(session_log, "SESSION_DIR", log_file.parent), patch.object(session_log, "SESSION_FILE", log_file):
                session_log.append_session_log("user_message", {"content": "你好", "value": object()})

            lines = log_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            event = json.loads(lines[0])
            self.assertEqual(event["type"], "user_message")
            self.assertEqual(event["data"]["content"], "你好")
            self.assertIn("time", event)

    def test_record_tool_usage_logs_tool_results_after_last_ai(self):
        calls = []
        state = {
            "thread_id": "s1",
            "user_id": "u1",
            "profile_name": "p1",
            "messages": [
                ToolMessage(content="old result", name="old_tool", tool_call_id="old"),
                AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}]),
                ToolMessage(content='{"ok": true}', name="web_search", tool_call_id="call_1"),
            ],
        }

        with patch("personal_research_agent.graph.append_session_log", side_effect=lambda event_type, data: calls.append((event_type, data))):
            result = record_tool_usage_node(state)

        self.assertEqual(result["tool_call_counts"], {"web_search": 1})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "tool_result")
        self.assertEqual(calls[0][1]["tool_name"], "web_search")
        self.assertEqual(calls[0][1]["thread_id"], "s1")
        self.assertIn('"ok": true', calls[0][1]["content"])


if __name__ == "__main__":
    unittest.main()
