import sys
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from personal_research_agent.api.app import app, send_history, iter_tool_display_messages_from_update, serialize_chat_messages, stream_graph_result
from personal_research_agent.api.panel import build_research_panel, folder_summary_from_sessions


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class StreamingGraph:
    def __init__(self, events):
        self.events = events

    def stream(self, _graph_input, config=None, stream_mode=None):
        yield from self.events


class HistoryGraph:
    def __init__(self, messages):
        self.messages = messages

    def get_state(self, _config):
        return type("State", (), {"values": {"messages": self.messages}})()


class ApiTests(unittest.TestCase):
    def test_health_endpoint(self):
        client = TestClient(app)
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_index_serves_vue_workspace(self):
        client = TestClient(app)
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("reexamAgent", response.text)
        self.assertIn("/static/app.js", response.text)

    def test_research_panel_shape_for_missing_session(self):
        panel = build_research_panel("missing-session-for-test")

        self.assertEqual(panel["session_id"], "missing-session-for-test")
        self.assertIn("current_task", panel)
        self.assertIn("candidate_sources", panel)
        self.assertIn("draft_status", panel)

    def test_folder_summary_scans_workspace_and_groups_sessions(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            folder = workspace / "test" / "A大学B专业"
            empty_folder = workspace / "test" / "空资料夹"
            ignored_folder = workspace / "test" / "__pycache__"
            folder.mkdir(parents=True, exist_ok=True)
            empty_folder.mkdir(parents=True, exist_ok=True)
            ignored_folder.mkdir(parents=True, exist_ok=True)
            sessions = [
                {
                    "session_id": "s1",
                    "title": "X 会话",
                    "output_dir": "test/A大学B专业",
                    "school": "A大学",
                    "major": "B专业",
                    "updated_at": "2026-01-02T00:00:00+00:00",
                },
                {
                    "session_id": "s2",
                    "title": "未归档",
                    "output_dir": "",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
            ]

            with patch("personal_research_agent.api.panel.PROJECT_ROOT", workspace):
                folders = folder_summary_from_sessions(sessions)

        by_name = {item["name"]: item for item in folders}
        self.assertIn("A大学B专业", by_name)
        self.assertIn("空资料夹", by_name)
        self.assertIn("未归档 Session", by_name)
        self.assertNotIn("__pycache__", by_name)
        self.assertEqual(by_name["A大学B专业"]["session_count"], 1)
        self.assertEqual(by_name["A大学B专业"]["sessions"][0]["session_id"], "s1")
        self.assertEqual(by_name["空资料夹"]["session_count"], 0)

    def test_select_sources_writes_selected_sources(self):
        client = TestClient(app)
        session = {
            "candidate_sources": [
                {
                    "query_id": "q1",
                    "source_index": 1,
                    "url": "https://example.com/a",
                    "title": "候选来源",
                    "source": "example.com",
                    "query_type": "experience",
                }
            ],
            "reviewed_sources": [
                {
                    "query_id": "q1",
                    "source_index": 1,
                    "url": "https://example.com/a",
                    "title": "候选来源",
                    "source": "example.com",
                    "relevance": "high",
                    "credibility_hint": "medium",
                    "risk_flags": ["需要人工核验"],
                    "next_action": "keep",
                }
            ],
        }
        saved = {}

        def fake_write(session_id, data):
            saved["session_id"] = session_id
            saved["data"] = data

        with patch("personal_research_agent.api.app.ensure_session_exists"), patch(
            "personal_research_agent.source_selection.read_research_session", return_value=session
        ), patch("personal_research_agent.source_selection.write_research_session", side_effect=fake_write):
            response = client.post(
                "/sessions/s1/selected-sources",
                json={"source_keys": ["q1|1|https://example.com/a"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(saved["session_id"], "s1")
        self.assertEqual(saved["data"]["selected_sources"][0]["url"], "https://example.com/a")
        self.assertEqual(saved["data"]["selected_sources"][0]["next_action"], "keep")

    def test_delete_session_removes_checkpoint_before_directory(self):
        client = TestClient(app)
        with TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "s1"
            session_path.mkdir()
            calls = []

            def fake_delete_checkpoints(session_id):
                calls.append(("checkpoint", session_id, session_path.exists()))

            with patch("personal_research_agent.api.app.session_dir", return_value=session_path), patch(
                "personal_research_agent.api.app.delete_thread_checkpoints", side_effect=fake_delete_checkpoints
            ):
                response = client.delete("/sessions/s1")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(calls, [("checkpoint", "s1", True)])
            self.assertFalse(session_path.exists())
            self.assertTrue(response.json()["checkpoint_deleted"])

    def test_delete_session_keeps_directory_when_checkpoint_cleanup_fails(self):
        client = TestClient(app)
        with TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "s1"
            session_path.mkdir()

            with patch("personal_research_agent.api.app.session_dir", return_value=session_path), patch(
                "personal_research_agent.api.app.delete_thread_checkpoints", side_effect=RuntimeError("db down")
            ):
                response = client.delete("/sessions/s1")

            self.assertEqual(response.status_code, 500)
            self.assertTrue(session_path.exists())
            self.assertIn("checkpoint 清理失败", response.text)

    def test_stream_graph_result_sends_token_deltas(self):
        websocket = FakeWebSocket()
        graph = StreamingGraph(
            [
                ("messages", (AIMessageChunk(content="你"), {"langgraph_node": "agent"})),
                ("messages", (AIMessageChunk(content="好"), {"langgraph_node": "agent"})),
            ]
        )

        with patch("personal_research_agent.api.app.send_panel", new=AsyncMock()):
            asyncio.run(stream_graph_result(websocket, "s1", graph, {}, {"configurable": {"thread_id": "s1"}}))

        self.assertEqual(websocket.sent[0]["type"], "message_start")
        self.assertEqual([item.get("delta") for item in websocket.sent if item["type"] == "message_delta"], ["你", "好"])
        self.assertEqual(websocket.sent[-1]["type"], "message_end")

    def test_stream_graph_result_ignores_internal_llm_tokens(self):
        websocket = FakeWebSocket()
        graph = StreamingGraph(
            [
                ("messages", (AIMessageChunk(content="内部摘要"), {"langgraph_node": "summarize_if_needed"})),
                ("messages", (AIMessageChunk(content='{"save": false}'), {"langgraph_node": "save_long_term_memory"})),
                ("messages", (AIMessageChunk(content="用户可见"), {"langgraph_node": "agent"})),
            ]
        )

        with patch("personal_research_agent.api.app.send_panel", new=AsyncMock()):
            asyncio.run(stream_graph_result(websocket, "s1", graph, {}, {"configurable": {"thread_id": "s1"}}))

        self.assertEqual([item.get("delta") for item in websocket.sent if item["type"] == "message_delta"], ["用户可见"])

    def test_stream_graph_result_falls_back_to_update_message(self):
        websocket = FakeWebSocket()
        graph = StreamingGraph(
            [
                ("updates", {"source_confirmation": {"messages": [AIMessage(content="请选择来源")]} }),
            ]
        )

        with patch("personal_research_agent.api.app.send_panel", new=AsyncMock()):
            asyncio.run(stream_graph_result(websocket, "s1", graph, {}, {"configurable": {"thread_id": "s1"}}))

        self.assertEqual([item.get("delta") for item in websocket.sent if item["type"] == "message_delta"], ["请选择来源"])

    def test_stream_graph_result_sends_tool_messages_from_updates(self):
        websocket = FakeWebSocket()
        graph = StreamingGraph(
            [
                (
                    "updates",
                    {
                        "agent": {
                            "messages": [
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "id": "call_1",
                                            "name": "evaluate_research_readiness",
                                            "args": {"persist": True},
                                        }
                                    ],
                                    id="ai_tool",
                                )
                            ]
                        }
                    },
                ),
                (
                    "updates",
                    {
                        "tools": {
                            "messages": [
                                ToolMessage(
                                    content='{"draft_ready": false}',
                                    name="evaluate_research_readiness",
                                    tool_call_id="call_1",
                                    id="tool_result",
                                )
                            ]
                        }
                    },
                ),
            ]
        )

        with patch("personal_research_agent.api.app.send_panel", new=AsyncMock()):
            asyncio.run(stream_graph_result(websocket, "s1", graph, {}, {"configurable": {"thread_id": "s1"}}))

        deltas = [item.get("delta", "") for item in websocket.sent if item["type"] == "message_delta"]
        self.assertTrue(any("模型请求工具调用" in item for item in deltas))
        self.assertTrue(any('"draft_ready": false' in item for item in deltas))

    def test_tool_display_messages_are_extracted_from_updates(self):
        update = {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "name": "evaluate_research_readiness",
                                "args": {"persist": True},
                            }
                        ],
                    )
                ]
            },
            "tools": {
                "messages": [
                    ToolMessage(
                        content='{"draft_ready": false}',
                        tool_call_id="call_1",
                        name="evaluate_research_readiness",
                    )
                ]
            },
        }

        items = list(iter_tool_display_messages_from_update(update))

        self.assertEqual(len(items), 2)
        self.assertIn("evaluate_research_readiness", items[0]["content"])
        self.assertIn('"draft_ready": false', items[1]["content"])

    def test_serialize_chat_messages_for_history(self):
        messages = [
            HumanMessage(content="你好", id="h1"),
            AIMessage(content="你好，我在", id="a1"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "x.md"}, "id": "call_1"}], id="a2"),
            ToolMessage(content="文件不存在", name="read_file", tool_call_id="call_1", id="t1"),
        ]

        result = serialize_chat_messages(messages)

        self.assertEqual([item["role"] for item in result], ["user", "assistant", "assistant", "tool"])
        self.assertIn("read_file", result[2]["content"])
        self.assertIn("文件不存在", result[3]["content"])

    def test_send_history_emits_history_reset(self):
        websocket = FakeWebSocket()
        graph = HistoryGraph([HumanMessage(content="历史问题", id="h1"), AIMessage(content="历史回答", id="a1")])

        asyncio.run(send_history(websocket, graph, {"configurable": {"thread_id": "s1"}}))

        self.assertEqual(websocket.sent[0]["type"], "history_reset")
        self.assertEqual([item["content"] for item in websocket.sent[0]["messages"]], ["历史问题", "历史回答"])


if __name__ == "__main__":
    unittest.main()
