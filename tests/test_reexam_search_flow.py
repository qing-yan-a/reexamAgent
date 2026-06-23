import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_research_agent.graph import (
    route_after_reexam_decision,
    route_after_reexam_gaps,
    route_after_reexam_parse,
    route_after_reexam_route_confirmation,
    route_after_reexam_router,
)
from personal_research_agent.research_outputs import research_output_dir_name
from personal_research_agent.reexam_search_flow import (
    build_gap_query,
    is_reexam_post_search_action,
    next_gap_query,
    normalize_search_decision,
    parse_reexam_goal_text,
    parse_reexam_router_output,
    record_search_iteration,
    reexam_goal_from_router,
    ensure_reexam_session,
)


class ReexamSearchFlowTests(unittest.TestCase):
    def test_parse_reexam_search_goal(self):
        goal = parse_reexam_goal_text("帮我搜索昆明理工大学计算机复试资料")

        self.assertTrue(goal["is_reexam_search"])
        self.assertEqual(goal["school"], "昆明理工大学")
        self.assertEqual(goal["major"], "计算机")
        self.assertEqual(goal["year"], "latest")

    def test_non_reexam_message_routes_to_normal_agent(self):
        self.assertEqual(route_after_reexam_parse({"is_reexam_search": False}), "agent")
        self.assertEqual(route_after_reexam_parse({"is_reexam_search": True}), "ensure_reexam_session")

    def test_post_search_actions_do_not_enter_search_router(self):
        self.assertTrue(is_reexam_post_search_action("抽取已选来源正文并生成草稿"))
        self.assertTrue(is_reexam_post_search_action("把保留来源整理成文档"))
        self.assertFalse(is_reexam_post_search_action("帮我搜索河南农业大学人工智能复试资料"))

    def test_llm_router_routes_to_confirmation_before_search_flow(self):
        self.assertEqual(route_after_reexam_router({"reexam_route_action": "normal"}), "agent")
        self.assertEqual(route_after_reexam_router({"reexam_route_action": "clarify"}), "summarize_if_needed")
        self.assertEqual(route_after_reexam_router({"reexam_route_action": "confirm"}), "confirm_reexam_route")
        self.assertEqual(route_after_reexam_route_confirmation({"reexam_route_confirmed": True}), "ensure_reexam_session")
        self.assertEqual(route_after_reexam_route_confirmation({"reexam_route_confirmed": False}), "summarize_if_needed")

    def test_router_goal_can_continue_from_active_session(self):
        router = {
            "intent": "reexam_search",
            "action": "continue_search",
            "school": "",
            "major": "",
            "year": "latest",
            "research_goal": "",
        }
        session = {
            "vertical": "postgraduate_reexam",
            "school": "河南农业大学",
            "major": "人工智能",
            "year": "2026",
        }

        goal = reexam_goal_from_router("搜索资料吧", router, session)

        self.assertTrue(goal["is_reexam_search"])
        self.assertEqual(goal["school"], "河南农业大学")
        self.assertEqual(goal["major"], "人工智能")
        self.assertEqual(goal["year"], "2026")

    def test_router_output_parser_accepts_json_inside_text(self):
        parsed = parse_reexam_router_output('```json\n{"intent":"normal_chat","action":"normal"}\n```')

        self.assertEqual(parsed["intent"], "normal_chat")

    def test_missing_past_questions_generates_past_question_query_first(self):
        session = {"school": "昆明理工大学", "major": "计算机", "year": "latest", "search_queries": []}
        query = next_gap_query(session, ["历年复试真题或题型线索不足"])

        self.assertEqual(query["query_type"], "past_questions")
        self.assertIn("真题", query["query"])
        self.assertIn("机试", query["query"])

    def test_gap_query_uses_year_when_present(self):
        session = {"school": "浙江工业大学", "major": "软件", "year": "2026"}

        query = build_gap_query(session, "official_verification")

        self.assertIn("2026", query)
        self.assertIn("复试方案", query)

    def test_user_decision_routes_loop_or_exit(self):
        self.assertEqual(normalize_search_decision("继续补搜"), "continue")
        self.assertEqual(normalize_search_decision("可以下一步"), "next")
        self.assertEqual(normalize_search_decision("停止"), "stop")
        self.assertEqual(route_after_reexam_decision({"reexam_next_action": "continue"}), "generate_gap_queries")
        self.assertEqual(route_after_reexam_decision({"reexam_next_action": "next"}), "source_confirmation")
        self.assertEqual(route_after_reexam_decision({"reexam_next_action": "stop"}), "stop_reexam_search")

    def test_pending_queries_do_not_skip_search_iteration(self):
        state = {
            "reexam_iteration_complete": False,
            "reexam_session": {
                "search_queries": [
                    {
                        "query_id": "experience_1",
                        "query": "河南农业大学 人工智能 2026 复试经验",
                        "query_type": "experience",
                        "status": "pending",
                    }
                ],
                "candidate_sources": [],
                "reviewed_sources": [],
            },
        }

        self.assertEqual(route_after_reexam_gaps(state), "generate_gap_queries")

    def test_record_search_iteration_appends_sources_and_marks_query_done(self):
        saved = {}
        session = {
            "search_queries": [{"query_id": "past_questions_1", "query": "q", "query_type": "past_questions", "status": "pending", "notes": ""}],
            "candidate_sources": [],
            "reviewed_sources": [],
            "notes": [],
        }

        def fake_write(session_id, data):
            saved["session_id"] = session_id
            saved["data"] = data

        with patch("personal_research_agent.reexam_search_flow.require_active_session_id", return_value="s1"), patch(
            "personal_research_agent.reexam_search_flow.read_research_session", return_value=session
        ), patch("personal_research_agent.reexam_search_flow.write_research_session", side_effect=fake_write):
            result = record_search_iteration(
                {"query_id": "past_questions_1", "query": "q", "query_type": "past_questions", "status": "pending", "notes": ""},
                {"results": [{"source_index": 0, "title": "复试真题", "url": "https://example.com", "source": "example.com"}]},
                {"reviews": [{"source_index": 0, "title": "复试真题", "url": "https://example.com", "next_action": "keep"}]},
            )

        self.assertEqual(result["status"], "done")
        self.assertEqual(saved["data"]["search_queries"][0]["status"], "done")
        self.assertEqual(len(saved["data"]["candidate_sources"]), 1)
        self.assertEqual(len(saved["data"]["reviewed_sources"]), 1)

    def test_reexam_session_records_test_output_dir(self):
        saved = {}
        session = {"research_goal": "", "notes": []}

        def fake_write(session_id, data):
            saved["session_id"] = session_id
            saved["data"] = data

        with patch("personal_research_agent.reexam_search_flow.require_active_session_id", return_value="s1"), patch(
            "personal_research_agent.reexam_search_flow.read_research_session", return_value=session
        ), patch("personal_research_agent.reexam_search_flow.write_research_session", side_effect=fake_write), patch(
            "personal_research_agent.reexam_search_flow.research_output_dir"
        ) as output_dir:
            output_dir.return_value = Path("E:/study/LangGraph/test/昆明理工大学计算机")
            result = ensure_reexam_session(
                {
                    "research_goal": "整理昆明理工大学计算机复试资料",
                    "school": "昆明理工大学",
                    "major": "计算机",
                    "year": "latest",
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(saved["data"]["output_dir"], "test/昆明理工大学计算机")
        self.assertIn("资料输出目录：test/昆明理工大学计算机", saved["data"]["notes"])

    def test_output_dir_name_uses_school_and_major(self):
        self.assertEqual(research_output_dir_name("A大学", "B专业"), "A大学B专业")


if __name__ == "__main__":
    unittest.main()
