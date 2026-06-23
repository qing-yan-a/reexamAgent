import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_research_agent.cli import parse_source_selection
from personal_research_agent.source_selection import select_sources_for_session, select_sources_from_session, source_key


class SourceSelectionTests(unittest.TestCase):
    def sample_session(self):
        return {
            "candidate_sources": [
                {
                    "query_id": "q1",
                    "query_type": "experience",
                    "source_index": 1,
                    "title": "旧来源",
                    "url": "https://example.com/old",
                    "source": "example.com",
                },
                {
                    "query_id": "q2",
                    "query_type": "official_verification",
                    "source_index": 1,
                    "title": "官网方案",
                    "url": "https://school.edu.cn/reexam",
                    "source": "school.edu.cn",
                },
            ],
            "reviewed_sources": [
                {
                    "query_id": "q2",
                    "query_type": "official_verification",
                    "source_index": 1,
                    "title": "官网复试方案",
                    "url": "https://school.edu.cn/reexam",
                    "source": "school.edu.cn",
                    "relevance": "high",
                    "credibility_hint": "high",
                    "risk_flags": ["需要人工核验发布时间"],
                    "next_action": "keep",
                }
            ],
        }

    def test_select_sources_from_session_uses_latest_index_and_review_fields(self):
        result = select_sources_from_session(self.sample_session(), source_indexes=[1], selection_method="test")

        self.assertEqual(result["selected_count"], 1)
        selected = result["selected_sources"][0]
        self.assertEqual(selected["title"], "官网复试方案")
        self.assertEqual(selected["url"], "https://school.edu.cn/reexam")
        self.assertEqual(selected["next_action"], "keep")
        self.assertEqual(selected["selection_method"], "test")

    def test_select_sources_from_session_accepts_source_key_and_url(self):
        session = self.sample_session()
        key = source_key(session["candidate_sources"][0])

        result = select_sources_from_session(
            session,
            source_keys=[key],
            urls=["https://school.edu.cn/reexam"],
            selection_method="mixed",
        )

        self.assertEqual(result["selected_count"], 2)
        self.assertEqual([item["url"] for item in result["selected_sources"]], ["https://example.com/old", "https://school.edu.cn/reexam"])

    def test_select_sources_for_session_writes_selected_sources(self):
        saved = {}

        def fake_write(session_id, data):
            saved["session_id"] = session_id
            saved["data"] = data

        with patch("personal_research_agent.source_selection.read_research_session", return_value=self.sample_session()), patch(
            "personal_research_agent.source_selection.write_research_session", side_effect=fake_write
        ):
            result = select_sources_for_session("s1", source_indexes=[1], selection_method="unit")

        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(saved["session_id"], "s1")
        self.assertEqual(saved["data"]["selected_sources"][0]["selection_method"], "unit")

    def test_parse_source_selection_cli_args(self):
        result = parse_source_selection("source_index=1,2 https://example.com/a q1|3|https://example.com/b")

        self.assertEqual(result["source_indexes"], [1, 2])
        self.assertEqual(result["urls"], ["https://example.com/a"])
        self.assertEqual(result["source_keys"], ["q1|3|https://example.com/b"])


if __name__ == "__main__":
    unittest.main()
