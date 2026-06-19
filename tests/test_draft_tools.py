import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_research_agent.tools.draft_tools import draft_markdown


class DraftToolsTests(unittest.TestCase):
    def test_draft_markdown_saves_to_test_output_dir(self):
        with TemporaryDirectory() as temp:
            project_root = Path(temp)
            session = {
                "school": "昆明理工大学",
                "major": "计算机",
                "output_dir": "test/昆明理工大学计算机",
                "reviewed_sources": [{"title": "复试真题", "review_note": "真题 复试经验 流程"}],
                "selected_sources": [{"title": "复试经验", "note": "面试题 复试流程"}],
            }
            with patch("personal_research_agent.tools.draft_tools.PROJECT_ROOT", project_root), patch(
                "personal_research_agent.tools.draft_tools.load_session", return_value=session
            ), patch(
                "personal_research_agent.tools.draft_tools.analyze_research_readiness",
                return_value={"draft_ready": True, "open_gaps": []},
            ):
                result = draft_markdown.invoke({"title": "昆明理工大学计算机复试资料", "content": "正文", "filename": "草稿.md"})

            self.assertEqual(result["path"], "test/昆明理工大学计算机/草稿.md")
            self.assertTrue((project_root / result["path"]).exists())


if __name__ == "__main__":
    unittest.main()
