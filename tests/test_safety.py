import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_research_agent.tools.command_tools import validate_command
from personal_research_agent.tools.file_tools import ensure_safe_path, resolve_workspace_path


class SafetyTests(unittest.TestCase):
    def test_file_paths_reject_absolute_paths(self):
        with self.assertRaises(ValueError):
            resolve_workspace_path(r"C:\Windows\system32\drivers\etc\hosts")

    def test_file_paths_reject_parent_escape(self):
        with self.assertRaises(ValueError):
            resolve_workspace_path("../outside.txt")

    def test_file_paths_reject_sensitive_paths_after_resolve(self):
        with self.assertRaises(ValueError):
            ensure_safe_path(resolve_workspace_path(".env"))

    def test_command_rejects_shell_strings(self):
        with self.assertRaises(ValueError):
            validate_command("python test.py")  # type: ignore[arg-type]

    def test_command_rejects_non_whitelisted_executable(self):
        with self.assertRaises(ValueError):
            validate_command(["powershell", "-Command", "Get-ChildItem"])

    def test_command_allows_py_compile_shape_for_existing_file(self):
        validate_command(["python", "-m", "py_compile", "src/personal_research_agent/__init__.py"])


if __name__ == "__main__":
    unittest.main()
