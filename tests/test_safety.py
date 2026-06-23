import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_research_agent.tools.command_tools import classify_command, validate_command
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

    def test_command_rejects_shell_executable(self):
        with self.assertRaises(ValueError):
            validate_command(["powershell", "-Command", "Get-ChildItem"])

    def test_command_allows_py_compile_shape_for_existing_file(self):
        validate_command(["python", "-m", "py_compile", "src/personal_research_agent/__init__.py"])

    def test_command_allows_common_read_only_and_verification_commands(self):
        self.assertEqual(classify_command(["rg", "route_after_agent", "src"]), "allow")
        self.assertEqual(classify_command(["git", "status", "--short"]), "allow")
        self.assertEqual(classify_command(["python", "-m", "unittest", "discover", "-s", "tests"]), "allow")

    def test_command_asks_for_scripts_and_project_runners(self):
        self.assertEqual(classify_command(["python", "check_pg.py"]), "ask")
        self.assertEqual(classify_command(["python", "-m", "pytest", "tests"]), "ask")
        self.assertEqual(classify_command(["npm", "run", "build"]), "ask")
        self.assertEqual(classify_command(["pip", "freeze"]), "ask")

    def test_command_asks_for_unknown_development_commands(self):
        self.assertEqual(classify_command(["javac", "src/Main.java"]), "ask")
        self.assertEqual(classify_command(["java", "Main"]), "ask")
        self.assertEqual(classify_command(["gcc", "src/main.c", "-o", "build/main.exe"]), "ask")
        self.assertEqual(classify_command(["make", "test"]), "ask")

    def test_command_rejects_destructive_or_install_commands(self):
        for command in [
            ["git", "reset", "--hard"],
            ["git", "clean", "-fd"],
            ["git", "push", "--force"],
            ["npm", "install"],
            ["python", "-m", "pip", "install", "requests"],
            ["rg", "SECRET", "--no-ignore", ".env"],
            ["python", "-c", "print(1)"],
        ]:
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    validate_command(command)


if __name__ == "__main__":
    unittest.main()
