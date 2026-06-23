import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_research_agent.cli import open_runtime


class RuntimeTests(unittest.TestCase):
    def test_open_runtime_does_not_swallow_body_errors_after_postgres_opens(self):
        @contextmanager
        def fake_postgres_runtime():
            yield object(), object()

        with patch("personal_research_agent.cli.open_postgres_runtime", fake_postgres_runtime):
            with self.assertRaisesRegex(ValueError, "graph failed"):
                with open_runtime(use_postgres=True):
                    raise ValueError("graph failed")

    def test_open_runtime_falls_back_when_postgres_enter_fails(self):
        @contextmanager
        def fake_postgres_runtime():
            raise RuntimeError("postgres down")
            yield

        with patch("personal_research_agent.cli.open_postgres_runtime", fake_postgres_runtime):
            with open_runtime(use_postgres=True) as (_checkpointer, _store, runtime_name):
                self.assertEqual(runtime_name, "memory")


if __name__ == "__main__":
    unittest.main()
