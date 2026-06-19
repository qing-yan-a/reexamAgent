import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage

from personal_research_agent.memory import (
    KEEP_FIRST_MESSAGES,
    KEEP_RECENT_MESSAGES,
    build_summary_updates,
    compact_long_term_memory,
    retrieve_long_term_memories,
    should_save_long_term_memory,
)


class FakeItem:
    def __init__(self, key, value, created_at):
        self.key = key
        self.value = value
        self.created_at = created_at
        self.updated_at = created_at
        self.score = 0.9


class FakeStore:
    def __init__(self):
        self.items = {}

    def put(self, namespace, key, value):
        self.items[(namespace, key)] = FakeItem(key, value, datetime.now(UTC))

    def get(self, namespace, key):
        return self.items.get((namespace, key))

    def delete(self, namespace, key):
        self.items.pop((namespace, key), None)

    def search(self, namespace, query=None, filter=None, limit=10):
        results = [item for (item_namespace, _), item in self.items.items() if item_namespace == namespace]
        if filter:
            results = [item for item in results if all(item.value.get(name) == value for name, value in filter.items())]
        return results[:limit]


class FakeModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        return type("FakeResponse", (), {"content": self.content})()


class MemoryTests(unittest.TestCase):
    def test_low_signal_messages_are_not_saved_to_long_term_memory(self):
        with patch("personal_research_agent.memory.build_model", return_value=FakeModel('{"save": false, "reason": "寒暄"}')):
            self.assertFalse(should_save_long_term_memory(HumanMessage(content="你好")))
        with patch("personal_research_agent.memory.build_model", return_value=FakeModel('{"save": true, "reason": "稳定偏好"}')):
            self.assertTrue(should_save_long_term_memory(HumanMessage(content="我长期偏好把复试资料按学校和专业归档")))

    def test_slash_commands_are_not_sent_to_memory_judge(self):
        with patch("personal_research_agent.memory.build_model") as build_model:
            self.assertFalse(should_save_long_term_memory(HumanMessage(content="/exit")))
        build_model.assert_not_called()

    def test_summary_compression_keeps_first_six_and_last_twenty(self):
        messages = [HumanMessage(content=f"message {i}", id=f"m{i}") for i in range(40)]
        with patch("personal_research_agent.memory.load_current_summary", return_value=""), patch(
            "personal_research_agent.memory.save_current_summary"
        ), patch("personal_research_agent.memory.summarize_messages", return_value="summary"):
            updates = build_summary_updates(messages)
        removals = updates["messages"]
        removed_ids = {item.id for item in removals}
        expected_removed = {f"m{i}" for i in range(KEEP_FIRST_MESSAGES, 40 - KEEP_RECENT_MESSAGES)}
        self.assertEqual(removed_ids, expected_removed)
        self.assertEqual(updates["working_summary"], "summary")

    def test_store_compaction_creates_summary_and_deletes_old_facts(self):
        store = FakeStore()
        namespace = ("memories", "u1")
        base = datetime.now(UTC) - timedelta(days=2)
        for index in range(6):
            item = FakeItem(
                f"k{index}",
                {"type": "conversation_fact", "text": f"长期事实 {index}", "source_thread": "t1"},
                base + timedelta(minutes=index),
            )
            store.items[(namespace, item.key)] = item

        result = compact_long_term_memory(
            store,
            "u1",
            threshold=4,
            keep_recent=2,
            summarizer=lambda old, facts, model_name: old + "\n" + "\n".join(facts),
        )

        self.assertTrue(result["compacted"])
        self.assertIsNotNone(store.get(namespace, "memory_summary"))
        remaining_facts = store.search(namespace, filter={"type": "conversation_fact"}, limit=10)
        self.assertEqual({item.key for item in remaining_facts}, {"k4", "k5"})

    def test_long_term_retrieval_prefers_summary_then_relevant_facts(self):
        store = FakeStore()
        namespace = ("memories", "u1")
        store.put(namespace, "memory_summary", {"type": "memory_summary", "text": "用户偏好按学校整理资料"})
        store.put(namespace, "fact1", {"type": "conversation_fact", "text": "用户喜欢西瓜", "source_thread": "t1"})

        memories = retrieve_long_term_memories(store, "u1", "学校资料", limit=2)

        self.assertEqual(memories[0]["type"], "memory_summary")
        self.assertEqual(memories[1]["type"], "conversation_fact")


if __name__ == "__main__":
    unittest.main()
