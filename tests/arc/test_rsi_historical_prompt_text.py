"""Reconstruct the task block used by historical code requests."""
import unittest

from src.arc.agent import _code_task_text
from src.arc.data import Task
from src.arc.rsi_historical_prompt_text import historical_task_texts


class HistoricalTaskTextTests(unittest.TestCase):
    def test_task_block_matches_code_agent(self):
        task = Task('a', (([[1, 2]], [[3, 4]]),), ([[5, 6]],))
        self.assertEqual(historical_task_texts([task]), {'a': _code_task_text(task)})

    def test_rejects_invalid_and_duplicate_tasks(self):
        task = Task('a', (([[1]], [[2]]),), ([[3]],))
        for tasks in ([], [task, task], [object()], [Task('b', (), ([[1]],))]):
            with self.assertRaises(ValueError):
                historical_task_texts(tasks)
