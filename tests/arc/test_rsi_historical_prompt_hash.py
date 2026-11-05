"""Construct historical prompt hashes for bounded exposure review."""
import hashlib
import unittest

from src.arc.agent import _code_analysis_prompt, _prompt_json
from src.arc.data import Task
from src.arc.rsi_historical_prompt_hash import historical_prompt_hashes


class HistoricalPromptHashTests(unittest.TestCase):
    def test_hashes_match_historical_requests(self):
        task = Task('example', (([[1]], [[2]]),), ([[3]],))
        direct = _prompt_json({'instruction': 'Return a JSON array of output grids, one per test input.', 'task': task.prompt_data()})
        analysis = _code_analysis_prompt(task)
        expected = {f'example/{name}': hashlib.sha256(value.encode()).hexdigest()
                    for name, value in [('direct', direct), ('analysis', analysis)]}
        self.assertEqual(historical_prompt_hashes([task]), expected)

    def test_reject_duplicate_or_empty_tasks(self):
        task = Task('example', (([[1]], [[2]]),), ([[3]],))
        for tasks in ([], [task, task], [object()]):
            with self.assertRaises(ValueError):
                historical_prompt_hashes(tasks)
