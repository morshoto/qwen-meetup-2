"""Construct historical prompt hashes for bounded exposure review."""
import hashlib
import json
import unittest

from src.arc.agent import _code_analysis_prompt, _prompt_json
from src.arc.data import Task
from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_historical_prompt_hash import historical_prompt_hashes


class HistoricalPromptHashTests(unittest.TestCase):
    def test_hashes_match_historical_requests(self):
        task = Task('example', (([[1]], [[2]]),), ([[3]],))
        direct = _prompt_json({'instruction': 'Return a JSON array of output grids, one per test input.', 'task': task.prompt_data()})
        analysis = _code_analysis_prompt(task)
        expected = {f'example/{name}': hashlib.sha256(value.encode()).hexdigest()
                    for name, value in [('direct', direct), ('analysis', analysis)]}
        for kind, text in [('direct', direct), ('analysis', analysis)]:
            for mode in (False, True):
                messages = [{'role': 'system', 'content': system_prompt_for_kind(kind, analysis_mode=mode)},
                            {'role': 'user', 'content': text}]
                expected[f'example/{kind}/wire_{mode}'] = hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.assertEqual(historical_prompt_hashes([task]), expected)

    def test_reject_duplicate_or_empty_tasks(self):
        task = Task('example', (([[1]], [[2]]),), ([[3]],))
        for tasks in ([], [task, task], [object()]):
            with self.assertRaises(ValueError):
                historical_prompt_hashes(tasks)
