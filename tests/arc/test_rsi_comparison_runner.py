import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.arc.data import Task, Verifier
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_comparison_audit import replay_comparison
from src.arc.rsi_comparison_budget import ComparisonBudget
from src.arc.rsi_comparison_freeze import summarize_frozen_comparison
from src.arc.rsi_comparison_runner import run_comparison


class ComparisonRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [f'task-{i:03}' for i in range(200)]
        self.tasks = {name: Task(name, (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for name in self.ids}
        self.verifier = Verifier({name: [[[4, 3]]] for name in self.ids})
        self.seeds = {name: [10000 + 2*i, 10001 + 2*i] for i, name in enumerate(self.ids)}
        self.budget = ComparisonBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        self.generate = Mock(return_value=FAKE_PROGRAMS[1])

    def run_comparison(self):
        return run_comparison(root=self.root, registration_sha256='a' * 64, generation=0,
            task_ids=self.ids, tasks=self.tasks, seeds=self.seeds, parent_adapter='none',
            candidate_adapter='b' * 64, generate=self.generate, sandbox=FakeE6Sandbox(),
            verifier=self.verifier, budget=self.budget)

    def test_two_arms_are_matched_and_frozen(self):
        result = self.run_comparison()
        self.assertEqual(result['status'], 'frozen_unverified')
        report = summarize_frozen_comparison(self.root, expected_sha256=result['sha256'])
        self.assertEqual((report['base_solved'], report['candidate_solved']), (200, 200))
        self.assertEqual(report['budget_usage']['model_calls'], 800)
        self.assertGreater(report['budget_usage']['reserved_tokens'], 0)
        self.assertEqual(report['budget_usage']['sandbox_runs'], 1600)
        self.assertEqual(self.budget.snapshot()['model_calls'], 800)
        self.assertEqual(self.budget.snapshot()['sandbox_runs'], 1600)
        calls = self.generate.call_args_list
        first_task = [c for c in calls if c.kwargs['task_id'] == self.ids[0]]
        self.assertEqual([c.kwargs['seed'] for c in first_task], [10000, 10001, 10000, 10001])
        self.assertTrue(all(c.args[0] == first_task[0].args[0] for c in first_task))
        self.assertNotIn('[[4, 3]]', first_task[0].args[0])
        replay = replay_comparison(self.root, expected_sha256=result['sha256'], tasks=self.tasks,
                                   sandbox=FakeE6Sandbox(), verifier=self.verifier)
        self.assertEqual(replay['status'], 'comparison_replay_consistent')
        self.assertFalse(replay['independently_verified'])
        changed_tasks = dict(self.tasks)
        changed_tasks[self.ids[0]] = Task(self.ids[0], (([[5, 6]], [[6, 5]]),), ([[3, 4]],))
        with self.assertRaises(ValueError):
            replay_comparison(self.root, expected_sha256=result['sha256'], tasks=changed_tasks,
                              sandbox=FakeE6Sandbox(), verifier=self.verifier)
        record_path = self.root / 'comparison-000-parent.json'
        original = record_path.read_text()
        changed = json.loads(original)
        changed['score']['solved'] = False
        record_path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            replay_comparison(self.root, expected_sha256=result['sha256'], tasks=self.tasks,
                              sandbox=FakeE6Sandbox(), verifier=self.verifier)
        record_path.write_text(original)
        record_path.unlink()
        with self.assertRaises(ValueError):
            replay_comparison(self.root, expected_sha256=result['sha256'], tasks=self.tasks,
                              sandbox=FakeE6Sandbox(), verifier=self.verifier)
        with self.assertRaises(FileExistsError):
            self.run_comparison()

    def test_interruption_records_incomplete_and_cannot_restart(self):
        self.generate.side_effect = TimeoutError('unavailable')
        result = self.run_comparison()
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['improvement_claim_supported'])
        self.assertEqual(self.budget.snapshot()['model_calls'], 1)
        self.assertEqual(result['budget_usage']['model_calls'], 1)
        self.assertTrue((self.root / 'comparison.incomplete.json').exists())
        self.assertFalse((self.root / 'comparison.frozen.json').exists())
        with self.assertRaises(FileExistsError):
            self.run_comparison()

    def test_capture_ids_are_bound_to_saved_samples(self):
        self.generate.side_effect = [{'code': FAKE_PROGRAMS[1], 'request_id': '1' * 32},
                                     {'code': FAKE_PROGRAMS[1], 'request_id': '2' * 32},
                                     TimeoutError('fixture stop')]
        outcome = self.run_comparison()
        self.assertEqual(outcome['status'], 'incomplete')
        saved = json.loads((self.root / 'comparison-000-parent.json').read_text())
        self.assertEqual(saved['result']['sample_request_ids'], ['1' * 32, '2' * 32])

    def test_invalid_completion_is_recorded_and_next_sample_runs(self):
        self.generate.side_effect = [{'code': None, 'request_id': '1' * 32, 'failure': 'invalid_model_output'},
                                     {'code': FAKE_PROGRAMS[1], 'request_id': '2' * 32},
                                     TimeoutError('fixture stop')]
        outcome = self.run_comparison()
        self.assertEqual(outcome['status'], 'incomplete')
        saved = json.loads((self.root / 'comparison-000-parent.json').read_text())
        self.assertEqual(saved['result']['sample_statuses'], ['invalid_model_output', 'accepted'])
        self.assertEqual(saved['result']['sample_request_ids'], ['1' * 32, '2' * 32])
        self.assertTrue(saved['score']['solved'])

    def test_duplicate_request_id_stops_before_saving_task(self):
        self.generate.side_effect = [{'code': FAKE_PROGRAMS[1], 'request_id': '1' * 32}] * 2
        outcome = self.run_comparison()
        self.assertEqual(outcome['status'], 'incomplete')
        self.assertEqual(outcome['failure_type'], 'ValueError')
        self.assertFalse((self.root / 'comparison-000-parent.json').exists())

    def test_invalid_input_fails_before_dispatch(self):
        self.seeds[self.ids[0]] = [1, 1]
        with self.assertRaises(ValueError):
            self.run_comparison()
        self.generate.assert_not_called()
        self.assertFalse((self.root / 'comparison.started.json').exists())
