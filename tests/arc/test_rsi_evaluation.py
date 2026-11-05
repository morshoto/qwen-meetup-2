import unittest
from unittest.mock import Mock

from src.arc.data import Task, Verifier
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_evaluation import InvalidModelOutput, evaluate_task, score_task


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.task = Task('a', (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        self.verifier = Verifier({'a': [[[4, 3]]]})
        self.generate = Mock(side_effect=[FAKE_PROGRAMS[2], FAKE_PROGRAMS[1]])

    def test_independent_samples_use_identical_label_free_prompt(self):
        result = evaluate_task(self.task, generate=self.generate, sandbox=FakeE6Sandbox())
        self.assertEqual(result['sample_statuses'], ['visible_demonstration_mismatch', 'accepted'])
        self.assertEqual(result['sample_sources'], [FAKE_PROGRAMS[2], FAKE_PROGRAMS[1]])
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.generate.call_args_list[0].args[0], self.generate.call_args_list[1].args[0])
        self.assertEqual([c.kwargs['sample'] for c in self.generate.call_args_list], [0, 1])
        self.assertNotIn('[[4, 3]]', self.generate.call_args_list[0].args[0])
        self.assertEqual(result['predictions'], [[[[4, 3]]]])
        self.assertTrue(score_task(self.task, result, self.verifier)['solved'])

    def test_invalid_model_completion_counts_as_failed_sample(self):
        result = evaluate_task(self.task, generate=Mock(side_effect=[InvalidModelOutput(), FAKE_PROGRAMS[1]]),
                               sandbox=FakeE6Sandbox())
        self.assertEqual(result['sample_statuses'], ['invalid_model_output', 'accepted'])
        self.assertEqual(result['sample_sources'][0], None)
        self.assertTrue(score_task(self.task, result, self.verifier)['solved'])

    def test_static_and_invalid_outputs_are_failed_attempts(self):
        result = evaluate_task(self.task, generate=Mock(side_effect=['import os', FAKE_PROGRAMS[1]]),
                               sandbox=Mock(run=Mock(return_value=[[11]])))
        self.assertEqual(result['sample_statuses'], ['static_rejected', 'invalid_output'])
        self.assertFalse(score_task(self.task, result, self.verifier)['solved'])

    def test_infrastructure_failure_stops_instead_of_scoring(self):
        with self.assertRaises(TimeoutError):
            evaluate_task(self.task, generate=self.generate, sandbox=Mock(run=Mock(side_effect=TimeoutError('sandbox unavailable'))))
        with self.assertRaises(RuntimeError):
            evaluate_task(self.task, generate=self.generate, sandbox=Mock(run=Mock(side_effect=RuntimeError('daemon unavailable'))))
        with self.assertRaises(ValueError):
            evaluate_task(self.task, generate=Mock(return_value=FAKE_PROGRAMS[1]), sandbox=Mock(run=Mock(side_effect=ValueError('sandbox returned invalid JSON'))))

    def test_more_than_ten_sandbox_starts_aborts_task(self):
        task = Task('a', tuple(([[1]], [[1]]) for _ in range(10)), ([[1]],))
        with self.assertRaises(TimeoutError):
            evaluate_task(task, generate=Mock(return_value=FAKE_PROGRAMS[0]), sandbox=FakeE6Sandbox())

    def test_scoring_can_use_different_samples_for_each_test_input(self):
        task = Task('a', (([[1]], [[1]]),), ([[2]], [[3]]))
        verifier = Verifier({'a': [[[4]], [[5]]]})
        result = {'status': 'complete', 'task_id': 'a', 'sample_statuses': ['accepted', 'accepted'],
                  'predictions': [[[[4]], [[0]]], [[[0]], [[5]]]]}
        self.assertTrue(score_task(task, result, verifier)['solved'])
        result['predictions'][1][1] = [[6]]
        self.assertFalse(score_task(task, result, verifier)['solved'])

    def test_incomplete_or_malformed_results_cannot_be_scored(self):
        for result in ({'status': 'incomplete'}, {'status': 'complete', 'task_id': 'a',
                       'sample_statuses': ['accepted'], 'predictions': []}):
            with self.assertRaises(ValueError):
                score_task(self.task, result, self.verifier)
