"""Paired task-level reporting, independent from model or verifier execution."""
import unittest

from src.arc.rsi_paired import paired_summary


class PairedSummaryTests(unittest.TestCase):
    def test_all_candidate_wins_show_positive_interval(self):
        base = {f't{i}': False for i in range(20)}
        candidate = {key: True for key in base}
        result = paired_summary(base, candidate)
        self.assertEqual(result['task_count'], 20)
        self.assertEqual(result['base_solved'], 0)
        self.assertEqual(result['candidate_solved'], 20)
        self.assertEqual(result['difference_percentage_points'], 100.0)
        self.assertEqual(result['bootstrap_95_percentage_points'], [100.0, 100.0])
        self.assertTrue(result['improvement_criterion_met'])

    def test_equal_results_are_inconclusive_and_repeatable(self):
        base = {'a': True, 'b': False}
        self.assertEqual(paired_summary(base, base), paired_summary(base, base))
        self.assertFalse(paired_summary(base, base)['improvement_criterion_met'])

    def test_rejects_missing_extra_or_nonboolean_task_outcomes(self):
        for base, candidate in [({'a': True}, {}), ({}, {}),
                                ({'a': True}, {'a': True, 'b': False}),
                                ({'a': 1}, {'a': True}),
                                ({'a': True}, {'a': None})]:
            with self.subTest(base=base, candidate=candidate), self.assertRaises(ValueError):
                paired_summary(base, candidate)

    def test_candidate_loss_does_not_meet_improvement(self):
        result = paired_summary({'a': True, 'b': True}, {'a': False, 'b': True})
        self.assertEqual(result['difference_percentage_points'], -50.0)
        self.assertFalse(result['improvement_criterion_met'])


if __name__ == '__main__':
    unittest.main()
