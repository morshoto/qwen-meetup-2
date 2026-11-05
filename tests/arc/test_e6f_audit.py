import unittest

from src.cli.e6f_audit import evaluated_rates


class AuditTests(unittest.TestCase):
    def test_budget_interruption_is_not_a_validation_failure(self):
        events = [
            {'event': 'attempt_finished', 'candidate_index': 0, 'attempt': 0, 'category': 'passed'},
            {'event': 'candidate_finished', 'candidate_index': 0, 'status': 'duplicate_oracle', 'validation_pass': True},
            {'event': 'attempt_finished', 'candidate_index': 1, 'attempt': 0, 'category': 'runtime_exception'},
            {'event': 'candidate_finished', 'candidate_index': 1, 'status': 'repair_limit', 'validation_pass': False},
            {'event': 'attempt_finished', 'candidate_index': 2, 'attempt': 0, 'category': 'budget_exhausted'},
            {'event': 'candidate_finished', 'candidate_index': 2, 'status': 'budget_exhausted', 'validation_pass': False},
        ]
        result = evaluated_rates(events)
        self.assertEqual(result['comparable_candidates'], 2)
        self.assertEqual(result['baseline_rate'], 0.5)
        self.assertEqual(result['after_repair_rate'], 0.5)
        self.assertEqual(result['excluded_incomplete_candidates'], 1)


if __name__ == '__main__':
    unittest.main()
