import unittest

from src.arc.data import Verifier
from src.arc.lt_phase_b_eligibility import assess_candidates


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.verifier = Verifier({'a': [[[1]], [[2]]]})
        self.row = {'task_id': 'a', 'model': '14b', 'status': 'complete',
                    'candidates': [[[[1]], [[2]]], [[[1]], [[2]]]],
                    'cleanup': {'leader_reaped': True, 'group_absent': True}}

    def test_duplicate_solutions_count_as_one_task(self):
        records = assess_candidates(self.row, self.verifier)
        self.assertEqual([r['verification_status'] for r in records], ['eligible', 'duplicate_task'])
        self.assertEqual(records[0]['solution_lineage_sha256'], records[1]['solution_lineage_sha256'])

    def test_partial_solution_and_invalid_are_rejected(self):
        row = dict(self.row, candidates=[[[[1]], [[0]]], None])
        records = assess_candidates(row, self.verifier)
        self.assertEqual([r['verification_status'] for r in records], ['incorrect', 'invalid'])
        self.assertFalse(any(r['task_exact_match'] for r in records))

    def test_incomplete_or_unverified_cleanup_cannot_become_training_data(self):
        for row in (dict(self.row, status='error'), dict(self.row, model='32b'),
                    dict(self.row, cleanup={'leader_reaped': True, 'group_absent': False})):
            with self.subTest(row=row), self.assertRaises(ValueError):
                assess_candidates(row, self.verifier)


if __name__ == '__main__': unittest.main()
