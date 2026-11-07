import unittest

from src.arc.data import Verifier
from src.arc.lt_scoring import score_comparison


class ComparisonScoringTests(unittest.TestCase):
    def setUp(self):
        self.verifier = Verifier({'a': [[[1]], [[2]]], 'b': [[[3]]]})

    def test_per_model_scores_preserve_missing_candidates_and_denominators(self):
        rows = [
            {'model': '14b', 'task_id': 'a', 'status': 'complete',
             'candidates': [[[[1]], [[0]]], [[[0]], [[2]]]],
             'tokens': 12, 'runtime_seconds': 2.5, 'peak_memory_bytes': 100, 'peak_swap_bytes': 20},
            {'model': '32b', 'task_id': 'a', 'status': 'error',
             'candidates': [[[[1]], [[2]]]],
             'tokens': None, 'runtime_seconds': 3.0, 'peak_memory_bytes': None, 'peak_swap_bytes': None},
        ]
        report = score_comparison(['a', 'b'], rows, self.verifier)
        first, missing, single = report['rows'][0], report['rows'][1], report['rows'][2]
        self.assertEqual((first['pass_at_1'], first['pass_at_2']), (False, False))
        self.assertEqual((first['output_pass_at_1'], first['output_pass_at_2'], first['output_count']), (1, 2, 2))
        self.assertEqual(missing['status'], 'not_attempted')
        self.assertIsNone(missing['pass_at_1'])
        self.assertTrue(single['pass_at_1'])
        self.assertIsNone(single['pass_at_2'])
        self.assertEqual(report['models']['14b']['selected'], 2)
        self.assertEqual(report['models']['14b']['attempted'], 1)
        self.assertEqual(report['models']['14b']['status'], 'incomplete')
        self.assertEqual(report['models']['14b']['tokens'], 12)
        self.assertIsNone(report['models']['32b']['tokens'])
        self.assertNotIn('answers', str(report))

    def test_invalid_and_timeout_predictions_are_scored_without_answers(self):
        rows = [{'model': '14b', 'task_id': 'b', 'status': 'complete',
                 'candidates': ['timeout', [[[99]]]], 'tokens': 0,
                 'runtime_seconds': 1, 'peak_memory_bytes': 0, 'peak_swap_bytes': 0}]
        report = score_comparison(['b'], rows, self.verifier)
        self.assertEqual(report['models']['14b']['valid_candidates'], 0)
        self.assertEqual(report['models']['14b']['candidate_denominator'], 2)
        result = report['rows'][0]
        self.assertFalse(result['pass_at_1'])
        self.assertFalse(result['pass_at_2'])
        self.assertEqual((result['timeouts'], result['invalid']), (1, 1))

    def test_rejects_duplicate_unknown_and_malformed_rows(self):
        row = {'model': '14b', 'task_id': 'b', 'status': 'error', 'candidates': [],
               'tokens': None, 'runtime_seconds': 0, 'peak_memory_bytes': None, 'peak_swap_bytes': None}
        for rows in ([row, row], [{**row, 'task_id': 'unknown'}], [{**row, 'model': 'other'}],
                     [{**row, 'tokens': -1}], [{**row, 'status': 'complete'}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                score_comparison(['b'], rows, self.verifier)


if __name__ == '__main__': unittest.main()
