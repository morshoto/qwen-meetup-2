import unittest
from src.arc.data import Verifier
from src.arc.lt_single_score import score_single


class SingleScoreTests(unittest.TestCase):
    def test_selected_model_scores_without_fabricating_second_model(self):
        row = {'model': 'single', 'task_id': 'a', 'status': 'complete',
               'candidates': [[[[2]]], [[[0]]]], 'tokens': 5, 'runtime_seconds': 1,
               'peak_memory_bytes': 10, 'peak_swap_bytes': None}
        result = score_single(['a', 'b'], [row], Verifier({'a': [[[2]]], 'b': [[[3]]]}))
        self.assertEqual(list(result['models']), ['single'])
        self.assertEqual(result['models']['single']['status'], 'incomplete')
        self.assertEqual(result['models']['single']['pass_at_1'], 1)
        self.assertEqual(result['models']['single']['selected'], 2)
        self.assertEqual(result['models']['single']['attempted'], 1)
        self.assertEqual({row['model'] for row in result['rows']}, {'single'})

    def test_rejects_foreign_model(self):
        with self.assertRaises(ValueError):
            score_single(['a'], [{'model': '14b'}], Verifier({'a': [[[1]]]}))
