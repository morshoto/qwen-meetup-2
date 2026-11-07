import unittest

from src.arc.data import Verifier
from src.arc.lt_phase_b_paired import score_paired


def row(task, state, candidates, status='complete'):
    return {'task_id': task, 'adapter_state': state, 'model': '14b',
            'status': status, 'candidates': candidates, 'tokens': 10,
            'runtime_seconds': 2., 'peak_memory_bytes': 100, 'peak_swap_bytes': None}


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.ids = ['a', 'b', 'c', 'd', 'missing']
        self.verifier = Verifier({task: [[[1]]] for task in self.ids})
        yes, no = [[[1]]], [[[0]]]
        self.rows = [row('a', 'off', [no, no]), row('a', 'on', [yes, yes]),
                     row('b', 'off', [yes, yes]), row('b', 'on', [no, no]),
                     row('c', 'off', [yes, yes]), row('c', 'on', [yes, yes]),
                     row('d', 'off', [no, yes]), row('d', 'on', [no, no]),
                     row('missing', 'off', [yes, yes]),
                     row('missing', 'on', [], status='error')]

    def test_pairs_and_two_attempts_use_full_task_scoring(self):
        report = score_paired(self.ids, self.rows, self.verifier)
        first = report['paired']['pass_at_1']
        self.assertEqual(first, {'denominator': 4, 'improved': 1, 'regressed': 1,
                                 'unchanged_right': 1, 'unchanged_wrong': 1})
        second = report['paired']['two_recorded_attempts']
        self.assertEqual(second, {'denominator': 4, 'improved': 1, 'regressed': 2,
                                  'unchanged_right': 1, 'unchanged_wrong': 0})
        self.assertEqual(report['states']['on']['status'], 'incomplete')
        self.assertEqual(report['states']['off']['candidate_denominator'], 10)
        self.assertEqual(report['identical_candidates'], {'off': 4, 'on': 4})
        self.assertEqual(report['rows'][-1]['pair_status'], 'incomplete')
        self.assertNotIn('answers', str(report))

    def test_rejects_unknown_state_model_duplicate_and_unselected_task(self):
        original = self.rows[0]
        for bad in (dict(original, adapter_state='bad'), dict(original, model='32b'),
                    dict(original, task_id='other')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_paired(self.ids, [bad], self.verifier)
        with self.assertRaises(ValueError):
            score_paired(self.ids, [original, original], self.verifier)

    def test_missing_and_failed_pairs_do_not_count_as_wrong(self):
        result = score_paired(self.ids, [row('a', 'off', [], 'timeout')], self.verifier)
        self.assertEqual(result['paired']['pass_at_1']['denominator'], 0)
        self.assertEqual(result['paired']['two_recorded_attempts']['denominator'], 0)
        self.assertEqual(result['rows'][0]['pair_status'], 'incomplete')


if __name__ == '__main__': unittest.main()
