import hashlib
import unittest

from src.arc.data import Verifier
from src.arc.lt_phase_b_paired import score_paired


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def registration(ids):
    return {'run_sha256': 'a' * 64, 'source_code_sha256': 'b' * 64,
            'adapter_sha256': 'c' * 64,
            'selected_task_sha256': {task: digest(task) for task in ids},
            'selected_prompt_sha256': {task: digest('prompt-' + task) for task in ids}}


def row(task, state, candidates, status='complete'):
    return {'task_id': task, 'adapter_state': state, 'model': '14b',
            'status': status, 'candidates': candidates, 'tokens': 10,
            'run_sha256': 'a' * 64, 'source_code_sha256': 'b' * 64,
            'adapter_sha256': 'c' * 64 if state == 'on' else None,
            'public_task_sha256': digest(task), 'prompt_sha256': digest('prompt-' + task),
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
        report = score_paired(self.ids, self.rows, self.verifier, registration=registration(self.ids))
        first = report['paired']['pass_at_1']
        self.assertEqual(first, {'denominator': 4, 'improved': 1, 'regressed': 1,
                                 'unchanged_right': 1, 'unchanged_wrong': 1})
        second = report['paired']['two_recorded_attempts']
        self.assertEqual(second, {'denominator': 4, 'improved': 1, 'regressed': 2,
                                  'unchanged_right': 1, 'unchanged_wrong': 0})
        self.assertEqual(report['states']['on']['status'], 'incomplete')
        self.assertEqual(report['states']['off']['candidate_denominator'], 10)
        self.assertEqual(report['identical_candidates'], {'off': 4, 'on': 4})
        self.assertEqual(report['uncertainty']['pass_at_1']['difference'], 0.0)
        self.assertEqual(report['uncertainty']['pass_at_1']['interval_95'], [-1.0, 1.0])
        self.assertEqual(report['rows'][-1]['pair_status'], 'incomplete')
        self.assertNotIn('answers', str(report))

    def test_rejects_unknown_state_model_duplicate_and_unselected_task(self):
        original = self.rows[0]
        for bad in (dict(original, adapter_state='bad'), dict(original, model='32b'),
                    dict(original, task_id='other')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_paired(self.ids, [bad], self.verifier, registration=registration(self.ids))
        with self.assertRaises(ValueError):
            score_paired(self.ids, [original, original], self.verifier, registration=registration(self.ids))

    def test_mismatched_registration_or_prompt_rejected(self):
        on = self.rows[1]
        for field, value in (('run_sha256', 'd' * 64), ('source_code_sha256', 'd' * 64),
                             ('public_task_sha256', 'd' * 64), ('adapter_sha256', 'd' * 64),
                             ('prompt_sha256', 'd' * 64)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                score_paired(self.ids, [self.rows[0], dict(on, **{field: value})],
                             self.verifier, registration=registration(self.ids))
        with self.assertRaises(ValueError):
            score_paired(self.ids, [dict(self.rows[0], prompt_sha256='d' * 64)],
                         self.verifier, registration=registration(self.ids))
        with self.assertRaises(ValueError):
            score_paired(self.ids, [dict(self.rows[0], adapter_sha256='c' * 64)],
                         self.verifier, registration=registration(self.ids))
        with self.assertRaises(ValueError):
            score_paired(self.ids, [row('a', 'off', [[[[1]]]], 'error')],
                         self.verifier, registration=registration(self.ids))

    def test_missing_and_failed_pairs_do_not_count_as_wrong(self):
        result = score_paired(self.ids, [row('a', 'off', [], 'timeout')], self.verifier, registration=registration(self.ids))
        self.assertEqual(result['paired']['pass_at_1']['denominator'], 0)
        self.assertEqual(result['paired']['two_recorded_attempts']['denominator'], 0)
        self.assertEqual(result['rows'][0]['pair_status'], 'incomplete')
        self.assertIsNone(result['uncertainty']['pass_at_1']['interval_95'])
        self.assertIsNone(result['uncertainty']['pass_at_1']['difference'])


if __name__ == '__main__': unittest.main()
