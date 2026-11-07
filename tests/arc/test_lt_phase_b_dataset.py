import json
import unittest

from src.arc.data import Task, Verifier
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_dataset import build_dataset


def task(name):
    value = ord(name[0]) % 10
    return Task(name, (([[value]], [[1]]),), ([[value]],))


def row(name, candidates):
    return {'task_id': name, 'model': '14b', 'status': 'complete', 'candidates': candidates,
            'calls': [{'candidate': i, 'raw': json.dumps(candidate),
                       'usage': {'completion_tokens': 2}, 'error': None if candidate else 'invalid'}
                      for i, candidate in enumerate(candidates)],
            'cleanup': {'leader_reaped': True, 'group_absent': True}}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tasks = {x: task(x) for x in ('a', 'b', 'c', 'd')}
        self.verifier = Verifier({x: [[[1]]] for x in self.tasks})
        self.source = {'model_id': 'mlx-community/Qwen3-14B-4bit',
                       'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                       'source_code_sha256': 'a' * 64, 'run_id': 'fixture-run'}
        self.good, self.bad = [[[1]]], [[[0]]]

    def build(self, rows, **kw):
        return build_dataset(self.tasks, rows, self.verifier, source=self.source,
                             comparison_tasks=kw.get('comparison_tasks', {'x': task('x')}))

    def test_full_correct_only_one_lineage_and_disjoint_split(self):
        rows = [row('a', [self.good, self.good]), row('b', [self.bad, self.good]),
                row('c', [self.good, self.bad]), row('d', [self.bad, None])]
        out = self.build(rows)
        self.assertEqual(out['yield_status'], 'disjoint_split')
        self.assertEqual((len(out['train']), len(out['valid'])), (2, 1))
        self.assertEqual(out['eligible_task_count'], 3)
        self.assertEqual(out['split_task_ids'], {'train': ['b', 'c'], 'valid': ['a']})
        self.assertEqual([x['verification_status'] for x in out['journal'][:2]],
                         ['eligible', 'duplicate_task'])
        self.assertEqual(len({r['solution_lineage_sha256'] for r in out['journal'] if r['verification_status'] == 'eligible'}), 3)
        example = (out['train'] + out['valid'])[0]
        messages = example['messages']
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
        self.assertEqual(json.loads(messages[0]['content'])['instruction'], PROMPT)
        self.assertEqual(json.loads(messages[1]['content']), self.good)
        self.assertNotIn('answers', str(out))
        self.assertEqual(out, self.build(rows))

    def test_zero_and_small_yields_are_explicit(self):
        empty = self.build([row(x, [self.bad, None]) for x in self.tasks])
        self.assertEqual(empty['yield_status'], 'insufficient_verified_self_solutions')
        self.assertEqual(empty['train'], [])
        small = self.build([row('a', [self.good, self.bad])] +
                           [row(x, [self.bad, None]) for x in ('b', 'c', 'd')])
        self.assertEqual(small['yield_status'], 'overfitting_only')
        self.assertEqual(small['valid'], [])

    def test_success_must_match_recorded_raw_response(self):
        bad = row('a', [self.good, self.bad])
        bad['calls'][0]['raw'] = json.dumps(self.bad)
        with self.assertRaises(ValueError): self.build([bad])
        bad = row('a', [self.good, self.bad])
        bad['calls'][0]['error'] = 'timeout'
        with self.assertRaises(ValueError): self.build([bad])

    def test_worker_failure_is_preserved_and_never_counted_as_zero_yield(self):
        failure = {'task_id': 'a', 'model': '14b', 'status': 'timeout',
                   'candidates': [], 'calls': [], 'worker_diagnostics': {'stderr': 'deadline'},
                   'cleanup': {'leader_reaped': True, 'group_absent': True}}
        rows = [failure] + [row(x, [self.good, self.bad]) for x in ('b', 'c', 'd')]
        result = self.build(rows)
        self.assertEqual(result['yield_status'], 'collection_failed')
        self.assertEqual(result['failed_task_count'], 1)
        self.assertEqual(result['eligible_task_count'], 3)
        self.assertEqual((result['train'], result['valid']), ([], []))
        record = result['journal'][0]
        self.assertEqual(record['verification_status'], 'worker_timeout')
        self.assertIsNone(record['candidate_index'])
        self.assertIsNone(record['prediction'])
        self.assertIsNone(record['task_exact_match'])
        self.assertEqual(record['worker_diagnostics']['stderr'], 'deadline')
        for bad in (dict(failure, model='32b'), dict(failure, candidates=[self.good]),
                    dict(failure, cleanup={'leader_reaped': True, 'group_absent': False})):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.build([bad])

    def test_incomplete_collection_never_declares_zero_yield(self):
        result = self.build([row('a', [self.bad, None])])
        self.assertEqual(result['yield_status'], 'collection_incomplete')
        self.assertEqual(result['train'], [])
        self.assertEqual(result['valid'], [])

    def test_reject_duplicate_content_even_with_different_ids_or_example_order(self):
        duplicate = Task('x', self.tasks['a'].train, self.tasks['a'].test_inputs)
        with self.assertRaises(ValueError): self.build([], comparison_tasks={'x': duplicate})
        duplicated = dict(self.tasks, b=Task('b', self.tasks['a'].train, self.tasks['a'].test_inputs))
        with self.assertRaises(ValueError):
            build_dataset(duplicated, [], self.verifier, source=self.source, comparison_tasks={'x': task('x')})
        a = Task('a', (([[0]], [[1]]), ([[2]], [[3]])), ([[4]], [[5]]))
        other = Task('x', tuple(reversed(a.train)), tuple(reversed(a.test_inputs)))
        with self.assertRaises(ValueError):
            build_dataset({'a': a}, [], self.verifier, source=self.source, comparison_tasks={'x': other})

    def test_reject_overlap_duplicate_unknown_and_incomplete_provenance(self):
        with self.assertRaises(ValueError): self.build([], comparison_tasks={'a': self.tasks['a']})
        with self.assertRaises(ValueError): self.build([row('a', [self.good, self.bad])] * 2)
        with self.assertRaises(ValueError): self.build([row('x', [self.good, self.bad])])
        for source in (dict(self.source, model_id='other'), dict(self.source, source_code_sha256='x'),
                       dict(self.source, run_id='')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                build_dataset(self.tasks, [], self.verifier, source=source, comparison_tasks={'x': task('x')})


if __name__ == '__main__': unittest.main()
