import unittest

from src.arc.data import Task, Verifier
from src.arc.agent import code_agent
from src.arc.sandbox import FakeSandbox, IDENTITY_CODE, ZERO_CODE
from src.arc.lt_comparison import make_manifest, summarize


class ComparisonTests(unittest.TestCase):
    def test_repair_prompts_never_include_held_out_answer(self):
        task = Task('x', (([[1]], [[1]]),), ([[2]],))
        class Backend:
            def __init__(self): self.prompts = []
            def generate(self, prompt, *, kind, candidate):
                self.prompts.append(prompt)
                if kind == 'analysis': return 'copy input'
                return ZERO_CODE if len(self.prompts) == 2 else IDENTITY_CODE
        backend = Backend()
        code_agent(task, backend, FakeSandbox(), 1, 0, prompt_variant='structured')
        self.assertEqual(len(backend.prompts), 3)
        self.assertTrue(all('9 9 9' not in prompt for prompt in backend.prompts))

    def test_manifest_is_deterministic_and_excludes_reserved_tasks(self):
        ids = [f'task-{i:03}' for i in range(230)]
        kwargs = dict(seed=7, development={'task-000'}, training={'task-001'},
                      exposures={'task-002'}, dataset_sha256='a' * 64)
        manifest = make_manifest(ids, **kwargs)
        self.assertEqual(manifest, make_manifest(list(reversed(ids)), **kwargs))
        self.assertEqual(len(manifest['task_ids']), 200)
        self.assertFalse(set(manifest['task_ids']) & {'task-000', 'task-001', 'task-002'})
        self.assertTrue(manifest['unknown_pretraining_exposure'])
        self.assertTrue(manifest['unknown_project_exposure'])

    def test_manifest_rejects_duplicates_and_insufficient_tasks(self):
        with self.assertRaises(ValueError):
            make_manifest(['x'] * 201, seed=1, development=set(), training=set(), exposures=set(), dataset_sha256='a' * 64)
        with self.assertRaises(ValueError):
            make_manifest(['x'], seed=1, development=set(), training=set(), exposures=set(), dataset_sha256='a' * 64)

    def test_summary_distinguishes_task_and_output_scores_and_missing(self):
        verifier = Verifier({'a': [[[1]], [[2]]], 'b': [[[3]]]})
        rows = [{'task_id': 'a', 'candidates': [[[[1]], [[0]]], [[[1]], [[2]]]], 'status': 'complete'}]
        result = summarize(['a', 'b'], rows, verifier)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['selected'], 2)
        self.assertEqual(result['attempted'], 1)
        self.assertEqual(result['pass_at_1'], 0)
        self.assertEqual(result['pass_at_2'], 1)
        self.assertEqual(result['first_candidate_correct_outputs'], 1)
        self.assertEqual(result['test_outputs_attempted'], 2)

    def test_summary_rejects_duplicate_and_unselected_rows(self):
        verifier = Verifier({'a': [[[1]]]})
        row = {'task_id': 'a', 'candidates': [], 'status': 'error'}
        with self.assertRaises(ValueError): summarize(['a'], [row, row], verifier)
        with self.assertRaises(ValueError): summarize(['a'], [{'task_id': 'b'}], verifier)


if __name__ == '__main__': unittest.main()
