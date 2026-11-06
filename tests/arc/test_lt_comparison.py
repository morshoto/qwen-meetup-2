import unittest

from src.arc.data import Task, Verifier
from src.arc.sandbox import FakeSandbox, IDENTITY_CODE
from src.arc.lt_comparison import make_manifest, summarize, freeze_manifest, verify_manifest, run_task, run_bounded_task, budget_for_model


class ComparisonTests(unittest.TestCase):
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

    def test_freeze_binds_manifest_config_and_detects_changes(self):
        ids = [f'task-{i:03}' for i in range(210)]
        proposal = make_manifest(ids, seed=1, development=set(), training=set(), exposures=set(), dataset_sha256='a' * 64)
        config = {'prompt_sha256': 'b' * 64, 'code_sha256': 'c' * 64,
                  'models': {'14b': '1' * 40, '32b': '2' * 40},
                  'model_names': {'14b': 'model14', '32b': 'model32'}, 'backend_url': 'http://127.0.0.1:8080/v1',
                  'sandbox': {'image': 'sha256:' + '9' * 64, 'timeout': 3, 'memory_mb': 128, 'cpus': 1, 'pids': 32},
                  'thinking': 'off', 'temperature': 0.7, 'candidates': 2,
                  'max_tokens': 1000, 'max_prompt_tokens': 5000, 'timeout_seconds': 60, 'seed': 42, 'max_repairs': 1,
                  'prompt_variant': 'legacy', 'model_files_sha256': {'14b': 'd' * 64, '32b': 'e' * 64},
                  'runtime_sha256': 'f' * 64, 'budget': {'max_calls': 2000,
                  'max_reserved_tokens': 20000000, 'max_sandbox_starts': 8000,
                  'max_seconds': 36000, 'max_calls_per_task': 10,
                  'max_reserved_tokens_per_task': 100000, 'max_sandbox_per_task': 40,
                  'max_seconds_per_task': 120}}
        frozen = freeze_manifest(proposal, config)
        verify_manifest(frozen)
        frozen['config']['max_tokens'] = 1001
        with self.assertRaises(ValueError): verify_manifest(frozen)

    def test_freeze_rejects_missing_execution_budget_and_seed(self):
        ids = [f'task-{i:03}' for i in range(210)]
        proposal = make_manifest(ids, seed=1, development=set(), training=set(), exposures=set(), dataset_sha256='a' * 64)
        incomplete = {'prompt_sha256': 'b' * 64, 'code_sha256': 'c' * 64,
                      'models': {'14b': '1' * 40, '32b': '2' * 40},
                  'model_names': {'14b': 'model14', '32b': 'model32'}, 'backend_url': 'http://127.0.0.1:8080/v1',
                  'sandbox': {'image': 'sha256:' + '9' * 64, 'timeout': 3, 'memory_mb': 128, 'cpus': 1, 'pids': 32},
                      'thinking': 'off', 'temperature': 0.7, 'candidates': 2,
                      'max_tokens': 1000, 'max_prompt_tokens': 5000, 'timeout_seconds': 60}
        with self.assertRaises(ValueError): freeze_manifest(proposal, incomplete)

    def test_models_receive_equal_independent_frozen_budget(self):
        ids = [f'task-{i:03}' for i in range(210)]
        proposal = make_manifest(ids, seed=1, development=set(), training=set(), exposures=set(), dataset_sha256='a' * 64)
        config = {'prompt_sha256': 'b' * 64, 'code_sha256': 'c' * 64,
                  'models': {'14b': '1' * 40, '32b': '2' * 40},
                  'model_names': {'14b': 'model14', '32b': 'model32'}, 'backend_url': 'http://127.0.0.1:8080/v1',
                  'sandbox': {'image': 'sha256:' + '9' * 64, 'timeout': 3, 'memory_mb': 128, 'cpus': 1, 'pids': 32},
                  'thinking': 'off', 'temperature': 0.7, 'candidates': 2,
                  'max_tokens': 1000, 'max_prompt_tokens': 5000, 'timeout_seconds': 60, 'seed': 42, 'max_repairs': 1,
                  'prompt_variant': 'legacy', 'model_files_sha256': {'14b': 'd' * 64, '32b': 'e' * 64},
                  'runtime_sha256': 'f' * 64, 'budget': {'max_calls': 2000,
                  'max_reserved_tokens': 20000000, 'max_sandbox_starts': 8000,
                  'max_seconds': 36000, 'max_calls_per_task': 10,
                  'max_reserved_tokens_per_task': 100000, 'max_sandbox_per_task': 40,
                  'max_seconds_per_task': 120}}
        frozen = freeze_manifest(proposal, config)
        first, second = budget_for_model(frozen, '14b'), budget_for_model(frozen, '32b')
        self.assertEqual(first.limits, second.limits)
        self.assertEqual(first.task_limits, second.task_limits)
        first.reserve_call('task', 100)
        self.assertEqual(second.finish()['calls'], 0)
        with self.assertRaises(ValueError): budget_for_model(frozen, 'other')

    def test_freeze_rejects_tampered_proposal(self):
        ids = [f'task-{i:03}' for i in range(210)]
        proposal = make_manifest(ids, seed=1, development=set(), training=set(), exposures=set(), dataset_sha256='a' * 64)
        proposal['task_ids'] = ['duplicate'] * 200
        config = {'prompt_sha256': 'b' * 64, 'code_sha256': 'c' * 64,
                  'models': {'14b': '1' * 40, '32b': '2' * 40},
                  'model_names': {'14b': 'model14', '32b': 'model32'}, 'backend_url': 'http://127.0.0.1:8080/v1',
                  'sandbox': {'image': 'sha256:' + '9' * 64, 'timeout': 3, 'memory_mb': 128, 'cpus': 1, 'pids': 32},
                  'thinking': 'off', 'temperature': 0.7, 'candidates': 2,
                  'max_tokens': 1000, 'max_prompt_tokens': 5000, 'timeout_seconds': 60, 'seed': 42, 'max_repairs': 1,
                  'prompt_variant': 'legacy', 'model_files_sha256': {'14b': 'd' * 64, '32b': 'e' * 64},
                  'runtime_sha256': 'f' * 64, 'budget': {'max_calls': 2000,
                  'max_reserved_tokens': 20000000, 'max_sandbox_starts': 8000,
                  'max_seconds': 36000, 'max_calls_per_task': 10,
                  'max_reserved_tokens_per_task': 100000, 'max_sandbox_per_task': 40,
                  'max_seconds_per_task': 120}}
        with self.assertRaises(ValueError): freeze_manifest(proposal, config)

    def test_error_row_cannot_make_run_complete(self):
        verifier = Verifier({'a': [[[1]]]})
        result = summarize(['a'], [{'task_id': 'a', 'candidates': [], 'status': 'error'}], verifier)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['errors'], 1)

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

    def test_run_task_uses_two_independent_candidate_indices(self):
        task = Task('x', (([[1]], [[1]]),), ([[2]],))
        class Backend:
            def __init__(self): self.candidates = []
            def generate(self, prompt, *, kind, candidate):
                self.candidates.append(candidate)
                return 'copy input' if kind == 'analysis' else IDENTITY_CODE
        backend = Backend()
        result = run_task(task, backend, FakeSandbox(), max_repairs=0)
        self.assertEqual(backend.candidates, [0, 0, 1, 1])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['candidates'], [[[[2]]], [[[2]]]])

    def test_run_task_records_backend_failure_without_fabricating_candidate(self):
        from src.arc.http_backend import BackendError
        class Backend:
            def generate(self, prompt, *, kind, candidate):
                raise BackendError('unavailable', 'test')
        task = Task('x', (([[1]], [[1]]),), ([[2]],))
        result = run_task(task, Backend(), FakeSandbox(), max_repairs=0)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['candidates'], [])

    def test_bounded_task_stops_before_exceeding_model_call_limit(self):
        from src.arc.search import SearchBudget
        class Backend:
            def __init__(self): self.calls = 0
            def generate(self, prompt, *, kind, candidate):
                self.calls += 1
                return 'copy input'
        task = Task('x', (([[1]], [[1]]),), ([[2]],))
        backend = Backend()
        budget = SearchBudget(max_seconds=60, max_calls=1, max_tokens=100000,
                              max_sandbox_starts=10, max_calls_per_task=1,
                              max_tokens_per_task=100000, max_sandbox_per_task=10,
                              max_seconds_per_task=60)
        result = run_bounded_task(task, backend, FakeSandbox(), budget, max_tokens=500, max_repairs=0)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(budget.finish()['calls'], 1)
        self.assertEqual(budget.finish()['sandbox_starts'], 0)

    def test_bounded_task_stops_before_exceeding_sandbox_limit(self):
        from src.arc.search import SearchBudget
        class Backend:
            def generate(self, prompt, *, kind, candidate):
                return 'copy input' if kind == 'analysis' else IDENTITY_CODE
        class Sandbox:
            def __init__(self): self.calls = 0
            def run(self, code, grid):
                self.calls += 1
                return grid
        sandbox = Sandbox()
        budget = SearchBudget(max_seconds=60, max_calls=10, max_tokens=100000,
                              max_sandbox_starts=0, max_calls_per_task=10,
                              max_tokens_per_task=100000, max_sandbox_per_task=0,
                              max_seconds_per_task=60)
        task = Task('x', (([[1]], [[1]]),), ([[2]],))
        result = run_bounded_task(task, Backend(), sandbox, budget, max_tokens=500, max_repairs=0)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(sandbox.calls, 0)

    def test_bounded_task_rejects_nested_sandbox_reservation(self):
        from src.arc.search import SearchBudget
        class Sandbox:
            reserve = lambda: None
        budget = SearchBudget(max_seconds=60, max_calls=10, max_tokens=100000,
                              max_sandbox_starts=10, max_calls_per_task=10,
                              max_tokens_per_task=100000, max_sandbox_per_task=10,
                              max_seconds_per_task=60)
        task = Task('x', (([[1]], [[1]]),), ([[2]],))
        with self.assertRaises(ValueError):
            run_bounded_task(task, object(), Sandbox(), budget, max_tokens=500, max_repairs=0)

    def test_missing_second_candidate_is_incomplete(self):
        verifier = Verifier({'a': [[[1]]]})
        row = {'task_id': 'a', 'candidates': [[[[1]]]], 'status': 'complete'}
        result = summarize(['a'], [row], verifier)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['pass_at_1'], 1)

    def test_scoring_invalid_and_timeout_are_not_matches(self):
        from src.arc.scoring import score
        verifier = Verifier({'a': [[[1]], [[2]]]})
        result = score(verifier, 'a', ['timeout', [[[1]]]])
        self.assertEqual((result.pass_at_1, result.pass_at_2, result.timeouts, result.invalid),
                         (False, False, 1, 1))

    def test_summary_rejects_duplicate_and_unselected_rows(self):
        verifier = Verifier({'a': [[[1]]]})
        row = {'task_id': 'a', 'candidates': [], 'status': 'error'}
        with self.assertRaises(ValueError): summarize(['a'], [row, row], verifier)
        with self.assertRaises(ValueError): summarize(['a'], [{'task_id': 'b'}], verifier)


if __name__ == '__main__': unittest.main()
