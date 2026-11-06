import unittest
from copy import deepcopy
from unittest.mock import patch

from src.arc.lt_comparison import budget_for_model, freeze_manifest, make_manifest, run_bounded_task
from src.arc.lt_components import prepare_components, check_guarded_metadata


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.proposal = make_manifest([f't{i:03}' for i in range(200)], seed=1,
                                      development=set(), training=set(), exposures=set(), dataset_sha256='a' * 64)
        self.config = {'prompt_sha256': 'b' * 64, 'code_sha256': 'c' * 64,
                       'models': {'14b': '1' * 40, '32b': '2' * 40},
                       'model_names': {'14b': 'model14', '32b': 'model32'},
                       'backend_url': 'http://127.0.0.1:8080/v1',
                       'thinking': 'off', 'temperature': 0.7, 'candidates': 2, 'max_tokens': 1000,
                       'timeout_seconds': 60, 'seed': 42, 'max_repairs': 1, 'prompt_variant': 'legacy',
                       'model_files_sha256': {'14b': 'd' * 64, '32b': 'e' * 64}, 'runtime_sha256': 'f' * 64,
                       'sandbox': {'image': 'sha256:' + '9' * 64, 'timeout': 3, 'memory_mb': 128,
                                   'cpus': 1, 'pids': 32},
                       'budget': {'max_calls': 2000, 'max_reserved_tokens': 20000000, 'max_sandbox_starts': 8000,
                                  'max_seconds': 36000, 'max_calls_per_task': 10, 'max_reserved_tokens_per_task': 100000,
                                  'max_sandbox_per_task': 40, 'max_seconds_per_task': 120}}

    def test_components_use_same_frozen_settings_and_separate_models(self):
        frozen = freeze_manifest(self.proposal, self.config)
        a, sandbox_a = prepare_components(frozen, '14b')
        b, sandbox_b = prepare_components(frozen, '32b')
        self.assertEqual((a.model, a.revision), ('model14', '1' * 40))
        self.assertEqual((b.model, b.revision), ('model32', '2' * 40))
        self.assertEqual((a.seed, a.temperature, a.max_tokens, a.thinking), (b.seed, b.temperature, b.max_tokens, b.thinking))
        self.assertTrue(a.guarded and b.guarded)
        self.assertEqual((sandbox_a.image, sandbox_a.timeout, sandbox_a.memory_mb, sandbox_a.cpus, sandbox_a.pids),
                         (sandbox_b.image, sandbox_b.timeout, sandbox_b.memory_mb, sandbox_b.cpus, sandbox_b.pids))
        self.assertEqual(sandbox_a.image, 'sha256:' + '9' * 64)

    def test_registered_components_drive_existing_bounded_agent_fixture(self):
        from src.arc.data import Task
        from src.arc.sandbox import IDENTITY_CODE
        frozen = freeze_manifest(self.proposal, self.config)
        task = Task('t000', (([[1]], [[1]]),), ([[2]],))
        for model in ('14b', '32b'):
            backend, sandbox = prepare_components(frozen, model)
            def reply(prompt, *, kind, candidate):
                return 'copy input' if kind == 'analysis' else IDENTITY_CODE
            with patch.object(backend, 'generate', side_effect=reply) as generate, \
                 patch.object(sandbox, 'run', side_effect=lambda code, grid: grid):
                row = run_bounded_task(task, backend, sandbox, budget_for_model(frozen, model),
                                       max_tokens=frozen['config']['max_tokens'],
                                       max_repairs=frozen['config']['max_repairs'])
            self.assertEqual(row['status'], 'complete')
            self.assertEqual([call.kwargs['candidate'] for call in generate.call_args_list], [0, 0, 1, 1])
            self.assertEqual(row['candidates'], [[[[2]]], [[[2]]]])

    def test_task_budget_does_not_shrink_later_task_timeouts(self):
        from src.arc.data import Task
        from src.arc.sandbox import IDENTITY_CODE
        frozen = freeze_manifest(self.proposal, self.config)
        backend, sandbox = prepare_components(frozen, '14b')
        original_timeout, original_sandbox_timeout = backend.timeout, sandbox.timeout
        task = Task('t000', (([[1]], [[1]]),), ([[2]],))
        with patch.object(backend, 'generate', side_effect=lambda prompt, **kwargs:
                          'copy input' if kwargs['kind'] == 'analysis' else IDENTITY_CODE), \
             patch.object(sandbox, 'run', side_effect=lambda code, grid: grid):
            run_bounded_task(task, backend, sandbox, budget_for_model(frozen, '14b'),
                             max_tokens=1000, max_repairs=1)
        self.assertEqual(backend.timeout, original_timeout)
        self.assertEqual(sandbox.timeout, original_sandbox_timeout)
        self.assertIsNone(sandbox.remaining)

    def test_guarded_metadata_must_match_model_and_budget(self):
        frozen = freeze_manifest(self.proposal, self.config)
        metadata = {'model': 'model14', 'revision': '1' * 40, 'token_limit': 20000000,
                    'reserved_tokens': 0, 'max_prompt_tokens': 5000, 'max_new_tokens': 1000,
                    'contract': 'same_process_token_ids_v1', 'hard_deadline': 'process_exit_v1',
                    'thinking_mode': 'off', 'thinking_modes': ['off', 'on'],
                    'thinking_canary': {'supported': True, 'different_token_ids': True},
                    'max_request_seconds': 60, 'adapter_sha256': 'none'}
        report = check_guarded_metadata(frozen, '14b', metadata)
        self.assertEqual(report['status'], 'metadata_consistent_unverified')
        self.assertFalse(report['execution_ready'])
        for change in ({'model': 'model32'}, {'revision': 'wrong'}, {'token_limit': 20000001},
                       {'reserved_tokens': 1}, {'thinking_canary': {'supported': True, 'different_token_ids': False}},
                       {'max_request_seconds': 61}, {'adapter_sha256': 'unexpected'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_guarded_metadata(frozen, '14b', {**metadata, **change})

    def test_registration_rejects_remote_url_and_unpinned_sandbox(self):
        for key, value in [('models', {'14b': 'short', '32b': 'also-short'}),
                           ('backend_url', 'https://example.com/v1'),
                           ('sandbox', {**self.config['sandbox'], 'image': 'arc-sandbox:local'})]:
            config = deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): freeze_manifest(self.proposal, config)


if __name__ == '__main__': unittest.main()
