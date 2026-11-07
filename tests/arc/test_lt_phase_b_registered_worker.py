import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_live_backend import PROMPT
from src.arc.lt_public import build_public_export
from src.arc.lt_phase_b_registered_worker import run_registered_task


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Backend:
    adapter_state, adapter_sha256 = 'off', None
    def generate(self, prompt, *, kind, candidate):
        assert kind == 'direct'
        assert all(set(x) == {'input'} for x in json.loads(prompt)['task']['test'])
        self.raw = '[[[1]]]'
        self.stats = {'prompt_tokens': 10, 'completion_tokens': 1, 'peak_memory_gb': 1, 'finish_reason': 'stop'}
        return self.raw


class RegisteredWorkerTests(unittest.TestCase):
    def setUp(self):
        self.challenges = {f'{i:08x}': {'train': [{'input': [[i // 10, i % 10]], 'output': [[1]]}],
                                              'test': [{'input': [[2]]}]} for i in range(100)}
        self.hashes = {key: sha(value) for key, value in self.challenges.items()}
        self.inventory = {'code_sha256': 'a' * 64}
        self.spec = {'protocol': 'lt-phase-b-collection-v1', 'status': 'frozen', 'run_id': 'fixture',
                     'source_code_sha256': 'a' * 64, 'allocation_audit_sha256': 'b' * 64,
                     'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                     'selected_task_sha256': self.hashes, 'model_manifest': {
                         'model_id': 'mlx-community/Qwen3-14B-4bit',
                         'revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                         'quantization': {'bits': 4, 'group_size': 64},
                         'chat_template_sha256': 'c' * 64, 'files_sha256': {'config.json': 'd' * 64}},
                     'settings': {'seed': 20261111, 'max_tokens': 4096, 'temperature': 0.6,
                                  'thinking': 'off', 'candidate_count': 2,
                                  'mlx_memory_limit_bytes': 48 * 1024**3,
                                  'mlx_lm_version': '0.30.7', 'mlx_version': '0.32.4'}}
        self.created = []
        self.patcher = patch('src.arc.lt_phase_b_registered_worker.verify_source_tree')
        self.verify_source = self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.versions = patch('src.arc.lt_phase_b_registered_worker.importlib.metadata.version',
                              side_effect=lambda x: {'mlx-lm': '0.30.7', 'mlx': '0.32.4'}[x])
        self.versions.start(); self.addCleanup(self.versions.stop)

    def run_task(self, spec=None, **kwargs):
        spec = self.spec if spec is None else spec
        run_sha = sha(spec)
        receipt = {'comparison_sha256': run_sha, 'selected_task_sha256': self.hashes}
        export = build_public_export(self.challenges, receipt)
        def factory(**options):
            self.created.append(options)
            return Backend()
        args = {'spec': spec, 'expected_run_sha256': run_sha, 'source_inventory': self.inventory,
                'source_root': '.', 'model_path': '/not-used', 'export': export, 'receipt': receipt,
                'task_id': '00000000', 'backend_factory': factory}
        args.update(kwargs)
        return run_registered_task(**args)

    def test_frozen_public_worker_binds_identity_before_and_after(self):
        result = self.run_task()
        self.assertEqual(result['run_sha256'], sha(self.spec))
        self.assertEqual(result['run_id'], 'fixture')
        self.assertEqual(result['source_code_sha256'], 'a' * 64)
        self.assertEqual(result['adapter_state'], 'off')
        self.assertEqual(len(self.created), 1)
        self.assertEqual(self.verify_source.call_count, 2)

    def test_invalid_registration_never_loads_model(self):
        for changed in (dict(self.spec, status='draft'), dict(self.spec, allocation_audit_sha256='bad'),
                        dict(self.spec, settings={**self.spec['settings'], 'candidate_count': 1}),
                        dict(self.spec, selected_task_sha256={}), dict(self.spec, source_code_sha256='c' * 64)):
            with self.subTest(changed=changed), self.assertRaises(ValueError): self.run_task(changed)
        self.assertEqual(self.created, [])

    def test_tampered_public_export_or_runtime_fails_before_model(self):
        with self.assertRaises(ValueError): self.run_task(expected_run_sha256='e' * 64)
        receipt = {'comparison_sha256': sha(self.spec), 'selected_task_sha256': self.hashes}
        export = build_public_export(self.challenges, receipt)
        export['tasks']['00000000']['test'][0]['output'] = [[1]]
        with self.assertRaises(ValueError): self.run_task(export=export)
        self.versions.stop()
        with patch('src.arc.lt_phase_b_registered_worker.importlib.metadata.version', return_value='wrong'):
            with self.assertRaises(ValueError): self.run_task()
        self.assertEqual(self.created, [])


if __name__ == '__main__': unittest.main()
