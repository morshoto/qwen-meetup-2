import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_registered_comparison import run_registered_comparison_task
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_public import build_public_export
from tests.arc import test_lt_phase_b_registered_worker as fixtures


class Backend(fixtures.Backend):
    def __init__(self, **kwargs):
        self.adapter_state = kwargs['adapter_state']
        self.adapter_sha256 = kwargs.get('adapter_sha256')


class RegisteredComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RegisteredWorkerTests('test_frozen_public_worker_binds_identity_before_and_after')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.collection = f.spec; self.collection_sha = _digest(f.spec)
        self.canary, self.canary_sha = {'fixture': 'canary'}, 'b' * 64
        self.training, self.training_sha = {'fixture': 'training'}, 'c' * 64
        challenges = {key: {'train': [{'input': [[i % 10]], 'output': [[1]]}], 'test': [{'input': [[2]]}]}
                      for i, key in enumerate(sorted(f.allocation['comparison_task_sha256']))}
        self.tasks = {key: _digest(item) for key, item in challenges.items()}
        self.allocation = {**f.allocation, 'comparison_task_sha256': self.tasks}
        self.comparison = {'protocol': 'lt-phase-b-comparison-v1', 'status': 'frozen', 'run_id': f.spec['run_id'],
            'collection_sha256': self.collection_sha, 'training_sha256': self.training_sha,
            'canary_sha256': self.canary_sha, 'source_code_sha256': 'a' * 64,
            'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(), 'selected_task_sha256': self.tasks,
            'adapter_sha256': 'f' * 64, 'settings': f.spec['settings']}
        self.sha = _digest(self.comparison)
        self.receipt = {'comparison_sha256': self.sha, 'selected_task_sha256': self.tasks}
        self.export = build_public_export(challenges, self.receipt)
        self.collection_receipt = {'comparison_sha256': self.collection_sha, 'selected_task_sha256': f.hashes}
        patchers = {'validate_canary_identity': (f.spec['settings'], f.spec['model_manifest']),
            'review_real_reload_attempt': {'status': 'real_adapter_reload_observed_unverified', 'reload_verified': False,
                'promotion_allowed': False, 'reload': {'adapter_sha256': 'f' * 64}}}
        self.mock = {}
        for name, value in patchers.items():
            p = patch('src.arc.lt_phase_b_registered_comparison.' + name, return_value=value)
            self.mock[name] = p.start(); self.addCleanup(p.stop)
        for name in ('verify_runtime_identity', 'verify_environment', 'verify_source_tree'):
            p = patch('src.arc.lt_phase_b_registered_comparison.' + name)
            self.mock[name] = p.start(); self.addCleanup(p.stop)
        p = patch('src.arc.lt_phase_b_registered_comparison.importlib.metadata.version',
                  side_effect=lambda name: {'mlx-lm': '0.30.7', 'mlx': '0.32.4'}[name])
        p.start(); self.addCleanup(p.stop)
        self.loads = []
        def backend(**kwargs):
            self.loads.append(kwargs)
            return Backend(**kwargs)
        self.backend = backend

    def invoke(self, state='off', **kwargs):
        f = self.fixture
        args = {'comparison_spec': self.comparison, 'expected_comparison_sha256': self.sha,
            'collection_spec': self.collection, 'expected_collection_sha256': self.collection_sha,
            'training_spec': self.training, 'expected_training_sha256': self.training_sha,
            'canary_spec': self.canary, 'expected_canary_sha256': self.canary_sha,
            'source_inventory': f.inventory, 'source_root': '.', 'run_dir': '/fixture/run', 'model_path': '/fixture/model',
            'allocation_receipt': self.allocation, 'collection_receipt': self.collection_receipt,
            'comparison_receipt': self.receipt, 'export': self.export, 'task_id': sorted(self.tasks)[0],
            'adapter_state': state, 'backend_factory': self.backend}
        args.update(kwargs)
        return run_registered_comparison_task(**args)

    def test_matched_states_use_same_generation_settings(self):
        off, on = self.invoke('off'), self.invoke('on')
        self.assertEqual(off['adapter_state'], 'off'); self.assertIsNone(off['adapter_sha256'])
        self.assertEqual(on['adapter_state'], 'on'); self.assertEqual(on['adapter_sha256'], 'f' * 64)
        self.assertEqual(off['prompt_sha256'], on['prompt_sha256'])
        for key in ('seed', 'max_tokens', 'temperature', 'model_manifest', 'model_path'):
            self.assertEqual(self.loads[0][key], self.loads[1][key])
        self.assertEqual(self.loads[1]['adapter_path'], '/fixture/run/gen1-adapter')

    def test_descriptor_or_adapter_mismatch_fails_before_load(self):
        for changed in ({'status': 'draft'}, {'adapter_sha256': '0' * 64}, {'prompt_sha256': '0' * 64},
                        {'selected_task_sha256': {}}, {'settings': {**self.comparison['settings'], 'seed': 1}}):
            spec = {**self.comparison, **changed}
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.invoke(comparison_spec=spec, expected_comparison_sha256=_digest(spec))
        self.assertEqual(self.loads, [])

    def test_reload_tampering_or_bad_state_never_loads(self):
        self.mock['review_real_reload_attempt'].side_effect = ValueError('changed')
        with self.assertRaises(ValueError): self.invoke('on')
        self.mock['review_real_reload_attempt'].side_effect = None
        with self.assertRaises(ValueError): self.invoke('bad')
        self.assertEqual(self.loads, [])


if __name__ == '__main__': unittest.main()
