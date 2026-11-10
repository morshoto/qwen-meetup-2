import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_registered_training import run_registered_training
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_training_config as fixtures


class RegisteredTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingConfigTests('test_disjoint_yield_proposes_fixed_final_step_configuration')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.build()
        self.run = self.fixture.fixture.run
        self.source = self.fixture.fixture.fixture.fixture.source
        self.collection_sha = self.source['run_sha256']
        self.collection = {'run_id': self.source['run_id'], 'source_code_sha256': self.source['source_code_sha256'],
            'selected_task_sha256': {key: 'f' * 64 for key in ('a', 'b', 'c', 'd')},
            'runtime_manifest': {}, 'environment_manifest': {}}
        self.settings = {'task_timeout_seconds': 300, 'resource_limits': {'timeout_seconds': 300}}
        self.canary = {'fixture': 'canary'}; self.canary_sha = _digest(self.canary)
        self.config = self.run / 'training-config.json'
        self.config.write_text(json.dumps(self.plan['trainer_config'])); self.config.chmod(0o600)
        self.config_sha = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.spec = {'protocol': 'lt-phase-b-real-training-v1', 'status': 'frozen',
            'run_id': self.source['run_id'], 'collection_sha256': self.collection_sha,
            'canary_sha256': self.canary_sha, 'source_code_sha256': self.source['source_code_sha256'],
            'dataset_manifest_sha256': self.plan['dataset_manifest_sha256'], 'config_sha256': self.config_sha,
            'solution_sha256': 'b' * 64,
            'yield_status': 'disjoint_split', 'checkpoint_policy': 'final_step_only',
            'timeout_seconds': 300, 'resource_limits': self.settings['resource_limits']}
        self.evidence = self.run / 'training-evidence'; self.evidence.mkdir(mode=0o700)
        patches = {'validate_canary_identity': (self.settings, {'fixture': 'model'}),
            'review_reload_attempt': {'status': 'canary_reload_observed_unverified', 'reload_verified': False,
                                      'promotion_allowed': False},
            '_trainer_runtime': ('trainer', 'policy', 'flatten', 'evaluate'),
            'run_real_instrumented_cli': {'status': 'tensor_change_consistent_unverified', 'training_verified': False}}
        self.mock = {}
        for name, result in patches.items():
            p = patch('src.arc.lt_phase_b_registered_training.' + name, return_value=result)
            self.mock[name] = p.start(); self.addCleanup(p.stop)
        for name in ('verify_runtime_identity', 'verify_environment', 'verify_model_manifest', 'verify_source_tree'):
            p = patch('src.arc.lt_phase_b_registered_training.' + name)
            self.mock[name] = p.start(); self.addCleanup(p.stop)

    def invoke(self, **kwargs):
        params = {'training_spec': self.spec, 'expected_training_sha256': _digest(self.spec),
            'canary_spec': self.canary, 'expected_canary_sha256': self.canary_sha,
            'collection_spec': self.collection, 'expected_collection_sha256': self.collection_sha,
            'source_inventory': {}, 'source_root': self.fixture.fixture.root, 'run_dir': self.run,
            'model_path': self.fixture.model, 'allocation_receipt': {}, 'receipt': {},
            'config_path': self.config, 'evidence_dir': self.evidence}
        params.update(kwargs)
        return run_registered_training(**params)

    def test_bound_worker_invokes_instrumentation_after_replay(self):
        result = self.invoke()
        self.assertEqual(result['training_sha256'], _digest(self.spec))
        self.assertFalse(result['training_verified'])
        self.assertFalse(result['promotion_allowed'])
        self.assertEqual(self.mock['review_reload_attempt'].call_count, 2)
        self.mock['run_real_instrumented_cli'].assert_called_once()
        for name in ('verify_runtime_identity', 'verify_environment', 'verify_model_manifest', 'verify_source_tree'):
            self.assertEqual(self.mock[name].call_count, 2)

    def test_invalid_descriptor_and_canary_fail_before_runtime(self):
        for changes in ({'status': 'draft'}, {'dataset_manifest_sha256': 'f' * 64},
                        {'config_sha256': 'f' * 64}, {'yield_status': 'overfitting_only'},
                        {'timeout_seconds': 301}, {'solution_sha256': 'invalid'}, {'canary_sha256': 'f' * 64}):
            spec = {**self.spec, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.invoke(training_spec=spec, expected_training_sha256=_digest(spec))
        self.mock['_trainer_runtime'].assert_not_called()

    def test_replay_and_unregistered_tasks_fail_before_runtime(self):
        self.mock['review_reload_attempt'].side_effect = ValueError('canary changed')
        with self.assertRaises(ValueError): self.invoke()
        self.mock['review_reload_attempt'].side_effect = None
        collection = {**self.collection, 'selected_task_sha256': {'d': 'f' * 64}}
        with self.assertRaises(ValueError): self.invoke(collection_spec=collection)
        self.mock['_trainer_runtime'].assert_not_called()

    def test_bad_instrumentation_result_cannot_be_promoted(self):
        self.mock['run_real_instrumented_cli'].return_value = {'status': 'ok', 'training_verified': True}
        with self.assertRaises(ValueError): self.invoke()


if __name__ == '__main__': unittest.main()
