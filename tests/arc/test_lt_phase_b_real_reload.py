import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_phase_b_real_reload import run_real_reload
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_training_dispatch as fixtures


class Backend:
    adapter_state = 'on'
    adapter_sha256 = 'f' * 64
    def __init__(self, **kwargs): self.kwargs = kwargs
    def generate(self, prompt, *, kind, candidate):
        self.raw = '[[[1]]]'
        self.stats = {'prompt_tokens': 20, 'completion_tokens': 4, 'peak_memory_gb': 1., 'finish_reason': 'stop'}


class RealReloadTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingDispatchTests('test_one_shot_train_records_reload_pending_and_replays')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture.fixture
        self.run = f.run
        self.adapter = self.run / 'gen1-adapter'; self.adapter.mkdir()
        self.config = self.fixture.config
        self.config.write_text(json.dumps({'model': str(f.model), 'adapter_path': str(self.adapter)}))
        self.config.chmod(0o600)
        self.training = {**self.fixture.spec, 'config_sha256': hashlib.sha256(self.config.read_bytes()).hexdigest()}
        self.training_sha = _digest(self.training)
        self.tensor = {'fixture': 'tensor'}
        self.summary = {'status': 'real_training_observed_reload_pending', 'training_verified': False,
            'promotion_allowed': False, 'training_sha256': self.training_sha,
            'collection_sha256': f.collection_sha, 'canary_sha256': f.canary_sha,
            'yield_status': 'disjoint_split', 'adapter': {'adapter_sha256': 'f' * 64,
                'reload_verified': False, 'promotion_allowed': False}}
        self.reload = {'status': 'canary_reload_observed_unverified', 'reload_verified': False, 'promotion_allowed': False}
        p = patch('src.arc.lt_phase_b_real_reload.validate_canary_identity', return_value=(f.spec['settings'], f.spec['model_manifest']))
        self.valid = p.start(); self.addCleanup(p.stop)
        p = patch('src.arc.lt_phase_b_real_reload.review_real_training_attempt', return_value=self.summary)
        self.review = p.start(); self.addCleanup(p.stop)
        p = patch('src.arc.lt_phase_b_real_reload.review_reload_attempt', return_value=self.reload)
        self.canary = p.start(); self.addCleanup(p.stop)
        for name in ('verify_source_tree', 'verify_runtime_identity', 'verify_environment', 'verify_model_manifest'):
            p = patch('src.arc.lt_phase_b_real_reload.' + name); p.start(); self.addCleanup(p.stop)
        self.loaded = []
        def factory(**kwargs):
            self.loaded.append(kwargs)
            return Backend(**kwargs)
        self.factory = factory
        (self.run / 'training-summary.json').write_text(json.dumps(self.summary))

    def invoke(self, **kwargs):
        f = self.fixture.fixture
        args = {'training_spec': self.training, 'expected_training_sha256': self.training_sha,
            'canary_spec': f.canary, 'expected_canary_sha256': f.canary_sha,
            'collection_spec': f.spec, 'expected_collection_sha256': f.collection_sha,
            'source_inventory': {}, 'source_root': f.fixture.root, 'run_dir': self.run,
            'allocation_receipt': {}, 'receipt': {}, 'config_path': self.config,
            'backend_factory': self.factory}
        args.update(kwargs)
        return run_real_reload(**args)

    def test_reloads_real_adapter_and_samples_without_promotion(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'real_reload_observed_unverified')
        self.assertEqual(result['adapter_sha256'], 'f' * 64)
        self.assertFalse(result['reload_verified']); self.assertFalse(result['promotion_allowed'])
        self.assertEqual(len(self.loaded), 1)
        self.assertEqual(Path(self.loaded[0]['adapter_path']).resolve(), self.adapter.resolve())
        self.assertEqual(self.review.call_count, 2); self.assertEqual(self.canary.call_count, 2)

    def test_changed_config_or_training_summary_fails_before_load(self):
        self.config.write_text('{}')
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.loaded, [])
        self.config.write_text(json.dumps({'model': str(self.fixture.fixture.model), 'adapter_path': str(self.adapter)}))
        (self.run / 'training-summary.json').write_text('{}')
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.loaded, [])

    def test_mismatched_state_or_adapter_fails(self):
        class Wrong(Backend): adapter_sha256 = '0' * 64
        with self.assertRaises(ValueError): self.invoke(backend_factory=Wrong)

    def test_changed_training_evidence_after_sample_fails(self):
        changed = {**self.summary, 'yield_status': 'overfitting_only'}
        self.review.side_effect = [self.summary, changed]
        with self.assertRaises(ValueError): self.invoke()


if __name__ == '__main__': unittest.main()
