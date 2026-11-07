import json
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_canary_reload import run_canary_reload
from tests.arc import test_lt_phase_b_canary_dispatch as fixtures


class ReloadBackend:
    adapter_state = 'on'
    adapter_sha256 = 'f' * 64
    raw = '[[[2]]]'
    stats = {'prompt_tokens': 20, 'completion_tokens': 4, 'peak_memory_gb': 1., 'finish_reason': 'stop'}
    def generate(self, prompt, *, kind, candidate):
        assert kind == 'direct' and candidate == 0
        return self.raw


class CanaryReloadTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CanaryDispatchTests('test_records_one_shot_observation_and_reload_pending')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.summary = self.fixture.invoke()
        self.fixture.adapter.mkdir()
        self.created = []
        self.checked = []
        for name in ('verify_source_tree', 'verify_runtime_identity', 'verify_environment', 'verify_model_manifest'):
            p = patch('src.arc.lt_phase_b_canary_reload.' + name)
            self.checked.append(p.start()); self.addCleanup(p.stop)
        self.inspect = patch('src.arc.lt_phase_b_canary_reload.inspect_adapter_artifact', return_value=self.summary['adapter'])
        self.inspect.start(); self.addCleanup(self.inspect.stop)

    def invoke(self, **kwargs):
        f = self.fixture
        def factory(**args):
            self.created.append(args)
            return ReloadBackend()
        args = {'canary_spec': f.canary, 'expected_canary_sha256': f.canary_sha,
                'collection_spec': f.spec, 'expected_collection_sha256': f.collection_sha,
                'source_inventory': f.fixture.fixture.inventory, 'source_root': f.fixture.root,
                'allocation_receipt': f.fixture.fixture.allocation,
                'receipt': json.loads((f.run / 'receipt.json').read_text()), 'config_path': f.config,
                'training_summary': self.summary,
                'tensor_before': json.loads((f.evidence / 'tensor-before.json').read_text()),
                'tensor_after': json.loads((f.evidence / 'tensor-after.json').read_text()),
                'backend_factory': factory}
        args.update(kwargs)
        return run_canary_reload(**args)

    def test_reload_binds_adapter_and_generates_unscored_sample(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'reload_observed_unverified')
        self.assertEqual(result['adapter_sha256'], 'f' * 64)
        self.assertEqual(result['usage']['completion_tokens'], 4)
        self.assertFalse(result['reload_verified'])
        self.assertFalse(result['promotion_allowed'])
        self.assertEqual(self.created[0]['adapter_state'], 'on')
        self.assertEqual(self.created[0]['max_tokens'], 64)
        self.assertTrue(all(mock.call_count == 2 for mock in self.checked))

    def test_bad_training_or_adapter_identity_fails_before_loading(self):
        for bad in (dict(self.summary, status='training_failed'),
                    dict(self.summary, canary_sha256='0' * 64),
                    dict(self.summary, adapter={**self.summary['adapter'], 'adapter_sha256': '0' * 64})):
            with self.subTest(bad=bad), self.assertRaises(ValueError): self.invoke(training_summary=bad)
        self.assertEqual(self.created, [])

    def test_invalid_sample_or_loaded_adapter_cannot_pass(self):
        class Invalid(ReloadBackend):
            adapter_sha256 = '0' * 64
        with self.assertRaises(ValueError): self.invoke(backend_factory=lambda **_: Invalid())
        class Empty(ReloadBackend):
            raw = ''
        with self.assertRaises(ValueError): self.invoke(backend_factory=lambda **_: Empty())


if __name__ == '__main__': unittest.main()
