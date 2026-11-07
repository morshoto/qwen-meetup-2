import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_phase_b_canary_config import build_canary_config
from src.arc.lt_phase_b_registered_canary import run_registered_canary
from tests.arc import test_lt_phase_b_registered_worker as fixtures


class RegisteredCanaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.model = root / 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'; self.model.mkdir()
        self.data = root / 'data'; self.data.mkdir(mode=0o700)
        self.evidence = root / 'evidence'; self.evidence.mkdir(mode=0o700)
        self.plan = build_canary_config(model_dir=self.model, data_dir=self.data, adapter_dir=root / 'adapter')
        (self.data / 'train.jsonl').write_text(self.plan['dataset_jsonl']); (self.data / 'train.jsonl').chmod(0o600)
        self.config = root / 'config.json'; self.config.write_text(json.dumps(self.plan['trainer_config'])); self.config.chmod(0o600)
        self.fixture = fixtures.RegisteredWorkerTests('test_frozen_public_worker_binds_identity_before_and_after')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.collection = self.fixture.spec
        self.collection_sha = fixtures.sha(self.collection)
        self.receipt = {'comparison_sha256': self.collection_sha, 'selected_task_sha256': self.fixture.hashes}
        self.canary = {'protocol': 'lt-phase-b-canary-v1', 'status': 'frozen', 'run_id': 'fixture-canary',
                       'collection_sha256': self.collection_sha, 'source_code_sha256': 'a' * 64,
                       'runtime_manifest': self.collection['runtime_manifest'],
                       'environment_manifest': self.collection['environment_manifest'],
                       'model_manifest': self.collection['model_manifest'],
                       'config_sha256': hashlib.sha256(self.config.read_bytes()).hexdigest(),
                       'dataset_sha256': self.plan['dataset_sha256'], 'thinking_mode': 'off',
                       'timeout_seconds': 300, 'resource_limits': self.collection['settings']['resource_limits']}
        for name in ('verify_source_tree', 'verify_runtime_identity', 'verify_environment', 'verify_model_manifest'):
            p = patch('src.arc.lt_phase_b_registered_canary.' + name)
            p.start(); self.addCleanup(p.stop)
        self.runtime = patch('src.arc.lt_phase_b_registered_canary._trainer_runtime', return_value=('trainer', 'policy', 'flatten', 'evaluate'))
        self.loader = self.runtime.start(); self.addCleanup(self.runtime.stop)
        self.instrument = patch('src.arc.lt_phase_b_registered_canary.run_instrumented_cli',
                                return_value={'status': 'tensor_change_consistent_unverified', 'training_verified': False})
        self.run = self.instrument.start(); self.addCleanup(self.instrument.stop)

    def invoke(self, **kwargs):
        args = {'canary_spec': self.canary, 'expected_canary_sha256': fixtures.sha(self.canary),
                'collection_spec': self.collection, 'expected_collection_sha256': self.collection_sha,
                'source_inventory': self.fixture.inventory, 'source_root': '.',
                'allocation_receipt': self.fixture.allocation, 'receipt': self.receipt,
                'config_path': self.config, 'evidence_dir': self.evidence}
        args.update(kwargs)
        return run_registered_canary(**args)

    def test_bound_canary_calls_instrumented_trainer_only_after_checks(self):
        result = self.invoke()
        self.assertEqual(result['canary_sha256'], fixtures.sha(self.canary))
        self.assertEqual(result['collection_sha256'], self.collection_sha)
        self.assertFalse(result['training_verified'])
        self.assertEqual(self.loader.call_count, 1)
        self.assertEqual(self.run.call_count, 1)

    def test_unfrozen_or_mismatched_canary_never_loads_trainer(self):
        for changed in (dict(self.canary, status='draft'), dict(self.canary, collection_sha256='f' * 64),
                        dict(self.canary, dataset_sha256='f' * 64), dict(self.canary, thinking_mode='on'),
                        dict(self.canary, timeout_seconds=301)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.invoke(canary_spec=changed, expected_canary_sha256=fixtures.sha(changed))
        self.assertEqual(self.loader.call_count, 0)
        self.assertEqual(self.run.call_count, 0)

    def test_changed_config_fails_before_runtime_load(self):
        self.config.write_text('{}')
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.loader.call_count, 0)


if __name__ == '__main__': unittest.main()
