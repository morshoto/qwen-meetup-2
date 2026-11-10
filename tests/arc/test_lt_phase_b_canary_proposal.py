import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_canary_proposal import propose_canary_registration
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_registered_canary as fixtures


class CanaryProposalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RegisteredCanaryTests('test_bound_canary_calls_instrumented_trainer_only_after_checks')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.checks = {}
        for name in ('verify_source_tree', 'verify_runtime_identity', 'verify_environment', 'verify_model_manifest'):
            p = patch('src.arc.lt_phase_b_canary_proposal.' + name)
            self.checks[name] = p.start(); self.addCleanup(p.stop)

    def invoke(self, **kwargs):
        f = self.fixture
        args = {'collection_spec': f.collection, 'expected_collection_sha256': f.collection_sha,
            'source_inventory': f.fixture.inventory, 'source_root': '.', 'allocation_receipt': f.fixture.allocation,
            'receipt': f.receipt, 'model_dir': f.model, 'data_dir': f.data, 'adapter_dir': f.data.parent / 'adapter',
            'python_executable': '/fixture/python', 'canary_run_id': 'fixture-canary'}
        args.update(kwargs)
        return propose_canary_registration(**args)

    def test_draft_binds_exact_config_and_synthetic_example(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'canary_registration_proposed_unverified')
        self.assertFalse(result['execution_ready']); self.assertFalse(result['promotion_allowed'])
        self.assertEqual(result['spec']['status'], 'draft')
        self.assertEqual(result['spec_sha256'], _digest(result['spec']))
        raw = (json.dumps(result['trainer_config'], sort_keys=True, allow_nan=False) + '\n').encode()
        self.assertEqual(result['spec']['config_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['spec']['dataset_sha256'], hashlib.sha256(result['dataset_jsonl'].encode()).hexdigest())
        f = self.fixture
        with self.assertRaises(ValueError):
            validate_canary_identity(canary_spec=result['spec'], expected_canary_sha256=result['spec_sha256'],
                collection_spec=f.collection, expected_collection_sha256=f.collection_sha,
                source_inventory=f.fixture.inventory, allocation_receipt=f.fixture.allocation, receipt=f.receipt)
        for check in self.checks.values(): self.assertEqual(check.call_count, 2)

    def test_bad_registration_or_existing_adapter_refuses_proposal(self):
        with self.assertRaises(ValueError): self.invoke(expected_collection_sha256='0' * 64)
        (self.fixture.data.parent / 'adapter').mkdir()
        with self.assertRaises(ValueError): self.invoke()

    def test_changed_or_extra_synthetic_data_is_rejected(self):
        path = self.fixture.data / 'train.jsonl'; path.write_text('changed')
        with self.assertRaises(ValueError): self.invoke()
        path.write_text(self.fixture.plan['dataset_jsonl'])
        (self.fixture.data / 'extra.jsonl').write_text('{}')
        with self.assertRaises(ValueError): self.invoke()


if __name__ == '__main__': unittest.main()
