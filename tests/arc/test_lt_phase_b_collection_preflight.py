import json
import unittest
from unittest.mock import patch

from src.arc.lt_public import build_public_export
from src.arc.lt_phase_b_collection_preflight import preflight_collection
from tests.arc import test_lt_phase_b_registered_worker as fixture_module


class CollectionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.RegisteredWorkerTests('test_frozen_public_worker_binds_identity_before_and_after')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.spec = self.fixture.spec
        self.run_sha = fixture_module.sha(self.spec)
        self.receipt = {'comparison_sha256': self.run_sha, 'selected_task_sha256': self.fixture.hashes}
        self.export = build_public_export(self.fixture.challenges, self.receipt)
        self.challenges = json.dumps(self.fixture.challenges).encode()
        self.audit = patch('src.arc.lt_phase_b_collection_preflight.derive_allocation',
                           return_value=self.fixture.allocation)
        self.derive = self.audit.start(); self.addCleanup(self.audit.stop)
        self.sources = patch('src.arc.lt_phase_b_collection_preflight.verify_source_tree')
        self.verify = self.sources.start(); self.addCleanup(self.sources.stop)

    def call(self, **kwargs):
        args = {'source_root': '.', 'spec': self.spec, 'expected_run_sha256': self.run_sha,
                'source_inventory': self.fixture.inventory, 'allocation_receipt': self.fixture.allocation,
                'export': self.export, 'receipt': self.receipt, 'challenge_bytes': self.challenges,
                'official_challenge_bytes': b'{}'}
        args.update(kwargs)
        return preflight_collection(**args)

    def test_recomputes_allocation_and_public_export(self):
        result = self.call()
        self.assertEqual(result['status'], 'preflight_consistent_unverified')
        self.assertFalse(result['execution_ready'])
        self.assertEqual(result['selected_tasks'], 100)
        self.assertEqual(self.derive.call_count, 1)
        self.assertEqual(self.verify.call_count, 2)

    def test_refuses_changed_allocation_export_or_run(self):
        for changed in ({'expected_run_sha256': 'f' * 64}, {'export': {'bad': True}},
                        {'allocation_receipt': {**self.fixture.allocation, 'execution_ready': True}}):
            with self.subTest(changed=changed), self.assertRaises(ValueError): self.call(**changed)
        self.derive.return_value = {**self.fixture.allocation, 'different': True}
        with self.assertRaises(ValueError): self.call()


if __name__ == '__main__': unittest.main()
