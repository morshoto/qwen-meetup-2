import copy
import json
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_registration_proposal import propose_collection_registration
from src.arc.lt_phase_b_registered_worker import _digest, validate_descriptor
from tests.arc import test_lt_phase_b_registered_worker as fixtures


class RegistrationProposalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RegisteredWorkerTests('test_frozen_public_worker_binds_identity_before_and_after')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.challenge = json.dumps(f.challenges).encode()
        self.mock = {}
        for name, value in (('derive_allocation', f.allocation), ('capture_source_tree', f.inventory),
                            ('snapshot_runtime_identity', f.spec['runtime_manifest']),
                            ('snapshot_environment', f.spec['environment_manifest']),
                            ('snapshot_model_manifest', f.spec['model_manifest'])):
            p = patch('src.arc.lt_phase_b_registration_proposal.' + name, return_value=value)
            self.mock[name] = p.start(); self.addCleanup(p.stop)

    def invoke(self, **kwargs):
        args = {'source_root': '.', 'challenge_bytes': self.challenge, 'official_challenge_bytes': b'{}',
                'training_task_ids': sorted(self.fixture.hashes), 'run_id': 'fixture',
                'python_executable': '/fixture/python', 'model_path': '/fixture/model'}
        args.update(kwargs)
        return propose_collection_registration(**args)

    def test_proposal_is_draft_and_not_an_executable_registration(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'collection_registration_proposed_unverified')
        self.assertFalse(result['execution_ready'])
        self.assertEqual(result['spec']['status'], 'draft')
        self.assertEqual(result['spec_sha256'], _digest(result['spec']))
        self.assertEqual(result['receipt']['comparison_sha256'], result['spec_sha256'])
        self.assertEqual(set(result['export']['tasks']), set(self.fixture.hashes))
        with self.assertRaises(ValueError):
            validate_descriptor(spec=result['spec'], expected_run_sha256=result['spec_sha256'],
                source_inventory=result['inventory'], receipt=result['receipt'], allocation_receipt=result['allocation'])
        self.assertEqual(self.mock['capture_source_tree'].call_count, 2)

    def test_allocation_failure_prevents_runtime_and_model_probes(self):
        self.mock['derive_allocation'].side_effect = ValueError('reserved tasks overlap')
        with self.assertRaises(ValueError): self.invoke()
        self.mock['snapshot_model_manifest'].assert_not_called()
        self.mock['snapshot_runtime_identity'].assert_not_called()

    def test_changed_source_and_bad_allocation_are_rejected(self):
        self.mock['capture_source_tree'].side_effect = [self.fixture.inventory, {'code_sha256': 'f'*64}]
        with self.assertRaises(ValueError): self.invoke()
        self.mock['capture_source_tree'].side_effect = None
        allocation = copy.deepcopy(self.fixture.allocation); allocation['execution_ready'] = True
        self.mock['derive_allocation'].return_value = allocation
        with self.assertRaises(ValueError): self.invoke()

    def test_invalid_ids_or_run_identifier_refused(self):
        for args in ({'training_task_ids': []}, {'training_task_ids': ['00000000'] * 100}, {'run_id': '../bad'}):
            with self.subTest(args=args), self.assertRaises(ValueError): self.invoke(**args)


if __name__ == '__main__': unittest.main()
