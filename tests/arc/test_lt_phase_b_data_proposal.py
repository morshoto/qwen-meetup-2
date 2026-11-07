import hashlib
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_data_proposal import propose_data_registration
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_training_preflight as fixtures


class DataProposalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingPreflightTests('test_replays_only_selected_training_answers')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.protocol = {'data': {'challenge_sha256': hashlib.sha256(f.challenges).hexdigest(),
                                  'solution_sha256': hashlib.sha256(f.solutions).hexdigest()}}
        p = patch('src.arc.lt_phase_b_data_proposal._read_protocol', return_value=self.protocol)
        self.protocol_read = p.start(); self.addCleanup(p.stop)
        p = patch('src.arc.lt_phase_b_data_proposal.preflight_collection', return_value={
            'status': 'preflight_consistent_unverified', 'execution_ready': False, 'run_sha256': f.collection_sha,
            'selected_tasks': 4, 'evidence_type': 'local_files_and_caller_supplied_registration'})
        self.preflight = p.start(); self.addCleanup(p.stop)

    def invoke(self, **kwargs):
        f = self.fixture
        args = {'collection_spec': f.collection, 'expected_collection_sha256': f.collection_sha,
            'source_root': f.fixture.fixture.root, 'source_inventory': {}, 'allocation_receipt': f.allocation,
            'receipt': f.receipt, 'export': f.export, 'challenge_bytes': f.challenges,
            'solution_bytes': f.solutions, 'official_challenge_bytes': b'{}',
            'python_executable': '/fixture/python', 'model_path': '/fixture/model'}
        args.update(kwargs)
        return propose_data_registration(**args)

    def test_returns_non_authorizing_draft_without_answers(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'data_registration_proposed_unverified')
        self.assertFalse(result['execution_ready']); self.assertFalse(result['training_allowed'])
        self.assertEqual(result['spec']['status'], 'draft')
        self.assertEqual(result['spec']['solution_sha256'], self.protocol['data']['solution_sha256'])
        self.assertEqual(result['spec_sha256'], _digest(result['spec']))
        self.assertNotIn('fixture', str(result))
        self.assertEqual(self.preflight.call_count, 2)

    def test_mismatched_data_or_changed_protocol_refuses_proposal(self):
        with self.assertRaises(ValueError): self.invoke(solution_bytes=b'{}')
        self.protocol_read.side_effect = [self.protocol, {'data': {'challenge_sha256': '0'*64}}]
        with self.assertRaises(ValueError): self.invoke()

    def test_preflight_failure_refuses_proposal(self):
        self.preflight.side_effect = ValueError('invalid allocation')
        with self.assertRaises(ValueError): self.invoke()


if __name__ == '__main__': unittest.main()
