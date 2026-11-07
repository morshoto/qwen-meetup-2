import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_comparison_proposal import propose_comparison_registration
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_registered_comparison as fixtures


class ComparisonProposalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RegisteredComparisonTests('test_matched_states_use_same_generation_settings')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.challenge = json.dumps(f.export['tasks']).encode()
        self.solutions = b'{"fixture":[]}'
        self.training = {'solution_sha256': hashlib.sha256(self.solutions).hexdigest(),
                         'dataset_manifest_sha256': 'd'*64, 'yield_status': 'disjoint_split'}
        self.training_sha = _digest(self.training)
        patches = {'validate_canary_identity': (f.collection['settings'], f.collection['model_manifest']),
            'review_real_reload_attempt': {'status': 'real_adapter_reload_observed_unverified', 'reload_verified': False,
                'promotion_allowed': False, 'reload': {'adapter_sha256': 'f'*64}}}
        self.mock = {}
        for name, value in patches.items():
            p = patch('src.arc.lt_phase_b_comparison_proposal.' + name, return_value=value)
            self.mock[name] = p.start(); self.addCleanup(p.stop)
        p = patch('src.arc.lt_phase_b_comparison_proposal.preflight_training_data', return_value={
            'status': 'training_data_consistent_unverified', 'training_allowed': False,
            'training_sha256': self.training_sha, 'solution_sha256': self.training['solution_sha256'],
            'dataset_manifest_sha256': self.training['dataset_manifest_sha256'], 'attempted_tasks': 100,
            'eligible_tasks': 3, 'yield_status': 'disjoint_split'})
        self.preflight = p.start(); self.addCleanup(p.stop)

    def invoke(self, **kwargs):
        f = self.fixture
        args = {'collection_spec': f.collection, 'expected_collection_sha256': f.collection_sha,
            'training_spec': self.training, 'expected_training_sha256': self.training_sha,
            'canary_spec': f.canary, 'expected_canary_sha256': f.canary_sha,
            'source_root': '.', 'run_dir': '/fixture/run', 'source_inventory': f.fixture.inventory,
            'allocation_receipt': f.allocation, 'collection_receipt': f.collection_receipt,
            'collection_export': {}, 'challenge_bytes': self.challenge, 'solution_bytes': self.solutions,
            'official_challenge_bytes': b'{}', 'python_executable': '/fixture/python', 'model_path': '/fixture/model'}
        args.update(kwargs)
        return propose_comparison_registration(**args)

    def test_proposes_only_public_tasks_and_matched_settings(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'comparison_registration_proposed_unverified')
        self.assertFalse(result['execution_ready'])
        self.assertEqual(result['spec']['status'], 'draft')
        self.assertEqual(result['spec_sha256'], _digest(result['spec']))
        self.assertEqual(result['receipt']['comparison_sha256'], result['spec_sha256'])
        self.assertEqual(result['spec']['adapter_sha256'], 'f'*64)
        self.assertEqual(result['spec']['settings'], self.fixture.collection['settings'])
        self.assertEqual(len(result['export']['tasks']), 50)
        self.assertTrue(all(set(item) == {'input'} for task in result['export']['tasks'].values() for item in task['test']))
        self.assertNotIn('fixture', str(result['export']))
        self.assertEqual(self.preflight.call_count, 2)

    def test_tampered_public_inventory_or_reload_fails(self):
        with self.assertRaises(ValueError): self.invoke(challenge_bytes=b'{}')
        self.mock['review_real_reload_attempt'].side_effect = ValueError('reload missing')
        with self.assertRaises(ValueError): self.invoke()

    def test_changed_evidence_or_answer_file_fails(self):
        with self.assertRaises(ValueError): self.invoke(solution_bytes=b'{}')
        first = self.mock['review_real_reload_attempt'].return_value
        self.mock['review_real_reload_attempt'].side_effect = [first, {**first, 'reload_verified': True}]
        with self.assertRaises(ValueError): self.invoke()


if __name__ == '__main__': unittest.main()
