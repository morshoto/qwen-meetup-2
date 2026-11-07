import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_training_proposal import propose_training_registration
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_training_preflight as fixtures


class TrainingProposalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingPreflightTests('test_replays_only_selected_training_answers')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture; self.run = f.fixture.fixture.run
        self.model = f.fixture.fixture.root / 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'; self.model.mkdir()
        self.canary = {'fixture': 'canary'}; self.canary_sha = _digest(self.canary)
        self.data = {'protocol': 'lt-phase-b-data-v1', 'status': 'frozen', 'run_id': f.collection['run_id'],
            'source_code_sha256': f.collection['source_code_sha256'], 'collection_sha256': f.collection_sha,
            'solution_sha256': hashlib.sha256(f.solutions).hexdigest()}
        self.settings = {'task_timeout_seconds': 300, 'resource_limits': f.fixture.limits}
        patches = {'validate_canary_identity': (self.settings, {'fixture': 'model'}),
            'review_reload_attempt': {'status': 'canary_reload_observed_unverified', 'reload_verified': False,
                                      'promotion_allowed': False}}
        self.mock = {}
        for name, result in patches.items():
            p = patch('src.arc.lt_phase_b_training_proposal.' + name, return_value=result)
            self.mock[name] = p.start(); self.addCleanup(p.stop)
        p = patch('src.arc.lt_phase_b_training_proposal.preflight_training_data')
        self.preflight = p.start(); self.addCleanup(p.stop)
        def check(**kwargs):
            spec = kwargs['training_spec']
            return {'status': 'training_data_consistent_unverified', 'training_allowed': False,
                'training_sha256': kwargs['expected_training_sha256'], 'solution_sha256': spec['solution_sha256'],
                'dataset_manifest_sha256': spec['dataset_manifest_sha256'], 'attempted_tasks': 4,
                'eligible_tasks': 3, 'yield_status': 'disjoint_split'}
        self.preflight.side_effect = check

    def invoke(self, **kwargs):
        f = self.fixture
        args = {'collection_spec': f.collection, 'expected_collection_sha256': f.collection_sha,
            'canary_spec': self.canary, 'expected_canary_sha256': self.canary_sha,
            'data_spec': self.data, 'expected_data_sha256': _digest(self.data),
            'source_root': f.fixture.fixture.root, 'run_dir': self.run, 'source_inventory': {},
            'allocation_receipt': f.allocation, 'receipt': f.receipt, 'export': f.export,
            'challenge_bytes': f.challenges, 'solution_bytes': f.solutions, 'official_challenge_bytes': b'{}',
            'python_executable': '/fixture/python', 'model_path': self.model}
        args.update(kwargs)
        return propose_training_registration(**args)

    def test_proposes_fixed_draft_after_replaying_data_and_canary(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'real_training_registration_proposed_unverified')
        self.assertFalse(result['execution_ready']); self.assertFalse(result['training_allowed'])
        self.assertEqual(result['spec']['status'], 'draft')
        self.assertEqual(result['spec']['yield_status'], 'disjoint_split')
        self.assertEqual(result['trainer_config']['iters'], 20)
        encoded = (json.dumps(result['trainer_config'], sort_keys=True, allow_nan=False) + '\n').encode()
        self.assertEqual(result['spec']['config_sha256'], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(result['spec_sha256'], _digest(result['spec']))
        self.assertEqual(self.preflight.call_count, 2)
        self.assertEqual(self.mock['review_reload_attempt'].call_count, 2)

    def test_proposal_replays_real_fixture_collection_and_dataset(self):
        from src.arc.lt_phase_b_training_preflight import preflight_training_data
        self.preflight.side_effect = preflight_training_data
        result = self.invoke()
        self.assertEqual(result['eligible_tasks'], 3)
        self.assertFalse(result['training_allowed'])
        self.assertEqual(result['spec']['dataset_manifest_sha256'], self.fixture.training['dataset_manifest_sha256'])

    def test_missing_canary_or_changed_dataset_refuses_proposal(self):
        self.mock['review_reload_attempt'].side_effect = ValueError('canary missing')
        with self.assertRaises(ValueError): self.invoke()
        self.mock['review_reload_attempt'].side_effect = None
        (self.run / 'sft-data/train.jsonl').write_text('tampered')
        with self.assertRaises(ValueError): self.invoke()

    def test_changed_canary_after_preflight_rejects_proposal(self):
        before = self.mock['review_reload_attempt'].return_value
        self.mock['review_reload_attempt'].side_effect = [before, {**before, 'reload_verified': True}]
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.run / 'training-spec.json').exists())

    def test_unfrozen_or_changed_data_registration_is_rejected(self):
        for change in ({'status': 'draft'}, {'solution_sha256': 'f'*64}, {'collection_sha256': 'f'*64}):
            data = {**self.data, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.invoke(data_spec=data, expected_data_sha256=_digest(data))


if __name__ == '__main__': unittest.main()
