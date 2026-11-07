import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_training_preflight import preflight_training_data
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_public import build_public_export
from tests.arc import test_lt_phase_b_collection_replay as fixtures
from tests.arc import test_lt_phase_b_dataset as examples


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CollectionReplayTests('test_rebuilds_verified_split_and_matches_materialized_hashes')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        challenges = {k: v.prompt_data() for k, v in f.f.tasks.items()}
        challenges['x'] = examples.task('x').prompt_data()
        self.challenges = json.dumps(challenges).encode()
        self.solutions = json.dumps({**{k: [[[1]]] for k in f.f.tasks}, 'x': [[[9]]]}).encode()
        self.collection_sha = f.source['run_sha256']
        self.selected = {k: _digest(challenges[k]) for k in f.f.tasks}
        self.comparison = {'x': _digest(challenges['x'])}
        self.collection = {'run_id': f.source['run_id'], 'source_code_sha256': f.source['source_code_sha256'],
                           'selected_task_sha256': self.selected, 'settings': {'resource_limits': f.limits}}
        self.receipt = {'comparison_sha256': self.collection_sha, 'selected_task_sha256': self.selected}
        self.export = build_public_export(challenges, self.receipt)
        self.allocation = {'comparison_task_sha256': self.comparison}
        data = f.fixture.run / 'sft-data/manifest.json'
        self.training = {'protocol': 'lt-phase-b-real-training-v1', 'status': 'frozen',
            'collection_sha256': self.collection_sha, 'solution_sha256': hashlib.sha256(self.solutions).hexdigest(),
            'dataset_manifest_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}
        patcher = patch('src.arc.lt_phase_b_training_preflight.preflight_collection', return_value={
            'status': 'preflight_consistent_unverified', 'execution_ready': False, 'run_sha256': self.collection_sha,
            'selected_tasks': 4, 'evidence_type': 'local_files_and_caller_supplied_registration'})
        self.collection_preflight = patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch('src.arc.lt_phase_b_training_preflight.require_protocol_solutions')
        self.protocol_solutions = patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch('src.arc.lt_phase_b_training_preflight.require_preparation_receipt')
        self.preparation_receipt = patcher.start(); self.addCleanup(patcher.stop)

    def invoke(self, **kwargs):
        args = {'training_spec': self.training, 'expected_training_sha256': _digest(self.training),
            'collection_spec': self.collection, 'expected_collection_sha256': self.collection_sha,
            'source_root': self.fixture.fixture.root, 'run_dir': self.fixture.fixture.run, 'source_inventory': {},
            'allocation_receipt': self.allocation, 'receipt': self.receipt, 'export': self.export,
            'challenge_bytes': self.challenges, 'solution_bytes': self.solutions, 'official_challenge_bytes': b'{}',
            'python_executable': '/fixture', 'model_path': '/fixture'}
        args.update(kwargs)
        return preflight_training_data(**args)

    def test_replays_only_selected_training_answers(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'training_data_consistent_unverified')
        self.assertEqual(result['eligible_tasks'], 3)
        self.assertFalse(result['training_allowed'])
        self.assertNotIn('answers', result)
        self.collection_preflight.assert_called_once()

    def test_changed_solution_or_descriptor_fails(self):
        with self.assertRaises(ValueError): self.invoke(solution_bytes=b'{}')
        for change in ({'status': 'draft'}, {'dataset_manifest_sha256': 'f' * 64}, {'collection_sha256': 'e' * 64}):
            training = {**self.training, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.invoke(training_spec=training, expected_training_sha256=_digest(training))

    def test_protocol_solution_mismatch_refuses_before_eligibility_replay(self):
        self.protocol_solutions.side_effect = ValueError('protocol solutions differ')
        with patch('src.arc.lt_phase_b_training_preflight.replay_collection_dataset') as replay:
            with self.assertRaises(ValueError): self.invoke()
            replay.assert_not_called()

    def test_rebound_descriptor_cannot_override_frozen_protocol_answers(self):
        from src.arc.lt_phase_b_protocol_gate import require_protocol_solutions
        root = self.fixture.fixture.root
        (root / 'configs').mkdir()
        protocol = {'protocol': 'lt-phase-b-exploratory', 'status': 'frozen',
                    'data': {'solution_sha256': hashlib.sha256(self.solutions).hexdigest()}}
        path = root / 'configs/lt.phase-b.protocol.json'; path.write_text(json.dumps(protocol))
        inventory = {'files': {'configs/lt.phase-b.protocol.json': hashlib.sha256(path.read_bytes()).hexdigest()}}
        alternate = json.dumps({**json.loads(self.solutions), 'a': [[[9]]]}).encode()
        training = {**self.training, 'solution_sha256': hashlib.sha256(alternate).hexdigest()}
        self.protocol_solutions.side_effect = require_protocol_solutions
        with patch('src.arc.lt_phase_b_training_preflight.replay_collection_dataset') as replay:
            with self.assertRaises(ValueError):
                self.invoke(solution_bytes=alternate, training_spec=training, expected_training_sha256=_digest(training),
                            source_inventory=inventory)
            replay.assert_not_called()

    def test_incomplete_preparation_never_reaches_collection_replay(self):
        self.preparation_receipt.side_effect = ValueError('preparation incomplete')
        with patch('src.arc.lt_phase_b_training_preflight.replay_collection_dataset') as replay:
            with self.assertRaises(ValueError): self.invoke()
            replay.assert_not_called()

    def test_wrong_training_answers_or_comparison_content_fail(self):
        values = json.loads(self.solutions); values['a'] = [[[9]]]
        bad = json.dumps(values).encode()
        training = {**self.training, 'solution_sha256': hashlib.sha256(bad).hexdigest()}
        with self.assertRaises(ValueError):
            self.invoke(solution_bytes=bad, training_spec=training, expected_training_sha256=_digest(training))
        with self.assertRaises(ValueError):
            self.invoke(allocation_receipt={'comparison_task_sha256': {'x': 'f' * 64}})


if __name__ == '__main__': unittest.main()
