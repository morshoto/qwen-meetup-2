import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_phase_b_proposal_dispatch import save_collection_proposal
from src.arc.lt_phase_b_registered_worker import _digest


class ProposalDispatchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); (self.root / 'artifacts').mkdir()
        self.run = self.root / 'artifacts/lt-phase-b-fixture'; self.run.mkdir(mode=0o700)
        self.pool = self.root / 'pool.json'; self.pool.write_text(json.dumps([f'{i:08x}' for i in range(100)]))
        self.challenge = self.root / 'challenge.json'; self.challenge.write_text('{}')
        self.archive = self.root / 'fixture.zip'
        with zipfile.ZipFile(self.archive, 'w') as archive: archive.writestr('arc-agi_evaluation_challenges.json', '{}')
        self.spec = {'status': 'draft', 'run_id': 'fixture'}
        self.proposal = {'status': 'collection_registration_proposed_unverified', 'execution_ready': False,
            'spec': self.spec, 'spec_sha256': _digest(self.spec), 'inventory': {}, 'allocation': {},
            'receipt': {}, 'export': {'tasks': {f'{i:08x}': {} for i in range(100)}}}
        p = patch('src.arc.lt_phase_b_proposal_dispatch.propose_collection_registration', return_value=self.proposal)
        self.propose = p.start(); self.addCleanup(p.stop)

    def invoke(self, **kwargs):
        args = {'source_root': self.root, 'run_dir': self.run, 'pool_path': self.pool, 'challenge_path': self.challenge,
                'official_archive': self.archive, 'python_executable': '/fixture/python', 'model_path': '/fixture/model'}
        args.update(kwargs)
        return save_collection_proposal(**args)

    def test_saves_private_draft_once_and_returns_only_summary(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'collection_registration_proposed_unverified')
        self.assertFalse(result['execution_ready'])
        self.assertEqual(result['task_count'], 100)
        self.assertNotIn('export', result)
        saved = self.run / 'registration-proposal.json'
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(saved.read_text()), self.proposal)
        with self.assertRaises(ValueError): self.invoke()
        self.propose.assert_called_once()

    def test_refused_allocation_writes_no_proposal(self):
        self.propose.side_effect = ValueError('training task intersects exclusion')
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.run / 'registration-proposal.json').exists())

    def test_never_persists_non_draft_or_changed_inputs(self):
        self.propose.return_value = {**self.proposal, 'spec': {'status': 'frozen', 'run_id': 'fixture'}}
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.run / 'registration-proposal.json').exists())
        def mutate(**kwargs):
            self.pool.write_text('[]')
            return self.proposal
        self.propose.side_effect = mutate
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.run / 'registration-proposal.json').exists())


if __name__ == '__main__': unittest.main()
