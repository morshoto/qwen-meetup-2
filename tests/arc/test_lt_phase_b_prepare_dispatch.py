import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_phase_b_prepare_dispatch import run_prepare_sft
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_prepare_sft as fixtures
from tests.arc.test_lt_phase_b_sft_preflight import Tokenizer, Dataset


class PrepareDispatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PrepareSFTTests('test_private_verified_examples_and_journal_are_written_once')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture.fixture; self.run = self.fixture.run
        for name, value in (('data-spec', self.fixture.spec), ('spec', f.collection), ('inventory', {}),
                            ('allocation', f.allocation), ('receipt', f.receipt), ('export', f.export)):
            (self.run / (name + '.json')).write_text(json.dumps(value))
        self.challenge = self.run / 'challenge-input.json'; self.challenge.write_bytes(f.challenges)
        self.solution = self.run / 'solution-input.json'; self.solution.write_bytes(f.solutions)
        self.archive = self.run / 'fixture.zip'
        with zipfile.ZipFile(self.archive, 'w') as z: z.writestr('arc-agi_evaluation_challenges.json', '{}')
        p = patch('src.arc.lt_phase_b_prepare_dispatch.preflight_collection', return_value={
            'status': 'preflight_consistent_unverified', 'execution_ready': False, 'run_sha256': f.collection_sha,
            'selected_tasks': 4, 'evidence_type': 'local_files_and_caller_supplied_registration'})
        self.preflight = p.start(); self.addCleanup(p.stop)
        p = patch('src.arc.lt_phase_b_prepare_dispatch._tokenizer_runtime', return_value=(Tokenizer(), Dataset))
        self.tokenizer = p.start(); self.addCleanup(p.stop)

    def invoke(self, **kwargs):
        f = self.fixture.fixture
        args = {'source_root': f.fixture.fixture.root, 'run_dir': self.run,
            'expected_data_sha256': _digest(self.fixture.spec), 'expected_collection_sha256': f.collection_sha,
            'challenge_path': self.challenge, 'solution_path': self.solution, 'official_archive': self.archive,
            'python_executable': sys.executable, 'model_path': '/fixture'}
        args.update(kwargs)
        return run_prepare_sft(**args)

    def test_replays_and_prepares_with_verified_tokenizer_runtime(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'private_sft_prepared_unverified')
        self.assertEqual(result['eligible_tasks'], 3)
        self.assertNotIn('records', result)
        self.tokenizer.assert_called_once()
        self.assertTrue((self.run / 'eligibility.json').exists())

    def test_bad_preflight_or_interpreter_never_loads_tokenizer(self):
        self.preflight.side_effect = ValueError('not registered')
        with self.assertRaises(ValueError): self.invoke()
        self.tokenizer.assert_not_called()
        self.preflight.side_effect = None
        with self.assertRaises(ValueError): self.invoke(python_executable='/not/a/python')
        self.tokenizer.assert_not_called()

    def test_bad_data_descriptor_is_rejected_before_tokenizer(self):
        with self.assertRaises(ValueError): self.invoke(expected_data_sha256='f' * 64)
        self.tokenizer.assert_not_called()


if __name__ == '__main__': unittest.main()
