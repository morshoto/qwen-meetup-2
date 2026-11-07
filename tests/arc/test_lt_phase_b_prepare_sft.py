import hashlib
import json
import shutil
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_prepare_sft import prepare_sft
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_training_preflight as fixtures
from tests.arc.test_lt_phase_b_sft_preflight import Tokenizer, Dataset


class PrepareSFTTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingPreflightTests('test_replays_only_selected_training_answers')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture; self.run = f.fixture.fixture.run
        shutil.rmtree(self.run / 'sft-data')
        self.spec = {'protocol': 'lt-phase-b-data-v1', 'status': 'frozen', 'run_id': f.collection['run_id'],
                     'collection_sha256': f.collection_sha, 'source_code_sha256': f.collection['source_code_sha256'],
                     'solution_sha256': hashlib.sha256(f.solutions).hexdigest()}
        p = patch('src.arc.lt_phase_b_prepare_sft.preflight_collection', return_value={
            'status': 'preflight_consistent_unverified', 'execution_ready': False, 'run_sha256': f.collection_sha,
            'selected_tasks': 4, 'evidence_type': 'local_files_and_caller_supplied_registration'})
        self.preflight = p.start(); self.addCleanup(p.stop)

    def invoke(self, **kwargs):
        f = self.fixture
        args = {'data_spec': self.spec, 'expected_data_sha256': _digest(self.spec),
            'collection_spec': f.collection, 'expected_collection_sha256': f.collection_sha,
            'source_root': f.fixture.fixture.root, 'run_dir': self.run, 'source_inventory': {},
            'allocation_receipt': f.allocation, 'receipt': f.receipt, 'export': f.export,
            'challenge_bytes': f.challenges, 'solution_bytes': f.solutions, 'official_challenge_bytes': b'{}',
            'python_executable': '/fixture', 'model_path': '/fixture', 'tokenizer': Tokenizer(), 'dataset_factory': Dataset}
        args.update(kwargs)
        return prepare_sft(**args)

    def test_private_verified_examples_and_journal_are_written_once(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'private_sft_prepared_unverified')
        self.assertEqual(result['eligible_tasks'], 3)
        self.assertFalse(result['training_allowed'])
        journal = self.run / 'eligibility.json'
        self.assertEqual(journal.stat().st_mode & 0o777, 0o600)
        records = json.loads(journal.read_text())['records']
        self.assertTrue(any(record['verification_status'] == 'incorrect' for record in records))
        self.assertTrue((self.run / 'sft-data/train.jsonl').is_file())
        with self.assertRaises(ValueError): self.invoke()

    def test_zero_yield_is_recorded_without_training_files(self):
        f = self.fixture
        values = json.loads(f.solutions)
        for task in f.selected: values[task] = [[[9]]]
        raw = json.dumps(values).encode()
        spec = {**self.spec, 'solution_sha256': hashlib.sha256(raw).hexdigest()}
        result = self.invoke(data_spec=spec, expected_data_sha256=_digest(spec), solution_bytes=raw)
        self.assertEqual(result['status'], 'insufficient_verified_self_solutions')
        self.assertEqual(result['eligible_tasks'], 0)
        self.assertFalse((self.run / 'sft-data').exists())
        self.assertTrue((self.run / 'eligibility.json').is_file())

    def test_changed_input_and_interrupted_artifact_refuse_preparation(self):
        with self.assertRaises(ValueError): self.invoke(solution_bytes=b'{}')
        (self.run / 'eligibility.json').write_text('{}')
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.run / 'sft-data').exists())


if __name__ == '__main__': unittest.main()
