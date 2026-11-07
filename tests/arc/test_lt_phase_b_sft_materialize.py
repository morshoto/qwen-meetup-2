import copy
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_phase_b_sft_materialize import materialize_sft
from tests.arc import test_lt_phase_b_sft_preflight as fixtures
from tests.arc import test_lt_phase_b_dataset as dataset_fixtures


class SFTMaterializeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); (self.root / 'artifacts').mkdir()
        self.run = self.root / 'artifacts/lt-phase-b-fixture'; self.run.mkdir(mode=0o700)
        self.fixture = fixtures.SFTPreflightTests('test_audits_exact_verified_examples_and_prefix')
        self.fixture.setUp()

    def materialize(self, dataset=None):
        return materialize_sft(source_root=self.root, run_dir=self.run,
            dataset=self.fixture.dataset if dataset is None else dataset,
            tasks=self.fixture.fixture.tasks, verifier=self.fixture.fixture.verifier, source=self.fixture.fixture.source,
            tokenizer=fixtures.Tokenizer(), dataset_factory=fixtures.Dataset)

    def test_writes_private_exact_jsonl_once(self):
        manifest = self.materialize()
        data = self.run / 'sft-data'
        self.assertEqual(stat.S_IMODE(data.stat().st_mode), 0o700)
        for name in ('train.jsonl', 'valid.jsonl', 'manifest.json'):
            self.assertEqual(stat.S_IMODE((data / name).stat().st_mode), 0o600)
        self.assertEqual(hashlib.sha256((data / 'train.jsonl').read_bytes()).hexdigest(), manifest['train_jsonl_sha256'])
        self.assertEqual(hashlib.sha256((data / 'valid.jsonl').read_bytes()).hexdigest(), manifest['valid_jsonl_sha256'])
        self.assertEqual(json.loads((data / 'manifest.json').read_text()), manifest)
        self.assertFalse(manifest['training_allowed'])
        with self.assertRaises(FileExistsError): self.materialize()

    def test_invalid_dataset_creates_no_training_directory(self):
        data = copy.deepcopy(self.fixture.dataset)
        data['train'][0]['messages'][1]['content'] = '[[[9]]]'
        with self.assertRaises(ValueError): self.materialize(data)
        self.assertFalse((self.run / 'sft-data').exists())

    def test_overfit_case_omits_empty_validation_file(self):
        f = self.fixture.fixture
        rows = [dataset_fixtures.row('a', [f.good, f.bad])] + [dataset_fixtures.row(x, [f.bad, None]) for x in ('b', 'c', 'd')]
        data = f.build(rows)
        manifest = self.materialize(data)
        self.assertEqual(manifest['yield_status'], 'overfitting_only')
        self.assertFalse((self.run / 'sft-data/valid.jsonl').exists())
        self.assertIsNone(manifest['valid_jsonl_sha256'])


if __name__ == '__main__': unittest.main()
