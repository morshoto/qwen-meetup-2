import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.rsi_canary_dataset import prepare_canary_dataset
from src.arc.rsi_local_canary import audit_local_canary
from tests.arc.test_rsi_canary_dataset import CanaryDatasetTests
from tests.arc.test_rsi_mlx_dataset import Dataset

REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


class LocalCanaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CanaryDatasetTests('test_private_separate_canary_is_hash_bound')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.model = self.root / REVISION
        self.model.mkdir()
        (self.model / 'tokenizer.json').write_bytes(b'tokenizer')
        (self.model / 'tokenizer_config.json').write_bytes(b'config')
        canary = prepare_canary_dataset(root=self.root, forbidden_tasks=self.fixture.fixture.tasks,
            tokenizer=self.fixture.fixture.tokenizer, render=self.fixture.fixture.render)
        self.args = dict(root=self.root, forbidden_tasks=self.fixture.fixture.tasks, model_dir=self.model,
            expected_tokenizer_sha256=hashlib.sha256(b'tokenizer').hexdigest(),
            expected_config_sha256=hashlib.sha256(b'config').hexdigest(),
            expected_dataset_sha256=canary['dataset_jsonl_sha256'])

    def components(self):
        def loader(path, tokenizer, config):
            rows = [json.loads(line) for line in (path / 'train.jsonl').read_text().splitlines()]
            return Dataset(rows, self.fixture.fixture.render), [], []
        return self.fixture.fixture.tokenizer, self.fixture.fixture.render, loader, '0.30.7'

    @patch('src.arc.rsi_local_canary._load_canary_components')
    def test_local_loader_checks_hashes_without_training(self, load):
        load.return_value = self.components()
        result = audit_local_canary(**self.args)
        self.assertEqual(result['status'], 'local_canary_loader_consistent')
        self.assertEqual(result['row_count'], 5)
        self.assertFalse(result['training_allowed'])

    @patch('src.arc.rsi_local_canary._load_canary_components')
    def test_32b_canary_checks_its_own_revision(self, load):
        revision = 'bcaaf7f538adf166c1080a2befdb4f6019f66639'
        model = self.root / revision
        model.mkdir()
        (model / 'tokenizer.json').write_bytes(b'tokenizer')
        (model / 'tokenizer_config.json').write_bytes(b'config')
        load.return_value = self.components()
        args = dict(self.args, model_dir=model, model_key='32b')
        result = audit_local_canary(**args)
        self.assertEqual(result['model_revision'], revision)
        self.assertFalse(result['training_allowed'])
        with self.assertRaises(ValueError):
            audit_local_canary(**dict(args, model_key='14b'))
        with self.assertRaises(ValueError):
            audit_local_canary(**dict(args, model_key='unknown'))

    @patch('src.arc.rsi_local_canary._load_canary_components')
    def test_wrong_file_runtime_and_mid_audit_drift_are_rejected(self, load):
        (self.model / 'tokenizer.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            audit_local_canary(**self.args)
        load.assert_not_called()
        (self.model / 'tokenizer.json').write_bytes(b'tokenizer')
        load.return_value = (*self.components()[:3], 'wrong')
        with self.assertRaises(ValueError):
            audit_local_canary(**self.args)
        def changed(_row):
            (self.model / 'tokenizer_config.json').write_bytes(b'changed')
            return ([10, 11, 12, 13, 14], 3)
        self.fixture.fixture.render.side_effect = changed
        load.return_value = self.components()
        with self.assertRaises(ValueError):
            audit_local_canary(**self.args)


if __name__ == '__main__':
    unittest.main()
