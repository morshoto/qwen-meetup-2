import hashlib
import json
import unittest

from src.arc.rsi_mlx_dataset import audit_mlx_dataset
from src.arc.rsi_sft_materialize import materialize_quarantined_dataset
import tests.arc.test_rsi_sft_format as format_fixtures


class Dataset:
    def __init__(self, rows, render):
        self.rows, self.process = rows, render

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class MLXDatasetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = format_fixtures.SFTFormatTests('test_rows_match_inference_prefix_and_mask')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.raw = (self.root / 'training.preparation.json').read_bytes()
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.args = dict(root=self.root, preparation_bytes=self.raw, expected_preparation_sha256=self.digest,
                         tasks=self.fixture.tasks, tokenizer=self.fixture.tokenizer, render=self.fixture.render)
        result = materialize_quarantined_dataset(**self.args)
        self.args['expected_dataset_sha256'] = result['dataset_jsonl_sha256']

    def loader(self, path, tokenizer, config):
        self.assertEqual(path, self.root / 'generation-sft.quarantine')
        self.assertIs(tokenizer, self.fixture.tokenizer)
        self.assertTrue(config.mask_prompt)
        rows = [json.loads(line) for line in (path / 'train.jsonl').read_text().splitlines()]
        return Dataset(rows, self.fixture.render), [], []

    def test_loaded_rows_and_masks_match_checked_artifact(self):
        report = audit_mlx_dataset(**self.args, load_local_dataset=self.loader)
        self.assertEqual(report['status'], 'dataset_loader_consistent_unverified')
        self.assertEqual(report['row_count'], 5)
        self.assertFalse(report['training_allowed'])

    def test_rejects_altered_rows_masks_and_extra_splits(self):
        def altered(path, tokenizer, config):
            train, valid, test = self.loader(path, tokenizer, config)
            train.rows[0]['messages'][2]['content'] = 'changed'
            return train, valid, test
        def unmasked(path, tokenizer, config):
            train, valid, test = self.loader(path, tokenizer, config)
            train.process = lambda row: ([10, 11, 12], 0)
            return train, valid, test
        def extra_split(path, tokenizer, config):
            train, valid, test = self.loader(path, tokenizer, config)
            return train, [1], test
        for loader in (altered, unmasked, extra_split):
            with self.subTest(loader=loader.__name__), self.assertRaises(ValueError):
                audit_mlx_dataset(**self.args, load_local_dataset=loader)

    def test_rejects_changed_artifact_before_loader(self):
        (self.root / 'generation-sft.quarantine' / 'train.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):
            audit_mlx_dataset(**self.args, load_local_dataset=self.loader)


if __name__ == '__main__':
    unittest.main()
