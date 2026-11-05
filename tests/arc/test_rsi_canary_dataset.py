import hashlib
import json
import stat
import unittest

from src.arc.data import Task
from src.arc.rsi_canary_dataset import canary_tasks, prepare_canary_dataset
import tests.arc.test_rsi_sft_format as fixtures


class CanaryDatasetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SFTFormatTests('test_rows_match_inference_prefix_and_mask')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def prepare(self, **changes):
        args = dict(root=self.root, forbidden_tasks=self.fixture.tasks,
                    tokenizer=self.fixture.tokenizer, render=self.fixture.render)
        args.update(changes)
        return prepare_canary_dataset(**args)

    def test_private_separate_canary_is_hash_bound(self):
        result = self.prepare()
        path = self.root / 'preflight-canary.quarantine'
        raw = (path / 'train.jsonl').read_bytes()
        audit = json.loads((path / 'audit.json').read_text())
        self.assertEqual(len(raw.splitlines()), 5)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result['dataset_jsonl_sha256'])
        self.assertEqual(audit['dataset_jsonl_sha256'], result['dataset_jsonl_sha256'])
        self.assertFalse(result['training_allowed'])
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((path / 'train.jsonl').stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_grid_overlap_or_mask_error_fails_before_writing(self):
        task = next(iter(canary_tasks().values()))
        overlap = Task('renamed', task.train, task.test_inputs)
        with self.assertRaises(ValueError):
            self.prepare(forbidden_tasks={'renamed': overlap})
        self.fixture.render.return_value = ([10, 11, 12], 1)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.root / 'preflight-canary.quarantine').exists())


if __name__ == '__main__':
    unittest.main()
