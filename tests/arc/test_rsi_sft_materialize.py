import hashlib
import json
import stat
import unittest
from unittest.mock import patch

from src.arc.rsi_sft_materialize import materialize_quarantined_dataset, verify_quarantined_dataset
import tests.arc.test_rsi_sft_format as format_fixtures


class SFTMaterializeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = format_fixtures.SFTFormatTests('test_rows_match_inference_prefix_and_mask')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.raw = (self.root / 'training.preparation.json').read_bytes()
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def run_materialize(self, **changes):
        args = dict(root=self.root, preparation_bytes=self.raw, expected_preparation_sha256=self.digest,
                    tasks=self.fixture.tasks, tokenizer=self.fixture.tokenizer, render=self.fixture.render)
        args.update(changes)
        return materialize_quarantined_dataset(**args)

    def test_private_dataset_and_audit_are_hash_bound(self):
        result = self.run_materialize()
        directory = self.root / 'generation-sft.quarantine'
        raw = (directory / 'train.jsonl').read_bytes()
        audit = json.loads((directory / 'audit.json').read_text())
        self.assertEqual(result['status'], 'dataset_quarantined')
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result['dataset_jsonl_sha256'])
        self.assertEqual(audit['preparation_sha256'], self.digest)
        self.assertEqual(audit['dataset_jsonl_sha256'], result['dataset_jsonl_sha256'])
        self.assertFalse(audit['training_allowed'])
        self.assertEqual(len(raw.splitlines()), 5)
        self.assertNotIn(b'query_outputs', raw)
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((directory / 'train.jsonl').stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE((directory / 'audit.json').stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            self.run_materialize()

    def test_independent_readback_detects_data_and_directory_changes(self):
        result = self.run_materialize()
        args = dict(root=self.root, preparation_bytes=self.raw, expected_preparation_sha256=self.digest,
                    expected_dataset_sha256=result['dataset_jsonl_sha256'], tasks=self.fixture.tasks,
                    tokenizer=self.fixture.tokenizer, render=self.fixture.render)
        verified = verify_quarantined_dataset(**args)
        self.assertEqual(verified['status'], 'dataset_consistent_unverified')
        self.assertFalse(verified['training_allowed'])
        directory = self.root / 'generation-sft.quarantine'
        data = directory / 'train.jsonl'
        original = data.read_bytes()
        data.write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            verify_quarantined_dataset(**args)
        data.write_bytes(original)
        extra = directory / 'valid.jsonl'
        extra.write_text('{}')
        with self.assertRaises(ValueError):
            verify_quarantined_dataset(**args)
        extra.unlink()
        audit = directory / 'audit.json'
        audit.write_text('{}')
        with self.assertRaises(ValueError):
            verify_quarantined_dataset(**args)

    def test_changed_preparation_or_mask_fails_before_write(self):
        with self.assertRaises(ValueError):
            self.run_materialize(preparation_bytes=self.raw + b' ')
        self.fixture.render.return_value = ([10, 11], 2)
        with self.assertRaises(ValueError):
            self.run_materialize()
        self.assertFalse((self.root / 'generation-sft.quarantine').exists())

    def test_interrupted_write_consumes_quarantine_directory(self):
        with patch('src.arc.rsi_sft_materialize._write_dataset', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                self.run_materialize()
        self.assertTrue((self.root / 'generation-sft.quarantine').is_dir())
        with self.assertRaises(FileExistsError):
            self.run_materialize()
