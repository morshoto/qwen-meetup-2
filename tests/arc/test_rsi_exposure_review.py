"""Audit identifiers in retained artifact formats outside the primary scanner."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_exposure_review import audit_extra_artifacts


class ExtraExposureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'artifacts/run').mkdir(parents=True)

    def test_finds_only_known_dataset_identifiers_in_extra_formats(self):
        (self.root / 'artifacts/run/results.jsonl').write_text(
            '{"task_id":"a"}\n{"task_ids":["b","external"]}\n')
        (self.root / 'artifacts/run/task_manifest.json').write_text('{"task_id":"c"}')
        result = audit_extra_artifacts(self.root, {'a', 'b', 'c'})
        self.assertEqual(result['task_ids'], ['a', 'b'])
        self.assertEqual(result['file_count'], 1)
        self.assertEqual(len(result['inventory_sha256']), 64)

    def test_missing_artifacts_or_invalid_data_fail_closed(self):
        (self.root / 'artifacts/run/results.jsonl').write_text('{')
        with self.assertRaisesRegex(ValueError, 'invalid'):
            audit_extra_artifacts(self.root, {'a'})
        with self.assertRaisesRegex(ValueError, 'missing'):
            audit_extra_artifacts(self.root / 'absent', {'a'})

    def test_inventory_change_during_scan_fails_closed(self):
        path = self.root / 'artifacts/run/results.jsonl'
        path.write_text('{"task_id":"a"}\n')
        original = Path.read_bytes
        calls = [0]
        def changing(p):
            result = original(p)
            if p == path:
                calls[0] += 1
                if calls[0] == 1:
                    path.write_text('{"task_id":"b"}\n')
            return result
        with patch.object(Path, 'read_bytes', changing):
            with self.assertRaisesRegex(ValueError, 'changed'):
                audit_extra_artifacts(self.root, {'a', 'b'})


if __name__ == '__main__':
    unittest.main()
