"""Fail closed when the retained task exposure inventory is incomplete."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_history import audit_history, provisional_snapshot


class HistoryAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'artifacts/run').mkdir(parents=True)
        (self.root / 'configs').mkdir()
        (self.root / 'artifacts/run/task_manifest.json').write_text(json.dumps({'task_ids': ['a']}))
        (self.root / 'artifacts/run/attempted_tasks.jsonl').write_text('{"task_id":"b"}\n')
        (self.root / 'configs/run.yml').write_text('p3c:\n  task_ids: [c]\n')

    def test_audits_all_recognized_files_and_collects_ids(self):
        result = audit_history(self.root, required_paths=['artifacts/run/task_manifest.json'])
        self.assertEqual(result['task_ids'], ['a', 'b', 'c'])
        self.assertEqual({row['path'] for row in result['files']}, {
            'artifacts/run/task_manifest.json', 'artifacts/run/attempted_tasks.jsonl', 'configs/run.yml'})
        self.assertTrue(all(len(row['sha256']) == 64 for row in result['files']))

    def test_rejects_missing_required_source(self):
        with self.assertRaisesRegex(ValueError, 'required'):
            audit_history(self.root, required_paths=['artifacts/missing/task_manifest.json'])

    def test_rejects_malformed_json_or_yaml(self):
        for relative, bad in [('artifacts/run/task_manifest.json', '{'),
                              ('artifacts/run/attempted_tasks.jsonl', '{'),
                              ('configs/run.yml', 'x: [')]:
            path = self.root / relative
            previous = path.read_text()
            path.write_text(bad)
            with self.subTest(path=relative), self.assertRaisesRegex(ValueError, 'unreadable or invalid'):
                audit_history(self.root, required_paths=[])
            path.write_text(previous)

    def test_rejects_changed_inventory_during_scan(self):
        from src.arc.exploration_registry import build_registry
        def mutate(*args, **kwargs):
            result = build_registry(*args, **kwargs)
            (self.root / 'artifacts/run/task_manifest.json').write_text('{}')
            return result
        with patch('src.arc.rsi_history.build_registry', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed'):
                audit_history(self.root, required_paths=[])

    def test_snapshot_combines_legacy_and_current_exposure(self):
        (self.root / 'data').mkdir()
        (self.root / 'data/tasks.json').write_text(json.dumps({key: {} for key in 'abcdef'}))
        registry = {'protocol': 'p3c_exploration_task_exclusions_v1', 'version': 1,
                    'exclusions': {'d': ['old']}, 'task_ids': ['d']}
        (self.root / 'configs/old.json').write_text(json.dumps(registry))
        snapshot = provisional_snapshot(self.root, task_path='data/tasks.json',
                                        registry_path='configs/old.json',
                                        required_paths=[], development_size=2)
        self.assertEqual(snapshot['partitions']['development'], ['e', 'f'])
        self.assertEqual(snapshot['known_exposed_in_dataset'], 4)
        self.assertEqual(snapshot['status'], 'provisional_not_preregistered')
        self.assertFalse(snapshot['independence_established'])

    def test_snapshot_rejects_invalid_legacy_registry(self):
        (self.root / 'data').mkdir()
        (self.root / 'data/tasks.json').write_text('{"a": {}, "b": {}}')
        (self.root / 'configs/old.json').write_text('{}')
        with self.assertRaises(ValueError):
            provisional_snapshot(self.root, task_path='data/tasks.json',
                                 registry_path='configs/old.json',
                                 required_paths=[], development_size=1)


if __name__ == '__main__':
    unittest.main()
