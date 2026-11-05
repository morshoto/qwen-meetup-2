"""Review reserved historical task IDs, including interrupted candidates."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_candidate_history import audit_candidate_journals


class CandidateHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'artifacts/a').mkdir(parents=True)

    def test_extracts_keys_and_ignores_unrelated_budgets(self):
        p = self.root / 'artifacts/a/budget.json'
        p.write_text(json.dumps({'candidate_keys': ['A:deadbeef:0', 'B:12345678:2']}))
        (self.root / 'artifacts/budget.json').write_text('{"model_calls": 1}')
        result = audit_candidate_journals(self.root, {'deadbeef', '12345678'})
        self.assertEqual(result['task_ids'], ['12345678', 'deadbeef'])
        self.assertEqual(result['journal_count'], 1)
        self.assertEqual(result['reservation_count'], 2)
        self.assertEqual(result['budget_file_count'], 2)

    def test_rejects_malformed_and_duplicate_keys(self):
        p = self.root / 'artifacts/a/budget.json'
        for keys in (['A:deadbeef:0', 'A:deadbeef:0'], ['A:broken'], ['D:deadbeef:1'], None):
            p.write_text(json.dumps({'candidate_keys': keys}))
            with self.assertRaises(ValueError):
                audit_candidate_journals(self.root, {'deadbeef'})

    def test_changed_inventory_rejected(self):
        p = self.root / 'artifacts/a/budget.json'
        p.write_text('{"candidate_keys": []}')
        original = Path.read_bytes
        def changing(path):
            data = original(path)
            if path.resolve() == p.resolve():
                p.write_text('{"candidate_keys": ["A:deadbeef:0"]}')
            return data
        with patch.object(Path, 'read_bytes', changing):
            with self.assertRaisesRegex(ValueError, 'changed'):
                audit_candidate_journals(self.root, {'deadbeef'})
