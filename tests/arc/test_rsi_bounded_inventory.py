"""A local union of historical evidence is still an unverified proposal."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_bounded_inventory import propose_bounded_inventory


class BoundedInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'data').mkdir()
        (self.root / 'data/tasks.json').write_text(json.dumps({k: {} for k in 'abcdef'}))
        (self.root / 'data/registry.json').write_text(json.dumps({
            'protocol':'p3c_exploration_task_exclusions_v1', 'version':1,
            'exclusions':{'a':['old']}, 'task_ids':['a']}))
        self.stack = [patch('src.arc.rsi_bounded_inventory.audit_history', return_value={'task_ids':['b'], 'files':[]}),
                      patch('src.arc.rsi_bounded_inventory.audit_extra_artifacts', return_value={'task_ids':['c'], 'file_count':1, 'inventory_sha256':'c'*64}),
                      patch('src.arc.rsi_bounded_inventory.audit_candidate_journals', return_value={'task_ids':['d'], 'journal_count':1, 'budget_file_count':1, 'reservation_count':1, 'inventory_sha256':'d'*64}),
                      patch('src.arc.rsi_bounded_inventory._code_hashes', return_value={'code':'f'*64})]
        for p in self.stack: p.start(); self.addCleanup(p.stop)

    def propose(self):
        return propose_bounded_inventory(self.root, task_path='data/tasks.json', registry_path='data/registry.json',
                                         required_paths=[], development_size=2)

    def test_unions_all_evidence_and_keeps_review_gate(self):
        result = self.propose()
        self.assertEqual(result['known_exposed_task_ids'], list('abcd'))
        self.assertEqual(result['partitions']['development'], list('ef'))
        self.assertFalse(result['execution_ready'])
        self.assertFalse(result['independent_review_completed'])
        self.assertFalse(result['historical_independence_established'])

    def test_requires_enough_eligible_tasks(self):
        with patch('src.arc.rsi_bounded_inventory.audit_extra_artifacts', return_value={'task_ids':['c','e'], 'file_count':1, 'inventory_sha256':'c'*64}):
            with self.assertRaisesRegex(ValueError, 'eligible'):
                self.propose()

    def test_detects_dataset_or_review_code_changes(self):
        p = self.root / 'data/tasks.json'
        original = p.read_bytes
        def change():
            b = original(); p.write_text('{"different": {}}'); return b
        with patch.object(Path, 'read_bytes', lambda path: change() if path == p.resolve() else original()):
            with self.assertRaises(ValueError): self.propose()
