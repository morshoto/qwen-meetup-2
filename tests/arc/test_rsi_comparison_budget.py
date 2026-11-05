import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_comparison_budget import ComparisonBudget


class ComparisonBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = ComparisonBudget.create(self.root, registration_sha256='a' * 64, generation=0)

    def test_model_call_and_token_caps(self):
        self.budget.reserve_model(16000000, kind='evaluation')
        with self.assertRaises(TimeoutError):
            self.budget.reserve_model(1, kind='evaluation')
        with self.assertRaises(ValueError):
            self.budget.reserve_model(1, kind='canary')
        self.assertEqual(self.budget.snapshot()['model_calls'], 1)

    def test_call_cap_and_durable_reopen(self):
        for _ in range(800):
            self.budget.reserve_model(1, kind='evaluation')
        reopened = ComparisonBudget.open(self.root, registration_sha256='a' * 64, generation=0)
        with self.assertRaises(TimeoutError):
            reopened.reserve_model(1, kind='evaluation')
        self.assertEqual(reopened.snapshot()['model_calls'], 800)

    def test_separate_collection_ledger_and_wall_limit(self):
        collection = CollectionBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        collection.reserve_model(1, kind='generation')
        self.assertEqual(self.budget.snapshot()['model_calls'], 0)
        state = self.budget.snapshot()
        with patch('src.arc.rsi_collection_budget.time.time', return_value=state['started_wall'] + 36000), patch('src.arc.rsi_collection_budget.time.monotonic', return_value=state['started_monotonic'] + 36000):
            with self.assertRaises(TimeoutError):
                self.budget.reserve_sandbox()

    def test_sandbox_limit(self):
        self.assertEqual(self.budget.max_sandbox_runs, 8000)
        with patch.object(self.budget, 'max_sandbox_runs', 1):
            self.budget.reserve_sandbox()
            with self.assertRaises(TimeoutError):
                self.budget.reserve_sandbox()
