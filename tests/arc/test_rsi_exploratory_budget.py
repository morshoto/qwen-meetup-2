"""Separate durable ceilings for the exploratory 50-task comparison."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_comparison_budget import ComparisonBudget
from src.arc.rsi_exploratory_budget import ExploratoryBudget


class ExploratoryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.budget=ExploratoryBudget.create(self.root,registration_sha256='a'*64,generation=0)
    def test_call_cap_and_reopen(self):
        for _ in range(200): self.budget.reserve_model(1,kind='evaluation')
        reopened=ExploratoryBudget.open(self.root,registration_sha256='a'*64,generation=0)
        with self.assertRaises(TimeoutError): reopened.reserve_model(1,kind='evaluation')
        self.assertEqual(reopened.snapshot()['model_calls'],200)
    def test_token_and_canary_limits(self):
        self.budget.reserve_model(4000000,kind='evaluation')
        with self.assertRaises(TimeoutError): self.budget.reserve_model(1,kind='evaluation')
        with self.assertRaises(ValueError): self.budget.reserve_model(1,kind='canary')
    def test_separate_ledger_and_wall_limit(self):
        other=ComparisonBudget.create(self.root,registration_sha256='a'*64,generation=0)
        other.reserve_model(1,kind='evaluation')
        self.assertEqual(self.budget.snapshot()['model_calls'],0)
        state=self.budget.snapshot()
        with patch('src.arc.rsi_collection_budget.time.time',return_value=state['started_wall']+9000), patch('src.arc.rsi_collection_budget.time.monotonic',return_value=state['started_monotonic']+9000):
            with self.assertRaises(TimeoutError): self.budget.reserve_sandbox()
    def test_sandbox_limit(self):
        self.assertEqual(self.budget.max_sandbox_runs,2000)
        with patch.object(self.budget,'max_sandbox_runs',1):
            self.budget.reserve_sandbox()
            with self.assertRaises(TimeoutError): self.budget.reserve_sandbox()
