import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_collection_budget import CollectionBudget

REG = 'a' * 64


class CollectionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = CollectionBudget.create(self.root, registration_sha256=REG, generation=0)

    def test_canaries_and_generation_share_call_and_token_caps(self):
        self.budget.reserve_model(1000, kind='canary')
        for _ in range(31):
            self.budget.reserve_model(1000, kind='generation')
        with self.assertRaises(TimeoutError):
            self.budget.reserve_model(1, kind='canary')
        self.assertEqual(self.budget.snapshot()['model_calls'], 32)

    def test_token_and_sandbox_limits_are_checked_before_reservation(self):
        self.budget.reserve_model(200000, kind='generation')
        with self.assertRaises(TimeoutError):
            self.budget.reserve_model(1, kind='generation')
        for _ in range(300):
            self.budget.reserve_sandbox()
        with self.assertRaises(TimeoutError):
            self.budget.reserve_sandbox()
        state = self.budget.snapshot()
        self.assertEqual((state['model_calls'], state['reserved_tokens'], state['sandbox_runs']), (1, 200000, 300))

    def test_reopen_preserves_reservations_and_identity(self):
        self.budget.reserve_model(100, kind='generation')
        resumed = CollectionBudget.open(self.root, registration_sha256=REG, generation=0)
        self.assertEqual(resumed.snapshot()['reserved_tokens'], 100)
        with self.assertRaises(ValueError):
            CollectionBudget.open(self.root, registration_sha256='b' * 64, generation=0)
        with self.assertRaises(FileExistsError):
            CollectionBudget.create(self.root, registration_sha256=REG, generation=0)

    def test_concurrent_reservations_never_overrun(self):
        def reserve(_):
            try:
                CollectionBudget.open(self.root, registration_sha256=REG, generation=0).reserve_model(1, kind='generation')
                return 1
            except TimeoutError:
                return 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            self.assertEqual(sum(pool.map(reserve, range(64))), 32)
        self.assertEqual(self.budget.snapshot()['model_calls'], 32)

    def test_expiry_and_clock_rollback_fail_closed(self):
        with patch('src.arc.rsi_collection_budget.time.time', return_value=10), patch('src.arc.rsi_collection_budget.time.monotonic', return_value=10):
            with self.assertRaises(TimeoutError):
                self.budget.reserve_sandbox()
        state = self.budget.snapshot()
        with patch('src.arc.rsi_collection_budget.time.time', return_value=state['started_wall'] + 3600), patch('src.arc.rsi_collection_budget.time.monotonic', return_value=state['started_monotonic'] + 3600):
            with self.assertRaises(TimeoutError):
                self.budget.reserve_sandbox()

    def test_remaining_budget_is_bounded_by_both_clocks(self):
        state = self.budget.snapshot()
        with patch('src.arc.rsi_collection_budget.time.time', return_value=state['started_wall'] + 100), patch('src.arc.rsi_collection_budget.time.monotonic', return_value=state['started_monotonic'] + 200):
            self.assertEqual(self.budget.remaining(), 3400)
        with patch('src.arc.rsi_collection_budget.time.time', return_value=state['started_wall'] + 3600), patch('src.arc.rsi_collection_budget.time.monotonic', return_value=state['started_monotonic'] + 200):
            with self.assertRaises(TimeoutError):
                self.budget.remaining()

    def test_rejects_invalid_inputs(self):
        for amount in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                self.budget.reserve_model(amount, kind='generation')
        with self.assertRaises(ValueError):
            self.budget.reserve_model(1, kind='unknown')
