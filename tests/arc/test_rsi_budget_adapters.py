import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.arc.rsi_budget_adapters import BudgetedSandbox, model_reserver
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_http_backend import CapturingBackend


class BudgetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = CollectionBudget.create(self.root, registration_sha256='a' * 64, generation=0)

    def test_sandbox_reserves_before_running_and_keeps_failed_charge(self):
        sandbox = Mock()
        sandbox.run.side_effect = RuntimeError('failure')
        wrapped = BudgetedSandbox(sandbox, self.budget)
        with self.assertRaises(RuntimeError):
            wrapped.run('code', [[1]])
        self.assertEqual(self.budget.snapshot()['sandbox_runs'], 1)
        for _ in range(299):
            self.budget.reserve_sandbox()
        with self.assertRaises(TimeoutError):
            wrapped.run('code', [[1]])
        self.assertEqual(sandbox.run.call_count, 1)

    def test_backend_reserves_before_capture_and_transport(self):
        client = CapturingBackend(model='fixture', revision='fixture', base_url='http://127.0.0.1:1/v1',
            temperature=0.2, max_tokens=512, seed=12000, timeout=1, thinking='off', guarded=True,
            log_path=self.root / 'log', request_capture_path=self.root / 'capture',
            reserve=model_reserver(self.budget, kind='generation'))
        self.budget.reserve_model(200000, kind='canary')
        with patch('src.arc.rsi_http_backend.urllib.request.build_opener') as opener:
            with self.assertRaises(TimeoutError):
                client.generate('prompt', kind='e5_generator', candidate=0)
            opener.assert_not_called()
        self.assertFalse((self.root / 'capture').exists())

    def test_model_reservation_does_not_authorize_calls(self):
        reserve = model_reserver(self.budget, kind='canary')
        reserve(100)
        self.assertEqual(self.budget.snapshot()['model_calls'], 1)
        with self.assertRaises(ValueError):
            model_reserver(self.budget, kind='unregistered')
