import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.arc.http_backend import BackendError
from src.arc.rsi_comparison_contract import ContractCheckedComparisonTransport
from src.arc.rsi_comparison_transport import CapturedComparisonTransport


class ComparisonContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent_url = 'http://127.0.0.1:10001/v1'
        self.candidate_url = 'http://127.0.0.1:10002/v1'
        self.transport = CapturedComparisonTransport(root=self.root, parent_url=self.parent_url,
            candidate_url=self.candidate_url, parent_adapter='none', candidate_adapter='b' * 64,
            max_request_seconds=2)
        self.states = {self.parent_url: {'ledger_id': '1' * 32, 'token_limit': 8000000, 'reserved_tokens': 0},
                       self.candidate_url: {'ledger_id': '2' * 32, 'token_limit': 8000000, 'reserved_tokens': 0}}

    def contract(self, url, model, revision, total_tokens, max_new_tokens, timeout, thinking, **kwargs):
        expected = 'none' if url == self.parent_url else 'b' * 64
        if kwargs['expected_adapter_sha256'] != expected or kwargs['required_thinking_modes'] != ('off',):
            raise BackendError('budget_unverified', 'wrong contract')
        return dict(self.states[url])

    @patch('src.arc.rsi_comparison_contract.verify_guarded_contract')
    def test_verifies_both_arms_before_each_dispatch(self, verify):
        verify.side_effect = self.contract
        wrapped = ContractCheckedComparisonTransport(self.transport, expected_max_prompt_tokens=4096)
        self.assertEqual(verify.call_count, 2)
        with patch.object(CapturedComparisonTransport, '__call__', return_value={'code': 'code', 'request_id': 'a' * 32}) as dispatch:
            self.states[self.parent_url]['reserved_tokens'] = 100
            wrapped('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=1)
            dispatch.assert_called_once()
            self.assertEqual(verify.call_count, 4)

    @patch('src.arc.rsi_comparison_contract.verify_guarded_contract')
    def test_rejects_ledger_or_budget_drift_before_dispatch(self, verify):
        verify.side_effect = self.contract
        wrapped = ContractCheckedComparisonTransport(self.transport, expected_max_prompt_tokens=4096)
        self.states[self.parent_url]['ledger_id'] = '3' * 32
        with patch.object(CapturedComparisonTransport, '__call__') as dispatch:
            with self.assertRaises(ValueError):
                wrapped('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=1)
            dispatch.assert_not_called()

    @patch('src.arc.rsi_comparison_contract.verify_guarded_contract')
    def test_rejects_shared_ledger_excess_budget_and_nonempty_start(self, verify):
        verify.side_effect = self.contract
        for change in ('shared', 'budget', 'used'):
            self.states[self.parent_url] = {'ledger_id': '1' * 32, 'token_limit': 8000000, 'reserved_tokens': 0}
            self.states[self.candidate_url] = {'ledger_id': '2' * 32, 'token_limit': 8000000, 'reserved_tokens': 0}
            if change == 'shared': self.states[self.candidate_url]['ledger_id'] = '1' * 32
            if change == 'budget': self.states[self.candidate_url]['token_limit'] += 1
            if change == 'used': self.states[self.parent_url]['reserved_tokens'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                ContractCheckedComparisonTransport(self.transport, expected_max_prompt_tokens=4096)
