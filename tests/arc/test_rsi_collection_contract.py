import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_collection_contract import ContractCheckedSlotTransport
from src.arc.rsi_slot_transport import CapturedSlotTransport


class CollectionContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = CollectionBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        self.transport = CapturedSlotTransport(base_url='http://127.0.0.1:10001/v1',
            log_path=self.root / 'requests.jsonl', request_capture_path=self.root / 'raw.jsonl', max_request_seconds=2)
        self.reservation = 512 + len(json.dumps([{'role': 'system', 'content': system_prompt_for_kind('e5_generator')},
                                                   {'role': 'user', 'content': 'prompt'}], ensure_ascii=False).encode()) + 1024
        self.contract = {'ledger_id': '1' * 32, 'token_limit': 200000, 'reserved_tokens': 0}

    def check_contract(self, url, model, revision, total_tokens, max_new_tokens, timeout, thinking, **kwargs):
        self.assertEqual(total_tokens, 200000)
        self.assertEqual(kwargs['expected_adapter_sha256'], 'none')
        self.assertEqual(kwargs['required_thinking_modes'], ('off',))
        return dict(self.contract)

    def wrap(self):
        return ContractCheckedSlotTransport(self.transport, budget=self.budget, registration_sha256='a' * 64,
            generation=0, parent_adapter='none', expected_max_prompt_tokens=4096,
            expected_ledger_id='1' * 32, expected_initial_reserved_tokens=0)

    @patch('src.arc.rsi_collection_contract.verify_guarded_contract')
    def test_requires_reservation_before_each_request(self, verify):
        verify.side_effect = self.check_contract
        wrapped = self.wrap()
        with patch.object(CapturedSlotTransport, '__call__', return_value={'code': 'code', 'request_id': 'b' * 32}) as dispatch:
            with self.assertRaises(ValueError):
                wrapped('prompt', seed=12000, timeout=1)
            dispatch.assert_not_called()
            self.budget.reserve_model(self.reservation, kind='generation')
            result = wrapped('prompt', seed=12000, timeout=1)
            self.assertEqual(result['request_id'], 'b' * 32)
            dispatch.assert_called_once()
            with self.assertRaises(ValueError):
                wrapped('prompt', seed=12001, timeout=1)

    @patch('src.arc.rsi_collection_contract.verify_guarded_contract')
    def test_ledger_drift_and_server_overspend_stop_dispatch(self, verify):
        verify.side_effect = self.check_contract
        wrapped = self.wrap()
        self.budget.reserve_model(self.reservation, kind='generation')
        self.contract['ledger_id'] = '2' * 32
        with patch.object(CapturedSlotTransport, '__call__') as dispatch:
            with self.assertRaises(ValueError):
                wrapped('prompt', seed=12000, timeout=1)
            self.contract['ledger_id'] = '1' * 32
            self.contract['reserved_tokens'] = self.reservation + 1
            with self.assertRaises(ValueError):
                wrapped('prompt', seed=12000, timeout=1)
            dispatch.assert_not_called()

    @patch('src.arc.rsi_collection_contract.verify_guarded_contract')
    def test_wrong_reservation_or_seed_cannot_dispatch(self, verify):
        verify.side_effect = self.check_contract
        wrapped = self.wrap()
        self.budget.reserve_model(self.reservation - 1, kind='generation')
        with patch.object(CapturedSlotTransport, '__call__') as dispatch:
            with self.assertRaises(ValueError):
                wrapped('prompt', seed=12000, timeout=1)
            with self.assertRaises(ValueError):
                wrapped('prompt', seed=12100, timeout=1)
            dispatch.assert_not_called()

    @patch('src.arc.rsi_collection_contract.verify_guarded_contract')
    def test_initial_identity_and_generation_are_checked(self, verify):
        verify.side_effect = self.check_contract
        self.contract['reserved_tokens'] = 1
        with self.assertRaises(ValueError):
            self.wrap()
        self.contract['reserved_tokens'] = 0
        with self.assertRaises(ValueError):
            ContractCheckedSlotTransport(self.transport, budget=self.budget, registration_sha256='a' * 64,
                generation=0, parent_adapter='b' * 64, expected_max_prompt_tokens=4096,
                expected_ledger_id='1' * 32, expected_initial_reserved_tokens=0)
