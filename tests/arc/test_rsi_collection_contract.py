import tempfile
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.guarded_contract import AUDITED_GENERATE_STEP, source_fingerprints
from src.arc.http_backend import BackendError
from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_collection_contract import ContractCheckedSlotTransport
from src.arc.rsi_slot_transport import CapturedSlotTransport
from src.arc.rsi_slot_collector import collect_slot
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_lineage_audit import probes_sha256


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
    def test_32b_contract_binds_transport_and_rejects_model_drift(self, verify):
        verify.side_effect = self.check_contract
        self.transport = CapturedSlotTransport(base_url='http://127.0.0.1:10001/v1',
            log_path=self.root/'requests.jsonl', request_capture_path=self.root/'raw.jsonl',
            max_request_seconds=2, model_key='32b')
        args = dict(budget=self.budget, registration_sha256='a'*64, generation=0, parent_adapter='none',
                    expected_max_prompt_tokens=4096, expected_ledger_id='1'*32, expected_initial_reserved_tokens=0)
        wrapped = ContractCheckedSlotTransport(self.transport, model_key='32b', **args)
        self.assertEqual(verify.call_args.args[1:3], ('mlx-community/Qwen3-32B-4bit',
                         'bcaaf7f538adf166c1080a2befdb4f6019f66639'))
        with self.assertRaises(ValueError): ContractCheckedSlotTransport(self.transport, model_key='14b', **args)
        self.transport.options['model'] = 'mlx-community/Qwen3-14B-4bit'
        self.budget.reserve_model(self.reservation, kind='generation')
        with patch.object(CapturedSlotTransport, '__call__') as dispatch:
            with self.assertRaises(ValueError): wrapped('prompt', seed=12000, timeout=1)
            dispatch.assert_not_called()

    def test_real_contract_parser_gates_fixture_slot(self):
        model = 'mlx-community/Qwen3-14B-4bit'
        revision = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                contract = {'contract': 'same_process_token_ids_v1', 'model': model, 'revision': revision,
                    'thinking_mode': 'off', 'thinking_modes': ['off', 'on'],
                    'thinking_canary': {'supported': True, 'different_token_ids': True,
                                        'off_sha256': 'a' * 64, 'on_sha256': 'b' * 64},
                    'adapter_sha256': self.server.adapter, 'hard_deadline': 'process_exit_v1',
                    'max_request_seconds': 2, 'token_limit': 200000, 'reserved_tokens': 0,
                    'max_prompt_tokens': 4096, 'max_new_tokens': 512, 'prefetch_allowance': 1,
                    'ledger_id': '1' * 32, 'source_sha256': source_fingerprints(),
                    'driver_type': 'src.arc.mlx_token_driver.MLXTokenDriver', 'driver_runtime_version': '0.30.7',
                    'driver_source_sha256': AUDITED_GENERATE_STEP['0.30.7']}
                raw = json.dumps(contract).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_POST(self):
                self.server.posts += 1
                self.rfile.read(int(self.headers['Content-Length']))
                raw = json.dumps({'model': model, 'thinking_mode': 'off',
                    'thinking_control': 'tokenizer_template_canary_passed',
                    'choices': [{'message': {'role': 'assistant', 'content': FAKE_PROGRAMS[1]},
                                 'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        server.adapter, server.posts = 'none', 0
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(worker.join)
        self.addCleanup(server.shutdown)
        self.transport = CapturedSlotTransport(base_url=f'http://127.0.0.1:{server.server_port}/v1',
            log_path=self.root / 'requests.jsonl', request_capture_path=self.root / 'raw.jsonl', max_request_seconds=2)
        wrapped = self.wrap()
        ids = [f'task-{i:02}' for i in range(20)]
        task = Task(ids[0], (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        public, private = _quality_grids(42, 5), _quality_grids(43, 5)
        result = collect_slot(root=self.root, registration_sha256='a' * 64, generation=0, slot=0,
            task_ids=ids, task=task, expected_manifest_sha256=manifest_sha256(ids), expected_task_sha256=task_sha256(task),
            public_probes=public, private_probes=private, expected_public_probes_sha256=probes_sha256(public),
            expected_private_probes_sha256=probes_sha256(private), reference_pool=[], transport=wrapped,
            sandbox=FakeE6Sandbox(), budget=self.budget)
        self.assertEqual(result['status'], 'novel_unverified')
        self.assertEqual(server.posts, 1)
        server.adapter = 'b' * 64
        self.budget.reserve_model(self.reservation, kind='generation')
        with self.assertRaises(BackendError):
            wrapped('prompt', seed=12003, timeout=1)
        self.assertEqual(server.posts, 1)

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
    def test_second_generation_requires_parent_adapter_and_seed(self, verify):
        def contract(url, model, revision, total_tokens, max_new_tokens, timeout, thinking, **kwargs):
            self.assertEqual(kwargs['expected_adapter_sha256'], 'b' * 64)
            return dict(self.contract)
        verify.side_effect = contract
        budget = CollectionBudget.create(self.root, registration_sha256='a' * 64, generation=1)
        wrapped = ContractCheckedSlotTransport(self.transport, budget=budget, registration_sha256='a' * 64,
            generation=1, parent_adapter='b' * 64, expected_max_prompt_tokens=4096,
            expected_ledger_id='1' * 32, expected_initial_reserved_tokens=0)
        budget.reserve_model(self.reservation, kind='generation')
        with patch.object(CapturedSlotTransport, '__call__', return_value={'code': 'code', 'request_id': 'b' * 32}) as dispatch:
            with self.assertRaises(ValueError):
                wrapped('prompt', seed=12000, timeout=1)
            wrapped('prompt', seed=12100, timeout=1)
            dispatch.assert_called_once()

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
