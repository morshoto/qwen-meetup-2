import tempfile
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.arc.data import Task, Verifier
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.http_backend import BackendError
from src.arc.guarded_contract import AUDITED_GENERATE_STEP, source_fingerprints
from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_comparison_budget import ComparisonBudget
from src.arc.rsi_comparison_contract import ContractCheckedComparisonTransport
from src.arc.rsi_comparison_transport import CapturedComparisonTransport
from src.arc.rsi_comparison_runner import run_comparison


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
        self.budget = ComparisonBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        self.reservation = 512 + len(json.dumps([{'role': 'system', 'content': system_prompt_for_kind('e5_self_solve')},
                                                   {'role': 'user', 'content': 'prompt'}], ensure_ascii=False).encode()) + 1024
        self.states = {self.parent_url: {'ledger_id': '1' * 32, 'token_limit': 8000000, 'reserved_tokens': 0},
                       self.candidate_url: {'ledger_id': '2' * 32, 'token_limit': 8000000, 'reserved_tokens': 0}}

    def contract(self, url, model, revision, total_tokens, max_new_tokens, timeout, thinking, **kwargs):
        expected = 'none' if url == self.parent_url else 'b' * 64
        if kwargs['expected_adapter_sha256'] != expected or kwargs['required_thinking_modes'] != ('off',):
            raise BackendError('budget_unverified', 'wrong contract')
        return dict(self.states[url])

    def wrap(self, transport=None):
        return ContractCheckedComparisonTransport(transport or self.transport, budget=self.budget,
            registration_sha256='a' * 64, generation=0, expected_max_prompt_tokens=4096)

    def test_real_contract_parser_checks_loopback_fixture(self):
        model = 'mlx-community/Qwen3-14B-4bit'
        revision = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                contract = {'contract': 'same_process_token_ids_v1', 'model': model, 'revision': revision,
                    'thinking_mode': 'off', 'thinking_modes': ['off', 'on'],
                    'thinking_canary': {'supported': True, 'different_token_ids': True,
                                        'off_sha256': 'a' * 64, 'on_sha256': 'b' * 64},
                    'adapter_sha256': self.server.adapter, 'hard_deadline': 'process_exit_v1',
                    'max_request_seconds': 2, 'token_limit': 8000000, 'reserved_tokens': 0,
                    'max_prompt_tokens': 4096, 'max_new_tokens': 512, 'prefetch_allowance': 1,
                    'ledger_id': self.server.ledger_id, 'source_sha256': source_fingerprints(),
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
                    'choices': [{'message': {'role': 'assistant', 'content': 'def transform(grid): return grid'},
                                 'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        servers = []
        for adapter, ledger in (('none', '1' * 32), ('b' * 64, '2' * 32)):
            server = HTTPServer(('127.0.0.1', 0), Handler)
            server.adapter, server.ledger_id, server.posts = adapter, ledger, 0
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            self.addCleanup(server.server_close)
            self.addCleanup(worker.join)
            self.addCleanup(server.shutdown)
            servers.append(server)
        urls = [f'http://127.0.0.1:{server.server_port}/v1' for server in servers]
        transport = CapturedComparisonTransport(root=self.root, parent_url=urls[0], candidate_url=urls[1],
            parent_adapter='none', candidate_adapter='b' * 64, max_request_seconds=2)
        checked = self.wrap(transport)
        self.budget.reserve_model(self.reservation, kind='evaluation')
        checked('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=2)
        self.assertEqual(servers[0].posts, 1)
        servers[1].adapter = 'c' * 64
        self.budget.reserve_model(self.reservation, kind='evaluation')
        with self.assertRaises(BackendError):
            checked('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=2)
        self.assertEqual(servers[0].posts, 1)

    @patch('src.arc.rsi_comparison_contract.verify_guarded_contract')
    def test_verifies_both_arms_before_each_dispatch(self, verify):
        verify.side_effect = self.contract
        wrapped = self.wrap()
        self.assertEqual(verify.call_count, 2)
        with patch.object(CapturedComparisonTransport, '__call__', return_value={'code': 'code', 'request_id': 'a' * 32}) as dispatch:
            self.states[self.parent_url]['reserved_tokens'] = 100
            self.budget.reserve_model(self.reservation, kind='evaluation')
            wrapped('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=1)
            dispatch.assert_called_once()
            self.assertEqual(verify.call_count, 4)

    @patch('src.arc.rsi_comparison_contract.verify_guarded_contract')
    def test_runner_reservations_drive_guarded_dispatch(self, verify):
        verify.side_effect = self.contract
        wrapped = self.wrap()
        ids = [f'task-{i:03}' for i in range(200)]
        tasks = {key: Task(key, (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for key in ids}
        verifier = Verifier({key: [[[4, 3]]] for key in ids})
        seeds = {key: [10000 + 2 * i, 10001 + 2 * i] for i, key in enumerate(ids)}
        replies = [{'code': FAKE_PROGRAMS[1], 'request_id': '1' * 32},
                   {'code': FAKE_PROGRAMS[1], 'request_id': '2' * 32}, TimeoutError('fixture stop')]
        with patch.object(CapturedComparisonTransport, '__call__', side_effect=replies) as dispatch:
            outcome = run_comparison(root=self.root, registration_sha256='a' * 64, generation=0,
                task_ids=ids, tasks=tasks, seeds=seeds, parent_adapter='none', candidate_adapter='b' * 64,
                generate=wrapped, sandbox=FakeE6Sandbox(), verifier=verifier, budget=self.budget)
        self.assertEqual(outcome['status'], 'incomplete')
        self.assertEqual(outcome['budget_usage']['model_calls'], 3)
        self.assertEqual(dispatch.call_count, 3)
        self.assertEqual(verify.call_count, 8)

    @patch('src.arc.rsi_comparison_contract.verify_guarded_contract')
    def test_rejects_ledger_or_budget_drift_before_dispatch(self, verify):
        verify.side_effect = self.contract
        wrapped = self.wrap()
        self.states[self.parent_url]['ledger_id'] = '3' * 32
        self.budget.reserve_model(self.reservation, kind='evaluation')
        with patch.object(CapturedComparisonTransport, '__call__') as dispatch:
            with self.assertRaises(ValueError):
                wrapped('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=1)
            dispatch.assert_not_called()

    @patch('src.arc.rsi_comparison_contract.verify_guarded_contract')
    def test_requires_exact_pre_dispatch_reservation(self, verify):
        verify.side_effect = self.contract
        wrapped = self.wrap()
        with patch.object(CapturedComparisonTransport, '__call__') as dispatch:
            with self.assertRaises(ValueError):
                wrapped('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=1)
            self.assertEqual(verify.call_count, 2)
            self.budget.reserve_model(self.reservation - 1, kind='evaluation')
            with self.assertRaises(ValueError):
                wrapped('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=1)
            dispatch.assert_not_called()

    @patch('src.arc.rsi_comparison_contract.verify_guarded_contract')
    def test_server_reservations_cannot_exceed_local_budget(self, verify):
        verify.side_effect = self.contract
        wrapped = self.wrap()
        self.budget.reserve_model(self.reservation, kind='evaluation')
        self.states[self.parent_url]['reserved_tokens'] = self.reservation + 1
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
                self.wrap()
