import base64
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from src.arc.data import Task, Verifier
from src.arc.e6_fake import FakeE6Sandbox
from src.arc.rsi_comparison_audit import audit_comparison_captures
from src.arc.rsi_comparison_budget import ComparisonBudget
from src.arc.rsi_comparison_runner import run_comparison
from src.arc.rsi_comparison_transport import CapturedComparisonTransport
from src.arc.rsi_transport_audit import audit_transport
from src.arc.rsi_transition_screen import screen_captured_rsi_chain
from src.arc.rsi_base_comparison import _paired_frozen_summary, summarize_captured_base_comparison

MODEL = 'mlx-community/Qwen3-14B-4bit'
CODE = 'def transform(grid):\n    return grid[::-1]'


class ComparisonTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.requests.append(self.rfile.read(int(self.headers['Content-Length'])))
                content = self.server.content
                if self.server.arm_difference and (len(self.server.requests) - 1) % 4 < 2:
                    content = 'def transform(grid):\n    return [row[:] for row in grid]'
                raw = json.dumps({'model': MODEL, 'thinking_mode': 'off',
                    'thinking_control': 'tokenizer_template_canary_passed',
                    'choices': [{'message': {'role': 'assistant', 'content': content}, 'finish_reason': ('length' if self.server.truncate_first and len(self.server.requests) == 1 else self.server.finish_reason)}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.server.requests = []
        self.server.content, self.server.finish_reason = CODE, 'stop'
        self.server.truncate_first = False
        self.server.arm_difference = False
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.worker.join)
        self.addCleanup(self.server.shutdown)
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.transport = CapturedComparisonTransport(root=self.root, parent_url=self.url, candidate_url=self.url,
            parent_adapter='none', candidate_adapter='b' * 64, max_request_seconds=2)

    def test_each_arm_has_matched_settings_and_captured_bytes(self):
        for arm, adapter in [('parent', 'none'), ('candidate', 'b' * 64)]:
            answer = self.transport('prompt', arm=arm, adapter=adapter, task_id='task', sample=0, seed=1024, timeout=1)
            ledger = json.loads((self.root / f'{arm}.responses' / 'requests.jsonl').read_text())
            capture = json.loads((self.root / f'{arm}.capture.jsonl').read_text())
            response = json.loads((self.root / f'{arm}.responses' / 'llm_responses.jsonl').read_text())
            raw = base64.b64decode(capture['body_base64'])
            self.assertEqual(answer['request_id'], ledger['request_id'])
            self.assertEqual(raw, self.server.requests[-1])
            audit_transport(raw, base64.b64decode(response['raw_http_body_base64']), ledger,
                            expected_prompt='prompt', expected_code=CODE, expected_seed=1024)
        self.assertEqual([json.loads(raw)['seed'] for raw in self.server.requests], [1024, 1024])

    def test_full_comparison_captures_join_and_replay(self):
        self.server.truncate_first = True
        ids = [f'task-{i:03}' for i in range(200)]
        tasks = {key: Task(key, (([[1], [2]], [[2], [1]]),), ([[3], [4]],)) for key in ids}
        verifier = Verifier({key: [[[4], [3]]] for key in ids})
        seeds = {key: [10000 + i * 2, 10001 + i * 2] for i, key in enumerate(ids)}
        budget = ComparisonBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        result = run_comparison(root=self.root, registration_sha256='a' * 64, generation=0, task_ids=ids,
            tasks=tasks, seeds=seeds, parent_adapter='none', candidate_adapter='b' * 64,
            generate=self.transport, sandbox=FakeE6Sandbox(), verifier=verifier, budget=budget)
        self.assertEqual(result['status'], 'frozen_unverified')
        first = json.loads((self.root / 'comparison-000-parent.json').read_text())
        self.assertEqual(first['result']['sample_statuses'][0], 'invalid_model_output')
        report = audit_comparison_captures(self.root, expected_sha256=result['sha256'], tasks=tasks,
                                           sandbox=FakeE6Sandbox(), verifier=verifier)
        self.assertEqual(report['checked_requests'], 800)
        self.assertEqual(report['budget_usage']['model_calls'], 800)
        self.assertGreater(report['budget_usage']['reserved_tokens'], 0)
        self.assertFalse(report['independently_verified'])
        screened = screen_captured_rsi_chain(first={'root': self.root, 'sha256': result['sha256']},
            tasks=tasks, sandbox=FakeE6Sandbox(), verifier=verifier)
        self.assertEqual(screened['status'], 'first_screen_nonpositive_unverified')
        self.assertEqual(screened['evidence_type'], 'caller_supplied_captures_sandbox_and_verifier')
        self.assertFalse(screened['improvement_claim_supported'])
        ledger_path = self.root / 'parent.responses' / 'requests.jsonl'
        ledger_lines = ledger_path.read_text().splitlines()
        changed = json.loads(ledger_lines[0])
        changed['token_reservation'] += 1
        ledger_path.write_text('\n'.join([json.dumps(changed), *ledger_lines[1:]]) + '\n')
        with self.assertRaises(ValueError):
            audit_comparison_captures(self.root, expected_sha256=result['sha256'], tasks=tasks,
                                      sandbox=FakeE6Sandbox(), verifier=verifier)

        with self.assertRaises(ValueError):
            screen_captured_rsi_chain(first={'root': self.root, 'sha256': result['sha256']},
                tasks=tasks, sandbox=FakeE6Sandbox(), verifier=verifier)
        ledger_path.write_text('\n'.join(ledger_lines) + '\n')
        capture = self.root / 'parent.capture.jsonl'
        lines = capture.read_text().splitlines()
        capture.write_text('\n'.join(reversed(lines)) + '\n')
        audit_comparison_captures(self.root, expected_sha256=result['sha256'], tasks=tasks,
                                  sandbox=FakeE6Sandbox(), verifier=verifier)
        corrupted = json.loads(lines[0])
        wire = json.loads(base64.b64decode(corrupted['body_base64']))
        wire['seed'] += 1
        changed = json.dumps(wire).encode()
        corrupted['body_base64'] = base64.b64encode(changed).decode()
        corrupted['body_sha256'] = hashlib.sha256(changed).hexdigest()
        altered_lines = [json.dumps(corrupted), *lines[1:]]
        capture.write_text('\n'.join(altered_lines) + '\n')
        with self.assertRaises(ValueError):
            audit_comparison_captures(self.root, expected_sha256=result['sha256'], tasks=tasks,
                                      sandbox=FakeE6Sandbox(), verifier=verifier)
        capture.write_text('\n'.join(lines[:-1]) + '\n')
        with self.assertRaises(ValueError):
            audit_comparison_captures(self.root, expected_sha256=result['sha256'], tasks=tasks,
                                      sandbox=FakeE6Sandbox(), verifier=verifier)
        capture.write_text('\n'.join(lines + [lines[0]]) + '\n')
        with self.assertRaises(ValueError):
            audit_comparison_captures(self.root, expected_sha256=result['sha256'], tasks=tasks,
                                      sandbox=FakeE6Sandbox(), verifier=verifier)

    def test_two_captured_transitions_require_both_capture_sets(self):
        self.server.arm_difference = True
        ids = [f'task-{i:03}' for i in range(200)]
        tasks = {key: Task(key, (([[1], [2]], [[2], [1]]),), ([[3], [4]],)) for key in ids}
        verifier = Verifier({key: [[[4], [3]]] for key in ids})
        seeds = {key: [10000 + i * 2, 10001 + i * 2] for i, key in enumerate(ids)}
        refs = []
        for generation, parent, candidate in ((0, 'none', 'b' * 64), (1, 'b' * 64, 'c' * 64)):
            root = self.root / f'run-{generation}'
            root.mkdir(mode=0o700)
            transport = CapturedComparisonTransport(root=root, parent_url=self.url, candidate_url=self.url,
                parent_adapter=parent, candidate_adapter=candidate, max_request_seconds=2)
            budget = ComparisonBudget.create(root, registration_sha256='a' * 64, generation=generation)
            result = run_comparison(root=root, registration_sha256='a' * 64, generation=generation,
                task_ids=ids, tasks=tasks, seeds=seeds, parent_adapter=parent, candidate_adapter=candidate,
                generate=transport, sandbox=FakeE6Sandbox(), verifier=verifier, budget=budget)
            refs.append({'root': root, 'sha256': result['sha256']})
        screened = screen_captured_rsi_chain(first=refs[0], second=refs[1], tasks=tasks,
            sandbox=FakeE6Sandbox(), verifier=verifier)
        self.assertEqual(screened['status'], 'second_screen_positive_unverified')
        self.assertFalse(screened['execution_ready'])
        summary = summarize_captured_base_comparison(first=refs[0], second=refs[1], tasks=tasks,
            sandbox=FakeE6Sandbox(), verifier=verifier)
        self.assertEqual(summary['base_solved'], 0)
        self.assertEqual(summary['candidate_solved'], 200)
        self.assertEqual(summary['difference_percentage_points'], 100.0)
        self.assertEqual(summary['selected_adapter'], 'c' * 64)
        self.assertFalse(summary['improvement_claim_supported'])
        first_record = json.loads((refs[0]['root'] / 'comparison.frozen.json').read_text())
        selected_record = json.loads((refs[1]['root'] / 'comparison.frozen.json').read_text())
        key = ids[0]
        selected_record['seeds'][key] = [90000, 90001]
        for arm in ('parent', 'candidate'):
            selected_record[arm][key]['seeds'] = [90000, 90001]
        with self.assertRaises(ValueError):
            _paired_frozen_summary(first_record, selected_record)
        (refs[1]['root'] / 'candidate.capture.jsonl').unlink()
        with self.assertRaises(ValueError):
            screen_captured_rsi_chain(first=refs[0], second=refs[1], tasks=tasks,
                sandbox=FakeE6Sandbox(), verifier=verifier)
        with self.assertRaises(ValueError):
            summarize_captured_base_comparison(first=refs[0], second=refs[1], tasks=tasks,
                sandbox=FakeE6Sandbox(), verifier=verifier)

    def test_truncated_and_malformed_completions_are_captured_invalid(self):
        self.server.finish_reason = 'length'
        first = self.transport('prompt', arm='parent', adapter='none', task_id='task', sample=0, seed=1, timeout=1)
        self.assertEqual(first['failure'], 'invalid_model_output')
        self.assertIsNone(first['code'])
        self.server.finish_reason, self.server.content = 'stop', '<think>unterminated'
        second = self.transport('prompt', arm='parent', adapter='none', task_id='task', sample=1, seed=2, timeout=1)
        self.assertEqual(second['failure'], 'invalid_model_output')
        self.assertIsNone(second['code'])
        self.assertNotEqual(first['request_id'], second['request_id'])
        self.assertEqual(len((self.root / 'parent.responses' / 'llm_responses.jsonl').read_text().splitlines()), 2)
        ledgers = [json.loads(line) for line in (self.root / 'parent.responses' / 'requests.jsonl').read_text().splitlines()]
        responses = [json.loads(line) for line in (self.root / 'parent.responses' / 'llm_responses.jsonl').read_text().splitlines()]
        captures = [json.loads(line) for line in (self.root / 'parent.capture.jsonl').read_text().splitlines()]
        for index in range(2):
            outcome = audit_transport(base64.b64decode(captures[index]['body_base64']),
                base64.b64decode(responses[index]['raw_http_body_base64']), ledgers[index],
                expected_prompt='prompt', expected_code=None, expected_seed=index + 1, expected_invalid=True)
            self.assertEqual(outcome['status'], 'transport_invalid_model_output')

    def test_invalid_route_or_seed_fails_before_dispatch(self):
        with self.assertRaises(ValueError):
            self.transport('prompt', arm='parent', adapter='b' * 64, task_id='task', sample=0, seed=1, timeout=1)
        with self.assertRaises(ValueError):
            self.transport('prompt', arm='candidate', adapter='b' * 64, task_id='task', sample=2, seed=1, timeout=1)
        self.assertEqual(self.server.requests, [])
