import base64
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from src.arc.rsi_comparison_transport import CapturedComparisonTransport
from src.arc.rsi_transport_audit import audit_transport

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
                raw = json.dumps({'model': MODEL, 'thinking_mode': 'off',
                    'thinking_control': 'tokenizer_template_canary_passed',
                    'choices': [{'message': {'role': 'assistant', 'content': CODE}, 'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.server.requests = []
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

    def test_invalid_route_or_seed_fails_before_dispatch(self):
        with self.assertRaises(ValueError):
            self.transport('prompt', arm='parent', adapter='b' * 64, task_id='task', sample=0, seed=1, timeout=1)
        with self.assertRaises(ValueError):
            self.transport('prompt', arm='candidate', adapter='b' * 64, task_id='task', sample=2, seed=1, timeout=1)
        self.assertEqual(self.server.requests, [])
