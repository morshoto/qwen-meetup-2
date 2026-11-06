"""Fixture HTTP requests demonstrate private, fail-closed request capture."""
import base64
import hashlib
import json
import stat
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_http_backend import CapturingBackend
from src.arc.rsi_transport_audit import audit_transport


MODEL = 'mlx-community/Qwen3-14B-4bit'
REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
CODE = 'def transform(grid):\n    return grid[::-1]'


class CaptureTests(unittest.TestCase):
    def test_existing_backend_source_remains_compatible_with_historical_seals(self):
        source = Path('src/arc/http_backend.py').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '46fddf994948c98e8c19ee4da7c84a0a3131a8ccdcde892f61f6d1092fd49c27')

    def test_captured_bytes_match_fixture_server_and_audit(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.observed = self.rfile.read(int(self.headers['Content-Length']))
                raw = json.dumps({'model': MODEL, 'thinking_mode': 'off',
                                  'thinking_control': 'tokenizer_template_canary_passed',
                                  'choices': [{'message': {'role': 'assistant', 'content': CODE},
                                               'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                logs = Path(tmp) / 'requests.jsonl'
                capture = Path(tmp) / 'private' / 'raw_requests.jsonl'
                client = CapturingBackend(model=MODEL, revision=REVISION,
                    base_url=f'http://127.0.0.1:{server.server_port}/v1', temperature=0.2,
                    max_tokens=512, seed=12000, timeout=2, thinking='off', guarded=True,
                    log_path=logs, request_capture_path=capture)
                self.assertEqual(client.generate('visible prompt', kind='e5_generator', candidate=0), CODE)
                request = json.loads(capture.read_text())
                ledger = json.loads(logs.read_text())
                response = json.loads((logs.parent / 'llm_responses.jsonl').read_text())
                wire = base64.b64decode(request['body_base64'], validate=True)
                self.assertEqual(wire, server.observed)
                self.assertEqual(request['request_id'], ledger['request_id'])
                self.assertEqual(hashlib.sha256(wire).hexdigest(), ledger['request_body_sha256'])
                self.assertEqual(stat.S_IMODE(capture.stat().st_mode), 0o600)
                result = audit_transport(wire, base64.b64decode(response['raw_http_body_base64']), ledger,
                                         expected_prompt='visible prompt', expected_code=CODE, expected_seed=12000)
                self.assertEqual(result['status'], 'transport_consistent')
                self.assertFalse(result['eligible_for_training'])
        finally:
            server.shutdown()
            worker.join()
            server.server_close()

    def test_capture_failure_stops_before_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingBackend(model=MODEL, revision=REVISION,
                base_url='http://127.0.0.1:1/v1', temperature=0.2, max_tokens=512,
                seed=12000, timeout=1, thinking='off', guarded=True,
                log_path=Path(tmp) / 'requests.jsonl', request_capture_path=Path(tmp) / 'raw.jsonl')
            with patch('src.arc.rsi_http_backend._append_private_jsonl', side_effect=OSError('capture failed')):
                with patch('src.arc.rsi_http_backend.urllib.request.build_opener') as opener:
                    with self.assertRaisesRegex(OSError, 'capture failed'):
                        client.generate('visible prompt', kind='e5_generator', candidate=0)
                    opener.assert_not_called()

    def test_capture_requires_guarded_mode_and_distinct_log_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'requests.jsonl'
            common = dict(model=MODEL, revision=REVISION, base_url='http://127.0.0.1:1/v1',
                          temperature=0.2, max_tokens=512, seed=12000, timeout=1, log_path=log)
            with self.assertRaises(ValueError):
                CapturingBackend(**common, request_capture_path=Path(tmp) / 'raw.jsonl')
            with self.assertRaises(ValueError):
                CapturingBackend(**common, guarded=True, request_capture_path=log)
            with self.assertRaises(ValueError):
                CapturingBackend(**common, guarded=True, allow_remote=True,
                                 request_capture_path=Path(tmp) / 'raw.jsonl')
            with self.assertRaises(ValueError):
                CapturingBackend(**common, guarded=True, thinking='on',
                                 request_capture_path=Path(tmp) / 'raw.jsonl')


if __name__ == '__main__':
    unittest.main()
