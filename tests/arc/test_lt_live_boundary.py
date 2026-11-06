"""Opt-in synthetic integration against the installed Docker daemon.

No model is loaded, and no real ARC task or generated model code is used.
"""
import json
import os
import subprocess
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from src.arc.data import Task
from src.arc.lt_comparison import budget_for_model, freeze_manifest, make_manifest, run_bounded_task
from src.arc.lt_components import prepare_components
from src.arc.sandbox import IDENTITY_CODE


@unittest.skipUnless(os.environ.get('LT_SANDBOX_IMAGE'), 'requires explicit local Docker image digest')
class DockerBoundaryTests(unittest.TestCase):
    def test_synthetic_http_agent_and_real_docker_for_both_models(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                content = IDENTITY_CODE if len(requests) % 2 == 0 else 'copy input'
                raw = json.dumps({'model': payload['model'], 'thinking_mode': 'off',
                    'thinking_control': 'tokenizer_template_canary_passed',
                    'choices': [{'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args): pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        identifiers = [uuid.uuid4() for _ in range(8)]
        names = {'arc-' + item.hex for item in identifiers}
        def present():
            listing = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'],
                                     check=True, capture_output=True, text=True, timeout=10)
            return names & set(listing.stdout.splitlines())
        try:
            self.assertFalse(present())
            proposal = make_manifest([f't{i:03}' for i in range(200)], seed=1, development=set(),
                                     training=set(), exposures=set(), dataset_sha256='a' * 64)
            config = {'prompt_sha256': 'b' * 64, 'code_sha256': 'c' * 64,
                'models': {'14b': '1' * 40, '32b': '2' * 40},
                'model_names': {'14b': 'model14', '32b': 'model32'},
                'backend_url': f'http://127.0.0.1:{server.server_port}/v1',
                'thinking': 'off', 'temperature': 0.7, 'candidates': 2, 'max_tokens': 1000,
                'max_prompt_tokens': 5000, 'timeout_seconds': 60, 'seed': 42, 'max_repairs': 0,
                'prompt_variant': 'legacy', 'model_files_sha256': {'14b': 'd' * 64, '32b': 'e' * 64},
                'runtime_sha256': 'f' * 64,
                'sandbox': {'image': os.environ['LT_SANDBOX_IMAGE'], 'timeout': 5, 'memory_mb': 128,
                            'cpus': 1, 'pids': 32},
                'budget': {'max_calls': 2000, 'max_reserved_tokens': 20000000, 'max_sandbox_starts': 8000,
                           'max_seconds': 36000, 'max_calls_per_task': 10, 'max_reserved_tokens_per_task': 100000,
                           'max_sandbox_per_task': 40, 'max_seconds_per_task': 120}}
            frozen = freeze_manifest(proposal, config)
            task = Task('t000', (([[1]], [[1]]),), ([[2]],))
            iterator = iter(identifiers)
            with patch('src.arc.sandbox.uuid', SimpleNamespace(uuid4=lambda: next(iterator))):
                for model in ('14b', '32b'):
                    backend, sandbox = prepare_components(frozen, model)
                    row = run_bounded_task(task, backend, sandbox, budget_for_model(frozen, model),
                                           max_tokens=1000, max_repairs=0)
                    self.assertEqual(row['status'], 'complete')
                    self.assertEqual(row['candidates'], [[[[2]]], [[[2]]]])
            self.assertEqual([item['seed'] for item in requests], [42, 42, 43, 43] * 2)
            self.assertEqual([item['model'] for item in requests], ['model14'] * 4 + ['model32'] * 4)
            self.assertFalse(present())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__': unittest.main()
