import base64
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_lineage_audit import inspect_lineage, probes_sha256
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_collected_slot_audit import audit_collected_slot
from src.arc.rsi_capture_draft import make_capture_draft
from src.arc.rsi_batch_precheck import precheck_batch
from src.arc.rsi_slot_collector import collect_slot
from src.arc.rsi_slot_transport import CapturedSlotTransport

MODEL = 'mlx-community/Qwen3-14B-4bit'


class SlotTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = self.rfile.read(int(self.headers['Content-Length']))
                self.server.requests.append(request)
                code = [FAKE_PROGRAMS[2], FAKE_PROGRAMS[1]][len(self.server.requests) - 1]
                raw = json.dumps({'model': MODEL, 'thinking_mode': 'off',
                    'thinking_control': 'tokenizer_template_canary_passed',
                    'choices': [{'message': {'role': 'assistant', 'content': code}, 'finish_reason': 'stop'}]}).encode()
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
        self.transport = CapturedSlotTransport(base_url=f'http://127.0.0.1:{self.server.server_port}/v1',
            log_path=self.root / 'requests.jsonl', request_capture_path=self.root / 'raw.jsonl', max_request_seconds=2)

    def test_collector_reaches_fixture_server_with_exact_schedule(self):
        ids = [f'task-{i:02}' for i in range(20)]
        task = Task(ids[0], (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        public, private = _quality_grids(42, 5), _quality_grids(43, 5)
        budget = CollectionBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        result = collect_slot(root=self.root, registration_sha256='a' * 64, generation=0, slot=0,
            task_ids=ids, task=task, expected_manifest_sha256=manifest_sha256(ids),
            expected_task_sha256=task_sha256(task), public_probes=public, private_probes=private,
            expected_public_probes_sha256=probes_sha256(public), expected_private_probes_sha256=probes_sha256(private),
            reference_pool=[], transport=self.transport, sandbox=FakeE6Sandbox(), budget=budget)
        self.assertEqual(result['status'], 'novel_unverified')
        wire = [json.loads(raw) for raw in self.server.requests]
        self.assertEqual([item['seed'] for item in wire], [12000, 12001])
        self.assertTrue(all(item['thinking_mode'] == 'off' and item['max_tokens'] == 512 for item in wire))
        self.assertTrue(all(item['deadline_monotonic'] > 0 for item in wire))
        captures = [json.loads(line) for line in (self.root / 'raw.jsonl').read_text().splitlines()]
        self.assertEqual([base64.b64decode(item['body_base64']) for item in captures], self.server.requests)
        self.assertEqual(budget.snapshot()['model_calls'], 2)
        ledger = [json.loads(line) for line in (self.root / 'requests.jsonl').read_text().splitlines()]
        responses = [json.loads(line) for line in (self.root / 'llm_responses.jsonl').read_text().splitlines()]
        evidence = json.loads((self.root / 'g0-slot-00.collection.json').read_text())
        self.assertEqual([item['request_id'] for item in evidence['attempts']], [item['request_id'] for item in ledger])
        self.assertEqual([item['request_id'] for item in evidence['attempts']], [item['request_id'] for item in captures])
        attempts = [{'code': attempt['source'], 'request_body': raw,
                     'response_body': base64.b64decode(response['raw_http_body_base64']), 'ledger': event}
                    for attempt, raw, response, event in zip(evidence['attempts'], self.server.requests, responses, ledger)]
        replay = inspect_lineage(ids, task, generation=0, slot=0, attempts=attempts,
            expected_manifest_sha256=manifest_sha256(ids), expected_task_sha256=task_sha256(task),
            public_probes=public, private_probes=private,
            expected_public_probes_sha256=probes_sha256(public), expected_private_probes_sha256=probes_sha256(private),
            sandbox=FakeE6Sandbox())
        self.assertEqual(replay['attempt_statuses'], ['visible_demonstration_mismatch', 'verified_candidate'])
        self.assertFalse(replay['eligible_for_training'])
        kwargs = dict(evidence_bytes=(self.root / 'g0-slot-00.collection.json').read_bytes(),
            evidence_sha256=result['evidence_sha256'], request_records=list(reversed(captures)),
            response_records=list(reversed(responses)), ledger_records=list(reversed(ledger)),
            task_ids=ids, task=task, generation=0, slot=0, expected_manifest_sha256=manifest_sha256(ids),
            expected_task_sha256=task_sha256(task), public_probes=public, private_probes=private,
            expected_public_probes_sha256=probes_sha256(public), expected_private_probes_sha256=probes_sha256(private),
            reference_pool=[], sandbox=FakeE6Sandbox())
        checked = audit_collected_slot(**kwargs)
        self.assertEqual(checked['status'], 'captured_slot_consistent')
        self.assertFalse(checked['eligible_for_training'])
        draft = make_capture_draft(**kwargs, parent_adapter='none')
        self.assertEqual(draft['status'], 'pending_independent_audit')
        self.assertEqual(draft['provenance']['execution_mode'], 'unverified_capture')
        self.assertEqual(draft['provenance']['request_id'], evidence['attempts'][-1]['request_id'])
        self.assertEqual(draft['provenance']['raw_response_sha256'], ledger[-1]['response_sha256'])
        self.assertFalse(draft['eligible_for_training'])
        self.assertNotIn('query_outputs', json.dumps(draft))
        with self.assertRaises(ValueError):
            precheck_batch([draft], generation=0, parent_adapter='none', allowed_tasks={task.id})
        with self.assertRaises(ValueError):
            make_capture_draft(**{**kwargs, 'evidence_sha256': 'b' * 64}, parent_adapter='none')
        with self.assertRaises(ValueError):
            audit_collected_slot(**{**kwargs, 'evidence_sha256': 'b' * 64})
        with self.assertRaises(ValueError):
            audit_collected_slot(**{**kwargs, 'request_records': captures + [captures[0]]})
        with self.assertRaises(ValueError):
            audit_collected_slot(**{**kwargs, 'response_records': responses[:1]})
        altered = dict(evidence)
        altered['status'] = 'sampled_behavior_match'
        changed = json.dumps(altered).encode()
        with self.assertRaises(ValueError):
            audit_collected_slot(**{**kwargs, 'evidence_bytes': changed,
                                    'evidence_sha256': hashlib.sha256(changed).hexdigest()})
        altered = dict(evidence)
        altered['novelty'] = {**evidence['novelty'], 'reference_pool_sha256': 'f' * 64}
        changed = json.dumps(altered).encode()
        with self.assertRaises(ValueError):
            audit_collected_slot(**{**kwargs, 'evidence_bytes': changed,
                                    'evidence_sha256': hashlib.sha256(changed).hexdigest()})
        corrupted = [dict(item) for item in captures]
        corrupted[0]['body_base64'] = base64.b64encode(b'bad').decode()
        with self.assertRaises(ValueError):
            audit_collected_slot(**{**kwargs, 'request_records': corrupted})

    def test_bad_seed_or_deadline_fails_before_dispatch(self):
        for seed, timeout in [(12030, 1), (11999, 1), (12000, 0), (True, 1)]:
            with self.assertRaises(ValueError):
                self.transport('prompt', seed=seed, timeout=timeout)
        self.assertEqual(self.server.requests, [])

    def test_rejects_remote_endpoint(self):
        with self.assertRaises(ValueError):
            CapturedSlotTransport(base_url='https://example.com/v1', log_path=self.root / 'a',
                                  request_capture_path=self.root / 'b', max_request_seconds=2)
