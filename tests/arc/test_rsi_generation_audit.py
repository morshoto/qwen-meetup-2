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
from src.arc.rsi_generation_audit import audit_generation
from src.arc.rsi_generation_drafts import prepare_generation_drafts
from src.arc.rsi_batch_precheck import precheck_batch
from src.arc.rsi_generation_stage import collect_generation
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_slot_transport import CapturedSlotTransport


class GenerationAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [f'task-{i:02}' for i in range(20)]
        self.tasks = {key: Task(key, (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for key in self.ids}
        self.hashes = {key: task_sha256(self.tasks[key]) for key in self.ids[:10]}
        self.public, self.private = _quality_grids(42, 5), _quality_grids(43, 5)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                code = self.server.codes.get(request['seed'], FAKE_PROGRAMS[1])
                raw = json.dumps({'model': 'mlx-community/Qwen3-14B-4bit', 'thinking_mode': 'off',
                    'thinking_control': 'tokenizer_template_canary_passed',
                    'choices': [{'message': {'role': 'assistant', 'content': code}, 'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.server.codes = {}
        worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(worker.join)
        self.addCleanup(self.server.shutdown)
        self.transport = CapturedSlotTransport(base_url=f'http://127.0.0.1:{self.server.server_port}/v1',
            log_path=self.root / 'requests.jsonl', request_capture_path=self.root / 'capture.jsonl', max_request_seconds=2)
        self.pool = []
        self.pool_hash = hashlib.sha256(b'[]').hexdigest()

    def test_nine_distinct_fixture_drafts_remain_quarantined(self):
        codes = {}
        for index, key in enumerate(self.ids[:10]):
            shift = (index % 9) + 1
            code = f'def transform(grid):\n    return [[(x + {shift}) % 10 for x in row] for row in grid]'
            codes[code] = shift
            self.server.codes[12000 + index * 3] = code
            if index == 9:
                self.server.codes[12001 + index * 3] = code
            self.tasks[key] = Task(key, (([[1, 2]], [[(1 + shift) % 10, (2 + shift) % 10]]),), ([[3, 4]],))
        self.hashes = {key: task_sha256(self.tasks[key]) for key in self.ids[:10]}
        class Sandbox:
            def run(self, code, grid):
                return [[(x + codes[code]) % 10 for x in row] for row in grid]
        common = dict(root=self.root, registration_sha256='a' * 64, generation=0, parent_adapter='none',
            task_ids=self.ids, tasks=self.tasks, expected_manifest_sha256=manifest_sha256(self.ids),
            expected_task_hashes=self.hashes, public_probes=self.public, private_probes=self.private,
            expected_public_probes_sha256=probes_sha256(self.public), expected_private_probes_sha256=probes_sha256(self.private),
            reference_pool=self.pool, expected_reference_pool_sha256=self.pool_hash)
        budget = CollectionBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        result = collect_generation(**common, transport=self.transport, sandbox=Sandbox(), budget=budget)
        self.assertEqual(result['counts'], {'novel_unverified': 9, 'repeated_code': 1})
        capture = [json.loads(line) for line in (self.root / 'capture.jsonl').read_text().splitlines()]
        ledger = [json.loads(line) for line in (self.root / 'requests.jsonl').read_text().splitlines()]
        response = [json.loads(line) for line in (self.root / 'llm_responses.jsonl').read_text().splitlines()]
        kwargs = dict(**common, request_records=capture, ledger_records=ledger, response_records=response, sandbox=Sandbox())
        with self.assertRaises(ValueError):
            prepare_generation_drafts(**{**kwargs, 'request_records': capture[:-1]})
        self.assertFalse((self.root / 'g0.stage.drafts.json').exists())
        report = prepare_generation_drafts(**kwargs)
        self.assertEqual(report['draft_count'], 9)
        self.assertTrue(report['provisional_minimum_count_met'])
        self.assertFalse(report['training_allowed'])
        saved = json.loads((self.root / 'g0.stage.drafts.json').read_text())
        self.assertEqual(len(saved['drafts']), 9)
        self.assertEqual(len(saved['synthetic_examples']), 9)
        self.assertEqual(len(saved['proposed_sft_rows']), 9)
        self.assertEqual(saved['mapping_rejections'], [])
        for draft, record in zip(saved['drafts'], saved['synthetic_examples']):
            self.assertEqual(record['parent_task_id'], draft['parent_task_id'])
            self.assertEqual(record['source_sha256'], draft['source_sha256'])
            self.assertEqual(record['request_id'], draft['provenance']['request_id'])
            self.assertEqual(record['slot_evidence_sha256'], draft['provenance']['slot_evidence_sha256'])
            self.assertEqual(len(record['examples']), 1)
            self.assertFalse(record['training_allowed'])
        for row in saved['proposed_sft_rows']:
            self.assertEqual(row['mapping'], 'generated_demos_holdout_first_visible_v1')
            self.assertFalse(row['training_allowed'])
        self.assertTrue(all(row['provenance']['execution_mode'] == 'unverified_capture' for row in saved['drafts']))
        self.assertNotIn('query_outputs', json.dumps(saved))
        with self.assertRaises(ValueError):
            precheck_batch(saved['drafts'], generation=0, parent_adapter='none', allowed_tasks=set(self.ids))
        with self.assertRaises(FileExistsError):
            prepare_generation_drafts(**kwargs)

    def test_captured_generation_replays_in_order_and_detects_tampering(self):
        common = dict(root=self.root, registration_sha256='a' * 64, generation=0, parent_adapter='none',
            task_ids=self.ids, tasks=self.tasks, expected_manifest_sha256=manifest_sha256(self.ids),
            expected_task_hashes=self.hashes, public_probes=self.public, private_probes=self.private,
            expected_public_probes_sha256=probes_sha256(self.public), expected_private_probes_sha256=probes_sha256(self.private),
            reference_pool=self.pool, expected_reference_pool_sha256=self.pool_hash)
        budget = CollectionBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        result = collect_generation(**common, transport=self.transport, sandbox=FakeE6Sandbox(), budget=budget)
        self.assertEqual(result['counts'], {'novel_unverified': 1, 'repeated_code': 9})
        capture = [json.loads(line) for line in (self.root / 'capture.jsonl').read_text().splitlines()]
        ledger = [json.loads(line) for line in (self.root / 'requests.jsonl').read_text().splitlines()]
        response = [json.loads(line) for line in (self.root / 'llm_responses.jsonl').read_text().splitlines()]
        audit_args = dict(**common, request_records=list(reversed(capture)), ledger_records=ledger,
                          response_records=response, sandbox=FakeE6Sandbox())
        audit = audit_generation(**audit_args)
        self.assertEqual(audit['status'], 'generation_replay_consistent')
        self.assertEqual(audit['checked_slots'], 10)
        self.assertEqual(audit['attempt_counts'], {'novel_unverified': 1, 'duplicate_source': 9, 'repeated_code': 9})
        self.assertFalse(audit['training_allowed'])
        self.assertEqual(audit['budget_usage']['model_calls'], 19)
        self.assertEqual(audit['budget_usage']['reserved_tokens'], sum(row['token_reservation'] for row in ledger))
        drafts = prepare_generation_drafts(**audit_args)
        self.assertEqual(drafts['draft_count'], 1)
        self.assertFalse(drafts['provisional_minimum_count_met'])
        self.assertFalse(drafts['training_allowed'])
        bad_ledger = [dict(row) for row in ledger]
        bad_ledger[0]['token_reservation'] += 1
        with self.assertRaises(ValueError):
            audit_generation(**{**audit_args, 'ledger_records': bad_ledger})
        with self.assertRaises(ValueError):
            audit_generation(**{**audit_args, 'request_records': capture[:-1]})
        with self.assertRaises(ValueError):
            audit_generation(**{**audit_args, 'request_records': capture + [capture[0]]})
        stage_path = self.root / 'g0.stage.completed.json'
        stage = json.loads(stage_path.read_text())
        original_attempts = dict(stage['attempt_counts'])
        stage['attempt_counts']['duplicate_source'] += 1
        stage_path.write_text(json.dumps(stage))
        with self.assertRaises(ValueError):
            audit_generation(**audit_args)
        stage['attempt_counts'] = original_attempts
        original = dict(stage['budget_usage'])
        stage['budget_usage']['reserved_tokens'] += 1
        stage_path.write_text(json.dumps(stage))
        with self.assertRaises(ValueError):
            audit_generation(**audit_args)
        stage['budget_usage'] = original
        stage['counts']['novel_unverified'] = 9
        stage_path.write_text(json.dumps(stage))
        with self.assertRaises(ValueError):
            audit_generation(**audit_args)
