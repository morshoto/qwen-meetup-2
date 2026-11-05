import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_generation_audit import audit_generation
from src.arc.rsi_generation_usage import audit_generation_usage
from src.arc.rsi_generation_drafts import audit_generation_drafts, prepare_generation_drafts
from src.arc.rsi_synthetic_mapping import map_synthetic_row
from src.arc.rsi_batch_precheck import precheck_batch
from src.arc.rsi_generation_stage import collect_generation
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_slot_transport import CapturedSlotTransport
from src.arc.rsi_synthetic_preparation import prepare_replayed_synthetic_training, prepare_synthetic_training
from src.arc.rsi_sft_format import audit_sft_format
from src.arc.rsi_synthetic_dataset import materialize_replayed_synthetic_dataset
from src.arc.rsi_canary_dataset import prepare_canary_dataset
from src.arc.rsi_preflight_spec import prepare_preflight_spec
from tests.arc.test_rsi_mlx_dataset import Dataset


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
                body = {'model': 'mlx-community/Qwen3-14B-4bit', 'thinking_mode': 'off',
                    'thinking_control': 'tokenizer_template_canary_passed',
                    'choices': [{'message': {'role': 'assistant', 'content': code}, 'finish_reason': 'stop'}]}
                if self.server.usage is not None:
                    body['usage'] = self.server.usage
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.server.codes = {}
        self.server.usage = None
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
        self.server.usage = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
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
        budget.reserve_model(1000, kind='canary')
        result = collect_generation(**common, transport=self.transport, sandbox=Sandbox(), budget=budget)
        self.assertEqual(result['counts'], {'novel_unverified': 9, 'repeated_code': 1})
        capture = [json.loads(line) for line in (self.root / 'capture.jsonl').read_text().splitlines()]
        ledger = [json.loads(line) for line in (self.root / 'requests.jsonl').read_text().splitlines()]
        response = [json.loads(line) for line in (self.root / 'llm_responses.jsonl').read_text().splitlines()]
        kwargs = dict(**common, request_records=capture, ledger_records=ledger, response_records=response, sandbox=Sandbox())
        usage = audit_generation_usage(**kwargs)
        self.assertEqual(usage['reported']['request_count'], len(ledger))
        self.assertEqual(usage['reported']['total_tokens_if_complete'], 15 * len(ledger))
        self.assertEqual(usage['reserved_tokens'], result['budget_usage']['reserved_tokens'])
        self.assertEqual(usage['setup_reservations']['model_calls'], 1)
        self.assertEqual(usage['setup_reservations']['reserved_tokens'], 1000)
        self.assertEqual(usage['stage_reservations_total']['reserved_tokens'],
                         1000 + result['budget_usage']['reserved_tokens'])
        self.assertIsNone(usage['stage_reported_total_tokens_if_complete'])
        self.assertEqual(usage['reported_scope'], 'captured_slot_attempts_only')
        self.assertIsNone(usage['monetary_cost'])
        altered_ledger = [dict(event) for event in ledger]
        altered_ledger[0]['usage'] = dict(altered_ledger[0]['usage'], prompt_tokens=11)
        with self.assertRaises(ValueError):
            audit_generation_usage(**{**kwargs, 'ledger_records': altered_ledger})
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
        raw = (self.root / 'g0.stage.drafts.json').read_bytes()
        replay = audit_generation_drafts(**kwargs, expected_sha256=report['sha256'])
        self.assertEqual(replay['status'], 'generation_drafts_replay_consistent')
        self.assertFalse(replay['training_allowed'])
        changed = json.loads(raw)
        changed['synthetic_examples'][0]['examples'][0]['output'] = [[9]]
        task_id = changed['drafts'][0]['parent_task_id']
        changed['proposed_sft_rows'][0] = map_synthetic_row(task=self.tasks[task_id],
            draft=changed['drafts'][0], generated=changed['synthetic_examples'][0])
        changed_raw = (json.dumps(changed, sort_keys=True, separators=(',', ':')) + '\n').encode()
        path = self.root / 'g0.stage.drafts.json'
        path.write_bytes(changed_raw)
        with self.assertRaises(ValueError):
            audit_generation_drafts(**kwargs, expected_sha256=hashlib.sha256(changed_raw).hexdigest())
        path.write_bytes(raw)
        altered = json.loads(raw)
        altered['proposed_sft_rows'][0]['messages'][1]['content'] += ' altered'
        altered_raw = json.dumps(altered).encode()
        args = dict(root=self.root, stage_bytes=raw, expected_stage_sha256=report['sha256'],
                    tasks=self.tasks, seed=42)
        with self.assertRaises(ValueError):
            prepare_synthetic_training(**{**args, 'expected_stage_sha256': '0' * 64})
        too_few = json.loads(raw)
        for key in ('drafts', 'synthetic_examples', 'proposed_sft_rows'):
            too_few[key] = too_few[key][:4]
        too_few['draft_count'] = 4
        too_few_raw = json.dumps(too_few).encode()
        insufficient = prepare_synthetic_training(**{**args, 'stage_bytes': too_few_raw,
            'expected_stage_sha256': hashlib.sha256(too_few_raw).hexdigest()})
        self.assertEqual(insufficient['status'], 'insufficient_mapped_records')
        self.assertFalse(insufficient['training_allowed'])
        duplicated = json.loads(raw)
        for key in ('drafts', 'synthetic_examples', 'proposed_sft_rows'):
            duplicated[key][1] = duplicated[key][0]
        duplicated_raw = json.dumps(duplicated).encode()
        with self.assertRaises(ValueError):
            prepare_synthetic_training(**{**args, 'stage_bytes': duplicated_raw,
                'expected_stage_sha256': hashlib.sha256(duplicated_raw).hexdigest()})
        with self.assertRaises(ValueError):
            prepare_synthetic_training(**{**args, 'stage_bytes': altered_raw,
                'expected_stage_sha256': hashlib.sha256(altered_raw).hexdigest()})
        self.assertFalse((self.root / 'synthetic.training.preparation.json').exists())
        marker_path = self.root / 'synthetic.training.replay.json'
        plan_path = self.root / 'synthetic.training.preparation.json'
        with patch('src.arc.rsi_generation_drafts.audit_generation_drafts', side_effect=[None, ValueError('drift')]):
            with self.assertRaises(ValueError):
                prepare_replayed_synthetic_training(**kwargs, expected_stage_sha256=report['sha256'], seed=42)
        self.assertTrue(plan_path.exists())
        self.assertFalse(marker_path.exists())
        plan_path.unlink()
        prepared = prepare_replayed_synthetic_training(**kwargs, expected_stage_sha256=report['sha256'], seed=42)
        self.assertFalse(prepared['training_allowed'])
        self.assertEqual(prepared['evidence_type'], 'caller_supplied_capture_and_sandbox')
        marker = json.loads(marker_path.read_text())
        self.assertEqual(marker['stage_sha256'], report['sha256'])
        self.assertEqual(marker['preparation_sha256'], hashlib.sha256(plan_path.read_bytes()).hexdigest())
        self.assertFalse(marker['training_allowed'])
        plan = json.loads((self.root / 'synthetic.training.preparation.json').read_text())
        self.assertEqual(plan['stage_sha256'], report['sha256'])
        self.assertEqual(len(plan['generation_training']['records']), 9)
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [10, 11]
        audited = audit_sft_format(plan, tasks=prepared['synthetic_tasks'], tokenizer=tokenizer,
                                   render=lambda row: ([10, 11, 12], 2))
        self.assertEqual(audited['row_count'], 9)
        self.assertFalse(audited['training_allowed'])
        dataset_args = dict(**kwargs, expected_stage_sha256=report['sha256'],
                            expected_preparation_sha256=marker['preparation_sha256'],
                            synthetic_tasks=prepared['synthetic_tasks'], tokenizer=tokenizer,
                            render=lambda row: ([10, 11, 12], 2))
        marker_raw = marker_path.read_bytes()
        marker_path.unlink()
        with self.assertRaises(ValueError):
            materialize_replayed_synthetic_dataset(**dataset_args)
        marker_path.write_bytes(marker_raw)
        marker_path.chmod(0o600)
        with self.assertRaises(ValueError):
            materialize_replayed_synthetic_dataset(**{**dataset_args, 'expected_preparation_sha256': '0' * 64})
        self.assertFalse((self.root / 'generation-sft.quarantine').exists())
        with patch('src.arc.rsi_synthetic_dataset.audit_generation_drafts', side_effect=[None, ValueError('drift')]):
            with self.assertRaises(ValueError):
                materialize_replayed_synthetic_dataset(**dataset_args)
        self.assertFalse((self.root / 'synthetic.dataset.replay.json').exists())
        self.assertTrue((self.root / 'generation-sft.quarantine').exists())
        shutil.rmtree(self.root / 'generation-sft.quarantine')
        dataset = materialize_replayed_synthetic_dataset(**dataset_args)
        self.assertEqual(dataset['row_count'], 9)
        self.assertFalse(dataset['training_allowed'])
        dataset_marker = json.loads((self.root / 'synthetic.dataset.replay.json').read_text())
        self.assertEqual(dataset_marker['dataset_jsonl_sha256'], dataset['dataset_jsonl_sha256'])
        self.assertFalse(dataset_marker['training_allowed'])
        render = lambda row: ([10, 11, 12], 2)
        canary = prepare_canary_dataset(root=self.root, forbidden_tasks=self.tasks, tokenizer=tokenizer, render=render)
        def loader(path, tokenizer, config):
            rows = [json.loads(line) for line in (path / 'train.jsonl').read_text().splitlines()]
            return Dataset(rows, render), [], []
        spec = prepare_preflight_spec(root=self.root, preparation_bytes=plan_path.read_bytes(),
            expected_preparation_sha256=marker['preparation_sha256'], expected_canary_sha256=canary['dataset_jsonl_sha256'],
            expected_generation_dataset_sha256=dataset['dataset_jsonl_sha256'], synthetic_tasks=prepared['synthetic_tasks'],
            forbidden_tasks=self.tasks, tokenizer=tokenizer, render=render, load_local_dataset=loader,
            model_dir=self.root / 'model', generation_data_dir=self.root / 'generation-sft.quarantine',
            adapter_dir=self.root / 'canary-adapter', previous_adapter_file=None,
            identity_hashes={'model_manifest': 'a' * 64, 'runtime_manifest': 'b' * 64,
                'environment_manifest': 'c' * 64, 'source_inventory': 'd' * 64,
                'tokenizer': 'e' * 64, 'tokenizer_config': 'f' * 64})
        self.assertEqual(spec['status'], 'preflight_proposed_unverified')
        self.assertFalse(spec['execution_ready'])
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
        usage = audit_generation_usage(**audit_args)
        self.assertEqual(usage['reported']['total_tokens_missing_requests'], 19)
        self.assertIsNone(usage['reported']['total_tokens_if_complete'])
        self.assertEqual(usage['setup_reservations']['model_calls'], 0)
        self.assertIsNone(usage['stage_reported_total_tokens_if_complete'])
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
