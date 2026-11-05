from __future__ import annotations

import ast
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FakeE6Sandbox
from src.arc.e6f_config import load_e6f_config
from src.pipelines.e6 import _fake_records
from src.pipelines.e5 import _seal_mutable_file
from src.pipelines.e6f import _audit_record, _probe_inputs, run_e6f
from src.pipelines import e6f as pipeline

ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.settings, self.base = load_e6f_config(ROOT / 'configs/e6f.cpu.fake.yml')

    def test_fake_e2e_repair_gate_seal_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            run_e6f(self.settings, self.base, run_dir=root)
            state = json.loads((root / 'run_state.json').read_text())
            metrics = json.loads((root / 'metrics.json').read_text())
            self.assertTrue(state['ready_for_e6_training'])
            self.assertFalse(state['training_invoked'])
            self.assertEqual(state['unique_verified_oracles'], 5)
            self.assertEqual((metrics['baseline_validation_successes'],
                              metrics['after_repair_validation_successes']), (4, 5))
            rows = [json.loads(x) for x in (root / 'sft_generator_distillation.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r['source_type'] == 'generator_distillation' and '_verifier_only' not in r for r in rows))
            self.assertEqual((root / 'sft_self_solved.jsonl').read_text(), '')
            identity = json.loads((root / 'identity.json').read_text())
            self.assertIn('src/arc/training.py', identity['implementation_sha256'])
            self.assertEqual((root / 'checkpoints/e6/request_cache.json').stat().st_mode & 0o777, 0o600)
            prior = (root / 'seal.json').read_bytes()
            run_e6f(self.settings, self.base, run_dir=root, resume=True)
            self.assertEqual((root / 'seal.json').read_bytes(), prior)
            original_sha = pipeline._sha
            def changed_filter(path):
                return '0' * 64 if str(path).endswith('/src/arc/training.py') else original_sha(path)
            with patch('src.pipelines.e6f._sha', side_effect=changed_filter):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    run_e6f(self.settings, self.base, run_dir=root, resume=True)
            (root / 'metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'seal mismatch'):
                run_e6f(self.settings, self.base, run_dir=root, resume=True)

    def test_insufficient_gate_rejects_repeat_program(self):
        settings = replace(self.settings, target_unique_oracles=6, initial_candidates=6)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            run_e6f(settings, self.base, run_dir=root)
            state = json.loads((root / 'run_state.json').read_text())
            self.assertEqual(state['status'], 'insufficient_verified_oracles')
            self.assertFalse(state['ready_for_e6_training'])
            self.assertEqual(state['unique_verified_oracles'], 5)
            self.assertIn('duplicate_oracle', (root / 'rejections.jsonl').read_text())

    def test_budget_exhaustion_stops_without_training(self):
        settings = replace(self.settings, max_sandbox_runs=1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            run_e6f(settings, self.base, run_dir=root)
            state = json.loads((root / 'run_state.json').read_text())
            metrics = json.loads((root / 'metrics.json').read_text())
            self.assertEqual(state['status'], 'budget_exhausted')
            self.assertEqual(metrics['budget']['sandbox_runs'], 1)
            self.assertFalse(state['training_invoked'])

    def test_live_requires_flags_before_any_execution(self):
        smoke, live_base = load_e6f_config(ROOT / 'configs/e6f.live.smoke.yml')
        collection, _ = load_e6f_config(ROOT / 'configs/e6f.live.collection.yml')
        with self.assertRaisesRegex(ValueError, 'requires --live'):
            run_e6f(smoke, live_base)
        with self.assertRaisesRegex(ValueError, 'requires --allow-collection'):
            run_e6f(collection, live_base, live=True)

    def test_interrupted_candidate_is_not_reissued_on_resume(self):
        settings = replace(self.settings, initial_candidates=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            with patch('src.pipelines.e6f.run_candidate', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_e6f(settings, self.base, run_dir=root)
            run_e6f(settings, self.base, run_dir=root, resume=True)
            rows = [json.loads(x) for x in (root / 'oracle_lineage.jsonl').read_text().splitlines()]
            started = [r['candidate_index'] for r in rows if r.get('event') == 'candidate_started']
            self.assertEqual(started, [0, 1])
            self.assertTrue(any(r.get('candidate_index') == 0 and r.get('status') == 'interrupted' for r in rows))

    def test_uncertain_model_request_fails_closed_on_resume(self):
        settings = replace(self.settings, initial_candidates=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'

            def interrupt(**kwargs):
                kwargs['generate']('fixture prompt', 0, 0)
                cache = root / 'checkpoints/e6/request_cache.json'
                state = json.loads(cache.read_text())
                state['pending'] = {'key': 'uncertain'}
                cache.write_text(json.dumps(state))
                _seal_mutable_file(cache)
                raise KeyboardInterrupt

            with patch('src.pipelines.e6f.run_candidate', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_e6f(settings, self.base, run_dir=root)
            run_e6f(settings, self.base, run_dir=root, resume=True)
            state = json.loads((root / 'run_state.json').read_text())
            metrics = json.loads((root / 'metrics.json').read_text())
            self.assertEqual(state['status'], 'interrupted_uncertain_request')
            self.assertEqual(metrics['budget']['model_calls'], 1)
            self.assertFalse(state['training_invoked'])

    def test_source_reaudit_checks_labels_and_identity(self):
        row = _fake_records(self.base, 0)[1]
        labels = {'task_id': row['task_id'], **row.pop('_verifier_only')}
        row['qwen_provenance']['execution_mode'] = 'live_model'
        with patch('src.pipelines.e6f._training_safe_record'):
            signature = _audit_record(row, labels, sandbox=FakeE6Sandbox(), probes=_quality_grids(88000, 5),
                                      revision=row['base_revision'], model_id=row['qwen_provenance']['model_id'],
                                      forbidden_ids=set())
            self.assertEqual(len(signature), 64)
            bad = dict(labels, outputs=[[[9]] for _ in labels['outputs']])
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                _audit_record(row, bad, sandbox=FakeE6Sandbox(), probes=_quality_grids(88000, 5),
                              revision=row['base_revision'], model_id=row['qwen_provenance']['model_id'],
                              forbidden_ids=set())

    def test_live_probe_inputs_are_distinct_and_have_a_matched_shape_pair(self):
        smoke, _ = load_e6f_config(ROOT / 'configs/e6f.live.smoke.yml')
        probes = _probe_inputs(smoke)
        self.assertEqual(len({json.dumps(grid) for grid in probes}), 5)
        self.assertTrue(any(len(a) == len(b) and len(a[0]) == len(b[0])
                            for i, a in enumerate(probes) for b in probes[i + 1:]))

    def test_execution_path_has_no_operator_search_import(self):
        for name in ('src/arc/e6f_repair.py', 'src/arc/e6f_config.py',
                     'src/pipelines/e6f.py', 'src/cli/e6f.py'):
            tree = ast.parse((ROOT / name).read_text())
            imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any('operator_search' in name for name in imports))


if __name__ == '__main__':
    unittest.main()
