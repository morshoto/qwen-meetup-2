from __future__ import annotations

import ast
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.e6_curriculum import (collection_rows, diagnose_e5_artifact,
                                   import_e5_oracle, make_distillation_record,
                                   verify_oracle)
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox, FakeE6Trainer
from src.arc.mlx_lora import _training_safe_record
from src.arc.sandbox import DockerSandbox
from src.core.config import load_experiment_config
from src.pipelines.e6 import (_claim_live_attempt, _fake_records, _write_collection,
                              run_e6)


ROOT = Path(__file__).resolve().parents[2]
E5_ROOT = ROOT / 'artifacts/e5-live-smoke-20261101-compact-feedback-final'
IDENTITY = FAKE_PROGRAMS[0]


class E6CurriculumTests(unittest.TestCase):
    def test_saved_e5_diagnostic_uses_traces_without_copying_response_text(self):
        seal_before = (E5_ROOT / 'root_artifact_hashes.json').read_bytes()
        result = diagnose_e5_artifact(E5_ROOT)
        self.assertEqual(result['categories']['training_mismatch'], 2)
        self.assertEqual(result['categories']['train_matched_heldout_mismatch'], 0)
        self.assertEqual(result['categories']['insufficient_evidence'], 0)
        self.assertTrue(all(item['train_mismatch_pair_indices'] == [1, 2]
                            for item in result['candidates']))
        self.assertTrue(all(item['thinking_mode_effective'] == 'off'
                            for item in result['candidates']))
        self.assertTrue(result['thinking_off_canary_recorded'])
        self.assertTrue(all('code' not in item for item in result['candidates']))
        self.assertEqual((E5_ROOT / 'root_artifact_hashes.json').read_bytes(), seal_before)

    def test_e5_oracle_import_requires_verified_qwen_response_provenance(self):
        from src.arc.data import validate_grid
        task_file = next((E5_ROOT / 'gen-0/synthetic').glob('*/task.json'))
        verifier_file = next((E5_ROOT / 'gen-0/synthetic').glob('*/private_verifier.json'))
        task_data = json.loads(task_file.read_text())
        verifier = json.loads(verifier_file.read_text())
        expected = {}
        for item in task_data['task']['train']:
            expected[json.dumps(item['input'], separators=(',', ':'))] = item['output']
        for source, target in zip(verifier['hidden_test_inputs'], verifier['hidden_test_outputs']):
            expected[json.dumps(source, separators=(',', ':'))] = target
        for source, target in zip(verifier['validation_inputs'], verifier['validation_outputs']):
            expected[json.dumps(source, separators=(',', ':'))] = target

        class ProvenanceFixtureSandbox:
            def run(self, code, grid):
                key = json.dumps(grid, separators=(',', ':'))
                return expected.get(key, [[0]])

        from src.arc import e6_curriculum
        original = e6_curriculum.verify_oracle

        def fake_verify(code, inputs, *, sandbox):
            return [sandbox.run(code, grid) for grid in inputs]

        seal_before = (E5_ROOT / 'root_artifact_hashes.json').read_bytes()
        with patch('src.arc.e6_curriculum.verify_oracle', side_effect=fake_verify):
            imported = import_e5_oracle(E5_ROOT, sandbox=ProvenanceFixtureSandbox())
        self.assertEqual(imported['provenance']['request_id'],
                         '00a4e02cff61435f89242185c8440be1')
        self.assertEqual(imported['provenance']['raw_response_sha256'],
                         'f3cb898570cb08b3d2dcc4c2dc6a23d5eacd1601fed48557b7902b00c456eccd')
        self.assertEqual(imported['provenance']['source_type'], 'generator_distillation')
        self.assertEqual((E5_ROOT / 'root_artifact_hashes.json').read_bytes(), seal_before)

    def test_generator_distillation_prompt_and_provenance_are_strict(self):
        task = Task('task-x', (([[1, 2]], [[2, 3]]),), ([[4, 5]],))
        provenance = {'source_type': 'generator_distillation', 'request_id': 'qwen-1',
                      'raw_response_sha256': 'a' * 64, 'model_revision': 'rev-1',
                      'model_id': 'fake-qwen', 'execution_mode': 'fake_fixture',
                      'thinking_mode': 'off'}
        heldout_input, heldout_output = [[6, 7]], [[7, 8]]
        record = make_distillation_record(task, FAKE_PROGRAMS[4], provenance=provenance,
            heldout_inputs=[heldout_input], heldout_outputs=[heldout_output], generation=0)
        self.assertEqual(record['messages'][1]['content'], solve_prompt(task))
        self.assertNotIn(json.dumps(heldout_output), record['messages'][1]['content'])
        self.assertEqual(record['data_source'], 'generator_distillation')
        self.assertEqual(record['qwen_provenance']['request_id'], 'qwen-1')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            make_distillation_record(task, FAKE_PROGRAMS[4], provenance={},
                heldout_inputs=[heldout_input], heldout_outputs=[heldout_output], generation=0)
        external = dict(provenance, source_type='external_or_human')
        with self.assertRaises(ValueError):
            make_distillation_record(task, FAKE_PROGRAMS[4], provenance=external,
                heldout_inputs=[heldout_input], heldout_outputs=[heldout_output], generation=0)
        with self.assertRaisesRegex(ValueError, 'verifier labels'):
            _training_safe_record(record, 'e6_generator_distillation')
        clean = {key: value for key, value in record.items() if key != '_verifier_only'}
        _training_safe_record(clean, 'e6_generator_distillation')
        missing_evidence = json.loads(json.dumps(clean))
        missing_evidence['qwen_provenance']['execution_mode'] = 'live_model'
        with self.assertRaisesRegex(ValueError, 'evidence is missing'):
            _training_safe_record(missing_evidence, 'e6_generator_distillation')

    def test_collection_deduplicates_task_oracle_and_near_copy_shapes(self):
        from src.arc.e6_curriculum import oracle_shape_sha256
        records = _fake_records(load_experiment_config(ROOT / 'configs/e6.fake.yml'), 0)
        self.assertEqual(len(records), 5)
        duplicate_task = dict(records[0], task_id=records[0]['task_id'])
        duplicate_oracle = dict(records[0], task_id='different-task-id')
        near_copy = dict(records[0], task_id='near-copy',
                         oracle_ast_sha256='f' * 64,
                         oracle_shape_sha256=records[0]['oracle_shape_sha256'])
        result = collection_rows([records[0], duplicate_task, duplicate_oracle, near_copy])
        self.assertEqual(len(result['accepted']), 1)
        self.assertEqual({item['reason'] for item in result['rejected']},
                         {'duplicate_task_id', 'duplicate_oracle_fingerprint', 'similar_oracle_shape'})
        self.assertEqual(len(oracle_shape_sha256(FAKE_PROGRAMS[1])), 64)

    def test_oracle_quality_filters_fixed_invalid_nondeterministic_and_timeout(self):
        inputs = [[[1, 2], [3, 4]], [[2, 1]], [[3, 4, 5]], [[6], [7]], [[9, 0]]]
        class ShapeOnly:
            def run(self, code, grid):
                return [[0 for _ in row] for row in grid]

        with self.assertRaisesRegex(ValueError, 'input_ignoring'):
            verify_oracle('def transform(grid):\n    return [[0 for _ in row] for row in grid]', inputs,
                          sandbox=ShapeOnly())

        class InvalidGrid:
            def run(self, code, grid):
                return [[10]]

        with self.assertRaisesRegex(ValueError, 'grid cells'):
            verify_oracle(IDENTITY, inputs, sandbox=InvalidGrid())

        class NonDeterministic:
            def __init__(self):
                self.calls = 0

            def run(self, code, grid):
                self.calls += 1
                return grid if self.calls % 2 else grid[::-1]

        with self.assertRaisesRegex(ValueError, 'nondeterministic'):
            verify_oracle(IDENTITY, inputs, sandbox=NonDeterministic())

        class TimedOut:
            def run(self, code, grid):
                raise TimeoutError('fixture timeout')

        with self.assertRaisesRegex(TimeoutError, 'fixture'):
            verify_oracle(IDENTITY, inputs, sandbox=TimedOut())

    def test_live_sandbox_never_falls_back_to_host_execution(self):
        with patch('src.arc.sandbox.shutil.which', return_value=None):
            with self.assertRaisesRegex(Exception, 'refusing to run code'):
                DockerSandbox().run(IDENTITY, [[1]])

    def test_e6_training_source_isolated_from_existing_e5_selfplay(self):
        record = _fake_records(load_experiment_config(ROOT / 'configs/e6.fake.yml'), 0)[0]
        clean = {key: value for key, value in record.items() if key != '_verifier_only'}
        with self.assertRaisesRegex(ValueError, 'E5 training rejects'):
            _training_safe_record(clean, 'e5_selfplay')


class E6PipelineTests(unittest.TestCase):
    def test_public_collection_manifest_redacts_dynamic_rejection_details(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_collection(root, [], [{
                'attempt': 0,
                'reason': 'sandbox execution failed: Traceback from runner with private details',
            }], generation=0, execution_mode='live_model')
            manifest = json.loads((root / 'gen-0/collection_manifest.json').read_text())
            private = (root / 'gen-0/rejections.jsonl').read_text()
            self.assertEqual(manifest['rejection_reason_counts'], {'sandbox_execution_failed': 1})
            self.assertNotIn('Traceback', json.dumps(manifest))
            self.assertIn('Traceback', private)

    def test_fake_e2e_gen0_to_gen2_separates_sources_and_fixed_data_control(self):
        config = load_experiment_config(ROOT / 'configs/e6.fake.yml')
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'e6'
            run_e6(config, run_dir=run)
            state = json.loads((run / 'run_state.json').read_text())
            comparison = json.loads((run / 'comparison.json').read_text())
            self.assertEqual(state['generations_completed'], [0, 1, 2])
            self.assertEqual([item['unique_verified_tasks'] for item in state['generations']], [5, 5, 5])
            self.assertEqual(comparison['conditions']['C1-SS']['status'],
                             'skipped_empty_self_solved_dataset')
            self.assertEqual(comparison['conditions']['D']['source'], 'fixed_gen0_data_only')
            self.assertEqual(comparison['conditions']['D']['sample_count'], 5)
            self.assertEqual(comparison['recursive_effect_evaluation_starts_at_generation'], 2)
            self.assertEqual((run / 'gen-0/sft_self_solved.jsonl').read_text(), '')
            self.assertEqual(len((run / 'gen-0/private/heldout_verifier.jsonl').read_text().splitlines()), 5)
            self.assertEqual((run / 'gen-0/private/heldout_verifier.jsonl').stat().st_mode & 0o777, 0o600)
            sft_rows = [json.loads(line) for line in
                        (run / 'gen-0/sft_generator_distillation.jsonl').read_text().splitlines()]
            self.assertEqual(len(sft_rows), 5)
            self.assertTrue(all(row['data_source'] == 'generator_distillation' for row in sft_rows))
            self.assertTrue(all('_verifier_only' not in row for row in sft_rows))
            before = (run / 'artifact_hashes.json').read_bytes()
            run_e6(config, run_dir=run, resume=True)
            self.assertEqual((run / 'artifact_hashes.json').read_bytes(), before)

    def test_fake_threshold_under_five_skips_training(self):
        config = load_experiment_config(ROOT / 'configs/e6.fake.yml')
        config = replace(config, e6=replace(config.e6, min_unique_verified_tasks=6))
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'e6'
            run_e6(config, run_dir=run)
            state = json.loads((run / 'run_state.json').read_text())
            self.assertTrue(all(item['training']['status'] == 'skipped_insufficient_data'
                                for item in state['generations']))
            self.assertFalse(state['trained'])

    def test_fake_trainer_failure_preserves_active_adapter(self):
        trainer = FakeE6Trainer(fail=True)
        result = trainer.train([{'id': 'one'}], steps=3, prior_adapter_hash='prior')
        self.assertEqual(trainer.calls, 1)
        self.assertFalse(result['weights_updated'])
        self.assertEqual(result['active_adapter_hash'], 'prior')
        self.assertTrue(result['training_invoked'])

    def test_e6_profiles_enforce_call_training_and_generation_limits(self):
        fake = load_experiment_config(ROOT / 'configs/e6.fake.yml')
        smoke = load_experiment_config(ROOT / 'configs/e6.local.smoke.yml')
        bootstrap = load_experiment_config(ROOT / 'configs/e6.local.bootstrap.yml')
        self.assertEqual(fake.e6.max_generations, 2)
        self.assertEqual(smoke.e6.max_training_steps, 0)
        self.assertEqual(bootstrap.e6.max_training_steps, 5)
        self.assertEqual(bootstrap.e6.max_generations, 1)

    def test_live_attempt_marker_allows_only_same_run_resume(self):
        config = load_experiment_config(ROOT / 'configs/e6.local.smoke.yml')
        with tempfile.TemporaryDirectory() as temp:
            config = replace(config, tracking=replace(config.tracking, output_dir=temp))
            identity = {'config_sha256': 'a' * 64}
            first = Path(temp) / 'run-one'
            second = Path(temp) / 'run-two'
            _claim_live_attempt(config, first, resume=False, identity=identity)
            _claim_live_attempt(config, first, resume=True, identity=identity)
            with self.assertRaisesRegex(RuntimeError, 'only one'):
                _claim_live_attempt(config, second, resume=False, identity=identity)

    def test_e6_runtime_has_no_operator_search_import(self):
        for path in (ROOT / 'src/arc/e6_curriculum.py', ROOT / 'src/arc/e6_fake.py',
                     ROOT / 'src/pipelines/e6.py', ROOT / 'src/cli/e6.py'):
            tree = ast.parse(path.read_text())
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or '')
            self.assertFalse(any('operator_search' in name for name in imported), path.name)


if __name__ == '__main__':
    unittest.main()
