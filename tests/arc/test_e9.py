from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.arc.e9_config import MODELS, load_e9_config
from src.pipelines.e9 import (_audit_sft, _e8_existing, _load_parent_pool, run,
                              run_model)
from src.pipelines.e6g import _seal_g, _verify_g
from src.cli.e9_audit import audit


class E9Tests(unittest.TestCase):
    def setUp(self):
        Path('artifacts/e9').mkdir(parents=True, exist_ok=True)

    def test_independent_fake_e2e_and_training_gate(self):
        settings = load_e9_config('configs/e9.fake.yml')
        with tempfile.TemporaryDirectory(dir='artifacts/e9') as tmp:
            roots = run(settings, Path(tmp) / 'fake')
            self.assertEqual(len(roots), 2)
            for model, root in zip(('14b', '32b'), roots):
                _verify_g(root)
                result = json.loads((root / 'result.json').read_text())
                protocol = json.loads((root / 'protocol.lock.yml').read_text())
                gate = json.loads((root / 'training-gate.json').read_text())
                self.assertEqual(result['model_key'], model)
                self.assertEqual(protocol['model_key'], model)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['completed_slots'], 2)
                self.assertEqual(result['verified_new_oracles'], 0)
                self.assertEqual(gate['strict_total'], 0)
                self.assertFalse(gate['trained'])
                self.assertFalse(gate['weights_updated'])
                self.assertEqual(gate['reason'], 'insufficient_strict_oracles')
                self.assertFalse(result['operator_search_used'])
                self.assertFalse(result['public_evaluation_run'])
                self.assertTrue((root / 'sft-audit.json').is_file())
                self.assertTrue((root / 'comparison.json').is_file())
            report = audit(roots[0], Path(tmp) / 'audit-14b')
            self.assertEqual(report['audit_status'], 'passed')
            self.assertTrue(report['checks']['protocol_and_tree_seals'])
            self.assertNotEqual(roots[0] / 'initial_pool.json', roots[1] / 'initial_pool.json')

    def test_fake_run_resume_never_repeats_inference(self):
        settings = load_e9_config('configs/e9.fake.yml')
        single = replace(settings, models=['14b'])
        with tempfile.TemporaryDirectory(dir='artifacts/e9') as tmp:
            root = Path(tmp) / '14b'
            run_model(single, '14b', root)
            with patch('src.arc.e6g_fake.FakeBackend.generate', side_effect=AssertionError('replayed')):
                self.assertEqual(run_model(single, '14b', root, resume=True), root.resolve())
            with self.assertRaisesRegex(ValueError, 'changed on resume'):
                run_model(replace(single, seed=44), '14b', root, resume=True)

    def test_extra_batch_requires_approval_and_parent(self):
        settings = load_e9_config('configs/e9.14b.extra30.yml')
        self.assertEqual((settings.slot_offset, settings.slots, settings.wall_seconds), (10, 30, 1200))
        with self.assertRaisesRegex(ValueError, 'explicit approval'):
            run_model(settings, '14b', Path('artifacts/e9/14b/unauthorized'), live=True)

    def test_parent_data_from_other_model_is_rejected(self):
        settings = load_e9_config('configs/e9.fake.yml')
        expected = {'model_identity': list(MODELS['14b']), 'sandbox_image_id': 'fake',
                    'initial_pool_sha256': 'pool', 'e9_source_hashes': {'src': 'hash'}}
        with tempfile.TemporaryDirectory(dir='artifacts/e9') as tmp:
            parent = Path(tmp) / 'parent'
            parent.mkdir()
            protocol = {'model_key': '14b', 'model_identity': expected['model_identity'],
                        'sandbox_image_id': 'fake', 'initial_pool_sha256': 'pool',
                        'e9_source_hashes': {'src': 'hash'}}
            (parent / 'protocol.lock.yml').write_text(json.dumps(protocol))
            (parent / 'strict_oracles.jsonl').write_text(json.dumps({
                'source_model_id': MODELS['32b'][0], 'model_revision': MODELS['32b'][1],
                'task_id': 'cross-model', 'behavior_fingerprint': 'fp',
                'record': {'task_id': 'cross-model'}}) + '\n')
            _seal_g(parent)
            with self.assertRaisesRegex(ValueError, 'cross-model'):
                _load_parent_pool(parent, '14b', expected)

    def test_sft_audit_rejects_mixed_model_hidden_labels_and_duplicates(self):
        settings = load_e9_config('configs/e9.14b.yml')
        model, revision, _ = MODELS['14b']
        checker = 'f' * 64
        manifest = {'chat_template_sha256': 'template'}
        base = {'source_model_id': model, 'model_revision': revision, 'generation': 0,
                'oracle_source': 'generator_distillation', 'task_id': 'task-1',
                'canonical_ast_hash': 'ast-1', 'behavior_fingerprint': 'behavior-1',
                'generator_prompt_hash': 'prompt', 'verifier_revision': checker,
                'static_checker_hash': checker, 'docker_image_digest': 'image', 'seal': 'seal',
                'record': {'task_id': 'task-1', 'base_revision': revision,
                           'adapter_revision': 'none', 'qwen_provenance': {
                               'model_id': model, 'model_revision': revision},
                           'messages': [{'role': 'system', 'content': 'sys'},
                                        {'role': 'user', 'content': 'question'},
                                        {'role': 'assistant', 'content': 'def f(grid): return grid'}]}}
        with patch('src.pipelines.e9._sha', return_value=checker):
            valid, audit = _audit_sft([base], '14b', 'image', manifest)
            self.assertEqual(len(valid), 1)
            self.assertTrue(valid[0]['mask_prompt'])
            self.assertEqual(valid[0]['loss_target'], 'assistant_completion_only')
            mixed = dict(base, source_model_id=MODELS['32b'][0])
            hidden_record = dict(base['record'], _verifier_only={'output': 'must not train'})
            hidden = dict(base, task_id='task-2', record=hidden_record,
                          canonical_ast_hash='ast-2', behavior_fingerprint='behavior-2')
            duplicate = dict(base, task_id='task-3')
            output, rejected = _audit_sft([mixed, hidden, base, duplicate], '14b', 'image', manifest)
        self.assertEqual(len(output), 1)
        self.assertEqual(rejected['rejected_records'], 3)
        self.assertIn('hidden_verifier_data_in_training_record', rejected['rejection_reasons'])

    def test_e8_existing_sources_are_reaudited_and_e7_is_excluded(self):
        protocol = json.loads(Path('artifacts/e8/full-rerun-preregistered-20261102/protocol.lock.yml').read_text())
        try:
            existing, pool, audit = _e8_existing('14b', False, protocol['sandbox_image_id'])
        except (FileNotFoundError, ValueError) as exc:
            self.skipTest(f'local sealed E8 source audit unavailable: {exc}')
        self.assertEqual(len(existing), 1)
        self.assertEqual(len(pool), 1)
        self.assertEqual(audit['eligible'], 1)
        self.assertEqual(audit['rejected'], 2)

    def test_operator_search_is_not_imported_by_e9(self):
        import subprocess
        import sys
        subprocess.run([sys.executable, '-c', 'import sys; import src.pipelines.e9; '
                        'assert "src.arc.operator_search" not in sys.modules'], check=True)


if __name__ == '__main__':
    unittest.main()
