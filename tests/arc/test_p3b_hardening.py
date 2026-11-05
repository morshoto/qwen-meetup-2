from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.arc.data import load_arc, split_tasks
from src.arc.mlx_chat_template import install_training_template_policy
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.task_manifests import load_diagnostic_task_ids
from src.cli.p3b_sft_audit import _validate_row
from src.core.config import P3CSettings, load_experiment_config
from src.pipelines.p3a import run_real
from src.arc.mlx_token_driver import adapter_sha256
from src.pipelines.p3b_ablation import run_ablation


ROOT = Path(__file__).resolve().parents[2]


class ChatTemplatePolicyTests(unittest.TestCase):
    def test_off_mode_is_passed_to_mlx_chat_template_calls(self):
        class FakeTokenizerWrapper:
            chat_template = 'enable_thinking toggle'

            def apply_chat_template(self, messages, **kwargs):
                return kwargs

        install_training_template_policy('off', FakeTokenizerWrapper)
        self.assertIs(FakeTokenizerWrapper().apply_chat_template([])['enable_thinking'], False)

    def test_off_mode_rejects_templates_without_the_qwen_switch(self):
        class FakeTokenizerWrapper:
            chat_template = 'simple template'

            def apply_chat_template(self, messages, **kwargs):
                return kwargs

        install_training_template_policy('off', FakeTokenizerWrapper)
        with self.assertRaisesRegex(ValueError, 'does not support'):
            FakeTokenizerWrapper().apply_chat_template([])


class SFTRowAuditTests(unittest.TestCase):
    def _row(self):
        prompt = json.dumps({'train': [{'input': [[0]], 'output': [[1]]}],
                             'test': [{'input': [[0]]}]})
        return {'task_id': 'exploration-task', 'split': 'exploration',
                'messages': [
                    {'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': 'def transform(grid):\n    return [[1]]'},
                ]}

    def test_accepts_completion_only_training_target_without_test_labels(self):
        task_id, completion, _ = _validate_row(self._row(), 0, 'exploration')
        self.assertEqual(task_id, 'exploration-task')
        self.assertEqual(completion, 'def transform(grid):\n    return [[1]]')

    def test_rejects_test_output_in_user_message(self):
        row = self._row()
        user = json.loads(row['messages'][1]['content'])
        user['test'][0]['output'] = [[1]]
        row['messages'][1]['content'] = json.dumps(user)
        with self.assertRaisesRegex(ValueError, 'test solution fields'):
            _validate_row(row, 0, 'exploration')

    def test_rejects_invalid_completion_and_non_exploration_split(self):
        row = self._row()
        row['messages'][-1]['content'] = 'Thoughts only, no transform.'
        with self.assertRaisesRegex(ValueError, 'transform function'):
            _validate_row(row, 0, 'exploration')
        row = self._row()
        row['split'] = 'promotion_gate'
        with self.assertRaisesRegex(ValueError, 'non-Exploration'):
            _validate_row(row, 0, 'exploration')


class P3CFixedBatchTests(unittest.TestCase):
    def test_checked_in_batch_is_fixed_unused_and_diagnostic_free(self):
        config = load_experiment_config(ROOT / 'configs/p3c.collect.qwen8b.offset285.yml')
        diagnostics, _ = load_diagnostic_task_ids(config.p3c.diagnostic_task_manifest)
        with (ROOT / config.data.challenges).open(encoding='utf-8') as stream:
            ids = list(json.load(stream))
        splits = split_tasks(ids, config.data.split_seed,
                             {'exploration': config.data.exploration,
                              'promotion_gate': config.data.promotion_gate,
                              'monitoring': config.data.monitoring})
        self.assertEqual(config.p3c.task_ids, splits['exploration'][285:305])
        self.assertTrue(set(diagnostics) <= set(splits['exploration']))
        self.assertFalse(set(config.p3c.task_ids) & set(diagnostics))
        self.assertEqual(config.p3c.max_code_attempts, config.agent.max_repairs + 1)
        self.assertEqual(config.p3b.insufficient_data.min_verified_unique_tasks, 5)

    def test_inference_rejects_diagnostic_overlap_before_creating_a_run(self):
        base = load_experiment_config(ROOT / 'configs/p3c.collect.qwen8b.offset285.yml')
        challenge = ROOT / 'tests/fixtures/arc_challenges.json'
        solutions = ROOT / 'tests/fixtures/arc_solutions.json'
        tasks, _ = load_arc(challenge, solutions)
        splits = split_tasks(list(tasks), base.data.split_seed,
                             {'exploration': 8, 'promotion_gate': 0, 'monitoring': 0})
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / 'diagnostic.json'
            manifest_path.write_text(json.dumps({
                'protocol': 'p3b_diagnostic_task_exclusion_v1', 'version': 1,
                'source_artifact': 'historical-test', 'source_protocol': 'p3b_unused_exploration_holdout_v1',
                'task_ids': [splits['exploration'][0]],
            }))
            data = replace(base.data, challenges=str(challenge), solutions=str(solutions),
                           exploration=8, promotion_gate=0, monitoring=0)
            model = replace(base.model, backend='fake')
            rsi = replace(base.rsi, exploration_per_generation=2, exploration_offset=0, trainer='dry_run')
            p3c = P3CSettings(enabled=True, task_ids=splits['exploration'][:2],
                              diagnostic_task_manifest=str(manifest_path), max_code_attempts=base.agent.max_repairs + 1)
            config = replace(base, data=data, model=model, rsi=rsi, p3c=p3c)
            with self.assertRaisesRegex(ValueError, 'previously used diagnostic task'):
                run_real(config)


class P3BAblationTests(unittest.TestCase):
    def test_all_fixed_candidates_share_one_record_and_never_promote(self):
        config = load_experiment_config(ROOT / 'configs/p3b.hardening.qlora.qwen8b.yml')
        record = {'id': 'record-1', 'task_id': 'task-1', 'split': 'exploration',
                  'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                               {'role': 'user', 'content': '{"test":[{"input":[[0]]}]}'}]}
        manifest = {'protocol': 'p3b_verified_exploration_v1', 'verified_unique_task_count': 1,
                    'verified_record_ids': ['record-1'], 'source_identity_sha256': 'a' * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_adapter = root / 'old-adapter'
            old_adapter.mkdir()
            (old_adapter / 'adapter_config.json').write_text('{}')
            (old_adapter / 'adapters.safetensors').write_bytes(b'old')
            ablation = replace(config.p3b.ablation, baseline_adapter_path=str(old_adapter))
            config = replace(config, p3b=replace(config.p3b, ablation=ablation))

            def train_candidate(candidate_config, records, source_manifest, output, *, source,
                                previous_adapter=None):
                adapter = Path(output) / 'adapter'
                adapter.mkdir(parents=True)
                (adapter / 'adapter_config.json').write_text('{}')
                (adapter / 'adapters.safetensors').write_bytes(str(candidate_config.p3b.training.max_steps).encode())
                return {'status': 'complete', 'trained': True, 'weights_updated': True,
                        'adapter_path': str(adapter), 'adapter_sha256': adapter_sha256(adapter),
                        'peak_rss_bytes': 100}

            canary_calls = []

            def fake_canary(config, *, adapter, expected_sha, output, remaining_seconds):
                canary_calls.append((adapter, expected_sha, remaining_seconds))
                return {'adapter_sha256': expected_sha, 'response_valid_rate': 1.0,
                        'thinking_format_error_count': 0, 'elapsed_seconds': 0.1,
                        'peak_memory_bytes': 200}, None

            with patch('src.pipelines.p3b_ablation.collect_verified_exploration_records',
                       return_value=([record], manifest)), \
                    patch('src.pipelines.p3b_ablation.train_p3b', side_effect=train_candidate), \
                    patch('src.pipelines.p3b_ablation._canary', side_effect=fake_canary):
                report = run_ablation(config, root / 'source', root / 'ablation')

            self.assertEqual(len(canary_calls), 5)
            self.assertFalse(report['promotion_enabled'])
            self.assertEqual(report['promoted_candidates'], [])
            self.assertEqual(len(report['results']), 5)
            with (root / 'ablation/ablation_results.csv').open(newline='') as stream:
                rows = list(__import__('csv').DictReader(stream))
            self.assertEqual(len({row['dataset_sha256'] for row in rows}), 1)
            self.assertTrue(all(row['promoted'] == 'False' for row in rows))
            hashes = json.loads((root / 'ablation/artifact_hashes.json').read_text())
            for relative, expected in hashes.items():
                self.assertEqual(hashlib.sha256((root / 'ablation' / relative).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
