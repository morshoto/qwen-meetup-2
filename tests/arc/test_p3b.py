import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.arc.data import exploration_window, load_arc, split_tasks
from src.arc.mlx_lora import _training_safe_record
from src.core.config import load_experiment_config
from src.pipelines.p3b import collect_verified_exploration_records, synthetic_pilot_records, train_p3b
from src.pipelines.p3b_compare import _normalize, compare_adapter, decide_promotion


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class P3BTests(unittest.TestCase):
    def test_p3b_config_parses_without_changing_legacy_profiles(self):
        legacy = load_experiment_config(ROOT / 'configs/p3a.local.mlx.yml')
        self.assertEqual(legacy.p3b.training.backend, 'none')
        config = load_experiment_config(ROOT / 'configs/p3b.local.qlora.yml')
        self.assertEqual(config.p3b.training.backend, 'mlx_lora')
        self.assertEqual(config.p3b.training.method, 'qlora_sft')
        self.assertEqual(config.p3b.insufficient_data.action, 'skip_training')
        self.assertEqual(config.p3b.insufficient_data.min_verified_unique_tasks, 5)
        compare = load_experiment_config(ROOT / 'configs/p3b.local.compare.yml')
        self.assertEqual(compare.experiment.conditions, ['B', 'C'])
        b = dataclasses.replace(compare, experiment=dataclasses.replace(compare.experiment, conditions=['B']))
        c = dataclasses.replace(compare, experiment=dataclasses.replace(compare.experiment, conditions=['C']),
                                model=dataclasses.replace(compare.model, adapter_sha256='a' * 64))
        self.assertEqual(_normalize(b), _normalize(c))

    def test_only_verified_exploration_records_are_admitted(self):
        record = {'id': 'id', 'task_id': 'toy', 'split': 'exploration', 'messages': [
            {'role': 'system', 'content': 'Follow the requested output format exactly. Return only the final answer; do not explain.'},
            {'role': 'user', 'content': json.dumps({'train': [], 'test': [{'input': [[0]]}]})},
            {'role': 'assistant', 'content': 'def transform(grid):\n    return grid'}]}
        record['id'] = hashlib.sha256((record['messages'][1]['content'] + '\0' + record['messages'][2]['content']).encode()).hexdigest()
        _training_safe_record(record, 'exploration')
        record['messages'][1]['content'] = json.dumps({'train': [], 'test': [{'input': [[0]], 'output': [[1]]}]})
        with self.assertRaisesRegex(ValueError, 'held-out test outputs'):
            _training_safe_record(record, 'exploration')

    def test_synthetic_pilot_records_are_explicitly_separate(self):
        records = synthetic_pilot_records()
        self.assertEqual(len(records), 8)
        for record in records:
            _training_safe_record(record, 'synthetic_pilot')
        with self.assertRaisesRegex(ValueError, 'Exploration records'):
            _training_safe_record(records[0], 'exploration')

    def test_collector_rebuilds_only_successes_from_a_complete_b_run(self):
        base = load_experiment_config(ROOT / 'configs/p3a.local.mlx.yml')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenges = ROOT / 'tests/fixtures/arc_challenges.json'
            solutions = ROOT / 'tests/fixtures/arc_solutions.json'
            data = dataclasses.replace(base.data, challenges=str(challenges), solutions=str(solutions),
                                       exploration=8, promotion_gate=0, monitoring=0)
            experiment = dataclasses.replace(base.experiment, conditions=['B'])
            rsi = dataclasses.replace(base.rsi, exploration_per_generation=1, exploration_offset=1)
            config = dataclasses.replace(base, data=data, experiment=experiment, rsi=rsi)
            tasks, verifier = load_arc(challenges, solutions)
            splits = split_tasks(list(tasks), data.split_seed,
                                 {'exploration': 8, 'promotion_gate': 0, 'monitoring': 0})
            task_id = splits['exploration'][1]
            solution_data = json.loads(solutions.read_text())
            result = {'outputs': solution_data[task_id], 'code': 'def transform(grid):\n    return grid',
                      'history': [], 'error': None, 'error_kind': None, 'raw_response': None}
            self.assertTrue(verifier.matches(task_id, result['outputs']))
            snapshot = root / 'config.snapshot.yml'
            snapshot.write_text(yaml.safe_dump(dataclasses.asdict(config), sort_keys=False))
            identity = {'config': dataclasses.asdict(config),
                        'data_sha256': {'challenges': sha(challenges), 'solutions': sha(solutions)},
                        'splits': splits}
            (root / 'identity.json').write_text(json.dumps(identity))
            (root / 'data_split.json').write_text(json.dumps(splits))
            (root / 'run_state.json').write_text(json.dumps({'status': 'complete', 'trained': False, 'weights_updated': False}))
            (root / 'environment.json').write_text(json.dumps({'sandbox_image_id': 'sha256:test-runner'}))
            rows = [{'condition': 'B', 'task_id': task_id,
                     'candidates': [{'result': result, 'error': None},
                                    {'result': result, 'error': None}]}]
            (root / 'results.json').write_text(json.dumps(rows))
            (root / 'metrics.csv').write_text('condition,pass_at_1\nB,1\n')
            (root / 'budget.json').write_text(json.dumps({'aggregate_hashes': {'results.json': sha(root / 'results.json')}}))
            class Sandbox:
                def __init__(self, *args, **kwargs):
                    pass
                def run(self, code, grid):
                    return [row[:] for row in grid]
            with patch('src.pipelines.p3b.load_experiment_config', return_value=config), \
                    patch('src.pipelines.p3b.shutil.which', return_value='/usr/bin/docker'), \
                    patch('src.pipelines.p3b.subprocess.check_output', return_value='sha256:test-runner\n'), \
                    patch('src.pipelines.p3b.DockerSandbox', Sandbox):
                records, manifest = collect_verified_exploration_records(root, required_tasks=1)
            self.assertEqual(len(records), 1)
            self.assertEqual(manifest['verified_record_count'], 1)
            self.assertEqual(manifest['verified_candidate_success_count'], 2)
            self.assertEqual(manifest['verified_unique_task_count'], 1)
            self.assertFalse(manifest['test_outputs_in_messages'])
            self.assertEqual(manifest['code_verification_mode'], 'docker_reexecution')
            user = json.loads(records[0]['messages'][-2]['content'])
            self.assertTrue(all(set(item) == {'input'} for item in user['test']))

    def test_holdout_excludes_noninitial_training_window(self):
        from src.pipelines.p3b_compare import _holdout_data
        base = load_experiment_config(ROOT / 'configs/p3b.local.compare.yml')
        challenges = ROOT / 'tests/fixtures/arc_challenges.json'
        solutions = ROOT / 'tests/fixtures/arc_solutions.json'
        data = dataclasses.replace(base.data, challenges=str(challenges), solutions=str(solutions),
                                   exploration=8, promotion_gate=0, monitoring=0)
        p3b = dataclasses.replace(base.p3b,
            training=dataclasses.replace(base.p3b.training, collection_tasks=2),
            promotion=dataclasses.replace(base.p3b.promotion, minimum_tasks=2))
        rsi = dataclasses.replace(base.rsi, exploration_offset=2, exploration_per_generation=2)
        config = dataclasses.replace(base, data=data, p3b=p3b, rsi=rsi)
        splits = split_tasks(list(json.loads(challenges.read_text())), data.split_seed,
                             {'exploration': 8, 'promotion_gate': 0, 'monitoring': 0})
        trained_ids = set(exploration_window(splits, 2, 2))
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _ = _holdout_data(config, Path(directory), trained_ids)
        self.assertEqual(manifest['task_ids'], splits['exploration'][4:6])
        self.assertTrue(set(manifest['task_ids']).isdisjoint(trained_ids))

    def test_holdout_excludes_diagnostic_ids_even_without_training_overlap(self):
        from src.pipelines.p3b_compare import _holdout_data
        base = load_experiment_config(ROOT / 'configs/p3b.local.compare.yml')
        challenges = ROOT / 'tests/fixtures/arc_challenges.json'
        solutions = ROOT / 'tests/fixtures/arc_solutions.json'
        data = dataclasses.replace(base.data, challenges=str(challenges), solutions=str(solutions),
                                   exploration=8, promotion_gate=0, monitoring=0)
        splits = split_tasks(list(json.loads(challenges.read_text())), data.split_seed,
                             {'exploration': 8, 'promotion_gate': 0, 'monitoring': 0})
        with tempfile.TemporaryDirectory() as directory:
            diagnostic_manifest = Path(directory) / 'diagnostic.json'
            diagnostic_ids = splits['exploration'][2:4]
            diagnostic_manifest.write_text(json.dumps({
                'protocol': 'p3b_diagnostic_task_exclusion_v1', 'version': 1,
                'source_artifact': 'historical-test', 'source_protocol': 'p3b_unused_exploration_holdout_v1',
                'task_ids': diagnostic_ids,
            }))
            promotion = dataclasses.replace(base.p3b.promotion, minimum_tasks=2,
                                            diagnostic_task_manifest=str(diagnostic_manifest),
                                            exclude_diagnostic_tasks=True)
            config = dataclasses.replace(base, data=data,
                                         p3b=dataclasses.replace(base.p3b, promotion=promotion),
                                         rsi=dataclasses.replace(base.rsi, exploration_offset=2))
            comparison_root = Path(directory) / 'comparison'
            comparison_root.mkdir()
            manifest, _, _ = _holdout_data(config, comparison_root, set())
        self.assertEqual(manifest['task_ids'], splits['exploration'][4:6])
        self.assertEqual(manifest['diagnostic_task_ids_excluded'], diagnostic_ids)

    def test_single_success_does_not_pass_the_five_unique_task_training_gate(self):
        config = load_experiment_config(ROOT / 'configs/p3b.local.qlora.yml')
        record = {'id': 'verified-record-1', 'task_id': 'unique-task-1'}
        manifest = {'protocol': 'p3b_verified_exploration_v1', 'test_outputs_in_messages': False,
                    'code_verification_mode': 'docker_reexecution', 'verified_record_ids': [record['id']]}
        with tempfile.TemporaryDirectory() as directory:
            result = train_p3b(config, [record], manifest, Path(directory) / 'single-record',
                               source='exploration')
        self.assertEqual(result['status'], 'skipped_insufficient_verified_tasks')
        self.assertEqual(result['verified_unique_task_count'], 1)
        self.assertEqual(result['minimum_verified_unique_tasks'], 5)
        self.assertTrue(result['previous_adapter_preserved'])

    def test_insufficient_real_data_writes_skip_without_importing_mlx(self):
        config = load_experiment_config(ROOT / 'configs/p3b.local.qlora.yml')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'skip-run'
            result = train_p3b(config, [], {'protocol': 'p3b_verified_exploration_v1',
                                            'test_outputs_in_messages': False,
                                            'code_verification_mode': 'docker_reexecution',
                                            'verified_record_ids': []}, output, source='exploration')
            self.assertEqual(result['status'], 'skipped_insufficient_verified_tasks')
            self.assertFalse(result['trained'])
            self.assertFalse(result['weights_updated'])
            self.assertTrue(result['previous_adapter_preserved'])
            self.assertFalse((output / 'training-run').exists())

    def test_comparison_skips_if_verified_exploration_training_was_skipped(self):
        training_config = load_experiment_config(ROOT / 'configs/p3b.local.qlora.yml')
        compare_config = load_experiment_config(ROOT / 'configs/p3b.local.compare.yml')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_dir = root / 'training'
            manifest = {'protocol': 'p3b_verified_exploration_v1', 'test_outputs_in_messages': False,
                        'code_verification_mode': 'docker_reexecution', 'verified_record_ids': []}
            trained = train_p3b(training_config, [], manifest, train_dir, source='exploration')
            self.assertEqual(trained['status'], 'skipped_insufficient_verified_tasks')
            result = compare_adapter(compare_config, train_dir, root / 'comparison')
            self.assertEqual(result['status'], 'skipped_no_verified_adapter')
            self.assertEqual(result['promotion'], 'keep_previous')
            self.assertFalse(result['weights_updated'])
            self.assertTrue((root / 'comparison/artifact_hashes.json').is_file())

    def test_promotion_gate_promotes_nonregression_and_rolls_back_errors(self):
        baseline = {'B': {'scored_tasks': '5', 'errors': '0', 'pass_at_1': '0.4'},
                    'C': {'scored_tasks': '5', 'errors': '0', 'pass_at_1': '0.4'}}
        self.assertEqual(decide_promotion(baseline, minimum_tasks=5, enabled=True,
                                          rollback_on_regression=True)['promotion'], 'promote_candidate')
        regression = {**baseline, 'C': {'scored_tasks': '5', 'errors': '0', 'pass_at_1': '0.2'}}
        self.assertEqual(decide_promotion(regression, minimum_tasks=5, enabled=True,
                                          rollback_on_regression=True)['promotion'], 'rollback')
        incomplete = {**baseline, 'C': {'scored_tasks': '4', 'errors': '1', 'pass_at_1': '0.0'}}
        self.assertEqual(decide_promotion(incomplete, minimum_tasks=5, enabled=True,
                                          rollback_on_regression=True)['promotion'], 'rollback')


if __name__ == '__main__':
    unittest.main()
