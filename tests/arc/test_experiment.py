import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from src.arc.agent import AgentResult, FakeLLM, code_agent
from src.arc.data import load_arc
from src.arc.sandbox import FakeSandbox
from src.arc.training import DryRunTrainer, make_record, acceptable_code
from src.core.config import load_experiment_config
from src.pipelines.experiment import run

class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_experiment_config('configs/pilot.cpu.yml')
        self.tasks, self.verifier = load_arc(self.config.data.challenges, self.config.data.solutions)

    def test_training_filter_and_dry_run(self):
        self.assertFalse(acceptable_code('not python !'))
        self.assertFalse(acceptable_code('def transform(grid):\n    return [[0]]'))
        task = self.tasks['toy_000']
        result = code_agent(task, FakeLLM(), FakeSandbox(), 2, 0)
        with self.assertRaises(ValueError): make_record(task, result, self.verifier, set(), 0)
        record = make_record(task, result, self.verifier, {task.id}, 0)
        self.assertTrue(all('output' not in x for x in json.loads(record['messages'][-2]['content'])['test']))
        self.assertIsNone(make_record(task, AgentResult([], None, [], 'failed'), self.verifier, {task.id}, 0))
        with tempfile.TemporaryDirectory() as d:
            meta = DryRunTrainer().train([record], Path(d), 'base', {})
            self.assertFalse(meta['trained']); self.assertFalse(meta['weights_updated'])
            with self.assertRaises(ValueError): DryRunTrainer().train([record, record], Path(d), 'base', {})

    def test_e2e_resume_and_insufficient_data(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = replace(self.config, tracking=replace(self.config.tracking, output_dir=d))
            path = run(cfg)
            gen0 = path / 'generations/gen_000/state.json'
            before = gen0.read_bytes()
            metrics = (path / 'metrics.csv').read_bytes()
            self.assertEqual(path, run(cfg, resume=True, run_dir=path))
            self.assertEqual(before, gen0.read_bytes()); self.assertEqual(metrics, (path / 'metrics.csv').read_bytes())
            self.assertEqual(json.loads(gen0.read_text())['training']['mode'], 'dry_run')
            with self.assertRaises(ValueError): run(replace(cfg, experiment=replace(cfg.experiment, seed=9)), resume=True, run_dir=path)
            small = replace(cfg, rsi=replace(cfg.rsi, min_samples=999))
            p = run(small)
            state = json.loads((p / 'generations/gen_000/state.json').read_text())
            self.assertTrue(state['active_checkpoint'].startswith('base:'))
            self.assertFalse(state['training']['trained'])
            self.assertEqual(state['promotion'], 'not_applicable')
            (path / 'generations/gen_000/replay.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'corrupt'): run(cfg, resume=True, run_dir=path)

    def test_trainer_failure_preserves_previous_checkpoint(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            cfg = replace(self.config, tracking=replace(self.config.tracking, output_dir=d))
            with patch('src.pipelines.experiment.DryRunTrainer.train', side_effect=RuntimeError('training failed')):
                path = run(cfg)
            state = json.loads((path / 'generations/gen_000/state.json').read_text())
            self.assertEqual(state['active_checkpoint'], state['previous_checkpoint'])
            self.assertIn('training failed', state['error'])
            self.assertFalse(state['training']['weights_updated'])

    def test_monitoring_schedule_and_training_isolation(self):
        import csv
        with tempfile.TemporaryDirectory() as d:
            cfg = replace(self.config, tracking=replace(self.config.tracking, output_dir=d),
                experiment=replace(self.config.experiment, generations=2, evaluation_generations=[0, 2], evaluation_split='monitoring'))
            path = run(cfg)
            with (path / 'metrics.csv').open() as metrics_file:
                rows = list(csv.DictReader(metrics_file))
            self.assertEqual({(row['condition'], int(row['generation'])) for row in rows}, {('A',0), ('B',0), ('C',0), ('C',2)})
            self.assertTrue(all(row['dataset_split'] == 'monitoring' for row in rows))
            split = json.loads((path / 'data_split.json').read_text())
            for generation in range(3):
                train = (path / f'generations/gen_{generation:03d}/training.jsonl').read_text().splitlines()
                self.assertTrue(all(json.loads(line)['task_id'] in split['exploration'] for line in train))

    def test_budget_interrupt_and_resume(self):
        import time
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            cfg = replace(self.config, tracking=replace(self.config.tracking, output_dir=d), budget=replace(self.config.budget, max_seconds=0.1))
            original = FakeLLM.generate
            def slow(self, *args, **kwargs):
                time.sleep(0.12)
                return original(self, *args, **kwargs)
            with patch.object(FakeLLM, 'generate', slow):
                path = run(cfg)
            state = json.loads((path / 'run_state.json').read_text())
            self.assertEqual(state['status'], 'budget_exhausted')
            self.assertFalse((path / 'generations/gen_000').exists())
            run(cfg, resume=True, run_dir=path)
            self.assertEqual(json.loads((path / 'run_state.json').read_text())['status'], 'complete')

    def test_no_success_and_same_gen0_candidates(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            cfg = replace(self.config, tracking=replace(self.config.tracking, output_dir=d))
            path = run(cfg)
            predictions = json.loads((path / 'generations/gen_000/predictions.json').read_text())
            b = {p['task_id']: p['candidates'] for p in predictions if p['condition'] == 'B'}
            c = {p['task_id']: p['candidates'] for p in predictions if p['condition'] == 'C'}
            self.assertEqual(b, c)
            with patch.object(FakeLLM, 'generate', return_value='bad code'):
                failed_path = run(cfg)
            state = json.loads((failed_path / 'generations/gen_000/state.json').read_text())
            self.assertEqual(state['new_verified_solutions'], 0)
            self.assertEqual(state['training_status'], 'insufficient_samples')
            self.assertEqual(state['active_checkpoint'], state['previous_checkpoint'])
            self.assertEqual((failed_path / 'generations/gen_000/training.jsonl').read_text(), '')
