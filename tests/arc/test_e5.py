from __future__ import annotations

import dataclasses
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.e5_plugins import FakePluginEvaluator, evaluate_and_store, validate_plugin_source
from src.arc.e5_selfplay import (generate_synthetic_task, make_sft_record,
                                 solve_prompt, solve_task)
from src.arc.mlx_lora import _training_safe_record
from src.core.config import load_experiment_config
from src.pipelines.e5 import (E5Budget, E5CachedBackend, FakeE5Backend,
                              _bounded_window, _replay_records, _train_generation,
                              _write_records, run_e5)
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[2]


class IncrementFixture:
    generator = (
        'def transform(grid):\n'
        '    return [[(cell + 1) % 10 for cell in row] for row in grid]'
    )
    solver = (
        'def transform(grid):\n'
        '    result = []\n'
        '    for row in grid:\n'
        '        output_row = []\n'
        '        for cell in row:\n'
        '            output_row.append((cell + 1) % 10)\n'
        '        result.append(output_row)\n'
        '    return result'
    )

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *, kind, candidate):
        self.prompts.append((kind, prompt))
        return self.generator if kind == 'e5_generator' else self.solver


class IncrementSandbox:
    def run(self, code, grid):
        if code not in {IncrementFixture.generator, IncrementFixture.solver}:
            raise ValueError('unexpected test fixture code')
        return [[(cell + 1) % 10 for cell in row] for row in grid]


class E5SelfPlayTests(unittest.TestCase):
    def test_generator_retry_gets_only_sanitized_failure_feedback(self):
        rejected = 'def transform(grid):\n    return grid[999][999]'
        repaired = IncrementFixture.generator

        class RetryBackend:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, *, kind, candidate):
                self.prompts.append((kind, candidate, prompt))
                return rejected if candidate == 0 else repaired

        class RetrySandbox:
            def run(self, code, grid):
                if code == rejected:
                    raise RuntimeError('private expected output must not reach the model')
                return [[(cell + 1) % 10 for cell in row] for row in grid]

        backend = RetryBackend()
        generated = generate_synthetic_task(
            backend, RetrySandbox(), seed=29, generation=0, index=0,
            generator_candidates=2, train_examples=2,
            hidden_test_inputs=1, validation_inputs=1)
        self.assertIsNotNone(generated)
        self.assertEqual(generated.generator_code, repaired)
        self.assertIn(rejected, backend.prompts[1][2])
        self.assertIn('sandbox_execution_RuntimeError', backend.prompts[1][2])
        self.assertIn('failing generated input shape', backend.prompts[1][2])
        self.assertNotIn('private expected output', backend.prompts[1][2])

    def test_solver_retry_feedback_does_not_repeat_training_grids(self):
        from src.arc.data import Task
        identity = 'def transform(grid):\n    return [row[:] for row in grid]'
        increment = IncrementFixture.solver

        class RetryBackend:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, *, kind, candidate):
                self.prompts.append(prompt)
                return identity if candidate == 0 else increment

        class RetrySandbox:
            def run(self, code, grid):
                if code == identity:
                    return [row[:] for row in grid]
                return [[(cell + 1) % 10 for cell in row] for row in grid]

        task = Task('repair', (([[1, 2]], [[2, 3]]),), ([[3, 4]],))
        backend = RetryBackend()
        result = solve_task(backend, RetrySandbox(), task, attempts=2)
        self.assertIsNone(result.error)
        feedback = backend.prompts[1].split(
            'Previous candidate code and training-only feedback:', 1)[1]
        self.assertIn("'pair': 0", feedback)
        self.assertIn('differed from the visible training target', feedback)
        self.assertNotIn('actual', feedback)
        self.assertNotIn('expected', feedback)

    def test_generator_is_hidden_and_only_verified_self_solution_becomes_sft(self):
        backend = IncrementFixture()
        sandbox = IncrementSandbox()
        generated = generate_synthetic_task(
            backend, sandbox, seed=29, generation=0, index=0,
            generator_candidates=1, train_examples=3,
            hidden_test_inputs=2, validation_inputs=2)
        self.assertIsNotNone(generated)
        assert generated is not None
        solve_input = solve_prompt(generated.task)
        self.assertNotIn(generated.generator_code, solve_input)
        for output in generated.hidden_outputs:
            self.assertNotIn(json.dumps(output, separators=(',', ':')), solve_input)
        self_solved = solve_task(
            backend, sandbox, generated.task, attempts=1,
            generator_ast_sha256=generated.generator_ast_sha256,
            hidden_outputs=generated.hidden_outputs,
            validation_inputs=generated.validation_inputs,
            validation_outputs=generated.validation_outputs)
        self.assertIsNone(self_solved.error)
        record = make_sft_record(generated.task, self_solved,
                                 data_source='self_solved_synthetic', generation=0,
                                 generator_sha256=generated.generator_sha256)
        self.assertIsNotNone(record)
        assert record is not None
        user = json.loads(record['messages'][1]['content'])
        self.assertTrue(all(set(item) == {'input'} for item in user['test']))
        _training_safe_record(record, 'e5_selfplay')
        distillation = dict(record, data_source='generator_distillation')
        with self.assertRaisesRegex(ValueError, 'generator_distillation'):
            _training_safe_record(distillation, 'e5_selfplay')

    def test_obvious_grid_memorization_and_unsafe_code_are_rejected(self):
        from src.arc.e5_selfplay import _safe_transform_code
        self.assertFalse(_safe_transform_code(
            'def transform(grid):\n    if grid == [[0]]:\n        return [[1]]\n    return grid'))
        self.assertFalse(_safe_transform_code(
            'import os\ndef transform(grid):\n    return grid'))
        self.assertTrue(_safe_transform_code(IncrementFixture.solver))

    def test_plugin_interface_writes_only_to_proposal_allowlist(self):
        source = 'def search_strategy(task_context):\n    return ""'
        self.assertEqual(validate_plugin_source(source)[0], True)
        self.assertFalse(validate_plugin_source('import os\ndef search_strategy(task_context):\n    return ""')[0])
        self.assertFalse(validate_plugin_source(
            'def search_strategy(task_context):\n    return __builtins__["eval"](task_context)')[0])
        with tempfile.TemporaryDirectory() as temp:
            decision = evaluate_and_store(temp, 'proposal_0', source, FakePluginEvaluator())
            self.assertTrue(decision.accepted)
            self.assertEqual(decision.sandbox_status, 'fake')
            self.assertTrue(Path(decision.path).is_relative_to((Path(temp) / 'plugins' / 'proposals').resolve()))
            with self.assertRaisesRegex(ValueError, 'invalid'):
                evaluate_and_store(temp, '../escape', source, FakePluginEvaluator())


class E5PipelineTests(unittest.TestCase):
    def test_exploration_and_promotion_windows_are_disjoint(self):
        task_ids = [f'task-{index}' for index in range(12)]
        exploration = [_bounded_window(task_ids, generation * 2, 2) for generation in range(3)]
        promotion = [_bounded_window(task_ids, generation * 2, 2) for generation in range(2)]
        self.assertEqual(len({item for window in exploration for item in window}), 6)
        self.assertEqual(len({item for window in promotion for item in window}), 4)
        self.assertEqual(_bounded_window(task_ids, len(task_ids), 0), [])
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            _bounded_window(task_ids, 11, 2)

    def test_profiles_load_and_live_requires_explicit_flag(self):
        for name in ('e5.cpu.fake.yml', 'e5.live.smoke.yml', 'e5.local.3gen.yml'):
            config = load_experiment_config(ROOT / 'configs' / name)
            self.assertTrue(config.e5.enabled)
        live = load_experiment_config(ROOT / 'configs/e5.live.smoke.yml')
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, '--live'):
            run_e5(live, run_dir=Path(temp) / 'run', live=False)

    def test_cpu_fake_runs_three_generations_and_resume_checks_hashes(self):
        config = load_experiment_config(ROOT / 'configs/e5.cpu.fake.yml')
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            run_e5(config, run_dir=run)
            state = json.loads((run / 'run_state.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['generations_completed'], [0, 1, 2])
            self.assertFalse(state['trained'])
            self.assertFalse(state['weights_updated'])
            identity = json.loads((run / 'identity.json').read_text())
            self.assertEqual(identity['execution_mode'], 'fake_fixture')
            self.assertFalse(identity['e4_operator_search_used'])
            self.assertFalse(identity['public_evaluation_accessed'])
            for source_path in (ROOT / 'src/pipelines/e5.py', ROOT / 'src/arc/e5_selfplay.py'):
                tree = ast.parse(source_path.read_text())
                imports = [alias.name for node in ast.walk(tree)
                           if isinstance(node, ast.Import) for alias in node.names]
                imports.extend(node.module for node in ast.walk(tree)
                               if isinstance(node, ast.ImportFrom))
                self.assertFalse(any(module and module.endswith('operator_search') for module in imports))
            self.assertEqual(state['budget']['model_calls'], 24)
            self.assertLessEqual(state['budget']['reserved_tokens'], config.e5.max_reserved_tokens)
            self.assertLessEqual(state['budget']['sandbox_runs'], config.e5.max_sandbox_runs)
            real_task_ids = []
            for generation in range(3):
                result = json.loads((run / f'gen-{generation}/generation.json').read_text())
                self.assertEqual(result['execution_mode'], 'fake_fixture')
                self.assertEqual(result['synthetic']['generated_tasks'], 2)
                self.assertEqual(result['synthetic']['valid_tasks'], 2)
                self.assertEqual(result['synthetic']['self_solved_tasks'], 2)
                self.assertEqual(result['synthetic']['sft_records'], 2)
                self.assertIn('C1', result['arc_metrics'])
                if generation == 0:
                    self.assertIn('A', result['arc_metrics'])
                real_task_ids.extend(result['arc_metrics']['C1']['task_ids'])
                self.assertTrue(result['c2']['accepted'])
                self.assertFalse(result['c2']['used_by_c1'])
                self.assertIn('self_solved_real_arc', result['sft']['sources'])
                self.assertIn('self_solved_synthetic', result['sft']['sources'])
            self.assertEqual(len(real_task_ids), len(set(real_task_ids)))
            run_e5(config, run_dir=run, resume=True)
            generation_zero = run / 'gen-0/generation.json'
            generation_zero.write_text(generation_zero.read_text() + 'tampered\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                run_e5(config, run_dir=run, resume=True)

    def test_promotion_gate_fails_closed_on_incomplete_evidence(self):
        from src.pipelines.e5 import _promotion_decision
        complete = {'scored_tasks': 5, 'errors': 0, 'pass_at_1': 0.4}
        improved = {'scored_tasks': 5, 'errors': 0, 'pass_at_1': 0.6}
        result = _promotion_decision(complete, improved, expected_tasks=5,
                                     minimum_delta=0.1, rollback_on_regression=True)
        self.assertEqual(result['decision'], 'promote')
        incomplete = dict(improved, errors=1)
        result = _promotion_decision(complete, incomplete, expected_tasks=5,
                                     minimum_delta=0.0, rollback_on_regression=True)
        self.assertEqual(result['decision'], 'rollback')

    def test_training_handoff_uses_only_eligible_self_solved_records(self):
        config = load_experiment_config(ROOT / 'configs/e5.cpu.fake.yml')
        config = dataclasses.replace(
            config,
            rsi=dataclasses.replace(config.rsi, trainer='mlx_lora'),
            p3b=dataclasses.replace(
                config.p3b,
                training=dataclasses.replace(config.p3b.training, backend='mlx_lora', max_steps=2,
                                             replay_buffer_enabled=False),
                insufficient_data=dataclasses.replace(config.p3b.insufficient_data,
                                                       min_verified_samples=1,
                                                       min_verified_unique_tasks=1)),
            e5=dataclasses.replace(config.e5, minimum_successful_tasks=1,
                                   max_training_steps=2, max_training_seconds=60))
        prompt = json.dumps({'train': [{'input': [[1]], 'output': [[2]]}],
                             'test': [{'input': [[3]]}]}, sort_keys=True, separators=(',', ':'))
        code = 'def transform(grid):\n    return [[(cell + 1) % 10 for cell in row] for row in grid]'
        record_id = hashlib.sha256(('self_solved_synthetic\0' + prompt + '\0' + code).encode()).hexdigest()
        record = {'id': record_id, 'task_id': 'syn-test', 'split': 'synthetic', 'synthetic': True,
                  'data_source': 'self_solved_synthetic', 'generator_sha256': 'b' * 64,
                  'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                               {'role': 'user', 'content': prompt},
                               {'role': 'assistant', 'content': code}]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root / 'parent-adapter'
            previous.mkdir()
            previous_hash = 'c' * 64
            budget = E5Budget(root / 'budget.json', config, resume=False)
            def fake_train(records, *, output, **kwargs):
                adapter = output / 'adapter'
                adapter.mkdir(parents=True)
                (adapter / 'adapters.safetensors').write_bytes(b'fake adapter')
                return {'status': 'complete', 'trained': True, 'weights_updated': True,
                        'source': kwargs['source'], 'sample_count': len(records),
                        'adapter_path': str(adapter), 'adapter_sha256': 'a' * 64,
                        'elapsed_seconds': 1.0}
            with patch('src.pipelines.e5.train_adapter', side_effect=fake_train) as trainer, \
                    patch('src.pipelines.e5.adapter_sha256', side_effect=lambda path: previous_hash if Path(path) == previous else 'a' * 64):
                result, active = _train_generation(config, budget, root / 'gen-0',
                                                   [record], [], previous, generation=0)
            self.assertEqual(result['source'], 'e5_selfplay')
            self.assertTrue(result['trained'])
            self.assertTrue(result['weights_updated'])
            self.assertEqual(active, (root / 'gen-0/training-run/adapter').resolve())
            self.assertEqual(budget.state['training_steps'], 2)
            trainer.assert_called_once()

    def test_mutable_journal_cache_and_replay_files_are_sealed(self):
        config = load_experiment_config(ROOT / 'configs/e5.cpu.fake.yml')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = E5Budget(root / 'budget.json', config, resume=False)
            cache_path = root / 'request_cache.json'
            backend = E5CachedBackend(FakeE5Backend(), budget, cache_path)
            backend.generate('fixture prompt', kind='fixture', candidate=0)
            cache_path.write_text(cache_path.read_text() + 'tampered\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                E5CachedBackend(FakeE5Backend(), budget, cache_path)

            records_path = root / 'replay.jsonl'
            _write_records(records_path, [{'id': 'fixture'}])
            self.assertEqual(_replay_records(records_path), [{'id': 'fixture'}])
            records_path.write_text(records_path.read_text() + '{}\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                _replay_records(records_path)

            budget_path = root / 'budget.json'
            budget_path.write_text(budget_path.read_text() + 'tampered\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                E5Budget(budget_path, config, resume=True)


if __name__ == '__main__':
    unittest.main()
