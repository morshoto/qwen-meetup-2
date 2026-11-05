import dataclasses
import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.arc.data import load_arc
from src.arc.http_backend import OpenAICompatibleBackend
from src.arc.sandbox import FakeSandbox, IDENTITY_CODE
from src.arc.search import SearchBudget, SearchBudgetError, run_strategy, static_code_check
from src.core.config import load_experiment_config
from src.pipelines.exploration import (SearchFakeBackend, _new_budget, _sandbox,
                                      _method_selection_metrics, _rank_methods,
                                      compare_strategies, collect_solutions)
from src.pipelines.exploration_diagnose import diagnose_run


ROOT = Path(__file__).resolve().parents[2]


class ExplorationSearchTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.verifier = load_arc(ROOT / 'tests/fixtures/exploration_cpu_challenges.json',
                                            ROOT / 'tests/fixtures/exploration_cpu_solutions.json')

    def test_independent_strategy_generates_distinct_candidates_and_uses_train_only(self):
        task = self.tasks['cpu_000']
        run = run_strategy(task, 'S1', SearchFakeBackend(), FakeSandbox(), max_repairs=2,
                           independent_candidates=3, base_candidate=0)
        hashes = [row['code_sha256'] for row in run.candidates]
        self.assertEqual(len(set(hashes)), 3)
        self.assertEqual(run.selected_index, 0)
        self.assertEqual(run.result.outputs, [[[0]], [[2]]])
        self.assertTrue(all(row['complete_train_match'] for row in run.candidates))
        self.assertNotIn('verifier', run.candidates[0])

    def test_s1_uses_first_runnable_candidate_while_s3_uses_train_match(self):
        first = 'def transform(grid):\n    return [list(reversed(row)) for row in grid]'
        second = IDENTITY_CODE
        class Backend:
            def generate(self, prompt, *, kind, candidate):
                return first if candidate % 100000 == 0 else second
        class Sandbox:
            def run(self, code, grid):
                if code == first:
                    return [list(reversed(row)) for row in grid]
                return [row[:] for row in grid]
        task = self.tasks['cpu_000']
        s1 = run_strategy(task, 'S1', Backend(), Sandbox(), max_repairs=1,
                          independent_candidates=3, base_candidate=0)
        s3 = run_strategy(task, 'S3', Backend(), Sandbox(), max_repairs=1,
                          independent_candidates=3, base_candidate=0)
        self.assertEqual(s1.selected_index, 0)
        self.assertEqual(s3.selected_index, 1)
        self.assertEqual(s1.candidates[0]['train_match_count'], 0)
        self.assertEqual(s3.candidates[1]['train_match_count'], 1)

    def test_static_filter_rejects_constant_input_keyed_and_external_code(self):
        self.assertIn('constant_output', static_code_check(
            'def transform(grid):\n    return [[1]]')[1])
        self.assertIn('constant_output', static_code_check(
            'def transform(grid):\n    answer = [[1]]\n    return answer')[1])
        self.assertIn('input_keyed_hardcode', static_code_check(
            'def transform(grid):\n    if grid == [[0]]:\n        return [[1]]\n    return grid')[1])
        self.assertIn('imports_forbidden', static_code_check(
            'import os\ndef transform(grid):\n    return grid')[1])
        self.assertIn('task_identifier_reference', static_code_check(
            'def transform(grid):\n    return task_id')[1])
        self.assertTrue(static_code_check(IDENTITY_CODE)[0])

    def test_search_budget_reserves_before_dispatch(self):
        budget = SearchBudget(max_seconds=10, max_calls=1, max_tokens=10000,
            max_sandbox_starts=10, max_calls_per_task=1, max_tokens_per_task=10000,
            max_sandbox_per_task=10, max_seconds_per_task=10)
        budget.reserve_call('task:S0', 100)
        with self.assertRaises(SearchBudgetError):
            budget.reserve_call('task:S0', 100)
        self.assertEqual(budget.state['calls'], 1)
        self.assertEqual(budget.state['reserved_tokens'], 100)

    def test_thinking_on_requires_a_backend_with_explicit_control(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            OpenAICompatibleBackend(model='m', revision='r', base_url='http://127.0.0.1:1/v1',
                temperature=0, max_tokens=8, seed=1, timeout=1, thinking='on', guarded=False)
        backend = OpenAICompatibleBackend(model='m', revision='r', base_url='http://127.0.0.1:1/v1',
            temperature=0, max_tokens=8, seed=1, timeout=1, thinking='on', guarded=True)
        self.assertEqual(backend.thinking, 'on')

    def test_live_sandbox_uses_the_frozen_docker_image_id(self):
        config = load_experiment_config(ROOT / 'configs/p3c.exploration.cpu.yml')
        with patch('src.pipelines.exploration.DockerSandbox') as docker_sandbox:
            sandbox = _sandbox(config, live=True, budget=_new_budget(config),
                               scope='cpu_000:S0', docker_image_id='sha256:frozen-image')
        self.assertEqual(docker_sandbox.return_value.image, 'sha256:frozen-image')
        self.assertEqual(sandbox.sandbox, docker_sandbox.return_value)


class ExplorationArtifactTests(unittest.TestCase):
    def _config(self, *, target=5):
        base = load_experiment_config(ROOT / 'configs/p3c.exploration.cpu.yml')
        search = dataclasses.replace(base.search, target_unique_verified_tasks=target)
        return dataclasses.replace(base, search=search)

    def test_offline_diagnosis_keeps_missing_evidence_and_response_failures_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'; root.mkdir()
            output = Path(temp) / 'diagnosis'
            result = {'code': IDENTITY_CODE, 'history': [{'code': IDENTITY_CODE, 'feedback': []}],
                      'error': None, 'error_kind': None}
            (root / 'results.json').write_text(json.dumps([
                {'task_id': 'cpu_000', 'condition': 'B', 'score': {'pass_at_1': False},
                 'candidates': [{'result': result, 'error': None, 'elapsed_seconds': 1.0}]},
                {'task_id': 'cpu_001', 'condition': 'B', 'score': None,
                 'candidates': [{'result': None, 'error': {'kind': 'malformed_thinking'}, 'elapsed_seconds': None}]},
            ]))
            (root / 'run_state.json').write_text(json.dumps({'status': 'complete'}))
            challenges = ROOT / 'tests/fixtures/exploration_cpu_challenges.json'
            solutions = ROOT / 'tests/fixtures/exploration_cpu_solutions.json'
            (root / 'config.snapshot.yml').write_text(yaml.safe_dump({'data': {
                'challenges': str(challenges), 'solutions': str(solutions)}}))
            (root / 'llm_requests.jsonl').write_text('{"request_id":"one"}\n{"request_id":"two"}\n')
            (root / 'llm_responses.jsonl').write_text(json.dumps({
                'diagnostic_category': 'thinking_unclosed',
                'generated_text': 'PRIVATE REASONING MUST NOT BE COPIED'}) + '\n')
            summary = diagnose_run(root, output)
            self.assertEqual(summary['metrics']['held_out_task_pass_at_1']['denominator'], 1)
            self.assertEqual(summary['metrics']['held_out_task_pass_at_1']['not_available_count'], 1)
            self.assertEqual(summary['response_event_failure_counts']['thinking_unclosed'], 1)
            self.assertIn('train_fit_heldout_mismatch', summary['failure_breakdown'])
            self.assertIn('thinking_unclosed', summary['failure_breakdown'])
            self.assertNotIn('PRIVATE REASONING', (output / 'diagnosis.md').read_text())
            self.assertNotIn('PRIVATE REASONING', (output / 'summary.json').read_text())

    def test_cpu_fake_e2_e3_verifies_five_unique_solutions_and_excludes_comparison_tasks(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as temp:
            e2_dir = Path(temp) / 'e2'
            e3_dir = Path(temp) / 'e3'
            comparison = compare_strategies(config, output_dir=e2_dir)
            self.assertEqual(comparison['status'], 'complete')
            self.assertEqual(comparison['selected_strategy'], 'S0')
            self.assertEqual(len(comparison['comparison_task_ids']), 3)
            manifest = json.loads((e2_dir / 'selection_manifest.json').read_text())
            self.assertEqual(set(row['strategy'] for row in manifest['strategies']), {'S0', 'S1', 'S2', 'S3'})
            result = collect_solutions(config, e2_dir, output_dir=e3_dir)
            self.assertEqual(result['status'], 'target_reached')
            self.assertEqual(result['verified_unique_tasks'], 5)
            self.assertEqual(result['sft_record_count'], 5)
            self.assertTrue(set(comparison['comparison_task_ids']).isdisjoint(result['attempted_task_ids']))
            sft = [json.loads(line) for line in (e3_dir / 'sft_train.jsonl').read_text().splitlines()]
            self.assertEqual(len(sft), 5)
            self.assertTrue(all('output' not in test for row in sft
                                for test in json.loads(row['messages'][1]['content'])['test']))
            collection_manifest = json.loads((e3_dir / 'collection_manifest.json').read_text())
            self.assertFalse(collection_manifest['test_outputs_in_sft_messages'])
            self.assertFalse(collection_manifest['trained'])
            self.assertFalse(collection_manifest['split_access']['promotion_gate'])
            self.assertFalse(collection_manifest['split_access']['monitoring'])
            for name in ('config.snapshot.yml', 'source_selection_manifest.json',
                         'attempted_tasks.jsonl', 'accepted_solutions.jsonl',
                         'rejected_candidates.jsonl', 'budget_ledger.json', 'artifact_seal.json'):
                self.assertTrue((e3_dir / name).is_file(), name)
            seal = json.loads((e3_dir / 'artifact_seal.json').read_text())
            for name, expected in seal['artifacts'].items():
                self.assertEqual(hashlib.sha256((e3_dir / name).read_bytes()).hexdigest(), expected)
            self.assertEqual(stat.S_IMODE((e2_dir / 'llm_requests.jsonl').stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((e3_dir / 'llm_requests.jsonl').stat().st_mode), 0o600)
            for name, expected in collection_manifest['artifact_sha256'].items():
                self.assertEqual(hashlib.sha256((e3_dir / name).read_bytes()).hexdigest(), expected)

    def test_resume_skips_uncertain_task_without_reissuing_its_request(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as temp:
            e2_dir = Path(temp) / 'e2'
            e3_dir = Path(temp) / 'e3'
            compare_strategies(config, output_dir=e2_dir)

            class InterruptOnce:
                def __init__(self, budget, scope):
                    self.budget, self.scope = budget, scope
                def generate(self, prompt, *, kind, candidate):
                    self.budget.reserve_call(self.scope, 1000)
                    raise KeyboardInterrupt()

            def interrupted_backend(config, *, live, thinking, log_path, budget, scope,
                                    safe_events, prompt_events):
                return InterruptOnce(budget, scope)

            with patch('src.pipelines.exploration._backend', side_effect=interrupted_backend):
                first = collect_solutions(config, e2_dir, output_dir=e3_dir)
            self.assertEqual(first['status'], 'interrupted')
            before = json.loads((e3_dir / 'checkpoint.json').read_text())
            uncertain_id = before['pending_task_id']
            self.assertIsNotNone(uncertain_id)
            resumed = collect_solutions(config, e2_dir, resume=True, run_dir=e3_dir)
            checkpoint = json.loads((e3_dir / 'checkpoint.json').read_text())
            self.assertEqual(resumed['status'], 'target_reached')
            self.assertEqual(checkpoint['uncertain_task_ids'], [uncertain_id])
            self.assertEqual(checkpoint['budget']['scopes'][uncertain_id + ':S0']['calls'], 1)
            traces = [json.loads(line) for line in (e3_dir / 'candidate_traces.jsonl').read_text().splitlines()]
            uncertain_trace = next(row for row in traces if row['task_id'] == uncertain_id)
            self.assertEqual(uncertain_trace['status'], 'uncertain_skipped_on_resume')
            self.assertIsNone(uncertain_trace['code'])
            self.assertEqual(len({row['task_id'] for row in traces}), 6)

    def test_e2_smoke_does_not_freeze_a_strategy(self):
        base = self._config()
        smoke_search = dataclasses.replace(base.search, comparison_task_ids=['cpu_000', 'cpu_001'],
                                           comparison_task_count=2)
        smoke = dataclasses.replace(base,
            tracking=dataclasses.replace(base.tracking, profile='e2_smoke'),
            search=smoke_search)
        with tempfile.TemporaryDirectory() as temp:
            result = compare_strategies(smoke, output_dir=Path(temp) / 'e2-smoke')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['selection_status'], 'not_established_smoke_only')
            self.assertIsNone(result['selected_strategy'])
            self.assertFalse((Path(result['output_dir']) / 'frozen_search_config.yml').exists())

    def test_selection_counts_truncated_response_as_parser_error(self):
        task_ids = [f'task_{index}' for index in range(5)]
        rows = []
        for task_id in task_ids:
            for strategy in ('S0', 'S1', 'S2', 'S3'):
                response_events = []
                if strategy == 'S2':
                    response_events = [{'diagnostic_category': 'generation_truncated',
                                        'error': {'kind': 'malformed_thinking'}}]
                reserved = {'S0': 300, 'S1': 200, 'S2': 100, 'S3': 200}[strategy]
                rows.append({'task_id': task_id, 'strategy': strategy,
                             'cell_status': 'complete', 'final_train_match_rate': 0.0,
                             'final_train_full_match': False,
                             'attempts': [], 'response_events': response_events,
                             'budget': {'reserved_tokens': reserved, 'calls': 1}})
        metrics, paired = _method_selection_metrics(rows, ['S0', 'S1', 'S2', 'S3'], task_ids)
        ranking = _rank_methods(metrics)
        by_strategy = {row['strategy']: row for row in metrics}
        self.assertEqual(paired, task_ids)
        self.assertEqual(by_strategy['S2']['parser_error_events'], 5)
        self.assertEqual(ranking[0]['strategy'], 'S1')

    def test_e2_resume_skips_uncertain_cell_without_reissuing_its_request(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'e2'
            class UncertainBackend:
                def __init__(self, budget, scope): self.budget, self.scope = budget, scope
                def generate(self, prompt, *, kind, candidate):
                    self.budget.reserve_call(self.scope, 1000)
                    raise KeyboardInterrupt()
            def interrupted_backend(config, *, live, thinking, log_path, budget, scope,
                                    safe_events, prompt_events, **callbacks):
                return UncertainBackend(budget, scope)
            with patch('src.pipelines.exploration._backend', side_effect=interrupted_backend):
                first = compare_strategies(config, output_dir=target)
            self.assertEqual(first['status'], 'interrupted')
            checkpoint = json.loads((target / 'checkpoint.json').read_text())
            pending = checkpoint['pending_cell']['cell_key']
            resumed = compare_strategies(config, resume=True, run_dir=target)
            final_checkpoint = json.loads((target / 'checkpoint.json').read_text())
            self.assertEqual(resumed['status'], 'complete')
            self.assertIn(pending, final_checkpoint['uncertain_cells'])
            self.assertEqual(final_checkpoint['budget']['scopes'][pending]['calls'], 1)
            self.assertEqual(resumed['selection_status'], 'not_established')


if __name__ == '__main__':
    unittest.main()
