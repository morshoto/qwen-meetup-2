from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.arc.e5_selfplay import _generator_prompt
from src.arc.e6f_fake import PROGRAMS
from src.arc.e6g_config import load_e6g_config
from src.arc.e6g_novelty import FEEDBACK_INSTRUCTION, NoveltyFeedback, classify_novelty
from src.arc.e6g_provenance import verify_response_body
from src.arc.http_backend import BackendError
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.pipelines.e5 import _seal_mutable_file
from src.pipelines import e6g
from src.pipelines.e6g import _fake_pool, run_e6g

ROOT = Path(__file__).resolve().parents[2]


class NoveltyTests(unittest.TestCase):
    def setUp(self):
        self.settings, self.base = load_e6g_config(ROOT / 'configs/e6g.fake.yml')
        self.probes = e6g._quality_grids(88000, 5)
        self.pool = _fake_pool(self.base, self.probes)

    def test_source_ast_shape_and_behavior_are_distinct_categories(self):
        code = self.pool[0]['record']['messages'][-1]['content']
        self.assertEqual(classify_novelty(code, None, self.pool), 'duplicate_source')
        self.assertEqual(classify_novelty(code + '\n# harmless comment', None, self.pool), 'duplicate_ast')
        renamed = code.replace('row', 'element')
        self.assertEqual(classify_novelty(renamed, None, self.pool), 'normalized_shape_suspect')
        behavior = self.pool[0]['behavior_sha256']
        self.assertEqual(classify_novelty(PROGRAMS[2], behavior, self.pool), 'sampled_behavior_suspect')
        with self.assertRaises(SyntaxError):
            classify_novelty('def transform(:', None, self.pool)

    def test_feedback_only_uses_proven_model_code_and_no_verifier_data(self):
        self.pool[0]['record']['_verifier_only']['outputs'] = [[[7, 7, 7]]]
        feedback = NoveltyFeedback(enabled=True, pool=self.pool, allow_fixture=True)
        prompt = feedback.prompt('generic base prompt')
        self.assertIn(FEEDBACK_INSTRUCTION, prompt)
        self.assertNotIn('[[[7, 7, 7]]]', prompt)
        self.assertNotIn(self.pool[0]['record']['messages'][1]['content'], prompt)
        self.assertFalse(any(word in FEEDBACK_INSTRUCTION.lower() for word in ('rotate', 'flip', 'color', 'reflect')))
        with self.assertRaisesRegex(ValueError, 'Qwen provenance'):
            NoveltyFeedback(enabled=True, pool=self.pool)

    def test_generated_text_must_match_the_hashed_http_body(self):
        code = PROGRAMS[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root/'logs/e6'; log.mkdir(parents=True)
            request_path, response_path = log/'llm_requests.jsonl', log/'llm_responses.jsonl'
            prompt = 'generic generator prompt'
            prompt_sha = hashlib.sha256(json.dumps([
                {'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            provenance = {'request_id': 'req', 'model_revision': 'revision', 'model_id': 'model',
                          'thinking_mode': 'off', 'request_log_path': str(request_path),
                          'response_log_path': str(response_path)}
            def write(body_content):
                body = json.dumps({'choices': [{'message': {'content': body_content}}]})
                digest = hashlib.sha256(body.encode()).hexdigest()
                provenance['raw_response_sha256'] = digest
                request = {'request_id': 'req', 'kind': 'e5_generator', 'candidate': 0,
                           'prompt_sha256': prompt_sha, 'response_sha256': digest,
                           'revision': 'revision', 'model': 'model', 'thinking_mode_effective': 'off'}
                response = {'request_id': 'req', 'raw_http_body_text': body, 'generated_text': code}
                request_path.write_text(json.dumps(request)+'\n')
                response_path.write_text(json.dumps(response)+'\n')
            write(code)
            verify_response_body(provenance, code)
            self.assertEqual(e6g._live_provenance(root, prompt, 0, code)['request_id'], 'req')
            write(PROGRAMS[1])
            with self.assertRaisesRegex(ValueError, 'differs from the hashed HTTP'):
                verify_response_body(provenance, code)
            with self.assertRaisesRegex(ValueError, 'differs from the hashed HTTP'):
                e6g._live_provenance(root, prompt, 0, code)


class PairedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.settings, self.base = load_e6g_config(ROOT / 'configs/e6g.fake.yml')

    def test_fake_pair_fixed_pool_feedback_and_resume_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            run_e6g(self.settings, self.base, run_dir=root)
            result = json.loads((root / 'comparison.json').read_text())
            g0, g1 = result['conditions']['G0_baseline'], result['conditions']['G1_novelty_feedback']
            self.assertEqual(g0['initial_pool_snapshot_sha256'], g1['initial_pool_snapshot_sha256'])
            self.assertEqual((g0['verified_novel_oracles'], g1['verified_novel_oracles']), (0, 2))
            self.assertEqual(g0['behavior_evaluated_candidates'], 3)
            self.assertEqual(g0['feedback_applied_candidates'], 0)
            self.assertEqual(g1['feedback_applied_candidates'], 3)
            self.assertFalse(result['training_invoked'])
            self.assertFalse(result['eligible_for_future_training'])
            baseline = [json.loads(x) for x in (root / 'conditions/G0_baseline/lineage.jsonl').read_text().splitlines()]
            hashes = {e['prompt_sha256'] for e in baseline if e.get('event') == 'generation_requested'}
            self.assertEqual(hashes, {hashlib.sha256(_generator_prompt().encode()).hexdigest()})
            self.assertIn('src/arc/training.py', json.loads((root/'experiment_protocol.json').read_text())['implementation_sha256'])
            before = (root / 'seal.json').read_bytes()
            run_e6g(self.settings, self.base, run_dir=root, resume=True)
            self.assertEqual((root / 'seal.json').read_bytes(), before)
            changed_base = replace(self.base, agent=replace(self.base.agent, timeout_seconds=self.base.agent.timeout_seconds + 1))
            with self.assertRaisesRegex(ValueError, 'identity'):
                run_e6g(self.settings, changed_base, run_dir=root, resume=True)
            original_sha = e6g._sha
            def changed_filter(path):
                return '0' * 64 if str(path).endswith('/src/arc/training.py') else original_sha(path)
            with patch('src.pipelines.e6g._sha', side_effect=changed_filter):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    run_e6g(self.settings, self.base, run_dir=root, resume=True)
            (root/'conditions/G0_baseline/seal.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'seal mismatch'):
                run_e6g(self.settings, self.base, run_dir=root, resume=True)

    def test_conditions_do_not_share_new_records_even_at_five_fixture_oracles(self):
        class ThreeNovel:
            def __init__(self, condition): pass
            def generate(self, prompt, *, kind, candidate):
                return PROGRAMS[candidate // 3 + 2]
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.arc.e6g_fake.FakeBackend', ThreeNovel), \
             patch('src.pipelines.e5.train_adapter', side_effect=AssertionError('trainer invoked')), \
             patch('src.pipelines.e6.train_adapter', side_effect=AssertionError('trainer invoked')):
            root = Path(temp) / 'run'
            run_e6g(self.settings, self.base, run_dir=root)
            result = json.loads((root/'comparison.json').read_text())
            self.assertEqual([result['conditions'][n]['verified_novel_oracles'] for n in e6g.CONDITIONS], [3, 3])
            self.assertEqual(result['fixture_union_count'], 5)
            self.assertEqual(len(result['cross_arm_rejections']), 3)
            self.assertFalse(result['training_invoked'])
            self.assertFalse(result['eligible_for_future_training'])

    def test_protocol_is_sealed_before_the_first_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            class CheckingBackend:
                def __init__(self, condition): pass
                def generate(self, prompt, *, kind, candidate):
                    protocol = root/'experiment_protocol.json'
                    seal = json.loads((root/'experiment_protocol.json.sha256.json').read_text())
                    assert protocol.is_file() and hashlib.sha256(protocol.read_bytes()).hexdigest() == seal['sha256']
                    return PROGRAMS[0]
            with patch('src.arc.e6g_fake.FakeBackend', CheckingBackend):
                run_e6g(self.settings, self.base, run_dir=root)

    def test_full_comparison_caps_can_be_exercised_with_fake(self):
        class DuplicateOnly:
            def __init__(self, condition): pass
            def generate(self, prompt, *, kind, candidate): return PROGRAMS[0]
        settings = replace(self.settings, candidates_per_condition=10, per_condition_calls=30,
                           per_condition_reserved_tokens=140000, per_condition_sandbox=100,
                           per_condition_wall_seconds=1200, total_calls=60,
                           total_reserved_tokens=280000, total_sandbox=200, total_wall_seconds=2400)
        with tempfile.TemporaryDirectory() as temp, patch('src.arc.e6g_fake.FakeBackend', DuplicateOnly):
            root = Path(temp) / 'full-fake'
            run_e6g(settings, self.base, run_dir=root)
            result = json.loads((root/'comparison.json').read_text())
            self.assertEqual(result['global_budget']['sandbox_runs'], 200)
            self.assertTrue(all(result['conditions'][n]['sandbox_runs'] == 100 for n in e6g.CONDITIONS))
            self.assertTrue(all(result['conditions'][n]['completed_candidates'] == 5 for n in e6g.CONDITIONS))
            self.assertFalse(result['training_invoked'])

    def test_no_novel_is_not_a_success_and_budget_is_symmetric(self):
        class DuplicateOnly:
            def __init__(self, condition): pass
            def generate(self, prompt, *, kind, candidate): return PROGRAMS[0]
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.arc.e6g_fake.FakeBackend', DuplicateOnly), \
             patch('src.pipelines.e5.train_adapter', side_effect=AssertionError('trainer invoked')), \
             patch('src.pipelines.e6.train_adapter', side_effect=AssertionError('trainer invoked')):
            root = Path(temp) / 'zero'
            run_e6g(self.settings, self.base, run_dir=root)
            result = json.loads((root/'comparison.json').read_text())
            self.assertEqual(result['decision'], 'no_demonstrated_gain')
            self.assertFalse(result['training_invoked'])
        settings = replace(self.settings, per_condition_sandbox=1, total_sandbox=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'budget'
            run_e6g(settings, self.base, run_dir=root)
            result = json.loads((root/'comparison.json').read_text())
            self.assertEqual(result['global_budget']['sandbox_runs'], 2)
            self.assertTrue(all(result['conditions'][n]['sandbox_runs'] == 1 for n in e6g.CONDITIONS))
            self.assertTrue(all(result['conditions'][n]['stop_reason'] == 'budget_exhausted' for n in e6g.CONDITIONS))

    def test_model_call_and_token_reservations_stop_before_overrun(self):
        settings = replace(self.settings, per_condition_calls=1, total_calls=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'calls'
            run_e6g(settings, self.base, run_dir=root)
            result = json.loads((root/'comparison.json').read_text())
            self.assertEqual(result['global_budget']['model_calls'], 2)
            self.assertTrue(all(result['conditions'][n]['model_calls'] == 1 for n in e6g.CONDITIONS))
        settings = replace(self.settings, per_condition_reserved_tokens=1000, total_reserved_tokens=2000)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'tokens'
            run_e6g(settings, self.base, run_dir=root)
            result = json.loads((root/'comparison.json').read_text())
            self.assertEqual(result['global_budget']['model_calls'], 0)
            self.assertEqual(result['global_budget']['reserved_tokens'], 0)

    def test_errors_and_model_repair_are_reported_per_condition(self):
        from src.arc.e6f_fake import BAD
        class RepairFixture:
            def __init__(self, condition): pass
            def generate(self, prompt, *, kind, candidate):
                index, attempt = divmod(candidate, 3)
                if index == 0: return BAD if attempt == 0 else PROGRAMS[2]
                if index == 1: return 'def transform(:' if attempt == 0 else PROGRAMS[0]
                return PROGRAMS[3]
        settings = replace(self.settings, per_condition_sandbox=80, total_sandbox=160)
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.arc.e6g_fake.FakeBackend', RepairFixture):
            root = Path(temp) / 'repair'
            run_e6g(settings, self.base, run_dir=root)
            result = json.loads((root/'comparison.json').read_text())
            for name in e6g.CONDITIONS:
                metrics = result['conditions'][name]
                self.assertEqual(metrics['repair_attempted_candidates'], 2)
                self.assertEqual(metrics['repair_validation_successes'], 2)
                self.assertEqual(metrics['repair_novel_successes'], 1)
                self.assertEqual(metrics['execution_errors'], 1)

    def test_behavior_rate_excludes_candidates_without_a_fingerprint(self):
        events = [
            {'event': 'candidate_finished', 'candidate_index': 0, 'completion': 'completed',
             'status': 'repair_limit', 'behavior_match': False},
            {'event': 'candidate_finished', 'candidate_index': 1, 'completion': 'completed',
             'status': 'duplicate_source', 'behavior_match': True, 'behavior_fingerprint': 'hash'},
        ]
        budget = {'model_calls': 0, 'reserved_tokens': 0, 'sandbox_runs': 0,
                  'elapsed_seconds': 0.0, 'training_steps': 0}
        with tempfile.TemporaryDirectory() as temp:
            metrics = e6g._condition_metrics('G0_baseline', events, budget, Path(temp), 'pool', 2)
            self.assertEqual(metrics['behavior_evaluated_candidates'], 1)
            self.assertEqual(metrics['sampled_behavior_suspect_rate'], 1.0)
            metrics = e6g._condition_metrics('G0_baseline', events[:1], budget, Path(temp), 'pool', 1)
            self.assertEqual(metrics['behavior_evaluated_candidates'], 0)
            self.assertIsNone(metrics['sampled_behavior_suspect_rate'])

    def test_sandbox_timeout_stops_both_conditions_without_training(self):
        pool = _fake_pool(self.base, e6g._quality_grids(88000, 5))
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.pipelines.e6g._fake_pool', return_value=pool), \
             patch('src.arc.e6g_fake.FakeSandbox.run', side_effect=TimeoutError('sandbox deadline exceeded')):
            root = Path(temp) / 'timeout'
            run_e6g(self.settings, self.base, run_dir=root)
            result = json.loads((root/'comparison.json').read_text())
            self.assertTrue(all(result['conditions'][n]['stop_reason'] == 'timeout_or_resource_limit'
                                for n in e6g.CONDITIONS))
            self.assertFalse(result['training_invoked'])

    def test_interrupted_trial_is_never_reissued(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            with patch('src.pipelines.e6g.run_candidate', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_e6g(self.settings, self.base, run_dir=root)
            run_e6g(self.settings, self.base, run_dir=root, resume=True)
            protocol = json.loads((root/'experiment_protocol.json').read_text())
            first = protocol['condition_order'][0]
            rows = [json.loads(x) for x in (root/'conditions'/first/'lineage.jsonl').read_text().splitlines()]
            self.assertEqual([r['candidate_index'] for r in rows if r.get('event') == 'candidate_started'], [0, 1, 2])
            self.assertTrue(any(r.get('status') == 'interrupted' for r in rows))

    def test_uncertain_request_is_never_reissued(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            def interrupt(**kwargs):
                kwargs['generate']('fixture prompt', 0, 0)
                order = json.loads((root/'experiment_protocol.json').read_text())['condition_order']
                cache = root/'conditions'/order[0]/'checkpoints/e6/request_cache.json'
                state = json.loads(cache.read_text()); state['pending'] = {'key': 'uncertain'}
                cache.write_text(json.dumps(state)); _seal_mutable_file(cache)
                raise KeyboardInterrupt
            with patch('src.pipelines.e6g.run_candidate', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_e6g(self.settings, self.base, run_dir=root)
            run_e6g(self.settings, self.base, run_dir=root, resume=True)
            result = json.loads((root/'comparison.json').read_text())
            first = result['condition_order'][0]
            self.assertEqual(result['conditions'][first]['stop_reason'], 'interrupted_uncertain_request')
            self.assertEqual(result['conditions'][first]['model_calls'], 1)

    def test_full_comparison_and_docker_require_explicit_guards(self):
        full, base = load_e6g_config(ROOT/'configs/e6g.live.compare.yml')
        with self.assertRaisesRegex(ValueError, '--approve-full-comparison'):
            run_e6g(full, base, live=True)
        smoke, base = load_e6g_config(ROOT/'configs/e6g.live.smoke.yml')
        with self.assertRaisesRegex(ValueError, '--live'):
            run_e6g(smoke, base)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(e6g, 'PROJECT', Path(temp)), \
             patch('src.pipelines.e6g._docker_image_id', side_effect=BackendError('sandbox_unavailable', 'no Docker')):
            with self.assertRaises(BackendError):
                run_e6g(smoke, base, live=True)
            self.assertFalse((Path(temp)/'artifacts/e6g/live-smoke').exists())

    def test_no_operator_search_in_transitive_import_graph(self):
        pending = ['src.pipelines.e6g', 'src.arc.e6g_novelty', 'src.cli.e6g']
        seen = set()
        while pending:
            module = pending.pop()
            if module in seen: continue
            seen.add(module)
            self.assertNotIn('operator_search', module)
            path = ROOT.joinpath(*module.split('.')).with_suffix('.py')
            if not path.is_file(): continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('src.'):
                    pending.append(node.module)
                if isinstance(node, ast.Import):
                    pending.extend(alias.name for alias in node.names if alias.name.startswith('src.'))


if __name__ == '__main__':
    unittest.main()
