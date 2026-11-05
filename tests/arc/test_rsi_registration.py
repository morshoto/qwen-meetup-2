"""Draft registration binds inputs without authorizing live execution."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from src.arc.rsi_registration import prepare_draft_registration, verify_draft_registration


class DraftRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('src', 'sandbox', 'configs', 'scripts', 'inputs'):
            (self.root / name).mkdir()
        for name in ('src/main.py', 'sandbox/runner.py', 'configs/run.yml', 'scripts/run.sh',
                     'pyproject.toml', 'uv.lock'):
            (self.root / name).write_text('fixture')
        ids = [f't{i:04}' for i in range(1000)]
        challenges = {task: {'train': [{'input': [[1]], 'output': [[2]]}],
                             'test': [{'input': [[3]]}]} for task in ids}
        solutions = {task: [[[4]]] for task in ids}
        values = {
            'challenges': challenges, 'solutions': solutions,
            'split': {'status': 'provisional_not_preregistered', 'independence_established': False,
                      'partitions': {'generation': ids[:800], 'development': ids[800:]}},
            'public_probes': [[[i, i], [i, i]] for i in range(5)],
            'private_probes': [[[i, i], [i, i]] for i in range(5, 10)],
            'reference_pool': [],
            'model_manifest': {'model_id': 'mlx-community/Qwen3-14B-4bit',
                'revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                'quantization': {'bits': 4, 'group_size': 64},
                'chat_template_sha256': 'a' * 64, 'files_sha256': {'weights': 'b' * 64}},
            'runtime_manifest': {'fixture': True}, 'sandbox_identity': {'digest': 'sha256:' + 'c' * 64},
            'design_registration': {'kind': 'design_only', 'execution_ready': False,
                                     'competitive_target_percent': 42.4}}
        self.paths = {}
        for key, value in values.items():
            relative = f'inputs/{key}.json'
            (self.root / relative).write_text(json.dumps(value))
            self.paths[key] = relative
        for directory in ('docs', 'research/rsi'):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / 'docs/rsi-research-protocol.md').write_text('v1 design')
        (self.root / 'configs/rsi-research-protocol.json').write_text('{}')
        (self.root / 'research/rsi/provisional-split.json').write_bytes(
            (self.root / self.paths['split']).read_bytes())
        digest = lambda path: hashlib.sha256((self.root / path).read_bytes()).hexdigest()
        v1paths = ('configs/rsi-research-protocol.json', 'docs/rsi-research-protocol.md',
                   'research/rsi/provisional-split.json')
        v1 = {'kind': 'design_only', 'execution_ready': False,
              'files': {name: digest(name) for name in v1paths}}
        (self.root / 'research/rsi/design-registration-v1.json').write_text(json.dumps(v1))
        (self.root / 'research/rsi/design-v2.md').write_text('v2 design')
        v2paths = ('research/rsi/design-registration-v1.json', 'research/rsi/design-v2.md')
        v2 = {'kind': 'design_only', 'execution_ready': False, 'competitive_target_percent': 42.4,
              'supersedes': 'research/rsi/design-registration-v1.json',
              'reference': {'model': 'Qwen3.8-27B', 'split': 'ARC-AGI-2 semi-private',
                            'reasoning_effort': 'xhigh', 'matched_compute': False},
              'files': {name: digest(name) for name in v2paths}}
        (self.root / 'research/rsi/design-registration-v2.json').write_text(json.dumps(v2))
        self.paths['design_registration'] = 'research/rsi/design-registration-v2.json'

    def test_draft_binds_inputs_schedule_and_remains_non_authorizing(self):
        result = prepare_draft_registration(self.root, input_paths=self.paths, generation=0)
        self.assertEqual(result['status'], 'draft_not_execution_ready')
        self.assertFalse(result['execution_ready'])
        self.assertEqual(len(result['slots']), 10)
        self.assertEqual(result['slots'][0]['task_id'], 't0000')
        self.assertEqual(result['slots'][0]['attempt_seeds'], [12000, 12001, 12002])
        self.assertEqual(set(result['input_sha256']), set(self.paths))
        self.assertNotIn('[[[4]]]', json.dumps(result))
        verify_draft_registration(self.root, result)

    def test_v3_design_chain_is_verified_without_authorizing_execution(self):
        digest = lambda path: hashlib.sha256((self.root / path).read_bytes()).hexdigest()
        (self.root / 'research/rsi/design-v3.md').write_text('v3 design')
        v3 = {'kind': 'design_only', 'execution_ready': False, 'competitive_target_percent': 42.4,
              'supersedes': 'research/rsi/design-registration-v2.json',
              'exposure_policy': 'bounded_review_unknown_project_exposure',
              'files': {name: digest(name) for name in ('research/rsi/design-registration-v2.json',
                                                       'research/rsi/design-v3.md')}}
        path = self.root / 'research/rsi/design-registration-v3.json'
        path.write_text(json.dumps(v3))
        self.paths['design_registration'] = 'research/rsi/design-registration-v3.json'
        result = prepare_draft_registration(self.root, input_paths=self.paths, generation=0)
        self.assertFalse(result['execution_ready'])
        self.assertIn('bounded exposure review not independently completed', result['blockers'])
        self.assertIn('research/rsi/design-v3.md', result['design_file_sha256'])
        verify_draft_registration(self.root, result)
        (self.root / 'research/rsi/design-v3.md').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'design'):
            verify_draft_registration(self.root, result)

    def test_v3_rejects_changed_target_or_exposure_policy(self):
        path = self.root / 'research/rsi/design-registration-v3.json'
        for field, value in [('competitive_target_percent', 0), ('exposure_policy', 'complete')]:
            v3 = {'kind': 'design_only', 'execution_ready': False, 'competitive_target_percent': 42.4,
                  'supersedes': 'research/rsi/design-registration-v2.json',
                  'exposure_policy': 'bounded_review_unknown_project_exposure', 'files': {}}
            v3[field] = value
            path.write_text(json.dumps(v3))
            self.paths['design_registration'] = 'research/rsi/design-registration-v3.json'
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_draft_registration(self.root, input_paths=self.paths, generation=0)

    def test_v4_design_chain_keeps_canaries_non_authorizing(self):
        digest = lambda name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
        (self.root / 'research/rsi/design-v3.md').write_text('v3 design')
        v3 = {'kind':'design_only','execution_ready':False,'competitive_target_percent':42.4,
              'supersedes':'research/rsi/design-registration-v2.json',
              'exposure_policy':'bounded_review_unknown_project_exposure',
              'files':{p:digest(p) for p in ('research/rsi/design-registration-v2.json','research/rsi/design-v3.md')}}
        (self.root / 'research/rsi/design-registration-v3.json').write_text(json.dumps(v3))
        (self.root / 'research/rsi/design-v4.md').write_text('v4 design')
        v4 = {'kind':'design_only','execution_ready':False,'competitive_target_percent':42.4,
              'supersedes':'research/rsi/design-registration-v3.json',
              'canary_policy':'independent_synthetic_only_no_promotion',
              'evaluation_policy':'separate_exploratory_50_then_development_200',
              'oracle_policy':'new_only_pending_historical_decision',
              'files':{p:digest(p) for p in ('research/rsi/design-registration-v3.json','research/rsi/design-v4.md')}}
        path=self.root / 'research/rsi/design-registration-v4.json'; path.write_text(json.dumps(v4))
        self.paths['design_registration']='research/rsi/design-registration-v4.json'
        result=prepare_draft_registration(self.root,input_paths=self.paths,generation=0)
        self.assertFalse(result['execution_ready'])
        self.assertIn('bounded exposure review not independently completed',result['blockers'])
        self.assertIn('research/rsi/design-v4.md',result['design_file_sha256'])
        verify_draft_registration(self.root,result)
        altered=dict(v4,oracle_policy='include_historical')
        path.write_text(json.dumps(altered))
        with self.assertRaises(ValueError):
            prepare_draft_registration(self.root,input_paths=self.paths,generation=0)
        path.write_text(json.dumps(v4))
        (self.root / 'research/rsi/design-v4.md').write_text('changed')
        with self.assertRaisesRegex(ValueError,'design'):verify_draft_registration(self.root,result)

    def test_tampered_or_missing_inputs_and_source_are_rejected(self):
        result = prepare_draft_registration(self.root, input_paths=self.paths, generation=0)
        (self.root / self.paths['solutions']).write_text('{}')
        with self.assertRaises(ValueError):
            verify_draft_registration(self.root, result)
        (self.root / self.paths['solutions']).unlink()
        with self.assertRaises(ValueError):
            verify_draft_registration(self.root, result)

    def test_rejects_overlap_wrong_model_and_path_escape(self):
        split_path = self.root / self.paths['split']
        split = json.loads(split_path.read_text())
        split['partitions']['development'][0] = split['partitions']['generation'][0]
        split_path.write_text(json.dumps(split))
        with self.assertRaises(ValueError):
            prepare_draft_registration(self.root, input_paths=self.paths, generation=0)
        with self.assertRaises(ValueError):
            prepare_draft_registration(self.root, input_paths=dict(self.paths, challenges='../outside.json'), generation=0)
        split['partitions']['development'][0] = 't0800'
        split_path.write_text(json.dumps(split))
        model = self.root / self.paths['model_manifest']
        value = json.loads(model.read_text())
        value['revision'] = 'wrong'
        model.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'model'):
            prepare_draft_registration(self.root, input_paths=self.paths, generation=0)

    def test_rejects_probe_overlap_with_scheduled_source_inputs(self):
        public = self.root / self.paths['public_probes']
        probes = json.loads(public.read_text())
        probes[0] = [[1]]
        public.write_text(json.dumps(probes))
        with self.assertRaisesRegex(ValueError, 'source.*probe'):
            prepare_draft_registration(self.root, input_paths=self.paths, generation=0)

    def test_source_change_after_registration_is_rejected(self):
        result = prepare_draft_registration(self.root, input_paths=self.paths, generation=0)
        (self.root / 'src/main.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_draft_registration(self.root, result)

    def test_rejects_forged_or_changed_design_registration_chain(self):
        v1 = self.root / 'research/rsi/design-registration-v1.json'
        v1.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'design'):
            prepare_draft_registration(self.root, input_paths=self.paths, generation=0)

    def test_rejects_unregistered_quantization(self):
        model = self.root / self.paths['model_manifest']
        value = json.loads(model.read_text())
        for quantization in ({'bits': 4, 'group_size': 128}, {'bits': 4.0, 'group_size': 64}):
            value['quantization'] = quantization
            model.write_text(json.dumps(value))
            with self.subTest(quantization=quantization), self.assertRaisesRegex(ValueError, 'model'):
                prepare_draft_registration(self.root, input_paths=self.paths, generation=0)

    def test_rejects_non_string_and_symlinked_input_paths(self):
        with self.assertRaises(ValueError):
            prepare_draft_registration(self.root, input_paths=dict(self.paths, challenges=[]), generation=0)
        link = self.root / 'inputs/challenge-link.json'
        link.symlink_to(self.root / self.paths['challenges'])
        with self.assertRaises(ValueError):
            prepare_draft_registration(self.root, input_paths=dict(self.paths, challenges='inputs/challenge-link.json'), generation=0)


if __name__ == '__main__':
    unittest.main()
