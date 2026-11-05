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
