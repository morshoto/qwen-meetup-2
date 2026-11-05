"""Draft registration binds inputs without authorizing live execution."""
import json
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
            'public_probes': [[[i]] for i in range(5)], 'private_probes': [[[i]] for i in range(5, 10)],
            'reference_pool': [],
            'model_manifest': {'model_id': 'mlx-community/Qwen3-14B-4bit',
                'revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                'chat_template_sha256': 'a' * 64, 'files_sha256': {'weights': 'b' * 64}},
            'runtime_manifest': {'fixture': True}, 'sandbox_identity': {'digest': 'sha256:' + 'c' * 64},
            'design_registration': {'kind': 'design_only', 'execution_ready': False,
                                     'competitive_target_percent': 42.4}}
        self.paths = {}
        for key, value in values.items():
            relative = f'inputs/{key}.json'
            (self.root / relative).write_text(json.dumps(value))
            self.paths[key] = relative

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
        model = self.root / self.paths['model_manifest']
        value = json.loads(model.read_text())
        value['revision'] = 'wrong'
        model.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            prepare_draft_registration(self.root, input_paths=self.paths, generation=0)


if __name__ == '__main__':
    unittest.main()
