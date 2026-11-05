import hashlib
import json
import unittest

from src.arc.rsi_canary_dataset import prepare_canary_dataset
from src.arc.rsi_preflight_spec import prepare_preflight_spec
from tests.arc.test_rsi_mlx_dataset import Dataset
import tests.arc.test_rsi_sft_format as fixtures


class PreflightSpecTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SFTFormatTests('test_rows_match_inference_prefix_and_mask')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        canary = prepare_canary_dataset(root=self.root, forbidden_tasks=self.fixture.tasks,
            tokenizer=self.fixture.tokenizer, render=self.fixture.render)
        raw = (self.root / 'training.preparation.json').read_bytes()
        def loader(path, tokenizer, config):
            rows = [json.loads(line) for line in (path / 'train.jsonl').read_text().splitlines()]
            return Dataset(rows, self.fixture.render), [], []
        self.args = dict(root=self.root, preparation_bytes=raw, expected_preparation_sha256=hashlib.sha256(raw).hexdigest(),
            expected_canary_sha256=canary['dataset_jsonl_sha256'], forbidden_tasks=self.fixture.tasks,
            tokenizer=self.fixture.tokenizer, render=self.fixture.render, load_local_dataset=loader,
            model_dir=self.root / 'model', generation_data_dir=self.root / 'generation',
            adapter_dir=self.root / 'canary-adapter', previous_adapter_file=None,
            identity_hashes={'model_manifest': 'a' * 64, 'runtime_manifest': 'b' * 64,
                'environment_manifest': 'c' * 64, 'source_inventory': 'd' * 64,
                'tokenizer': 'e' * 64, 'tokenizer_config': 'f' * 64})

    def test_binds_canary_configuration_and_fixed_limits(self):
        result = prepare_preflight_spec(**self.args)
        path = self.root / 'preflight.proposal.json'
        spec = json.loads(path.read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), result['sha256'])
        self.assertEqual(spec['config']['iters'], 1)
        self.assertEqual(spec['config']['data'], str((self.root / 'preflight-canary.quarantine').resolve()))
        self.assertEqual(spec['canary_dataset_sha256'], self.args['expected_canary_sha256'])
        self.assertEqual(spec['resource_limits']['timeout_seconds'], 300)
        self.assertFalse(spec['execution_ready'])
        self.assertFalse(result['training_allowed'])
        with self.assertRaises(FileExistsError):
            prepare_preflight_spec(**self.args)

    def test_rejects_changed_preparation_canary_or_overlapping_output(self):
        with self.assertRaises(ValueError):
            prepare_preflight_spec(**{**self.args, 'preparation_bytes': self.args['preparation_bytes'] + b' '})
        with self.assertRaises(ValueError):
            prepare_preflight_spec(**{**self.args, 'adapter_dir': self.root / 'preflight-canary.quarantine' / 'adapter'})
        (self.root / 'preflight-canary.quarantine' / 'train.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):
            prepare_preflight_spec(**self.args)
        self.assertFalse((self.root / 'preflight.proposal.json').exists())


if __name__ == '__main__':
    unittest.main()
