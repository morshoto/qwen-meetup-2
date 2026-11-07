import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.arc.lt_phase_b_canary_config import build_canary_config
from src.arc.lt_phase_b_canary_trainer import run_instrumented_cli
from tests.arc.test_lt_phase_b_tensor_evidence import Model


class CanaryTrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.model_dir = root / 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'; self.model_dir.mkdir()
        self.data = root / 'data'; self.data.mkdir(mode=0o700)
        self.evidence = root / 'evidence'; self.evidence.mkdir(mode=0o700)
        self.adapter = root / 'adapter'
        self.plan = build_canary_config(model_dir=self.model_dir, data_dir=self.data, adapter_dir=self.adapter)
        (self.data / 'train.jsonl').write_text(self.plan['dataset_jsonl'])
        (self.data / 'train.jsonl').chmod(0o600)
        self.config = root / 'config.json'
        self.config.write_text(json.dumps(self.plan['trainer_config'], sort_keys=True))
        self.config.chmod(0o600)
        self.config_sha = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.model = Model(); self.steps = 0; self.modes = []
        self.train_args = SimpleNamespace(iters=1, batch_size=1, max_seq_length=512,
            steps_per_report=1, steps_per_save=1, grad_accumulation_steps=1, val_batches=0,
            adapter_file=self.adapter.resolve() / 'adapters.safetensors')
        self.trainer = SimpleNamespace()
        def train(**kwargs):
            self.steps += 1
            self.model.params['layer.lora_b'][0, 0] = .25
        self.trainer.train = train
        def main():
            self.assertEqual(sys.argv, ['mlx_lm.lora', '--config', str(self.config)])
            self.trainer.train(model=self.model, args=self.train_args, val_dataset=[])
        self.trainer.main = main
        self.shapes = patch('src.arc.lt_phase_b_tensor_evidence._expected_shapes',
                            return_value={'layer.lora_a': [2, 1], 'layer.lora_b': [1, 2]})
        self.shapes.start(); self.addCleanup(self.shapes.stop)

    def run_cli(self):
        return run_instrumented_cli(config_path=self.config, expected_config_sha256=self.config_sha,
            evidence_dir=self.evidence, trainer_module=self.trainer, install_policy=self.modes.append,
            flatten=lambda x: list(x.items()), evaluate=lambda _: None)

    def test_single_step_is_instrumented_and_hook_restored(self):
        original, argv = self.trainer.train, sys.argv
        result = self.run_cli()
        self.assertEqual(self.steps, 1)
        self.assertEqual(self.modes, ['off'])
        self.assertIs(self.trainer.train, original)
        self.assertIs(sys.argv, argv)
        self.assertEqual(result['changed_adapter_tensor_count'], 1)
        self.assertFalse(result['training_verified'])
        self.assertTrue((self.evidence / 'tensor-before.json').exists())
        self.assertTrue((self.evidence / 'tensor-after.json').exists())

    def test_bad_step_config_and_dataset_fail_before_training(self):
        self.train_args.iters = 2
        original = self.trainer.train
        with self.assertRaises(ValueError): self.run_cli()
        self.assertEqual(self.steps, 0)
        self.assertIs(self.trainer.train, original)
        self.train_args.iters = 1
        (self.data / 'train.jsonl').write_text('wrong')
        with self.assertRaises(ValueError): self.run_cli()
        self.assertEqual(self.steps, 0)

    def test_config_tampering_and_validation_dataset_fail_closed(self):
        self.config.write_text('{}')
        with self.assertRaises(ValueError): self.run_cli()
        self.assertEqual(self.steps, 0)
        self.config.write_text(json.dumps(self.plan['trainer_config'], sort_keys=True))
        self.config.chmod(0o600)
        (self.data / 'valid.jsonl').write_text('{}')
        with self.assertRaises(ValueError): self.run_cli()
        self.assertEqual(self.steps, 0)


if __name__ == '__main__': unittest.main()
