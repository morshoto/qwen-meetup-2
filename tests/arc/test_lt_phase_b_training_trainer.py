import hashlib
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.arc.lt_phase_b_training_trainer import run_real_instrumented_cli
from tests.arc import test_lt_phase_b_training_config as fixtures
from tests.arc.test_lt_phase_b_tensor_evidence import Model


class RealTrainerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingConfigTests('test_disjoint_yield_proposes_fixed_final_step_configuration')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.build()
        self.run = self.fixture.fixture.run
        self.config = self.run / 'training-config.json'
        self.config.write_text(json.dumps(self.plan['trainer_config'], sort_keys=True)); self.config.chmod(0o600)
        self.sha = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.evidence = self.run / 'training-evidence'; self.evidence.mkdir(mode=0o700)
        self.model = Model(); self.calls = 0; self.modes = []
        self.args = SimpleNamespace(iters=20, batch_size=1, max_seq_length=512, steps_per_report=1,
            steps_per_save=20, grad_accumulation_steps=1, val_batches=-1,
            adapter_file=self.run / 'gen1-adapter/adapters.safetensors')
        self.trainer = SimpleNamespace()
        def train(**kwargs):
            self.calls += 1
            self.model.params['layer.lora_b'][0, 0] = .25
        self.trainer.train = train
        self.trainer.main = lambda: self.trainer.train(model=self.model, args=self.args,
            train_dataset=[None] * self.fixture.manifest['train_count'],
            val_dataset=[None] * self.fixture.manifest['valid_count'])
        self.shapes = patch('src.arc.lt_phase_b_tensor_evidence._expected_shapes',
                            return_value={'layer.lora_a': [2, 1], 'layer.lora_b': [1, 2]})
        self.shapes.start(); self.addCleanup(self.shapes.stop)

    def execute(self):
        return run_real_instrumented_cli(source_root=self.fixture.fixture.root, run_dir=self.run,
            model_dir=self.fixture.model, expected_source=self.fixture.fixture.fixture.fixture.source,
            config_path=self.config, expected_config_sha256=self.sha, evidence_dir=self.evidence,
            trainer_module=self.trainer, install_policy=self.modes.append,
            flatten=lambda x: list(x.items()), evaluate=lambda _: None)

    def test_fixed_trainer_is_instrumented_and_restored(self):
        train, argv = self.trainer.train, sys.argv
        result = self.execute()
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.modes, ['off'])
        self.assertIs(self.trainer.train, train); self.assertIs(sys.argv, argv)
        self.assertEqual(result['changed_adapter_tensor_count'], 1)
        self.assertFalse(result['training_verified'])
        self.assertTrue((self.evidence / 'tensor-after.json').exists())

    def test_changed_settings_fail_before_trainer(self):
        self.args.iters = 19
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(self.calls, 0)
        self.assertFalse((self.evidence / 'tensor-before.json').exists())

    def test_bad_dataset_count_fails_before_trainer(self):
        self.trainer.main = lambda: self.trainer.train(model=self.model, args=self.args, train_dataset=[], val_dataset=[])
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(self.calls, 0)

    def test_post_training_data_mutation_fails_and_restores_hook(self):
        original = self.trainer.train
        def corrupt(**kwargs):
            original(**kwargs)
            (self.run / 'sft-data/train.jsonl').write_text('changed')
        self.trainer.train = corrupt
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(self.calls, 1)
        self.assertIs(self.trainer.train, corrupt)

    def test_config_changes_fail_before_training(self):
        self.config.write_text('{}')
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(self.calls, 0)


if __name__ == '__main__': unittest.main()
