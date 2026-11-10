import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_phase_b_canary_config import build_canary_config


class CanaryConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.model = root / 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'; self.model.mkdir()
        self.data = root / 'synthetic-data'; self.data.mkdir(mode=0o700)
        self.adapter = root / 'canary-adapter'

    def build(self, **changes):
        args = {'model_dir': self.model, 'data_dir': self.data, 'adapter_dir': self.adapter}
        args.update(changes)
        return build_canary_config(**args)

    def test_fixed_synthetic_one_step_configuration(self):
        result = self.build()
        self.assertEqual(result['status'], 'canary_configuration_proposed_unverified')
        self.assertFalse(result['execution_ready'])
        self.assertFalse(result['promotion_allowed'])
        self.assertEqual(result['thinking_mode'], 'off')
        config = result['trainer_config']
        self.assertEqual((config['iters'], config['num_layers'], config['batch_size'], config['max_seq_length']), (1, 8, 1, 512))
        self.assertEqual(config['lora_parameters'], {'rank': 8, 'scale': 2.0, 'dropout': 0.0})
        self.assertEqual((config['seed'], config['learning_rate'], config['mask_prompt']), (20261111, 1e-4, True))
        self.assertEqual(config['optimizer'], 'adam')
        self.assertFalse(config['test'])
        data = result['dataset_jsonl'].encode()
        self.assertEqual(result['dataset_sha256'], hashlib.sha256(data).hexdigest())
        messages = json.loads(data)['messages']
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
        self.assertNotIn('grid', str(messages).lower())
        self.assertEqual(result, self.build())

    def test_rejects_wrong_model_overlapping_paths_or_existing_adapter(self):
        with self.assertRaises(ValueError): self.build(model_dir=self.data)
        with self.assertRaises(ValueError): self.build(adapter_dir=self.data / 'adapter')
        with self.assertRaises(ValueError): self.build(data_dir=self.model)
        self.adapter.mkdir()
        with self.assertRaises(ValueError): self.build()


if __name__ == '__main__': unittest.main()
