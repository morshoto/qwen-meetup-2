import copy
import unittest
from pathlib import Path

from src.arc.rsi_mlx_config import preview_mlx_configuration
import tests.arc.test_rsi_sft_format as format_fixtures


class MLXConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = format_fixtures.SFTFormatTests('test_rows_match_inference_prefix_and_mask')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.plan = self.fixture.plan
        self.paths = dict(model_dir=self.root / 'model', generation_data_dir=self.root / 'generation',
                          canary_data_dir=self.root / 'canary', adapter_dir=self.root / 'adapter')

    def preview(self, **changes):
        args = dict(plan=self.plan, phase='generation', previous_adapter_file=None, **self.paths)
        args.update(changes)
        return preview_mlx_configuration(**args)

    def test_generation_config_matches_fixed_qlora_parameters(self):
        result = self.preview()
        config = result['config']
        self.assertFalse(result['training_allowed'])
        self.assertEqual(result['timeout_seconds'], 1800)
        self.assertEqual(config['data'], str(self.paths['generation_data_dir'].resolve()))
        self.assertEqual(config['iters'], 20)
        self.assertEqual(config['num_layers'], 2)
        self.assertEqual(config['learning_rate'], 0.0001)
        self.assertEqual(config['lora_parameters'], {'rank': 8, 'scale': 2.0, 'dropout': 0.0})
        self.assertTrue(config['mask_prompt'])
        self.assertIsNone(config['resume_adapter_file'])

    def test_preflight_uses_only_separate_canary_and_one_step(self):
        result = self.preview(phase='preflight')
        self.assertEqual(result['timeout_seconds'], 300)
        self.assertEqual(result['config']['iters'], 1)
        self.assertEqual(result['config']['data'], str(self.paths['canary_data_dir'].resolve()))
        with self.assertRaises(ValueError):
            self.preview(phase='preflight', canary_data_dir=self.paths['generation_data_dir'])

    def test_second_generation_requires_parent_adapter_path(self):
        plan = copy.deepcopy(self.plan)
        plan.update(generation=1, parent_adapter='b' * 64)
        with self.assertRaises(ValueError):
            self.preview(plan=plan)
        parent = self.root / 'parent' / 'adapters.safetensors'
        result = self.preview(plan=plan, previous_adapter_file=parent)
        self.assertEqual(result['config']['resume_adapter_file'], str(parent.resolve()))
        self.assertEqual(result['parent_adapter_claim'], 'b' * 64)
        self.assertFalse(result['parent_adapter_verified'])

    def test_changed_parameters_and_overlapping_output_fail(self):
        changed = copy.deepcopy(self.plan)
        changed['generation_training']['lora_alpha'] = 32
        with self.assertRaises(ValueError):
            self.preview(plan=changed)
        with self.assertRaises(ValueError):
            self.preview(adapter_dir=self.paths['generation_data_dir'] / 'adapter')
        with self.assertRaises(ValueError):
            self.preview(previous_adapter_file=self.root / 'parent.safetensors')


if __name__ == '__main__':
    unittest.main()
