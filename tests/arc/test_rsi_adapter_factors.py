import shutil
import struct
import unittest
from pathlib import Path

from src.arc.mlx_token_driver import adapter_sha256
from src.arc.rsi_adapter_factors import compare_adapter_factors, probe_adapter_products
from tests.arc.test_rsi_adapter_structure import AdapterStructureTests


class AdapterFactorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = AdapterStructureTests('test_valid_synthetic_structure_is_not_reload_or_training_proof')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.candidate = self.fixture.root
        self.parent = Path(self.fixture.tmp.name + '-parent')
        shutil.copytree(self.candidate, self.parent)
        self.addCleanup(shutil.rmtree, self.parent)

    def write_value(self, root, suffix, value, index=0):
        path = root / 'adapters.safetensors'
        name = next(key for key in self.fixture.header if key.endswith(suffix))
        with path.open('r+b') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            stream.seek(8 + size + self.fixture.header[name]['data_offsets'][0] + 4 * index)
            stream.write(struct.pack('<f', value))

    def compare(self, *, parent=False):
        return compare_adapter_factors(self.candidate, expected_candidate_sha256=adapter_sha256(self.candidate),
            parent_dir=self.parent if parent else None,
            expected_parent_sha256=adapter_sha256(self.parent) if parent else None)

    def test_32b_factors_use_32b_shapes_without_claiming_reload(self):
        self.fixture.test_32b_structure_uses_its_actual_layer_and_projection_dimensions()
        self.write_value(self.candidate, 'q_proj.lora_a', 1.0)
        self.write_value(self.candidate, 'q_proj.lora_b', 2.0)
        digest = adapter_sha256(self.candidate)
        result = compare_adapter_factors(self.candidate, expected_candidate_sha256=digest, model_key='32b')
        self.assertEqual(result['status'], 'adapter_factors_changed_unverified')
        probe = probe_adapter_products(self.candidate, expected_candidate_sha256=digest, model_key='32b')
        self.assertEqual(probe['candidate_product'], 'nonzero_witness')
        self.assertFalse(probe['reload_verified'])
        with self.assertRaises(ValueError):
            compare_adapter_factors(self.candidate, expected_candidate_sha256=digest, model_key='14b')

    def test_requires_nonzero_pair_in_same_module(self):
        self.assertEqual(self.compare()['status'], 'adapter_factors_rejected')
        self.write_value(self.candidate, 'q_proj.lora_a', 1.0)
        self.assertEqual(self.compare()['status'], 'adapter_factors_rejected')
        self.write_value(self.candidate, 'k_proj.lora_b', 2.0)
        self.assertEqual(self.compare()['status'], 'adapter_factors_rejected')
        self.write_value(self.candidate, 'q_proj.lora_b', 2.0)
        result = self.compare()
        self.assertEqual(result['status'], 'adapter_factors_changed_unverified')
        self.assertFalse(result['effective_update_verified'])

    def test_product_probe_distinguishes_cancellation_and_parent_difference(self):
        for root in (self.candidate, self.parent):
            self.write_value(root, 'q_proj.lora_a', 1.0, 0)
            self.write_value(root, 'q_proj.lora_a', 1.0, 1)
            self.write_value(root, 'q_proj.lora_b', 1.0, 0)
            self.write_value(root, 'q_proj.lora_b', -1.0, 5120)
        def probe():
            return probe_adapter_products(self.candidate, expected_candidate_sha256=adapter_sha256(self.candidate),
                parent_dir=self.parent, expected_parent_sha256=adapter_sha256(self.parent))
        result = probe()
        self.assertEqual(result['candidate_product'], 'inconclusive')
        self.assertEqual(result['parent_difference'], 'inconclusive')
        self.assertFalse(result['reload_verified'])
        self.write_value(self.candidate, 'q_proj.lora_a', 2.0, 0)
        self.write_value(self.candidate, 'q_proj.lora_a', 2.0, 1)
        self.write_value(self.candidate, 'q_proj.lora_b', 0.5, 0)
        self.write_value(self.candidate, 'q_proj.lora_b', -0.5, 5120)
        self.assertEqual(self.compare(parent=True)['status'], 'adapter_factors_changed_unverified')
        self.assertEqual(probe()['parent_difference'], 'inconclusive')
        self.write_value(self.candidate, 'q_proj.lora_b', -0.25, 5120)
        result = probe()
        self.assertEqual(result['candidate_product'], 'nonzero_witness')
        self.assertEqual(result['parent_difference'], 'nonzero_witness')
        self.assertEqual(result['parent_witness']['row'], 0)
        self.assertEqual(result['parent_witness']['column'], 0)
        self.assertFalse(result['training_allowed'])
        self.assertFalse(result['training_allowed'])

    def test_parent_comparison_checks_values_not_header_or_signed_zero(self):
        for root in (self.candidate, self.parent):
            self.write_value(root, 'q_proj.lora_a', 1.0)
            self.write_value(root, 'q_proj.lora_b', 2.0)
        self.write_value(self.candidate, 'k_proj.lora_a', -0.0)
        self.assertEqual(self.compare(parent=True)['status'], 'adapter_factors_rejected')
        self.write_value(self.candidate, 'q_proj.lora_b', 3.0)
        result = self.compare(parent=True)
        self.assertEqual(result['status'], 'adapter_factors_changed_unverified')
        self.assertGreater(result['changed_tensor_count'], 0)
        self.assertFalse(result['effective_update_verified'])


if __name__ == '__main__':
    unittest.main()
