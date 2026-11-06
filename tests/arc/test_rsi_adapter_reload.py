"""Reload comparison is file-bound evidence, never authorization."""
import unittest

import numpy as np

from src.arc.mlx_token_driver import adapter_sha256
from tests.arc.test_rsi_adapter_structure import AdapterStructureTests


class ReloadComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = AdapterStructureTests('test_valid_synthetic_structure_is_not_reload_or_training_proof')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_both_models_compare_every_loaded_tensor_and_reject_drift(self):
        from src.arc.rsi_adapter_factors import _read_factors
        from src.arc.rsi_adapter_reload import compare_reloaded_factors
        from src.arc.rsi_adapter_structure import _expected_shapes
        for key in ('14b', '32b'):
            if key == '32b':
                self.fixture.test_32b_structure_uses_its_actual_layer_and_projection_dimensions()
            root = self.fixture.root
            digest = adapter_sha256(root)
            factors = _read_factors(root, digest, key)
            shapes = _expected_shapes(key)
            loaded = {name: value.copy().reshape(shapes[name]) for name, value in factors.items()}
            result = compare_reloaded_factors(root, expected_sha256=digest, model_key=key, loaded_factors=loaded)
            self.assertEqual(result['status'], 'loaded_adapter_factors_match_unverified')
            self.assertEqual(result['matched_tensor_count'], 28)
            self.assertFalse(result['reload_verified'])
            self.assertFalse(result['training_allowed'])
            name = next(iter(loaded))
            loaded[name] = loaded[name].reshape(-1)
            with self.assertRaises(ValueError):
                compare_reloaded_factors(root, expected_sha256=digest, model_key=key, loaded_factors=loaded)
            loaded[name] = loaded[name].reshape(shapes[name])
            loaded[name].flat[0] = 1.0
            with self.assertRaises(ValueError):
                compare_reloaded_factors(root, expected_sha256=digest, model_key=key, loaded_factors=loaded)
            del loaded[name]
            with self.assertRaises(ValueError):
                compare_reloaded_factors(root, expected_sha256=digest, model_key=key, loaded_factors=loaded)
