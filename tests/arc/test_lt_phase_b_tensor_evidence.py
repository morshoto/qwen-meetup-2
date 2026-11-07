import unittest
from unittest.mock import patch

import numpy as np

from src.arc.lt_phase_b_tensor_evidence import capture_tensor_state, compare_tensor_states


class Model:
    def __init__(self):
        self.params = {'base.weight': np.array([[1, 2]], dtype=np.uint32),
                       'layer.lora_a': np.array([[1], [2]], dtype=np.float32),
                       'layer.lora_b': np.array([[0, 0]], dtype=np.float32)}
    def parameters(self): return self.params
    def trainable_parameters(self): return {k: v for k, v in self.params.items() if '.lora_' in k}


class TensorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.expected = {'layer.lora_a': [2, 1], 'layer.lora_b': [1, 2]}
        self.shapes = patch('src.arc.lt_phase_b_tensor_evidence._expected_shapes', return_value=self.expected)
        self.shapes.start(); self.addCleanup(self.shapes.stop)

    def capture(self):
        return capture_tensor_state(self.model, flatten=lambda tree: list(tree.items()), evaluate=lambda _: None)

    def test_changed_adapter_and_unchanged_base_are_separate(self):
        before = self.capture()
        self.model.params['layer.lora_b'][0, 0] = .25
        after = self.capture()
        result = compare_tensor_states(before, after)
        self.assertEqual(result['status'], 'tensor_change_consistent_unverified')
        self.assertEqual(result['changed_adapter_tensor_count'], 1)
        self.assertEqual(result['base_tensor_count'], 1)
        self.assertFalse(result['training_verified'])
        self.assertNotEqual(result['initial_adapter_sha256'], result['final_adapter_sha256'])

    def test_unchanged_adapter_or_changed_base_fails_closed(self):
        before = self.capture()
        with self.assertRaises(ValueError): compare_tensor_states(before, self.capture())
        self.model.params['base.weight'][0, 0] = 9
        self.model.params['layer.lora_b'][0, 0] = .25
        with self.assertRaises(ValueError): compare_tensor_states(before, self.capture())

    def test_missing_wrong_shape_nonfinite_and_trainable_base_are_rejected(self):
        for bad in ('missing', 'shape', 'nonfinite', 'base_trainable'):
            self.model = Model()
            if bad == 'missing': del self.model.params['layer.lora_b']
            elif bad == 'shape': self.model.params['layer.lora_b'] = np.zeros((2, 1), dtype=np.float32)
            elif bad == 'nonfinite': self.model.params['layer.lora_a'][0, 0] = np.nan
            else: self.model.trainable_parameters = lambda: self.model.params
            with self.subTest(bad=bad), self.assertRaises(ValueError): self.capture()

    def test_changed_tensor_structure_or_snapshot_tampering_is_rejected(self):
        before = self.capture()
        self.model.params['layer.lora_b'][0, 0] = .25
        after = self.capture()
        after['adapter']['layer.lora_a']['shape'] = [1, 2]
        with self.assertRaises(ValueError): compare_tensor_states(before, after)
        before['adapter_sha256'] = '0' * 64
        with self.assertRaises(ValueError): compare_tensor_states(before, self.capture())


if __name__ == '__main__': unittest.main()
