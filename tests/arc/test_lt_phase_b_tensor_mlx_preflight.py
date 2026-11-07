import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('LT_PHASE_B_MLX_PREFLIGHT') == '1', 'local MLX preflight not requested')
class MLXTensorPreflightTests(unittest.TestCase):
    def test_bfloat_base_is_hashed_losslessly(self):
        import mlx.core as mx
        from src.arc.lt_phase_b_tensor_evidence import capture_tensor_state, compare_tensor_states
        class Model:
            def __init__(self):
                self.p = {'base.weight': mx.array([1, 2], dtype=mx.bfloat16),
                          'layer.lora_a': mx.array([[1]], dtype=mx.float32),
                          'layer.lora_b': mx.array([[0]], dtype=mx.float32)}
            def parameters(self): return self.p
            def trainable_parameters(self): return {k: v for k, v in self.p.items() if '.lora_' in k}
        model = Model()
        with patch('src.arc.lt_phase_b_tensor_evidence._expected_shapes',
                   return_value={'layer.lora_a': [1, 1], 'layer.lora_b': [1, 1]}):
            before = capture_tensor_state(model, flatten=lambda x: list(x.items()), evaluate=mx.eval)
            self.assertEqual(before['base']['base.weight']['dtype'], 'bfloat16')
            model.p['layer.lora_b'] = mx.array([[.25]], dtype=mx.float32)
            after = capture_tensor_state(model, flatten=lambda x: list(x.items()), evaluate=mx.eval)
            self.assertEqual(compare_tensor_states(before, after)['changed_adapter_tensor_count'], 1)
            model.p['base.weight'] = mx.array([1, 3], dtype=mx.bfloat16)
            with self.assertRaises(ValueError):
                compare_tensor_states(before, capture_tensor_state(model, flatten=lambda x: list(x.items()), evaluate=mx.eval))


if __name__ == '__main__': unittest.main()
