import json
import struct
import tempfile
import unittest
from pathlib import Path

from src.arc.mlx_token_driver import adapter_sha256
from src.arc.rsi_adapter_structure import inspect_adapter_structure


def fixture_header():
    dimensions = {'self_attn.q_proj': (5120, 5120), 'self_attn.k_proj': (5120, 1024),
                  'self_attn.v_proj': (5120, 1024), 'self_attn.o_proj': (5120, 5120),
                  'mlp.gate_proj': (5120, 17408), 'mlp.up_proj': (5120, 17408),
                  'mlp.down_proj': (17408, 5120)}
    header, offset = {}, 0
    for layer in (38, 39):
        for name, (inp, out) in dimensions.items():
            for suffix, shape in (('lora_a', [inp, 8]), ('lora_b', [8, out])):
                length = shape[0] * shape[1] * 4
                header[f'model.layers.{layer}.{name}.{suffix}'] = {
                    'dtype': 'F32', 'shape': shape, 'data_offsets': [offset, offset + length]}
                offset += length
    return header, offset


class AdapterStructureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {'fine_tune_type': 'lora', 'num_layers': 2,
                       'lora_parameters': {'rank': 8, 'scale': 2.0, 'dropout': 0.0}}
        self.header, self.size = fixture_header()
        self.write_fixture()

    def write_fixture(self):
        (self.root / 'adapter_config.json').write_text(json.dumps(self.config))
        raw = json.dumps(self.header, separators=(',', ':')).encode()
        with (self.root / 'adapters.safetensors').open('wb') as stream:
            stream.write(struct.pack('<Q', len(raw)) + raw)
            stream.truncate(8 + len(raw) + self.size)

    def inspect(self):
        return inspect_adapter_structure(self.root, expected_sha256=adapter_sha256(self.root))

    def test_valid_synthetic_structure_is_not_reload_or_training_proof(self):
        result = self.inspect()
        self.assertEqual(result['status'], 'adapter_structure_consistent_unverified')
        self.assertEqual(result['tensor_count'], 28)
        self.assertFalse(result['reload_verified'])
        self.assertFalse(result['training_allowed'])

    def test_missing_shape_overlap_and_wrong_config_are_rejected(self):
        mutations = ('missing', 'shape', 'offset', 'config', 'truncated')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.header, self.size = fixture_header()
                self.config['num_layers'] = 2
                first = next(iter(self.header))
                if mutation == 'missing':
                    del self.header[first]
                elif mutation == 'shape':
                    self.header[first]['shape'][1] = 9
                elif mutation == 'offset':
                    self.header[first]['data_offsets'][0] = 1
                elif mutation == 'config':
                    self.config['num_layers'] = 3
                self.write_fixture()
                if mutation == 'truncated':
                    with (self.root / 'adapters.safetensors').open('r+b') as stream:
                        stream.truncate(100)
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_nonfinite_weights_and_wrong_expected_digest_are_rejected(self):
        path = self.root / 'adapters.safetensors'
        with path.open('r+b') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            stream.seek(8 + length)
            stream.write(struct.pack('<f', float('nan')))
        with self.assertRaises(ValueError):
            self.inspect()
        with self.assertRaises(ValueError):
            inspect_adapter_structure(self.root, expected_sha256='0' * 64)


if __name__ == '__main__':
    unittest.main()
