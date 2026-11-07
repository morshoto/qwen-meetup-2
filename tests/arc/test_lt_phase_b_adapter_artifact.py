import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_phase_b_adapter_artifact import inspect_adapter_artifact


class AdapterArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / 'adapter_config.json'
        self.config.write_text(json.dumps({'fine_tune_type': 'lora', 'num_layers': 8,
            'lora_parameters': {'rank': 8, 'scale': 2.0, 'dropout': 0.0}}))
        header = {'layer.lora_a': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]},
                  'layer.lora_b': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [4, 8]}}
        raw = json.dumps(header).encode()
        self.weights = self.root / 'adapters.safetensors'
        self.weights.write_bytes(struct.pack('<Q', len(raw)) + raw + struct.pack('<ff', 1, 2))
        self.shapes = patch('src.arc.lt_phase_b_adapter_artifact._expected_shapes',
                            return_value={'layer.lora_a': [1, 1], 'layer.lora_b': [1, 1]})
        self.shapes.start(); self.addCleanup(self.shapes.stop)

    def test_hashes_exact_files_and_records_nonpromotion(self):
        result = inspect_adapter_artifact(self.root)
        self.assertEqual(result['weights_sha256'], hashlib.sha256(self.weights.read_bytes()).hexdigest())
        self.assertEqual(result['config_sha256'], hashlib.sha256(self.config.read_bytes()).hexdigest())
        combined = hashlib.sha256()
        for p in (self.config, self.weights):
            combined.update(p.name.encode() + b'\0'); combined.update(p.read_bytes())
        self.assertEqual(result['adapter_sha256'], combined.hexdigest())
        self.assertFalse(result['reload_verified'])
        self.assertFalse(result['promotion_allowed'])

    def test_rejects_missing_malformed_and_symlink_files(self):
        self.weights.write_bytes(b'bad')
        with self.assertRaises(ValueError): inspect_adapter_artifact(self.root)
        self.weights.unlink()
        with self.assertRaises(ValueError): inspect_adapter_artifact(self.root)
        self.weights.symlink_to(self.config)
        with self.assertRaises(ValueError): inspect_adapter_artifact(self.root)


if __name__ == '__main__': unittest.main()
