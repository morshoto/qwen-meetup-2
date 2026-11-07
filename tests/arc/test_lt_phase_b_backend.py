import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.arc.lt_live_backend import MLXBackend
from src.arc.lt_phase_b_backend import PhaseBBackend


def digest(root):
    h = hashlib.sha256()
    for name in ('adapter_config.json', 'adapters.safetensors'):
        h.update(name.encode() + b'\0')
        h.update((root / name).read_bytes())
    return h.hexdigest()


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'adapter_config.json').write_text(json.dumps({'fine_tune_type': 'lora',
            'num_layers': 8, 'lora_parameters': {'rank': 8, 'scale': 2.0, 'dropout': 0.0}}))
        (self.root / 'adapters.safetensors').write_bytes(b'synthetic')
        self.hash = digest(self.root)
        self.saved = {'layer.lora_a': np.array([[1]], dtype=np.float32),
                      'layer.lora_b': np.array([[2]], dtype=np.float32)}
        self.loaded = {k: v.copy() for k, v in self.saved.items()}
        self.calls = []
        self.mx = SimpleNamespace(set_memory_limit=lambda x: self.calls.append(('limit', x)),
                                  load=lambda p: self.saved, eval=lambda *a: None)
        def loader(path, **kwargs):
            self.calls.append(('load', path, kwargs))
            return SimpleNamespace(parameters=lambda: self.loaded if kwargs.get('adapter_path') else {'base.weight': np.array([1])}), object()
        self.runtime = (self.mx, loader, lambda *a, **k: None, lambda **k: None,
                        lambda x: list(x.items()))
        for name, obj in [('src.arc.lt_phase_b_backend._runtime', lambda: self.runtime),
                          ('src.arc.lt_phase_b_backend.verify_model_manifest', lambda *a, **k: None),
                          ('src.arc.lt_phase_b_backend._expected_shapes', lambda: {k: [1, 1] for k in self.saved})]:
            p = patch(name, obj); p.start(); self.addCleanup(p.stop)

    def test_adapter_off_and_on_share_frozen_generation(self):
        self.assertIs(PhaseBBackend.generate, MLXBackend.generate)
        off = PhaseBBackend('model', model_manifest={}, adapter_state='off')
        self.assertEqual(off.adapter_state, 'off')
        self.assertNotIn('adapter_path', self.calls[-1][2])
        on = PhaseBBackend('model', model_manifest={}, adapter_state='on',
                           adapter_path=self.root, adapter_sha256=self.hash)
        self.assertEqual(on.adapter_sha256, self.hash)
        self.assertEqual(self.calls[-1][2]['adapter_path'], str(self.root.resolve()))

    def test_reject_wrong_identity_state_configuration_and_loaded_values(self):
        for kwargs in ({'adapter_state': 'off', 'adapter_path': self.root, 'adapter_sha256': self.hash},
                       {'adapter_state': 'on'}, {'adapter_state': 'on', 'adapter_path': self.root, 'adapter_sha256': '0' * 64},
                       {'adapter_state': 'other'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PhaseBBackend('model', model_manifest={}, **kwargs)
        self.loaded['layer.lora_b'][0, 0] = 7
        with self.assertRaises(ValueError):
            PhaseBBackend('model', model_manifest={}, adapter_state='on', adapter_path=self.root, adapter_sha256=self.hash)
        self.loaded.pop('layer.lora_b')
        with self.assertRaises(ValueError):
            PhaseBBackend('model', model_manifest={}, adapter_state='on', adapter_path=self.root, adapter_sha256=self.hash)
        (self.root / 'adapter_config.json').write_text('{}')
        with self.assertRaises(ValueError):
            PhaseBBackend('model', model_manifest={}, adapter_state='on', adapter_path=self.root, adapter_sha256=digest(self.root))

    def test_off_rejects_unexpected_adapter_tensors(self):
        mx, _, stream, sampler, flatten = self.runtime
        def loader(*args, **kwargs):
            return SimpleNamespace(parameters=lambda: self.loaded), object()
        with patch('src.arc.lt_phase_b_backend._runtime', return_value=(mx, loader, stream, sampler, flatten)):
            with self.assertRaises(ValueError):
                PhaseBBackend('model', model_manifest={}, adapter_state='off')

    def test_changed_adapter_during_load_is_rejected(self):
        mx, loader, stream, sampler, flatten = self.runtime
        def changed(*args, **kwargs):
            result = loader(*args, **kwargs)
            (self.root / 'adapters.safetensors').write_bytes(b'changed')
            return result
        with patch('src.arc.lt_phase_b_backend._runtime', return_value=(mx, changed, stream, sampler, flatten)):
            with self.assertRaises(ValueError):
                PhaseBBackend('model', model_manifest={}, adapter_state='on', adapter_path=self.root, adapter_sha256=self.hash)

    def test_reject_nonfinite_extra_and_wrong_shapes(self):
        for replacement in ({**self.saved, 'extra': np.ones((1, 1), dtype=np.float32)},
                            {**self.saved, 'layer.lora_a': np.array([[float('nan')]], dtype=np.float32)},
                            {**self.saved, 'layer.lora_a': np.ones((2, 1), dtype=np.float32)}):
            self.saved = replacement
            with self.assertRaises(ValueError):
                PhaseBBackend('model', model_manifest={}, adapter_state='on', adapter_path=self.root, adapter_sha256=self.hash)


class ShapeTests(unittest.TestCase):
    def test_expected_factors_cover_last_eight_layers(self):
        from src.arc.lt_phase_b_backend import _expected_shapes
        shapes = _expected_shapes()
        self.assertEqual(len(shapes), 112)
        for layer in range(32, 40):
            self.assertEqual(shapes[f'model.layers.{layer}.self_attn.q_proj.lora_a'], [5120, 8])
            self.assertEqual(shapes[f'model.layers.{layer}.mlp.down_proj.lora_b'], [8, 5120])
        self.assertNotIn('model.layers.31.self_attn.q_proj.lora_a', shapes)


if __name__ == '__main__': unittest.main()
