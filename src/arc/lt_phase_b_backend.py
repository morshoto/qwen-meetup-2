"""Adapter-aware version of the frozen direct-grid backend.

File and tensor checks bind this local load to caller-supplied identities. They
are not independent process attestation or evidence of training or improvement.
"""
from pathlib import Path

from src.arc.lt_live_backend import MLXBackend
from src.arc.rsi_adapter_structure import _expected_shapes as _rsi_shapes
from src.arc.rsi_model_identity import _hash_file, _json_file, verify_adapter_identity, verify_model_manifest


def _runtime():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    return mx, load, stream_generate, make_sampler, tree_flatten


def _expected_shapes():
    # Qwen3-14B has 40 transformer blocks; adapt the final eight.
    template = {name: shape for name, shape in _rsi_shapes('14b').items() if '.layers.38.' in name}
    return {name.replace('.layers.38.', f'.layers.{layer}.'): shape
            for layer in range(32, 40) for name, shape in template.items()}


def _configuration(root):
    path = root / 'adapter_config.json'
    config = _json_file(path, _hash_file(path))
    parameters = config.get('lora_parameters')
    if (config.get('fine_tune_type') != 'lora' or type(config.get('num_layers')) is not int or
            config['num_layers'] != 8 or not isinstance(parameters, dict) or
            parameters != {'rank': 8, 'scale': 2.0, 'dropout': 0.0} or
            type(parameters.get('rank')) is not int or
            type(parameters.get('scale')) not in (int, float) or
            type(parameters.get('dropout')) not in (int, float)):
        raise ValueError('adapter configuration differs')


def _match_factors(mx, flatten, model, saved, shapes):
    import numpy as np
    if not isinstance(saved, dict) or set(saved) != set(shapes) or not shapes:
        raise ValueError('saved adapter tensors differ')
    factors = {}
    for name, value in flatten(model.parameters()):
        if name.endswith(('.lora_a', '.lora_b')):
            if name in factors:
                raise ValueError('duplicate loaded tensor')
            factors[name] = value
    if set(factors) != set(shapes):
        raise ValueError('loaded adapter tensors differ')
    for name, shape in shapes.items():
        mx.eval(saved[name], factors[name])
        stored, actual = np.asarray(saved[name]), np.asarray(factors[name])
        if (stored.dtype != np.float32 or actual.dtype != np.float32 or
                list(stored.shape) != shape or list(actual.shape) != shape or
                not np.isfinite(stored).all() or not np.isfinite(actual).all() or
                not np.array_equal(stored, actual)):
            raise ValueError('adapter tensor content differs')


class PhaseBBackend(MLXBackend):
    def __init__(self, model_path, *, model_manifest, adapter_state, adapter_path=None,
                 adapter_sha256=None, seed=20261111, max_tokens=4096, temperature=0.6):
        if adapter_state not in ('off', 'on'):
            raise ValueError('invalid adapter state')
        if adapter_state == 'off' and (adapter_path is not None or adapter_sha256 is not None):
            raise ValueError('adapter forbidden in off state')
        if adapter_state == 'on' and (adapter_path is None or adapter_sha256 is None):
            raise ValueError('adapter identity required')
        verify_model_manifest(model_path, model_manifest, model_key='14b')
        root = Path(adapter_path).expanduser().resolve() if adapter_state == 'on' else None
        if root is not None:
            verify_adapter_identity(root, adapter_sha256)
            _configuration(root)
        mx, load, stream, sampler, flatten = _runtime()
        mx.set_memory_limit(48 * 1024**3)
        kwargs = {'adapter_path': str(root)} if root is not None else {}
        model, tokenizer = load(str(model_path), **kwargs)
        if root is not None:
            saved = mx.load(str(root / 'adapters.safetensors'))
            _match_factors(mx, flatten, model, saved, _expected_shapes())
            verify_adapter_identity(root, adapter_sha256)
        elif any(name.endswith(('.lora_a', '.lora_b')) for name, _ in flatten(model.parameters())):
            raise ValueError('adapter tensors present in off state')
        verify_model_manifest(model_path, model_manifest, model_key='14b')
        self.mx, self.stream_generate, self.make_sampler = mx, stream, sampler
        self.model, self.tokenizer = model, tokenizer
        self.seed, self.max_tokens, self.temperature = seed, max_tokens, temperature
        self.raw = self.stats = None
        self.adapter_state, self.adapter_sha256 = adapter_state, adapter_sha256
