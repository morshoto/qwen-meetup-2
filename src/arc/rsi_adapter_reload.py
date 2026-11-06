"""Compare caller-supplied loaded LoRA tensors against saved adapter files.

This is not independent process attestation and does not load model weights.
"""
from pathlib import Path

import numpy as np

from src.arc.rsi_adapter_factors import _read_factors
from src.arc.rsi_model_identity import verify_adapter_identity, verify_model_manifest
from src.arc.rsi_adapter_structure import _expected_shapes, inspect_adapter_structure


def compare_reloaded_factors(adapter_dir, *, expected_sha256, model_key, loaded_factors):
    root = Path(adapter_dir).expanduser().resolve()
    saved = _read_factors(root, expected_sha256, model_key)
    if not isinstance(loaded_factors, dict) or set(loaded_factors) != set(saved):
        raise ValueError('loaded adapter tensor names differ')
    shapes = _expected_shapes(model_key)
    for name, expected in saved.items():
        actual = np.asarray(loaded_factors[name])
        if actual.dtype != np.dtype('float32') or list(actual.shape) != shapes[name]:
            raise ValueError('loaded adapter tensor type or size differs')
        if not np.array_equal(actual.reshape(-1), expected):
            raise ValueError('loaded adapter tensor values differ')
    verify_adapter_identity(root, expected_sha256)
    return {'status': 'loaded_adapter_factors_match_unverified', 'model_key': model_key,
            'adapter_sha256': expected_sha256, 'matched_tensor_count': len(saved),
            'reload_verified': False, 'training_allowed': False, 'promotion_allowed': False,
            'evidence_type': 'caller_supplied_loaded_tensors_and_local_adapter_files'}


def observe_model_reload(*, model_key, model_dir, adapter_dir, expected_adapter_sha256, model_manifest,
                         load_model, flatten_parameters, evaluate, to_numpy):
    """Observe one supplied model load; independent process provenance remains external."""
    if any(not callable(item) for item in (load_model, flatten_parameters, evaluate, to_numpy)):
        raise ValueError('reload probe runtime unavailable')
    verify_model_manifest(model_dir, model_manifest, model_key=model_key)
    inspect_adapter_structure(adapter_dir, expected_sha256=expected_adapter_sha256, model_key=model_key)
    model, _tokenizer = load_model(str(model_dir), adapter_path=str(adapter_dir),
                                   tokenizer_config={'trust_remote_code': False})
    loaded = {}
    for name, value in flatten_parameters(model.parameters()):
        if not isinstance(name, str):
            raise ValueError('loaded parameter name invalid')
        if name.endswith(('.lora_a', '.lora_b')):
            if name in loaded:
                raise ValueError('duplicate loaded adapter tensor')
            evaluate(value)
            loaded[name] = to_numpy(value)
    if not loaded:
        raise ValueError('loaded model has no adapter tensors')
    checked = compare_reloaded_factors(adapter_dir, expected_sha256=expected_adapter_sha256,
                                      model_key=model_key, loaded_factors=loaded)
    verify_model_manifest(model_dir, model_manifest, model_key=model_key)
    verify_adapter_identity(adapter_dir, expected_adapter_sha256)
    return {'status': 'model_reload_observed_unverified', 'model_key': model_key,
            'matched_tensor_count': checked['matched_tensor_count'], 'adapter_sha256': expected_adapter_sha256,
            'reload_verified': False, 'training_allowed': False, 'promotion_allowed': False,
            'evidence_type': 'caller_supplied_runtime_and_local_files'}
