"""Compare caller-supplied loaded LoRA tensors against saved adapter files.

This is not independent process attestation and does not load model weights.
"""
from pathlib import Path

import numpy as np

from src.arc.rsi_adapter_factors import _read_factors
from src.arc.rsi_model_identity import verify_adapter_identity
from src.arc.rsi_adapter_structure import _expected_shapes


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
