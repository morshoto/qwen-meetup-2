"""Compare local LoRA factors without loading a model or proving improvement."""
import os
import struct
from pathlib import Path

import numpy as np

from src.arc.rsi_adapter_structure import _expected_shapes, inspect_adapter_structure
from src.arc.rsi_model_identity import _signature, verify_adapter_identity
from src.arc.rsi_transport_audit import _json_object


def _read_factors(root, expected):
    inspect_adapter_structure(root, expected_sha256=expected)
    try:
        fd = os.open(root / 'adapters.safetensors', os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            raw = stream.read(64 * 1024 * 1024 + 1)
            if len(raw) != before.st_size or _signature(os.fstat(stream.fileno())) != _signature(before):
                raise ValueError('adapter changed during factor read')
        length = struct.unpack('<Q', raw[:8])[0]
        header = _json_object(raw[8:8 + length])
        data = memoryview(raw)[8 + length:]
        factors = {}
        for name in _expected_shapes():
            start, end = header[name]['data_offsets']
            factors[name] = np.frombuffer(data[start:end], dtype='<f4')
    except (OSError, ValueError, KeyError, struct.error) as exc:
        raise ValueError('adapter factor data unavailable') from exc
    verify_adapter_identity(root, expected)
    return factors


def compare_adapter_factors(candidate_dir, *, expected_candidate_sha256, parent_dir=None,
                            expected_parent_sha256=None):
    if (parent_dir is None) != (expected_parent_sha256 is None):
        raise ValueError('parent adapter and digest must be supplied together')
    candidate = _read_factors(Path(candidate_dir).expanduser().resolve(), expected_candidate_sha256)
    parent = (_read_factors(Path(parent_dir).expanduser().resolve(), expected_parent_sha256)
              if parent_dir is not None else None)
    pair_count = sum(bool(np.any(value != 0)) and bool(np.any(candidate[name[:-1] + 'b'] != 0))
                     for name, value in candidate.items() if name.endswith('.lora_a'))
    changed = (sum(not np.array_equal(value, parent[name]) for name, value in candidate.items())
               if parent is not None else None)
    rejected = pair_count == 0 or (parent is not None and changed == 0)
    return {'status': 'adapter_factors_rejected' if rejected else 'adapter_factors_changed_unverified',
            'training_allowed': False, 'effective_update_verified': False, 'nonzero_factor_pair_count': pair_count,
            'changed_tensor_count': changed, 'candidate_sha256': expected_candidate_sha256,
            'parent_sha256': expected_parent_sha256, 'evidence_type': 'local_adapter_files_and_caller_digests'}
