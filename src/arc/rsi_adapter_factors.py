"""Compare local LoRA factors without loading a model or proving improvement."""
import os
import math
import struct
from pathlib import Path

import numpy as np

from src.arc.rsi_adapter_structure import _expected_shapes, inspect_adapter_structure
from src.arc.rsi_model_identity import _signature, verify_adapter_identity
from src.arc.rsi_transport_audit import _json_object


def _read_factors(root, expected, model_key):
    inspect_adapter_structure(root, expected_sha256=expected, model_key=model_key)
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
        for name in _expected_shapes(model_key):
            start, end = header[name]['data_offsets']
            factors[name] = np.frombuffer(data[start:end], dtype='<f4')
    except (OSError, ValueError, KeyError, struct.error) as exc:
        raise ValueError('adapter factor data unavailable') from exc
    verify_adapter_identity(root, expected)
    return factors


def compare_adapter_factors(candidate_dir, *, expected_candidate_sha256, parent_dir=None,
                            expected_parent_sha256=None, model_key='14b'):
    if (parent_dir is None) != (expected_parent_sha256 is None):
        raise ValueError('parent adapter and digest must be supplied together')
    candidate = _read_factors(Path(candidate_dir).expanduser().resolve(), expected_candidate_sha256, model_key)
    parent = (_read_factors(Path(parent_dir).expanduser().resolve(), expected_parent_sha256, model_key)
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


def probe_adapter_products(candidate_dir, *, expected_candidate_sha256, parent_dir=None,
                           expected_parent_sha256=None, model_key='14b'):
    """Find bounded file-level witnesses; failure to find one is inconclusive."""
    if (parent_dir is None) != (expected_parent_sha256 is None):
        raise ValueError('parent adapter and digest must be supplied together')
    candidate = _read_factors(Path(candidate_dir).expanduser().resolve(), expected_candidate_sha256, model_key)
    parent = (_read_factors(Path(parent_dir).expanduser().resolve(), expected_parent_sha256, model_key)
              if parent_dir is not None else None)
    shapes = _expected_shapes(model_key)
    candidate_witness, parent_witness = None, None
    checked = 0
    for name in sorted(key for key in shapes if key.endswith('.lora_a')):
        b_name = name[:-1] + 'b'
        a = candidate[name].reshape(shapes[name])
        b = candidate[b_name].reshape(shapes[b_name])
        pa = parent[name].reshape(shapes[name]) if parent is not None else None
        pb = parent[b_name].reshape(shapes[b_name]) if parent is not None else None
        rows = np.any(a != 0, axis=1)
        cols = np.any(b != 0, axis=0)
        if parent is not None:
            rows |= np.any(pa != 0, axis=1)
            cols |= np.any(pb != 0, axis=0)
        for row in np.flatnonzero(rows)[:8]:
            for col in np.flatnonzero(cols)[:8]:
                current_terms = [float(a[row, k]) * float(b[k, col]) for k in range(8)]
                current = math.fsum(current_terms)
                checked += 1
                if current != 0 and candidate_witness is None:
                    candidate_witness = {'module': name[:-7], 'row': int(row), 'column': int(col), 'value': current}
                if parent is not None:
                    previous_terms = [float(pa[row, k]) * float(pb[k, col]) for k in range(8)]
                    difference = math.fsum(current_terms + [-value for value in previous_terms])
                    if difference != 0 and parent_witness is None:
                        parent_witness = {'module': name[:-7], 'row': int(row), 'column': int(col),
                                          'difference': difference}
    verify_adapter_identity(Path(candidate_dir).expanduser().resolve(), expected_candidate_sha256)
    if parent is not None:
        verify_adapter_identity(Path(parent_dir).expanduser().resolve(), expected_parent_sha256)
    return {'status': 'bounded_adapter_product_probe_unverified', 'training_allowed': False,
            'reload_verified': False, 'checked_entries': checked,
            'candidate_product': 'nonzero_witness' if candidate_witness else 'inconclusive',
            'candidate_witness': candidate_witness,
            'parent_difference': ('not_applicable' if parent is None else
                                  'nonzero_witness' if parent_witness else 'inconclusive'),
            'parent_witness': parent_witness, 'candidate_sha256': expected_candidate_sha256,
            'parent_sha256': expected_parent_sha256, 'evidence_type': 'local_adapter_files_and_caller_digests'}
