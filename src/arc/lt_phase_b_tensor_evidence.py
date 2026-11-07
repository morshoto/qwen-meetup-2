"""Hash every base and trainable tensor around a synthetic canary step.

Caller-supplied model and runtime observations are not independent
attestation, a saved-adapter reload, or evidence of ARC improvement.
"""
import hashlib
import json
import re

import numpy as np

from src.arc.lt_phase_b_backend import _expected_shapes


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _metadata(value, evaluate, *, adapter):
    evaluate(value)
    if str(getattr(value, 'dtype', '')) == 'mlx.core.bfloat16':
        import mlx.core as mx
        if not isinstance(value, mx.array) or value.dtype != mx.bfloat16 or adapter:
            raise ValueError('unsupported bfloat16 tensor')
        # NumPy cannot view MLX BF16 directly; uint16 preserves the exact bits.
        array = np.asarray(value.view(mx.uint16))
        dtype = 'bfloat16'
    else:
        array = np.asarray(value)
        dtype = str(array.dtype)
    if array.dtype.hasobject or (adapter and (array.dtype != np.float32 or not np.isfinite(array).all())):
        raise ValueError('unsupported or nonfinite tensor')
    contiguous = np.ascontiguousarray(array)
    return {'shape': list(array.shape), 'dtype': dtype,
            'sha256': hashlib.sha256(memoryview(contiguous).cast('B')).hexdigest()}


def _flatten(tree, flatten):
    values = {}
    for name, value in flatten(tree):
        if not isinstance(name, str) or not name or name in values:
            raise ValueError('invalid or duplicate tensor name')
        values[name] = value
    return values


def capture_tensor_state(model, *, flatten, evaluate):
    if not callable(flatten) or not callable(evaluate):
        raise ValueError('tensor inspection runtime unavailable')
    params = _flatten(model.parameters(), flatten)
    trainable = _flatten(model.trainable_parameters(), flatten)
    expected = _expected_shapes()
    if not expected or set(trainable) != set(expected) or not set(trainable) <= set(params):
        raise ValueError('trainable tensor names differ')
    base, adapter = {}, {}
    for name, value in params.items():
        if name in expected:
            item = _metadata(value, evaluate, adapter=True)
            if item['shape'] != expected[name] or _metadata(trainable[name], evaluate, adapter=True) != item:
                raise ValueError('trainable tensor content or shape differs')
            adapter[name] = item
        else:
            if name.endswith(('.lora_a', '.lora_b')):
                raise ValueError('unexpected adapter tensor')
            base[name] = _metadata(value, evaluate, adapter=False)
    if not base:
        raise ValueError('base tensors missing')
    return {'base': base, 'adapter': adapter, 'base_sha256': _digest(base), 'adapter_sha256': _digest(adapter)}


def _validate(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != {'base', 'adapter', 'base_sha256', 'adapter_sha256'}:
        raise ValueError('invalid tensor snapshot')
    base, adapter = snapshot['base'], snapshot['adapter']
    if (not isinstance(base, dict) or not base or not isinstance(adapter, dict) or
            set(adapter) != set(_expected_shapes()) or set(base) & set(adapter)):
        raise ValueError('invalid tensor snapshot names')
    for name, item in list(base.items()) + list(adapter.items()):
        if (not isinstance(name, str) or not name or not isinstance(item, dict) or
                set(item) != {'shape', 'dtype', 'sha256'} or not isinstance(item['shape'], list) or
                any(type(n) is not int or n < 0 for n in item['shape']) or
                not isinstance(item['dtype'], str) or not item['dtype'] or
                not isinstance(item['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
            raise ValueError('invalid tensor metadata')
        if name in adapter and (item['shape'] != _expected_shapes()[name] or item['dtype'] != 'float32'):
            raise ValueError('adapter tensor metadata differs')
    if snapshot['base_sha256'] != _digest(base) or snapshot['adapter_sha256'] != _digest(adapter):
        raise ValueError('tensor snapshot digest differs')


def compare_tensor_states(before, after):
    _validate(before)
    _validate(after)
    if before['base'] != after['base']:
        raise ValueError('base tensor changed')
    changed = 0
    for name, first in before['adapter'].items():
        last = after['adapter'][name]
        if first['shape'] != last['shape'] or first['dtype'] != last['dtype']:
            raise ValueError('adapter tensor metadata changed')
        changed += first['sha256'] != last['sha256']
    if not changed:
        raise ValueError('adapter tensors did not change')
    return {'status': 'tensor_change_consistent_unverified', 'training_verified': False,
            'base_tensor_count': len(before['base']), 'changed_adapter_tensor_count': changed,
            'base_sha256': before['base_sha256'], 'initial_adapter_sha256': before['adapter_sha256'],
            'final_adapter_sha256': after['adapter_sha256'],
            'evidence_type': 'caller_supplied_model_and_tensor_runtime'}
