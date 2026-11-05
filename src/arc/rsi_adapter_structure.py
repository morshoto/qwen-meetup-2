"""Read-only structural precheck for proposed Qwen3 MLX adapters."""
import os
import stat
import struct
from pathlib import Path

from src.arc.rsi_model_identity import _hash_file, _json_file, _signature, verify_adapter_identity
from src.arc.rsi_transport_audit import _json_object


def _expected_shapes(model_key='14b'):
    if model_key not in ('14b', '32b'):
        raise ValueError('unsupported adapter model')
    dimensions = {'self_attn.q_proj': (5120, 5120), 'self_attn.k_proj': (5120, 1024),
                  'self_attn.v_proj': (5120, 1024), 'self_attn.o_proj': (5120, 5120),
                  'mlp.gate_proj': (5120, 17408), 'mlp.up_proj': (5120, 17408),
                  'mlp.down_proj': (17408, 5120)}
    if model_key == '32b':
        dimensions.update({'self_attn.q_proj': (5120, 8192), 'self_attn.o_proj': (8192, 5120),
                           'mlp.gate_proj': (5120, 25600), 'mlp.up_proj': (5120, 25600),
                           'mlp.down_proj': (25600, 5120)})
    result = {}
    for layer in ((38, 39) if model_key == '14b' else (62, 63)):
        for name, (input_size, output_size) in dimensions.items():
            prefix = f'model.layers.{layer}.{name}'
            result[prefix + '.lora_a'] = [input_size, 8]
            result[prefix + '.lora_b'] = [8, output_size]
    return result


def inspect_adapter_structure(adapter_dir, *, expected_sha256, model_key='14b'):
    shapes = _expected_shapes(model_key)
    root = Path(adapter_dir).expanduser().resolve()
    verify_adapter_identity(root, expected_sha256)
    config_path, weights_path = root / 'adapter_config.json', root / 'adapters.safetensors'
    config = _json_file(config_path, _hash_file(config_path))
    if (config.get('fine_tune_type') != 'lora' or type(config.get('num_layers')) is not int or
            config['num_layers'] != 2 or config.get('lora_parameters') != {'rank': 8, 'scale': 2.0, 'dropout': 0.0} or
            type(config['lora_parameters'].get('rank')) is not int or
            type(config['lora_parameters'].get('scale')) not in (int, float) or
            type(config['lora_parameters'].get('dropout')) not in (int, float)):
        raise ValueError('adapter configuration differs from proposed LoRA')
    try:
        fd = os.open(weights_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 8 < before.st_size <= 64 * 1024 * 1024:
                raise ValueError('invalid adapter weights file')
            length_bytes = stream.read(8)
            length = struct.unpack('<Q', length_bytes)[0]
            if not 2 <= length <= 1024 * 1024 or 8 + length > before.st_size:
                raise ValueError('invalid adapter header length')
            header = _json_object(stream.read(length))
            metadata = header.pop('__metadata__', {})
            if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                     for k, v in metadata.items()):
                raise ValueError('invalid adapter metadata')
            if set(header) != set(shapes):
                raise ValueError('adapter tensor names differ')
            ranges = []
            for name, shape in shapes.items():
                entry = header[name]
                if (not isinstance(entry, dict) or set(entry) != {'dtype', 'shape', 'data_offsets'} or
                        entry['dtype'] != 'F32' or entry['shape'] != shape or
                        any(type(item) is not int for item in entry['shape']) or
                        not isinstance(entry['data_offsets'], list) or len(entry['data_offsets']) != 2 or
                        any(type(item) is not int for item in entry['data_offsets'])):
                    raise ValueError('adapter tensor structure differs')
                start, end = entry['data_offsets']
                if start < 0 or end - start != shape[0] * shape[1] * 4:
                    raise ValueError('invalid adapter tensor extent')
                ranges.append((start, end))
            position = 0
            for start, end in sorted(ranges):
                if start != position:
                    raise ValueError('adapter tensor extents overlap or contain gaps')
                position = end
            if position != before.st_size - 8 - length:
                raise ValueError('adapter tensor data length differs')
            remaining = position
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block or len(block) % 4:
                    raise ValueError('adapter tensor data truncated')
                if any(value & 0x7f800000 == 0x7f800000 for (value,) in struct.iter_unpack('<I', block)):
                    raise ValueError('adapter contains nonfinite tensor values')
                remaining -= len(block)
            if _signature(os.fstat(stream.fileno())) != _signature(before):
                raise ValueError('adapter weights changed during check')
    except (OSError, struct.error) as exc:
        raise ValueError('adapter weights unavailable') from exc
    verify_adapter_identity(root, expected_sha256)
    return {'status': 'adapter_structure_consistent_unverified', 'training_allowed': False,
            'reload_verified': False, 'tensor_count': len(shapes), 'adapter_sha256': expected_sha256,
            'evidence_type': 'local_adapter_files_and_caller_digest'}
