"""Inspect and hash a local eight-layer canary adapter without loading it.

File consistency is not training proof, independent reload, or promotion.
"""
import hashlib
import os
import stat
import struct
from pathlib import Path

from src.arc.lt_phase_b_backend import _check_header, _configuration, _expected_shapes
from src.arc.rsi_model_identity import _hash_file, _signature, verify_adapter_identity


def inspect_adapter_artifact(adapter_dir):
    root = Path(adapter_dir)
    if root.is_symlink():
        raise ValueError('adapter directory may not be a symlink')
    root = root.resolve(strict=True)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError('invalid adapter directory')
    aggregate = hashlib.sha256()
    hashes = {}
    for name in ('adapter_config.json', 'adapters.safetensors'):
        aggregate.update(name.encode() + b'\0')
        hashes[name] = _hash_file(root / name, aggregate)
    digest = aggregate.hexdigest()
    _configuration(root)
    _check_header(root / 'adapters.safetensors', _expected_shapes())
    try:
        fd = os.open(root / 'adapters.safetensors', os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            length = struct.unpack('<Q', stream.read(8))[0]
            stream.seek(8 + length)
            remaining = before.st_size - 8 - length
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block or len(block) % 4:
                    raise ValueError('adapter tensor data truncated')
                if any(value & 0x7f800000 == 0x7f800000 for (value,) in struct.iter_unpack('<I', block)):
                    raise ValueError('adapter contains nonfinite tensors')
                remaining -= len(block)
            if _signature(os.fstat(stream.fileno())) != _signature(before):
                raise ValueError('adapter changed during inspection')
    except (OSError, struct.error) as exc:
        raise ValueError('adapter tensor data unavailable') from exc
    verify_adapter_identity(root, digest)
    return {'status': 'adapter_files_consistent_unverified', 'adapter_sha256': digest,
            'weights_sha256': hashes['adapters.safetensors'], 'config_sha256': hashes['adapter_config.json'],
            'reload_verified': False, 'promotion_allowed': False}
