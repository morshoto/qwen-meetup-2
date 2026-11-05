"""Hash local model and adapter files without claiming server attestation."""
import hashlib
import json
import os
import re
import stat
from pathlib import Path

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _hash_file(path, aggregate=None):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('artifact must be a regular file')
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
                if aggregate is not None:
                    aggregate.update(block)
            after = os.fstat(stream.fileno())
        if _signature(before) != _signature(after) or _signature(after) != _signature(os.stat(path, follow_symlinks=False)):
            raise ValueError('artifact changed while hashing')
        return digest.hexdigest()
    except OSError as exc:
        raise ValueError('artifact file unavailable') from exc


def _json_file(path, expected):
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('artifact changed while parsing')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate artifact key')
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError('artifact JSON invalid or changed') from exc
    if not isinstance(value, dict):
        raise ValueError('artifact JSON must be an object')
    return value


def snapshot_model_manifest(model_dir):
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir() or root.name != _REVISION or root.parent.name != 'snapshots':
        raise ValueError('model must be the proposed local snapshot')
    repo = root.parent.parent
    names = sorted(item.name for item in root.iterdir())
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    if not required <= set(names) or not names:
        raise ValueError('model snapshot is incomplete')
    hashes, paths = {}, {}
    for name in names:
        item = root / name
        try:
            target = item.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError('snapshot file unavailable') from exc
        shared_cache = (repo.name == 'models--mlx-community--Qwen3-14B-4bit' and
                        repo.parent.name == 'hub' and target.is_relative_to(repo.parent / 'blobs'))
        if not target.is_relative_to(root) and not target.is_relative_to(repo / 'blobs') and not shared_cache:
            raise ValueError('snapshot file escapes its repository')
        paths[name] = target
        hashes[name] = _hash_file(target)
    if sorted(item.name for item in root.iterdir()) != names:
        raise ValueError('snapshot file set changed')
    config = _json_file(paths['config.json'], hashes['config.json'])
    if config.get('model_type') != 'qwen3' or config.get('quantization') != {'bits': 4, 'group_size': 64}:
        raise ValueError('model quantization differs')
    if any(type(value) is not int for value in config['quantization'].values()):
        raise ValueError('invalid quantization types')
    index = _json_file(paths['model.safetensors.index.json'], hashes['model.safetensors.index.json'])
    weight_map = index.get('weight_map')
    if (not isinstance(weight_map, dict) or not weight_map or
            any(not isinstance(value, str) for value in weight_map.values()) or
            set(weight_map.values()) != {name for name in names if name.endswith('.safetensors')}):
        raise ValueError('weight index differs from snapshot')
    tokenizer = _json_file(paths['tokenizer_config.json'], hashes['tokenizer_config.json'])
    template = tokenizer.get('chat_template')
    if not isinstance(template, str) or not template:
        raise ValueError('chat template unavailable')
    for name in names:
        if (root / name).resolve(strict=True) != paths[name]:
            raise ValueError('snapshot link changed')
    return {'model_id': _MODEL, 'revision': _REVISION, 'quantization': {'bits': 4, 'group_size': 64},
            'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(), 'files_sha256': hashes}


def verify_model_manifest(model_dir, expected):
    if not isinstance(expected, dict) or snapshot_model_manifest(model_dir) != expected:
        raise ValueError('model snapshot differs from expected manifest')
    return {'status': 'local_model_files_consistent', 'execution_ready': False,
            'evidence_type': 'local_files_and_caller_supplied_manifest'}


def verify_adapter_identity(adapter_dir, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('invalid expected adapter digest')
    root = Path(adapter_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('adapter directory unavailable')
    digest = hashlib.sha256()
    for name in ('adapter_config.json', 'adapters.safetensors'):
        path = root / name
        if path.is_symlink():
            raise ValueError('adapter file may not be a symlink')
        digest.update(name.encode() + b'\0')
        _hash_file(path, digest)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('adapter digest differs')
    return {'status': 'local_adapter_files_consistent', 'execution_ready': False,
            'evidence_type': 'local_files_and_caller_supplied_digest'}
