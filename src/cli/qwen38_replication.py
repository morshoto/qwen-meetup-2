"""Guard a separately registered, supervised replication run."""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_replication_freeze import verify_pinned_files
from src.arc.qwen38_weight_allocation import freeze_registration


def verify_runtime(repo, model_path, execution):
    """Recheck pinned code, installed packages and model file contents."""
    verify_pinned_files(repo, execution['source_files_sha256'])
    if {name: metadata.version(name) for name in execution['packages']} != execution['packages']:
        raise ValueError('runtime package versions changed')
    model = Path(model_path)
    if not model.is_dir(): raise ValueError('model snapshot missing')
    files = {}
    for path in model.iterdir():
        if not path.is_file(): raise ValueError('unexpected model snapshot entry')
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                value.update(block)
        files[path.name] = value.hexdigest()
    if files != execution['model']['files_sha256']:
        raise ValueError('model snapshot contents changed')
    return {'status': 'runtime_matches'}


def _registration(root, rebuild):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077 or not callable(rebuild):
        raise ValueError('replication run directory must be private')
    descriptor = rebuild()
    if (not isinstance(descriptor, dict) or descriptor.get('status') != 'evaluation_ready' or
        descriptor.get('inference_authorized') is not True or descriptor.get('run_dir') != str(root.resolve()) or
        descriptor.get('sha256') != digest({k: v for k, v in descriptor.items() if k != 'sha256'})):
        raise ValueError('replayed replication registration is not ready')
    return descriptor


def freeze_replication(root, rebuild):
    descriptor = _registration(root, rebuild)
    path = Path(root) / 'registration.json'
    if path.is_symlink(): raise ValueError('registration cannot be linked')
    return freeze_registration(path, {k: v for k, v in descriptor.items() if k != 'sha256'})


def verify_replication(root, rebuild):
    descriptor = _registration(root, rebuild)
    path = Path(root) / 'registration.json'
    if path.is_symlink() or not path.is_file() or json.loads(path.read_bytes()) != descriptor:
        raise ValueError('registered replication sources changed')
    return descriptor
