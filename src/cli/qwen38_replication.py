"""Guard a separately registered, supervised replication run."""
from __future__ import annotations

import json
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import freeze_registration


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
