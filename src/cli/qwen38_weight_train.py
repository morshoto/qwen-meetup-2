"""Guarded training helpers for source-replayed exploratory LoRA data."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


def materialize_trainer_dataset(bundle, training_root, expected_sha256):
    """Copy only the previously replayed JSONL into a new private dataset dir."""
    training_root, bundle = Path(training_root).resolve(), Path(bundle).resolve()
    source = bundle / 'train.jsonl'
    if (not training_root.is_dir() or training_root.stat().st_mode & 0o077 or
        bundle != training_root / 'sft-v1' or not bundle.is_dir() or bundle.stat().st_mode & 0o077 or source.is_symlink() or
        not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None):
        raise ValueError('invalid private SFT source')
    data = source.read_bytes()
    if not data or len(data) > 16 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('SFT dataset hash mismatch')
    target = training_root / 'trainer-data-v1'
    target.mkdir(mode=0o700, exist_ok=False)
    fd = os.open(target / 'train.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    dir_fd = os.open(target, os.O_RDONLY)
    try: os.fsync(dir_fd)
    finally: os.close(dir_fd)
    return target


from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_supervisor import run_supervised
from src.cli.qwen38_weight_prepare_sft import replay_bundle


def _write_decision(path, value):
    import json
    payload = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def execute_training(**kwargs):
    root = Path(kwargs['run_dir']).resolve()
    manifest = kwargs['manifest']
    if manifest.get('run_dir') != str(root):
        raise ValueError('training run does not match the frozen collection')
    selection, registration, replay = replay_bundle(**kwargs)
    if replay.get('status') != 'matches_supplied_replay':
        raise ValueError('SFT source replay failed')
    training = root / 'training'
    if not training.is_dir() or training.is_symlink() or training.stat().st_mode & 0o077:
        raise ValueError('training directory is not private')
    if selection['status'] == 'insufficient_verified_data':
        result = {'status': 'insufficient_verified_data', 'mode': selection['mode'], 'counts': selection['counts'],
                  'execution_sha256': manifest['sha256'], 'sft_manifest_sha256': replay['manifest_sha256']}
        result['sha256'] = digest(result)
        _write_decision(training / 'training-decision-v1.json', result)
        return result
    raise NotImplementedError('source-replayed real training controller is not complete')
