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
