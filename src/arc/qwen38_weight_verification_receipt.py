"""Bind offline outcomes to exact journal contents and frozen dataset hashes."""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_fresh_verify import verify_attempts


def build_receipt(manifest, challenges_bytes, solutions_bytes, rows):
    result = verify_attempts(manifest, challenges_bytes, solutions_bytes, rows)
    outcomes = [{'task_id': task_id, 'candidate_index': index, 'correct': correct}
                for (task_id, index), correct in sorted(result['outcomes'].items())]
    receipt = {'protocol': 'qwen38-weight-offline-verification-v1', 'manifest_sha256': manifest['sha256'],
               'challenge_sha256': manifest['input_hashes']['challenges'],
               'solution_sha256': manifest['input_hashes']['solutions'], 'journal_sha256': digest(rows),
               'counts': result['counts'], 'outcomes': outcomes}
    receipt['sha256'] = digest(receipt)
    return receipt


def verify_receipt(receipt, manifest, challenges_bytes, solutions_bytes, rows):
    expected = build_receipt(manifest, challenges_bytes, solutions_bytes, rows)
    if receipt != expected:
        raise ValueError('offline verification receipt differs from replay')
    return {(entry['task_id'], entry['candidate_index']): entry['correct'] for entry in expected['outcomes']}


def write_receipt(path, receipt):
    path = Path(path)
    if path.parent.stat().st_mode & 0o077:
        raise ValueError('verification receipt directory must be private')
    payload = (json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(parent)
    finally: os.close(parent)
