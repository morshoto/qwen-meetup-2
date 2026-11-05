"""Write checked MLX chat rows to a private, non-authorizing quarantine."""
import hashlib
import json
import os
import re

from src.arc.rsi_sft_format import audit_sft_format
from src.arc.rsi_slot_lifecycle import _create, _directory
from src.arc.rsi_transport_audit import _json_object


def _write_dataset(directory, raw):
    fd = os.open('train.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(directory)


def materialize_quarantined_dataset(*, root, preparation_bytes, expected_preparation_sha256,
                                    tasks, tokenizer, render):
    if (not isinstance(preparation_bytes, bytes) or not 1 <= len(preparation_bytes) <= 8 * 1024 * 1024 or
            not isinstance(expected_preparation_sha256, str) or
            not re.fullmatch(r'[0-9a-f]{64}', expected_preparation_sha256) or
            hashlib.sha256(preparation_bytes).hexdigest() != expected_preparation_sha256):
        raise ValueError('training preparation digest differs')
    with _directory(root):
        pass
    plan = _json_object(preparation_bytes)
    checked = audit_sft_format(plan, tasks=tasks, tokenizer=tokenizer, render=render)
    raw = checked['dataset_jsonl'].encode('utf-8')
    if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != checked['dataset_jsonl_sha256']:
        raise ValueError('checked dataset digest differs')
    audit = {key: value for key, value in checked.items() if key != 'dataset_jsonl'}
    audit.update({'status': 'dataset_quarantined', 'preparation_sha256': expected_preparation_sha256,
                  'training_allowed': False})
    with _directory(root) as parent:
        os.mkdir('generation-sft.quarantine', 0o700, dir_fd=parent)
        os.fsync(parent)
        directory = os.open('generation-sft.quarantine', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            os.fchmod(directory, 0o700)
            _write_dataset(directory, raw)
            _create(directory, 'audit.json', audit)
        finally:
            os.close(directory)
    return {'status': 'dataset_quarantined', 'training_allowed': False,
            'row_count': audit['row_count'], 'preparation_sha256': expected_preparation_sha256,
            'dataset_jsonl_sha256': audit['dataset_jsonl_sha256']}
