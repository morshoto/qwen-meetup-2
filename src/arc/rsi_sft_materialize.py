"""Write checked MLX chat rows to a private, non-authorizing quarantine."""
import hashlib
import json
import os
import re
import stat

from src.arc.rsi_comparison_audit import _read
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


def verify_quarantined_dataset(*, root, preparation_bytes, expected_preparation_sha256,
                               expected_dataset_sha256, tasks, tokenizer, render):
    if (not isinstance(preparation_bytes, bytes) or not 1 <= len(preparation_bytes) <= 8 * 1024 * 1024 or
            not isinstance(expected_preparation_sha256, str) or
            not re.fullmatch(r'[0-9a-f]{64}', expected_preparation_sha256) or
            hashlib.sha256(preparation_bytes).hexdigest() != expected_preparation_sha256 or
            not isinstance(expected_dataset_sha256, str) or
            not re.fullmatch(r'[0-9a-f]{64}', expected_dataset_sha256)):
        raise ValueError('invalid expected dataset or preparation digest')
    checked = audit_sft_format(_json_object(preparation_bytes), tasks=tasks, tokenizer=tokenizer, render=render)
    expected = checked['dataset_jsonl'].encode('utf-8')
    if (len(expected) > 8 * 1024 * 1024 or hashlib.sha256(expected).hexdigest() != expected_dataset_sha256 or
            checked['dataset_jsonl_sha256'] != expected_dataset_sha256):
        raise ValueError('expected dataset digest differs')
    expected_audit = {key: value for key, value in checked.items() if key != 'dataset_jsonl'}
    expected_audit.update({'status': 'dataset_quarantined', 'preparation_sha256': expected_preparation_sha256,
                           'training_allowed': False})
    try:
        with _directory(root) as parent:
            directory = os.open('generation-sft.quarantine', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                info = os.fstat(directory)
                if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                    raise ValueError('quarantine directory must be owner-only')
                if set(os.listdir(directory)) != {'train.jsonl', 'audit.json'}:
                    raise ValueError('quarantine directory contents differ')
                fd = os.open('train.jsonl', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                with os.fdopen(fd, 'rb') as stream:
                    info = os.fstat(stream.fileno())
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
                            stat.S_IMODE(info.st_mode) != 0o600):
                        raise ValueError('dataset must be owner-only')
                    actual = stream.read(8 * 1024 * 1024 + 1)
                _, audit = _read(directory, 'audit.json')
                if actual != expected or audit != expected_audit:
                    raise ValueError('quarantined dataset or audit differs')
            finally:
                os.close(directory)
    except OSError as exc:
        raise ValueError('quarantined dataset unavailable') from exc
    return {'status': 'dataset_consistent_unverified', 'training_allowed': False,
            'row_count': checked['row_count'], 'dataset_jsonl_sha256': expected_dataset_sha256,
            'evidence_type': 'caller_supplied_tokenizer_and_renderer'}
