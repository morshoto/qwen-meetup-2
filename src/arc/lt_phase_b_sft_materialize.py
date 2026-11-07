"""Create private, verified SFT files without starting or authorizing training.

The caller must separately bind the verified source, allocation, runtime,
canary and frozen training registration. Interrupted output is retained and
must be reconciled; this function never overwrites it.
"""
import copy
import fcntl
import hashlib
import os
import stat

from src.arc.lt_phase_b_dispatch import _private_run_directory
from src.arc.lt_phase_b_sft_preflight import _json, audit_sft_dataset


def _write(directory_fd, name, raw):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def materialize_sft(*, source_root, run_dir, dataset, tasks, verifier, source, tokenizer, dataset_factory, _locked_fd=None):
    _, directory = _private_run_directory(source_root, run_dir)
    if not isinstance(source, dict) or directory.name != 'lt-phase-b-' + str(source.get('run_id')):
        raise ValueError('run directory identity differs')
    owns_lock = _locked_fd is None
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600) if owns_lock else _locked_fd
    try:
        held, current = os.fstat(lockfd), os.stat(directory / 'run.lock', follow_symlinks=False)
        if not stat.S_ISREG(held.st_mode) or (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError('run lock identity differs')
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_data, frozen_source = copy.deepcopy(dataset), copy.deepcopy(source)
        report = audit_sft_dataset(dataset=frozen_data, tasks=tasks, verifier=verifier, source=frozen_source,
                                   tokenizer=tokenizer, dataset_factory=dataset_factory)
        raw = {name: ''.join(_json(example) + '\n' for example in frozen_data[name]).encode()
               for name in ('train', 'valid')}
        for name in raw:
            if hashlib.sha256(raw[name]).hexdigest() != report[name + '_jsonl_sha256']:
                raise ValueError('dataset bytes differ from token audit')
        lineages = {item['task_id']: item['solution_lineage_sha256'] for item in frozen_data['journal']
                    if item.get('verification_status') == 'eligible'}
        manifest = {'status': 'private_sft_materialized_unverified', 'training_allowed': False,
            'yield_status': frozen_data['yield_status'], 'source': frozen_source,
            'split_task_ids': frozen_data['split_task_ids'], 'solution_lineage_sha256': lineages,
            'train_count': report['train_count'], 'valid_count': report['valid_count'],
            'maximum_tokens': report['maximum_tokens'], 'train_jsonl_sha256': report['train_jsonl_sha256'],
            'valid_jsonl_sha256': report['valid_jsonl_sha256'] if report['valid_count'] else None,
            'token_audit_status': report['status']}
        runfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(runfd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError('run directory changed')
            os.mkdir('sft-data', 0o700, dir_fd=runfd)
            os.fsync(runfd)
            datafd = os.open('sft-data', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=runfd)
            try:
                os.fchmod(datafd, 0o700)
                _write(datafd, 'train.jsonl', raw['train'])
                if report['valid_count']:
                    _write(datafd, 'valid.jsonl', raw['valid'])
                _write(datafd, 'manifest.json', (_json(manifest) + '\n').encode())
                os.fsync(datafd)
            finally:
                os.close(datafd)
        finally:
            os.close(runfd)
        return manifest
    finally:
        if owns_lock: os.close(lockfd)
