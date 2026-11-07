"""Dispatch a preregistered public collection through a bounded local journal.

Preflight establishes local consistency, not independent registration or
isolation. This function never reads solution files or official evaluation
answers; escaped descendants and system-owned services remain unproven.
"""
import fcntl
import os
import stat
import zipfile
from pathlib import Path

from src.arc.lt_batch_journal import BatchJournal
from src.arc.lt_phase_b_collection import collect_batch
from src.arc.lt_phase_b_collection_preflight import preflight_collection
from src.arc.lt_phase_b_worker_supervisor import run_monitored_worker
from src.arc.rsi_model_identity import _hash_file, _json_file

_BOOTSTRAP = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
              'from src.cli.lt_phase_b_worker import main; main()')


def _read_object(path):
    return _json_file(path, _hash_file(path))


def _private_run_directory(source_root, run_dir):
    root = Path(source_root).resolve(strict=True)
    artifacts = root / 'artifacts'
    directory = Path(run_dir)
    if artifacts.is_symlink() or directory.is_symlink() or not artifacts.is_dir():
        raise ValueError('invalid artifact directory')
    directory = directory.resolve(strict=True)
    info = directory.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or
            directory.parent != artifacts.resolve(strict=True)):
        raise ValueError('run directory must be private and directly under artifacts')
    return root, directory


def run_collection(*, source_root, run_dir, expected_run_sha256, challenge_path, official_archive,
                   python_executable, model_path, limit):
    root, directory = _private_run_directory(source_root, run_dir)
    if type(limit) is not int or not 0 < limit <= 100:
        raise ValueError('invalid batch limit')
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = {name: _read_object(directory / (name + '.json'))
                  for name in ('spec', 'inventory', 'allocation', 'export', 'receipt')}
        spec = inputs['spec']
        if directory.name != 'lt-phase-b-' + str(spec.get('run_id')):
            raise ValueError('run directory identity differs')
        with zipfile.ZipFile(official_archive) as archive:
            info = archive.getinfo('arc-agi_evaluation_challenges.json')
            if info.file_size > 16 * 1024**2:
                raise ValueError('official challenge inventory too large')
            official = archive.read(info)
        challenges = Path(challenge_path).read_bytes()
        preflight = preflight_collection(source_root=root, spec=spec, expected_run_sha256=expected_run_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'],
            export=inputs['export'], receipt=inputs['receipt'], challenge_bytes=challenges,
            official_challenge_bytes=official, python_executable=python_executable, model_path=model_path)
        if preflight != {'status': 'preflight_consistent_unverified', 'execution_ready': False,
                         'run_sha256': expected_run_sha256, 'selected_tasks': 100,
                         'evidence_type': 'local_files_and_caller_supplied_registration'}:
            raise ValueError('collection preflight result differs')
        settings = spec['settings']
        journal = BatchJournal(directory / 'collection', sorted(spec['selected_task_sha256']),
                               expected_run_sha256 + ':14b:collection')

        def worker(task, output, on_start):
            argv = [str(python_executable), '-I', '-B', '-c', _BOOTSTRAP, str(root),
                    '--spec', str(directory / 'spec.json'), '--inventory', str(directory / 'inventory.json'),
                    '--allocation', str(directory / 'allocation.json'), '--export', str(directory / 'export.json'),
                    '--receipt', str(directory / 'receipt.json'), '--output', str(output), '--task', task,
                    '--expected-run-sha256', expected_run_sha256, '--model-path', str(model_path)]
            return run_monitored_worker(argv, root=directory, cwd=root,
                timeout=settings['task_timeout_seconds'], on_start=on_start,
                resource_limits=settings['resource_limits'], interval_seconds=settings['monitor_interval_seconds'],
                sample_timeout_seconds=settings['sample_timeout_seconds'],
                terminate_grace_seconds=settings['termination_grace_seconds'])

        return collect_batch(journal, limit=limit, worker=worker, resource_limits=settings['resource_limits'])
    finally:
        os.close(lockfd)
