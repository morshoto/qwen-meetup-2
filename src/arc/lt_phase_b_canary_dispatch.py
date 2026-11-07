"""Supervise one registered synthetic 14B canary training attempt.

A locally observed step is not an independently attested canary, adapter reload,
Gen 1, or RSI improvement. Interrupted attempts require reconciliation.
"""
import fcntl
import hashlib
import math
import os
import zipfile
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_adapter_artifact import inspect_adapter_artifact
from src.arc.lt_phase_b_collection_preflight import preflight_collection
from src.arc.lt_phase_b_dispatch import _private_run_directory, _read_object
from src.arc.lt_phase_b_registered_canary import validate_canary_descriptor
from src.arc.lt_phase_b_tensor_evidence import compare_tensor_states
from src.arc.lt_phase_b_worker_supervisor import run_monitored_worker
from src.arc.rsi_loss_watch import TrainingLossWatch
from src.arc.rsi_training_resources import training_resource_decision

_BOOTSTRAP = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
              'from src.cli.lt_phase_b_canary_worker import main; main()')


def review_training_attempt(directory, *, canary, expected_canary_sha256, expected_collection_sha256):
    """Replay local training evidence without claiming independent attestation."""
    directory = Path(directory)
    started = _read_object(directory / 'canary-started.json')
    process = _read_object(directory / 'canary-process.json')
    if (started != {'canary_sha256': expected_canary_sha256, 'collection_sha256': expected_collection_sha256} or
            set(process) != {'pid', 'canary_sha256'} or type(process.get('pid')) is not int or process['pid'] <= 0 or
            process.get('canary_sha256') != expected_canary_sha256):
        raise ValueError('canary start identity differs')
    observation = _read_object(directory / 'canary-observation.json')
    if observation.get('status') != 'complete':
        raise ValueError('canary training did not complete')
    evidence = directory / 'canary-evidence'
    monitor = observation.get('observation')
    if not isinstance(monitor, dict):
        raise ValueError('canary monitoring unavailable')
    elapsed, loss = monitor.get('elapsed_seconds'), monitor.get('loss')
    if (observation.get('leader_reaped') is not True or observation.get('group_absent') is not True or
            monitor.get('stop_reason') != 'process_exited' or type(monitor.get('exit_code')) is not int or monitor['exit_code'] != 0 or
            monitor.get('stream_complete') is not True or monitor.get('post_reap_group_absent') is not True or
            type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= canary['timeout_seconds'] or
            not isinstance(loss, dict) or loss.get('status') != 'ok' or
            type(loss.get('last_step')) is not int or loss['last_step'] != 1 or
            type(loss.get('last_loss')) not in (int, float) or not math.isfinite(loss['last_loss']) or
            training_resource_decision(observation.get('baseline'), monitor.get('samples'),
                                       limits=canary['resource_limits']) is not None):
        raise ValueError('canary monitoring evidence differs')
    text = monitor.get('log')
    if not isinstance(text, str):
        raise ValueError('canary training log unavailable')
    log_bytes = text.encode()
    if len(log_bytes) > 4 * 1024**2 or hashlib.sha256(log_bytes).hexdigest() != monitor.get('log_sha256'):
        raise ValueError('canary training log identity differs')
    if monitor['samples'][-1]['monotonic_seconds'] - observation['baseline']['monotonic_seconds'] > canary['timeout_seconds']:
        raise ValueError('canary sample window exceeded')
    watch = TrainingLossWatch()
    for offset in range(0, len(log_bytes), 8192):
        watch.feed(log_bytes[offset:offset + 8192])
    watch.finish()
    if watch.snapshot() != loss:
        raise ValueError('canary loss differs from retained log')
    before = _read_object(evidence / 'tensor-before.json')
    after = _read_object(evidence / 'tensor-after.json')
    checked = compare_tensor_states(before, after)
    if _read_object(evidence / 'tensor-evidence.json') != checked:
        raise ValueError('canary tensor evidence differs')
    result = _read_object(directory / 'canary-result.json')
    if result != {**checked, 'canary_sha256': expected_canary_sha256,
                  'collection_sha256': expected_collection_sha256, 'canary_run_id': canary['run_id'],
                  'promotion_allowed': False}:
        raise ValueError('canary worker result differs')
    adapter = inspect_adapter_artifact(directory / 'canary-adapter', expected_tensors=after['adapter'])
    summary = {'status': 'training_observed_reload_pending', 'training_verified': False, 'promotion_allowed': False,
               'canary_sha256': expected_canary_sha256, 'collection_sha256': expected_collection_sha256,
               'loss': loss, 'tensor_evidence': checked, 'adapter': adapter}
    return summary


def run_canary(*, source_root, run_dir, expected_collection_sha256, expected_canary_sha256,
               challenge_path, official_archive, python_executable, model_path):
    root, directory = _private_run_directory(source_root, run_dir)
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('canary-started.json', 'canary-process.json', 'canary-observation.json',
                     'canary-result.json', 'canary-training-summary.json'):
            marker = directory / name
            if marker.exists() or marker.is_symlink():
                raise ValueError('canary attempt requires reconciliation')
        inputs = {name: _read_object(directory / (name + '.json'))
                  for name in ('spec', 'inventory', 'allocation', 'export', 'receipt', 'canary-spec')}
        spec, canary = inputs['spec'], inputs['canary-spec']
        if directory.name != 'lt-phase-b-' + str(spec.get('run_id')):
            raise ValueError('run directory identity differs')
        with zipfile.ZipFile(official_archive) as archive:
            info = archive.getinfo('arc-agi_evaluation_challenges.json')
            if info.file_size > 16 * 1024**2:
                raise ValueError('official challenge inventory too large')
            official = archive.read(info)
        preflight = preflight_collection(source_root=root, spec=spec, expected_run_sha256=expected_collection_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'], export=inputs['export'],
            receipt=inputs['receipt'], challenge_bytes=Path(challenge_path).read_bytes(),
            official_challenge_bytes=official, python_executable=python_executable, model_path=model_path)
        if preflight != {'status': 'preflight_consistent_unverified', 'execution_ready': False,
                         'run_sha256': expected_collection_sha256, 'selected_tasks': 100,
                         'evidence_type': 'local_files_and_caller_supplied_registration'}:
            raise ValueError('collection preflight result differs')
        config_path = directory / 'canary-config.json'
        settings, _, config = validate_canary_descriptor(canary_spec=canary, expected_canary_sha256=expected_canary_sha256,
            collection_spec=spec, expected_collection_sha256=expected_collection_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'],
            receipt=inputs['receipt'], config_path=config_path)
        evidence = directory / 'canary-evidence'
        if (Path(config['model']).resolve() != Path(model_path).resolve() or
                Path(config['data']).resolve() != (directory / 'canary-data').resolve() or
                Path(config['adapter_path']).resolve() != (directory / 'canary-adapter').resolve() or
                evidence.is_symlink() or not evidence.is_dir() or evidence.stat().st_uid != os.getuid() or
                evidence.stat().st_mode & 0o077 or any(evidence.iterdir())):
            raise ValueError('canary paths or evidence directory differ')
        _create(directory / 'canary-started.json', {'canary_sha256': expected_canary_sha256,
                                                    'collection_sha256': expected_collection_sha256})
        argv = [str(python_executable), '-I', '-B', '-c', _BOOTSTRAP, str(root),
                '--canary', str(directory / 'canary-spec.json'), '--collection', str(directory / 'spec.json'),
                '--inventory', str(directory / 'inventory.json'), '--allocation', str(directory / 'allocation.json'),
                '--receipt', str(directory / 'receipt.json'), '--config', str(config_path),
                '--evidence-dir', str(evidence), '--output', str(directory / 'canary-result.json'),
                '--expected-canary-sha256', expected_canary_sha256,
                '--expected-collection-sha256', expected_collection_sha256]
        def on_start(pid):
            if type(pid) is not int or pid <= 0:
                raise ValueError('invalid canary worker pid')
            _create(directory / 'canary-process.json', {'pid': pid, 'canary_sha256': expected_canary_sha256})
        observation = run_monitored_worker(argv, root=directory, cwd=root, timeout=canary['timeout_seconds'],
            on_start=on_start, resource_limits=canary['resource_limits'],
            interval_seconds=settings['monitor_interval_seconds'], sample_timeout_seconds=settings['sample_timeout_seconds'],
            terminate_grace_seconds=settings['termination_grace_seconds'], training=True)
        _create(directory / 'canary-observation.json', observation)
        if observation.get('status') != 'complete':
            return {'status': 'training_failed', 'training_verified': False, 'promotion_allowed': False,
                    'reason': observation.get('reason') or observation.get('observation', {}).get('stop_reason')}
        summary = review_training_attempt(directory, canary=canary,
            expected_canary_sha256=expected_canary_sha256, expected_collection_sha256=expected_collection_sha256)
        _create(directory / 'canary-training-summary.json', summary)
        return summary
    finally:
        os.close(lockfd)
