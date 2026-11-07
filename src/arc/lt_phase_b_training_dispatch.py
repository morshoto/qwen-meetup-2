"""Supervise one registered real-data training attempt with retained evidence.

Local observations do not establish independent attestation, complete escaped
process cleanup, adapter reload or ARC improvement. Failed attempts must be
reconciled; this dispatcher never retries or promotes an adapter.
"""
import fcntl
import hashlib
import math
import os
import re
import zipfile
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_adapter_artifact import inspect_adapter_artifact
from src.arc.lt_phase_b_dispatch import _private_run_directory, _read_object
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_reload_dispatch import review_reload_attempt
from src.arc.lt_phase_b_tensor_evidence import compare_tensor_states
from src.arc.lt_phase_b_training_preflight import preflight_training_data
from src.arc.lt_phase_b_training_config import _private_file
from src.arc.lt_phase_b_worker_supervisor import run_monitored_worker
from src.arc.rsi_loss_watch import TrainingLossWatch
from src.arc.rsi_training_resources import training_resource_decision

_BOOTSTRAP = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
              'from src.cli.lt_phase_b_training_worker import main; main()')


def review_real_training_attempt(directory, *, training, expected_training_sha256):
    directory = Path(directory)
    if not isinstance(training, dict) or _digest(training) != expected_training_sha256:
        raise ValueError('training descriptor differs')
    started, process = (_read_object(directory / name) for name in ('training-started.json', 'training-process.json'))
    if (started != {'training_sha256': expected_training_sha256, 'collection_sha256': training['collection_sha256'],
                    'canary_sha256': training['canary_sha256']} or
            set(process) != {'pid', 'training_sha256'} or type(process.get('pid')) is not int or process['pid'] <= 0 or
            process.get('training_sha256') != expected_training_sha256):
        raise ValueError('training start evidence differs')
    observed = _read_object(directory / 'training-observation.json')
    monitor = observed.get('observation')
    if observed.get('status') != 'complete' or not isinstance(monitor, dict):
        raise ValueError('training observation incomplete')
    elapsed, loss = monitor.get('elapsed_seconds'), monitor.get('loss')
    if (observed.get('leader_reaped') is not True or observed.get('group_absent') is not True or
            monitor.get('stop_reason') != 'process_exited' or type(monitor.get('exit_code')) is not int or
            monitor['exit_code'] != 0 or monitor.get('stream_complete') is not True or
            monitor.get('post_reap_group_absent') is not True or
            type(elapsed) not in (int, float) or not math.isfinite(elapsed) or
            not 0 <= elapsed <= training['timeout_seconds'] or not isinstance(loss, dict) or
            loss.get('status') != 'ok' or type(loss.get('last_step')) is not int or loss['last_step'] != 20 or
            type(loss.get('last_loss')) not in (int, float) or not math.isfinite(loss['last_loss']) or
            training_resource_decision(observed.get('baseline'), monitor.get('samples'),
                                       limits=training['resource_limits']) is not None):
        raise ValueError('training resource or loss evidence differs')
    if monitor['samples'][-1]['monotonic_seconds'] - observed['baseline']['monotonic_seconds'] > training['timeout_seconds']:
        raise ValueError('training sample window exceeded')
    log = monitor.get('log')
    if not isinstance(log, str) or len(log.encode()) > 4 * 1024**2 or hashlib.sha256(log.encode()).hexdigest() != monitor.get('log_sha256'):
        raise ValueError('training log differs')
    watch = TrainingLossWatch(); raw = log.encode()
    for offset in range(0, len(raw), 8192): watch.feed(raw[offset:offset + 8192])
    watch.finish()
    if watch.snapshot() != loss:
        raise ValueError('training loss differs from log')
    evidence = directory / 'training-evidence'
    before, after = (_read_object(evidence / name) for name in ('tensor-before.json', 'tensor-after.json'))
    checked = compare_tensor_states(before, after)
    if _read_object(evidence / 'tensor-evidence.json') != checked:
        raise ValueError('training tensor evidence differs')
    expected = {**checked, 'training_sha256': expected_training_sha256,
        'collection_sha256': training['collection_sha256'], 'canary_sha256': training['canary_sha256'],
        'training_run_id': training['run_id'], 'yield_status': training['yield_status'], 'promotion_allowed': False}
    if _read_object(directory / 'training-result.json') != expected:
        raise ValueError('training worker result differs')
    adapter = inspect_adapter_artifact(directory / 'gen1-adapter', expected_tensors=after['adapter'])
    if (not isinstance(adapter, dict) or set(adapter) != {'status', 'adapter_sha256', 'weights_sha256',
            'config_sha256', 'matched_tensor_count', 'reload_verified', 'promotion_allowed'} or
            adapter['status'] != 'adapter_files_consistent_unverified' or
            type(adapter['matched_tensor_count']) is not int or adapter['matched_tensor_count'] != len(after['adapter']) or
            adapter['reload_verified'] is not False or adapter['promotion_allowed'] is not False or
            any(not isinstance(adapter[k], str) or not re.fullmatch(r'[0-9a-f]{64}', adapter[k])
                for k in ('adapter_sha256', 'weights_sha256', 'config_sha256'))):
        raise ValueError('training adapter evidence differs')
    return {'status': 'real_training_observed_reload_pending', 'training_verified': False, 'promotion_allowed': False,
            'training_sha256': expected_training_sha256, 'collection_sha256': training['collection_sha256'],
            'canary_sha256': training['canary_sha256'], 'yield_status': training['yield_status'],
            'loss': loss, 'tensor_evidence': checked, 'adapter': adapter}


def run_training(*, source_root, run_dir, expected_training_sha256, expected_collection_sha256,
                 expected_canary_sha256, challenge_path, solution_path, official_archive, python_executable, model_path):
    root, directory = _private_run_directory(source_root, run_dir)
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('training-started.json', 'training-process.json', 'training-observation.json',
                     'training-result.json', 'training-summary.json'):
            if (directory / name).exists() or (directory / name).is_symlink():
                raise ValueError('training attempt requires reconciliation')
        inputs = {name: _read_object(directory / (name + '.json')) for name in
                  ('spec', 'inventory', 'allocation', 'export', 'receipt', 'canary-spec', 'training-spec')}
        spec, canary, training = inputs['spec'], inputs['canary-spec'], inputs['training-spec']
        if (training.get('status') != 'frozen' or _digest(training) != expected_training_sha256 or
                training.get('collection_sha256') != expected_collection_sha256 or
                training.get('canary_sha256') != expected_canary_sha256 or
                directory.name != 'lt-phase-b-' + str(spec.get('run_id'))):
            raise ValueError('training registration differs')
        with zipfile.ZipFile(official_archive) as archive:
            info = archive.getinfo('arc-agi_evaluation_challenges.json')
            if info.file_size > 16 * 1024**2: raise ValueError('official challenge inventory too large')
            official = archive.read(info)
        challenge, solutions = Path(challenge_path).read_bytes(), Path(solution_path).read_bytes()
        def preflight():
            return preflight_training_data(training_spec=training, expected_training_sha256=expected_training_sha256,
                collection_spec=spec, expected_collection_sha256=expected_collection_sha256,
                source_root=root, run_dir=directory, source_inventory=inputs['inventory'],
                allocation_receipt=inputs['allocation'], receipt=inputs['receipt'], export=inputs['export'],
                challenge_bytes=challenge, solution_bytes=solutions, official_challenge_bytes=official,
                python_executable=python_executable, model_path=model_path)
        checked = preflight()
        if (checked.get('status') != 'training_data_consistent_unverified' or checked.get('training_allowed') is not False or
                checked.get('training_sha256') != expected_training_sha256 or checked.get('attempted_tasks') != 100 or
                checked.get('solution_sha256') != training.get('solution_sha256') or
                checked.get('dataset_manifest_sha256') != training.get('dataset_manifest_sha256') or
                type(checked.get('eligible_tasks')) is not int or checked['eligible_tasks'] <= 0 or
                checked.get('yield_status') != training.get('yield_status')):
            raise ValueError('training data preflight differs')
        settings, _ = validate_canary_identity(canary_spec=canary, expected_canary_sha256=expected_canary_sha256,
            collection_spec=spec, expected_collection_sha256=expected_collection_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'], receipt=inputs['receipt'])
        canary_result = review_reload_attempt(directory, canary=canary, expected_canary_sha256=expected_canary_sha256,
                                              expected_collection_sha256=expected_collection_sha256)
        if (canary_result.get('status') != 'canary_reload_observed_unverified' or
                canary_result.get('reload_verified') is not False or canary_result.get('promotion_allowed') is not False):
            raise ValueError('canary reload evidence differs')
        if (training.get('protocol') != 'lt-phase-b-real-training-v1' or training.get('run_id') != spec['run_id'] or
                training.get('source_code_sha256') != spec['source_code_sha256'] or
                type(training.get('timeout_seconds')) is not int or training['timeout_seconds'] != settings['task_timeout_seconds'] or
                _digest(training.get('resource_limits')) != _digest(settings['resource_limits']) or
                training.get('checkpoint_policy') != 'final_step_only'):
            raise ValueError('training limits or identity differ')
        evidence, config = directory / 'training-evidence', directory / 'training-config.json'
        if (evidence.is_symlink() or not evidence.is_dir() or evidence.stat().st_uid != os.getuid() or
                evidence.stat().st_mode & 0o077 or any(evidence.iterdir()) or
                (directory / 'gen1-adapter').exists() or (directory / 'gen1-adapter').is_symlink() or
                _private_file(config) != training.get('config_sha256')):
            raise ValueError('training output paths differ')
        _create(directory / 'training-started.json', {'training_sha256': expected_training_sha256,
            'collection_sha256': expected_collection_sha256, 'canary_sha256': expected_canary_sha256})
        argv = [str(python_executable), '-I', '-B', '-c', _BOOTSTRAP, str(root),
            '--training', str(directory / 'training-spec.json'), '--canary', str(directory / 'canary-spec.json'),
            '--collection', str(directory / 'spec.json'), '--inventory', str(directory / 'inventory.json'),
            '--allocation', str(directory / 'allocation.json'), '--receipt', str(directory / 'receipt.json'),
            '--run-dir', str(directory), '--model-path', str(model_path), '--config', str(config),
            '--evidence-dir', str(evidence), '--output', str(directory / 'training-result.json'),
            '--expected-training-sha256', expected_training_sha256, '--expected-canary-sha256', expected_canary_sha256,
            '--expected-collection-sha256', expected_collection_sha256]
        def on_start(pid):
            if type(pid) is not int or pid <= 0: raise ValueError('invalid training worker pid')
            _create(directory / 'training-process.json', {'pid': pid, 'training_sha256': expected_training_sha256})
        observed = run_monitored_worker(argv, root=directory, cwd=root, timeout=training['timeout_seconds'],
            on_start=on_start, resource_limits=training['resource_limits'], interval_seconds=settings['monitor_interval_seconds'],
            sample_timeout_seconds=settings['sample_timeout_seconds'], terminate_grace_seconds=settings['termination_grace_seconds'],
            training=True, expected_training_steps=20)
        _create(directory / 'training-observation.json', observed)
        if observed.get('status') != 'complete':
            return {'status': 'training_failed', 'training_verified': False, 'promotion_allowed': False,
                    'reason': observed.get('reason') or observed.get('observation', {}).get('stop_reason')}
        with zipfile.ZipFile(official_archive) as archive:
            if archive.getinfo('arc-agi_evaluation_challenges.json').file_size > 16 * 1024**2:
                raise ValueError('official challenge inventory too large')
            official_after = archive.read('arc-agi_evaluation_challenges.json')
        if (Path(challenge_path).read_bytes() != challenge or Path(solution_path).read_bytes() != solutions or
                official_after != official or preflight() != checked or review_reload_attempt(directory, canary=canary,
                expected_canary_sha256=expected_canary_sha256, expected_collection_sha256=expected_collection_sha256) != canary_result):
            raise ValueError('training inputs changed during attempt')
        summary = review_real_training_attempt(directory, training=training, expected_training_sha256=expected_training_sha256)
        _create(directory / 'training-summary.json', summary)
        return summary
    finally:
        os.close(lockfd)
