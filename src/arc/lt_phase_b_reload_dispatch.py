"""Supervise a separate synthetic canary reload and inference observation.

Local process and file checks do not establish escaped-descendant cleanup,
independent attestation, ARC improvement, or permission to promote an adapter.
"""
import fcntl
import math
import os
import zipfile
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_phase_b_adapter_artifact import inspect_adapter_artifact
from src.arc.lt_phase_b_canary_dispatch import review_training_attempt
from src.arc.lt_phase_b_collection_preflight import preflight_collection
from src.arc.lt_phase_b_dispatch import _private_run_directory, _read_object
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_worker_supervisor import run_monitored_worker
from src.arc.lt_source import verify_source_tree
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_model_identity import _json_file, verify_model_manifest
from src.arc.rsi_runtime_identity import verify_runtime_identity
from src.arc.rsi_training_resources import training_resource_decision

_BOOTSTRAP = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
              'from src.cli.lt_phase_b_reload_worker import main; main()')


def review_reload_attempt(directory, *, canary, expected_canary_sha256, expected_collection_sha256,
                          require_summary=True):
    """Replay retained local observations; this does not verify independent provenance."""
    directory = Path(directory)
    if not isinstance(canary, dict) or _digest(canary) != expected_canary_sha256:
        raise ValueError('canary descriptor differs')
    started = _read_object(directory / 'canary-reload-started.json')
    process = _read_object(directory / 'canary-reload-process.json')
    if (started != {'canary_sha256': expected_canary_sha256, 'collection_sha256': expected_collection_sha256} or
            set(process) != {'pid', 'canary_sha256'} or type(process.get('pid')) is not int or
            process['pid'] <= 0 or process.get('canary_sha256') != expected_canary_sha256):
        raise ValueError('reload start identity differs')
    summary = review_training_attempt(directory, canary=canary, expected_canary_sha256=expected_canary_sha256,
                                      expected_collection_sha256=expected_collection_sha256)
    if _read_object(directory / 'canary-training-summary.json') != summary:
        raise ValueError('saved training summary differs')
    observation = _read_object(directory / 'canary-reload-observation.json')
    if observation.get('status') != 'complete':
        raise ValueError('reload did not complete')
    monitor = observation.get('observation')
    if not isinstance(monitor, dict):
        raise ValueError('reload monitoring unavailable')
    elapsed = monitor.get('elapsed_seconds')
    if (observation.get('leader_reaped') is not True or observation.get('group_absent') is not True or
            monitor.get('stop_reason') != 'process_exited' or type(monitor.get('exit_code')) is not int or
            monitor['exit_code'] != 0 or monitor.get('stream_complete') is not True or
            monitor.get('post_reap_group_absent') is not True or
            type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= canary['timeout_seconds'] or
            training_resource_decision(observation.get('baseline'), monitor.get('samples'),
                                       limits=canary['resource_limits']) is not None):
        raise ValueError('reload monitoring evidence differs')
    if monitor['samples'][-1]['monotonic_seconds'] - observation['baseline']['monotonic_seconds'] > canary['timeout_seconds']:
        raise ValueError('reload sample window exceeded')
    result = _read_object(directory / 'canary-reload-result.json')
    usage, raw = result.get('usage'), result.get('sample_raw')
    if (result.get('status') != 'reload_observed_unverified' or result.get('reload_verified') is not False or
            result.get('promotion_allowed') is not False or result.get('canary_sha256') != expected_canary_sha256 or
            result.get('collection_sha256') != expected_collection_sha256 or
            result.get('adapter_sha256') != summary['adapter']['adapter_sha256'] or
            result.get('sample_prompt_sha256') != canary['reload_settings']['sample_prompt_sha256'] or
            not isinstance(raw, str) or not raw or len(raw.encode()) > 32768 or not isinstance(usage, dict) or
            type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] < 0 or
            type(usage.get('completion_tokens')) is not int or
            not 0 < usage['completion_tokens'] <= canary['reload_settings']['max_tokens'] or
            type(usage.get('peak_memory_gb')) not in (int, float) or not math.isfinite(usage['peak_memory_gb']) or
            not 0 <= usage['peak_memory_gb'] <= (48 * 1024**3 / 1e9) or
            usage.get('finish_reason') not in ('stop', 'length')):
        raise ValueError('reload result differs')
    try:
        normalize_direct(raw, 1)
        format_valid = True
    except ValueError:
        format_valid = False
    if result.get('sample_format_valid') is not format_valid:
        raise ValueError('reload sample format differs')
    evidence = directory / 'canary-evidence'
    after = _read_object(evidence / 'tensor-after.json')
    if inspect_adapter_artifact(directory / 'canary-adapter', expected_tensors=after['adapter']) != summary['adapter']:
        raise ValueError('canary adapter changed during reload')
    output = {'status': 'canary_reload_observed_unverified', 'reload_verified': False, 'promotion_allowed': False,
              'canary_sha256': expected_canary_sha256, 'collection_sha256': expected_collection_sha256,
              'training': summary, 'reload': result}
    saved = directory / 'canary-reload-summary.json'
    if (type(require_summary) is not bool or saved.is_symlink() or
            (require_summary and not saved.exists()) or (saved.exists() and _read_object(saved) != output)):
        raise ValueError('saved reload summary differs')
    return output


def run_reload(*, source_root, run_dir, expected_collection_sha256, expected_canary_sha256,
               challenge_path, official_archive, python_executable, model_path):
    root, directory = _private_run_directory(source_root, run_dir)
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('canary-reload-started.json', 'canary-reload-process.json', 'canary-reload-observation.json',
                     'canary-reload-result.json', 'canary-reload-summary.json'):
            marker = directory / name
            if marker.exists() or marker.is_symlink():
                raise ValueError('canary reload attempt requires reconciliation')
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
        settings, model = validate_canary_identity(canary_spec=canary, expected_canary_sha256=expected_canary_sha256,
            collection_spec=spec, expected_collection_sha256=expected_collection_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'], receipt=inputs['receipt'])
        config_path = directory / 'canary-config.json'
        config = _json_file(config_path, canary['config_sha256'])
        if (Path(config['model']).resolve() != Path(model_path).resolve() or
                Path(config['data']).resolve() != (directory / 'canary-data').resolve() or
                Path(config['adapter_path']).resolve() != (directory / 'canary-adapter').resolve()):
            raise ValueError('canary paths differ')
        summary = review_training_attempt(directory, canary=canary, expected_canary_sha256=expected_canary_sha256,
                                          expected_collection_sha256=expected_collection_sha256)
        if _read_object(directory / 'canary-training-summary.json') != summary:
            raise ValueError('saved training summary differs')
        _create(directory / 'canary-reload-started.json', {'canary_sha256': expected_canary_sha256,
                                                           'collection_sha256': expected_collection_sha256})
        evidence = directory / 'canary-evidence'
        argv = [str(python_executable), '-I', '-B', '-c', _BOOTSTRAP, str(root),
                '--canary', str(directory / 'canary-spec.json'), '--collection', str(directory / 'spec.json'),
                '--inventory', str(directory / 'inventory.json'), '--allocation', str(directory / 'allocation.json'),
                '--receipt', str(directory / 'receipt.json'), '--config', str(config_path),
                '--summary', str(directory / 'canary-training-summary.json'),
                '--before', str(evidence / 'tensor-before.json'), '--after', str(evidence / 'tensor-after.json'),
                '--output', str(directory / 'canary-reload-result.json'),
                '--expected-canary-sha256', expected_canary_sha256,
                '--expected-collection-sha256', expected_collection_sha256]
        def on_start(pid):
            if type(pid) is not int or pid <= 0:
                raise ValueError('invalid reload worker pid')
            _create(directory / 'canary-reload-process.json', {'pid': pid, 'canary_sha256': expected_canary_sha256})
        observation = run_monitored_worker(argv, root=directory, cwd=root, timeout=canary['timeout_seconds'],
            on_start=on_start, resource_limits=canary['resource_limits'],
            interval_seconds=settings['monitor_interval_seconds'], sample_timeout_seconds=settings['sample_timeout_seconds'],
            terminate_grace_seconds=settings['termination_grace_seconds'])
        _create(directory / 'canary-reload-observation.json', observation)
        if observation.get('status') != 'complete':
            return {'status': 'reload_failed', 'reload_verified': False, 'promotion_allowed': False,
                    'reason': observation.get('reason') or observation.get('observation', {}).get('stop_reason')}
        verify_runtime_identity(python_executable, canary['runtime_manifest'])
        verify_environment(python_executable, canary['environment_manifest'])
        verify_model_manifest(model_path, model, model_key='14b')
        verify_source_tree(root, inputs['inventory'])
        output = review_reload_attempt(directory, canary=canary, expected_canary_sha256=expected_canary_sha256,
                                       expected_collection_sha256=expected_collection_sha256, require_summary=False)
        _create(directory / 'canary-reload-summary.json', output)
        return output
    finally:
        os.close(lockfd)
