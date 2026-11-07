"""Supervise a single separate real-adapter reload and unscored sample.

Local observations do not establish independent attestation, escaped-process
cleanup, adapter promotion, or ARC improvement. Failed attempts are retained.
"""
import fcntl
import math
import os
import zipfile
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_phase_b_dispatch import _private_run_directory, _read_object
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_reload_dispatch import review_reload_attempt
from src.arc.lt_phase_b_training_dispatch import review_real_training_attempt
from src.arc.lt_phase_b_training_preflight import preflight_training_data
from src.arc.lt_phase_b_training_config import _private_file
from src.arc.rsi_model_identity import _json_file
from src.arc.lt_phase_b_worker_supervisor import run_monitored_worker
from src.arc.rsi_training_resources import training_resource_decision

_BOOTSTRAP = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
              'from src.cli.lt_phase_b_real_reload_worker import main; main()')


def run_real_reload_dispatch(*, source_root, run_dir, expected_training_sha256, expected_canary_sha256,
                             expected_collection_sha256, challenge_path, solution_path, official_archive,
                             python_executable, model_path):
    root, directory = _private_run_directory(source_root, run_dir)
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('real-reload-started.json', 'real-reload-process.json', 'real-reload-observation.json',
                     'real-reload-result.json', 'real-reload-summary.json'):
            if (directory / name).exists() or (directory / name).is_symlink():
                raise ValueError('real reload requires reconciliation')
        inputs = {name: _read_object(directory / (name + '.json')) for name in
                  ('spec', 'inventory', 'allocation', 'export', 'receipt', 'canary-spec', 'training-spec')}
        spec, canary, training = inputs['spec'], inputs['canary-spec'], inputs['training-spec']
        if (training.get('status') != 'frozen' or _digest(training) != expected_training_sha256 or
                training.get('collection_sha256') != expected_collection_sha256 or
                training.get('canary_sha256') != expected_canary_sha256 or
                directory.name != 'lt-phase-b-' + str(spec.get('run_id'))):
            raise ValueError('real reload registration differs')
        def read_inputs():
            with zipfile.ZipFile(official_archive) as archive:
                info = archive.getinfo('arc-agi_evaluation_challenges.json')
                if info.file_size > 16 * 1024**2: raise ValueError('official challenge inventory too large')
                official = archive.read(info)
            return Path(challenge_path).read_bytes(), Path(solution_path).read_bytes(), official
        challenge, solutions, official = read_inputs()
        def preflight():
            return preflight_training_data(training_spec=training, expected_training_sha256=expected_training_sha256,
                collection_spec=spec, expected_collection_sha256=expected_collection_sha256,
                source_root=root, run_dir=directory, source_inventory=inputs['inventory'],
                allocation_receipt=inputs['allocation'], receipt=inputs['receipt'], export=inputs['export'],
                challenge_bytes=challenge, solution_bytes=solutions, official_challenge_bytes=official,
                python_executable=python_executable, model_path=model_path)
        data = preflight()
        if (data.get('status') != 'training_data_consistent_unverified' or data.get('training_allowed') is not False or
                data.get('training_sha256') != expected_training_sha256 or data.get('attempted_tasks') != 100 or
                data.get('solution_sha256') != training.get('solution_sha256') or
                data.get('dataset_manifest_sha256') != training.get('dataset_manifest_sha256') or
                data.get('yield_status') != training.get('yield_status')):
            raise ValueError('training data preflight differs')
        settings, _ = validate_canary_identity(canary_spec=canary, expected_canary_sha256=expected_canary_sha256,
            collection_spec=spec, expected_collection_sha256=expected_collection_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'], receipt=inputs['receipt'])
        def review():
            before = review_real_training_attempt(directory, training=training, expected_training_sha256=expected_training_sha256)
            synthetic = review_reload_attempt(directory, canary=canary, expected_canary_sha256=expected_canary_sha256,
                                               expected_collection_sha256=expected_collection_sha256)
            if (_read_object(directory / 'training-summary.json') != before or
                    before.get('status') != 'real_training_observed_reload_pending' or
                    before.get('training_verified') is not False or before.get('promotion_allowed') is not False or
                    synthetic.get('status') != 'canary_reload_observed_unverified' or
                    synthetic.get('reload_verified') is not False or synthetic.get('promotion_allowed') is not False):
                raise ValueError('prior training or canary evidence differs')
            return before, synthetic
        previous = review()
        config_path = directory / 'training-config.json'
        if _private_file(config_path) != training.get('config_sha256'):
            raise ValueError('real reload configuration changed')
        config = _json_file(config_path, training['config_sha256'])
        if (Path(config['model']).resolve() != Path(model_path).resolve() or
                Path(config['adapter_path']).resolve() != (directory / 'gen1-adapter').resolve()):
            raise ValueError('real reload paths differ')
        _create(directory / 'real-reload-started.json', {'training_sha256': expected_training_sha256,
            'canary_sha256': expected_canary_sha256, 'collection_sha256': expected_collection_sha256})
        argv = [str(python_executable), '-I', '-B', '-c', _BOOTSTRAP, str(root),
            '--training', str(directory / 'training-spec.json'), '--canary', str(directory / 'canary-spec.json'),
            '--collection', str(directory / 'spec.json'), '--inventory', str(directory / 'inventory.json'),
            '--allocation', str(directory / 'allocation.json'), '--receipt', str(directory / 'receipt.json'),
            '--run-dir', str(directory), '--config', str(directory / 'training-config.json'),
            '--output', str(directory / 'real-reload-result.json'), '--expected-training-sha256', expected_training_sha256,
            '--expected-canary-sha256', expected_canary_sha256, '--expected-collection-sha256', expected_collection_sha256]
        def on_start(pid):
            if type(pid) is not int or pid <= 0: raise ValueError('invalid reload worker pid')
            _create(directory / 'real-reload-process.json', {'pid': pid, 'training_sha256': expected_training_sha256})
        observed = run_monitored_worker(argv, root=directory, cwd=root, timeout=canary['timeout_seconds'],
            on_start=on_start, resource_limits=canary['resource_limits'], interval_seconds=settings['monitor_interval_seconds'],
            sample_timeout_seconds=settings['sample_timeout_seconds'], terminate_grace_seconds=settings['termination_grace_seconds'])
        _create(directory / 'real-reload-observation.json', observed)
        if observed.get('status') != 'complete':
            return {'status': 'real_reload_failed', 'reload_verified': False, 'promotion_allowed': False,
                    'reason': observed.get('reason') or observed.get('observation', {}).get('stop_reason')}
        monitor = observed.get('observation')
        if not isinstance(monitor, dict): raise ValueError('reload monitoring unavailable')
        elapsed = monitor.get('elapsed_seconds')
        if (observed.get('leader_reaped') is not True or observed.get('group_absent') is not True or
                monitor.get('stop_reason') != 'process_exited' or type(monitor.get('exit_code')) is not int or
                monitor['exit_code'] != 0 or monitor.get('stream_complete') is not True or
                monitor.get('post_reap_group_absent') is not True or type(elapsed) not in (int, float) or
                not math.isfinite(elapsed) or not 0 <= elapsed <= canary['timeout_seconds'] or
                training_resource_decision(observed.get('baseline'), monitor.get('samples'), limits=canary['resource_limits']) is not None or
                monitor['samples'][-1]['monotonic_seconds'] - observed['baseline']['monotonic_seconds'] > canary['timeout_seconds']):
            raise ValueError('real reload resource evidence differs')
        result = _read_object(directory / 'real-reload-result.json')
        usage, raw, reload_settings = result.get('usage'), result.get('sample_raw'), canary['reload_settings']
        if (result.get('status') != 'real_reload_observed_unverified' or result.get('reload_verified') is not False or
                result.get('promotion_allowed') is not False or result.get('training_sha256') != expected_training_sha256 or
                result.get('collection_sha256') != expected_collection_sha256 or result.get('canary_sha256') != expected_canary_sha256 or
                result.get('adapter_sha256') != previous[0]['adapter']['adapter_sha256'] or
                result.get('sample_prompt_sha256') != reload_settings['sample_prompt_sha256'] or
                not isinstance(raw, str) or not raw or len(raw.encode()) > 32768 or not isinstance(usage, dict) or
                type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] < 0 or
                type(usage.get('completion_tokens')) is not int or not 0 < usage['completion_tokens'] <= reload_settings['max_tokens'] or
                type(usage.get('peak_memory_gb')) not in (int, float) or not math.isfinite(usage['peak_memory_gb']) or
                not 0 <= usage['peak_memory_gb'] <= 48 * 1024**3 / 1e9 or usage.get('finish_reason') not in ('stop', 'length')):
            raise ValueError('real reload result differs')
        try:
            normalize_direct(raw, 1); valid = True
        except ValueError:
            valid = False
        if result.get('sample_format_valid') is not valid:
            raise ValueError('real reload format differs')
        if (read_inputs() != (challenge, solutions, official) or preflight() != data or review() != previous or
                _private_file(config_path) != training['config_sha256']):
            raise ValueError('real reload inputs changed')
        summary = {'status': 'real_adapter_reload_observed_unverified', 'reload_verified': False,
                   'promotion_allowed': False, 'training_sha256': expected_training_sha256,
                   'collection_sha256': expected_collection_sha256, 'canary_sha256': expected_canary_sha256,
                   'training': previous[0], 'reload': result}
        _create(directory / 'real-reload-summary.json', summary)
        return summary
    finally:
        os.close(lockfd)
