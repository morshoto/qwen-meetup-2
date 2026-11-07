"""Journal matched public predictions sequentially under parent supervision.

Caller must bind the frozen allocation, public export and runtime and serialize
access. Local observations are not independent attestation or scored results.
"""
import math
import os
import re
import stat

from src.arc.lt_batch_journal import BatchJournal
from src.arc.lt_phase_b_collection import collect_batch
from src.arc.lt_phase_b_sft_preflight import _json
from src.arc.lt_phase_b_training_config import _private_file
from src.arc.rsi_model_identity import _json_file
from src.arc.rsi_training_resources import training_resource_decision, valid_resource_limits


def _read(path):
    return _json_file(path, _private_file(path))


def _check_worker(row, task, state, registration):
    if (not isinstance(row, dict) or row.get('task_id') != task or row.get('model') != '14b' or
            row.get('status') != 'complete' or not isinstance(row.get('candidates'), list) or len(row['candidates']) != 2 or
            row.get('adapter_state') != state or 'adapter_sha256' not in row or
            row['adapter_sha256'] != (registration['adapter_sha256'] if state == 'on' else None) or
            row.get('run_sha256') != registration['run_sha256'] or
            row.get('source_code_sha256') != registration['source_code_sha256'] or
            row.get('public_task_sha256') != registration['selected_task_sha256'][task] or
            row.get('prompt_sha256') != registration['selected_prompt_sha256'][task]):
        raise ValueError('comparison worker identity differs')


def _replay(journal, state, registration, limits):
    journal.pending(1)
    stopped = None
    missing = False
    for task in journal.ids:
        result_path = journal.root / (task + '.result.json')
        if not result_path.exists():
            missing = True
            if any((journal.root / (task + suffix)).exists() or (journal.root / (task + suffix)).is_symlink()
                   for suffix in ('.process.json', '.resources.json', '.worker.json')):
                raise ValueError('orphan comparison artifact requires reconciliation')
            continue
        if missing or stopped:
            raise ValueError('comparison results are not a sequential prefix')
        marker, process, resources, result = (_read(journal.root / (task + suffix)) for suffix in
            ('.started.json', '.process.json', '.resources.json', '.result.json'))
        if (marker != {'task_id': task, 'identity': journal.identity} or
                set(process) != {'task_id', 'identity', 'pid'} or process['task_id'] != task or
                process['identity'] != journal.identity or type(process['pid']) is not int or process['pid'] <= 0 or
                resources.get('leader_reaped') is not True or resources.get('group_absent') is not True):
            raise ValueError('comparison process evidence differs')
        status = resources.get('status')
        if status == 'complete':
            monitor = resources.get('observation')
            if not isinstance(monitor, dict): raise ValueError('comparison monitoring unavailable')
            elapsed = monitor.get('elapsed_seconds', resources.get('elapsed_seconds'))
            if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= limits['timeout_seconds'] or
                    monitor.get('stop_reason') != 'process_exited' or type(monitor.get('exit_code')) is not int or
                    monitor['exit_code'] != 0 or monitor.get('stream_complete') is not True or
                    monitor.get('post_reap_group_absent') is not True or
                    not isinstance(monitor.get('cleanup_observation'), dict) or
                    training_resource_decision(resources.get('baseline'), monitor.get('samples'), limits=limits) is not None or
                    monitor['samples'][-1]['monotonic_seconds'] - resources['baseline']['monotonic_seconds'] > limits['timeout_seconds']):
                raise ValueError('comparison resource evidence differs')
            row = _read(journal.root / (task + '.worker.json'))
            _check_worker(row, task, state, registration)
        elif status in ('error', 'timeout'):
            row = {'task_id': task, 'model': '14b', 'status': status, 'candidates': [], 'tokens': None,
                   'runtime_seconds': resources.get('elapsed_seconds'), 'peak_memory_bytes': None,
                   'peak_swap_bytes': None, 'calls': []}
            stopped = 'worker_' + status
        else:
            raise ValueError('comparison worker status differs')
        expected = {**row, 'worker_diagnostics': {'stdout': resources.get('stdout'), 'stderr': resources.get('stderr')},
                    'cleanup': {'leader_reaped': True, 'group_absent': True, 'returncode': resources.get('returncode')},
                    'identity': journal.identity}
        if _json(result) != _json(expected): raise ValueError('comparison result differs from worker')
    return stopped


def compare_batch(*, off_journal, on_journal, registration, limit, worker, resource_limits):
    keys = {'run_sha256', 'source_code_sha256', 'adapter_sha256', 'selected_task_sha256', 'selected_prompt_sha256'}
    valid = lambda value: isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value))
    if (not isinstance(off_journal, BatchJournal) or not isinstance(on_journal, BatchJournal) or
            off_journal.root == on_journal.root or off_journal.ids != on_journal.ids or
            off_journal.ids != sorted(off_journal.ids) or not isinstance(registration, dict) or set(registration) != keys or
            any(not valid(registration[k]) for k in ('run_sha256', 'source_code_sha256', 'adapter_sha256')) or
            any(not isinstance(registration[k], dict) or set(registration[k]) != set(off_journal.ids) or
                any(not valid(v) for v in registration[k].values()) for k in ('selected_task_sha256', 'selected_prompt_sha256')) or
            off_journal.identity != registration['run_sha256'] + ':14b:off' or
            on_journal.identity != registration['run_sha256'] + ':14b:on' or
            type(limit) is not int or limit <= 0 or not callable(worker) or not valid_resource_limits(resource_limits)):
        raise ValueError('invalid comparison journal registration')
    for journal in (off_journal, on_journal):
        info = journal.root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('comparison journal must be private')
    off_stop = _replay(off_journal, 'off', registration, resource_limits)
    on_stop = _replay(on_journal, 'on', registration, resource_limits)
    off_pending, on_pending = off_journal.pending(1), on_journal.pending(1)
    if off_pending and len(on_pending) == 0:
        raise ValueError('on state ran before off state completed')
    if off_pending and any((on_journal.root / (x + '.started.json')).exists() for x in on_journal.ids):
        raise ValueError('on state started before off state completed')
    if off_stop: return {'completed': 0, 'stop_reason': off_stop, 'state': 'off'}
    if on_stop: return {'completed': 0, 'stop_reason': on_stop, 'state': 'on'}
    state, journal = ('off', off_journal) if off_pending else ('on', on_journal)
    if not journal.pending(1): return {'completed': 0, 'stop_reason': 'complete', 'state': state}
    def guarded(task, output, on_start):
        observation = worker(task, state, output, on_start)
        if isinstance(observation, dict) and observation.get('status') == 'complete':
            _check_worker(_read(output), task, state, registration)
        return observation
    result = collect_batch(journal, limit=limit, worker=guarded, resource_limits=resource_limits)
    reason = result['stop_reason']
    if state == 'off' and reason == 'complete': reason = 'state_complete'
    return {'completed': result['completed'], 'stop_reason': reason, 'state': state}
