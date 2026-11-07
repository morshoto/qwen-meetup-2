"""Run one owned worker with bounded resource monitoring.

The sampled process tree and group cleanup are observations, not proof that
escaped descendants or system-owned Metal services have terminated. This helper
is not a frozen protocol or model identity attestation.
"""
import math
import os
import subprocess

from src.arc.rsi_process_monitor import _stop_group
from src.arc.rsi_process_sampling import sample_baseline, sample_process
from src.arc.rsi_reload_stream import watch_reload_stream
from src.arc.rsi_training_resources import training_resource_decision, valid_resource_limits


def _post_reap_group_absent(pgid):
    # Signal zero does not affect a process; this is only a momentary group observation.
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return None
    return False


def run_monitored_worker(argv, *, root, timeout, on_start=None, resource_limits=None,
                         baseline_sampler=sample_baseline, process_sampler=sample_process,
                         interval_seconds=1.0, sample_timeout_seconds=1.0, terminate_grace_seconds=1.0):
    if (not isinstance(argv, list) or not argv or
            any(not isinstance(x, str) or not x or '\x00' in x for x in argv) or
            type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300 or
            any(type(x) not in (int, float) or not math.isfinite(x) or not 0 < x <= 5
                for x in (interval_seconds, sample_timeout_seconds, terminate_grace_seconds)) or
            (on_start is not None and not callable(on_start)) or not callable(baseline_sampler) or
            not callable(process_sampler) or (resource_limits is not None and
                (not valid_resource_limits(resource_limits) or timeout > resource_limits['timeout_seconds'] or
                 interval_seconds > resource_limits['sample_interval_seconds'] or
                 sample_timeout_seconds > resource_limits['sample_interval_seconds']))):
        raise ValueError('invalid monitored worker settings')
    try:
        baseline = baseline_sampler(root)
    except Exception:
        return {'status': 'preflight_refused', 'reason': 'monitor_unavailable', 'baseline': None}
    reason = training_resource_decision(baseline, [baseline], limits=resource_limits)
    if reason is not None:
        return {'status': 'preflight_refused', 'reason': reason, 'baseline': baseline}
    process = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    try:
        if on_start is not None:
            on_start(process.pid)
        observation = watch_reload_stream(process, baseline=baseline, sampler=lambda pid: process_sampler(root, pid),
            timeout_seconds=timeout, interval_seconds=interval_seconds,
            sample_timeout_seconds=sample_timeout_seconds, terminate_grace_seconds=terminate_grace_seconds,
            resource_limits=resource_limits)
    finally:
        try:
            if process.returncode is None:
                _stop_group(process, terminate_grace_seconds)
        finally:
            process.stdout.close()
    post_reap = _post_reap_group_absent(process.pid) if process.returncode is not None else None
    observation['post_reap_group_absent'] = post_reap
    group_absent = post_reap is True
    finished = (observation.get('stop_reason') == 'process_exited' and observation.get('exit_code') == 0 and
                observation.get('stream_complete') is True and group_absent)
    status = 'complete' if finished else ('timeout' if observation.get('stop_reason') == 'wall_clock_limit' else 'error')
    return {'status': status, 'baseline': baseline, 'observation': observation,
            'elapsed_seconds': observation.get('elapsed_seconds'), 'returncode': process.returncode,
            'leader_reaped': process.returncode is not None, 'group_absent': group_absent,
            'stdout': observation.get('output'), 'stderr': None}
