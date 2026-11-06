"""Watch an explicitly owned child session; never start or authorize training.

Sampling evidence is caller supplied. Escaped sessions, model identity, adapter
integrity, and training success require separate verification. Do not use this
module as a training gate.
"""
import math
import ctypes
import os
import platform
import queue
import signal
import subprocess
import threading
import time

from src.arc.rsi_training_resources import training_resource_decision, valid_resource_limits

_FIELDS = ('monotonic_seconds', 'disk_free_bytes', 'memory_free_percent',
           'swap_used_mib', 'process_tree_rss_bytes')


def _exited_unreaped(process: subprocess.Popen):
    # WNOWAIT retains the child PID until group cleanup, preventing PID reuse.
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    if hasattr(os, 'waitid'):
        return os.waitid(os.P_PID, process.pid, flags)
    if platform.system() != 'Darwin' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError('unreaped child observation is unavailable')
    # Darwin's public sys/signal.h siginfo_t layout on 64-bit macOS.
    class SigInfo(ctypes.Structure):
        _fields_ = [('signo', ctypes.c_int), ('error', ctypes.c_int), ('code', ctypes.c_int),
                    ('pid', ctypes.c_int), ('uid', ctypes.c_uint), ('status', ctypes.c_int),
                    ('addr', ctypes.c_void_p), ('value', ctypes.c_long), ('band', ctypes.c_long),
                    ('padding', ctypes.c_ulong * 7)]
    info = SigInfo()
    libc = ctypes.CDLL(None, use_errno=True)
    waitid = libc.waitid
    waitid.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.POINTER(SigInfo), ctypes.c_int)
    waitid.restype = ctypes.c_int
    if waitid(os.P_PID, process.pid, ctypes.byref(info), flags) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if info.pid not in (0, process.pid):
        raise OSError('waitid returned an unexpected child')
    return info if info.pid else None


def _stop_group(process: subprocess.Popen, grace: float) -> tuple[int, bool, dict]:
    group_signals_confirmed = True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        group_signals_confirmed = False
    deadline = time.monotonic() + grace
    while _exited_unreaped(process) is None and time.monotonic() < deadline:
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    # The leader remains unreaped until after this signal, so its PID cannot be
    # reassigned to an unrelated group between ownership validation and cleanup.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        group_signals_confirmed = False
    # Capture only a bounded, unverified listing while the leader PID remains reserved.
    # It neither changes the signal result nor establishes escaped-process cleanup.
    try:
        from src.arc.rsi_group_observation import observe_exited_group
        observation = observe_exited_group(process)
    except (OSError, ValueError):
        observation = {'status': 'group_observation_unavailable', 'group_cleanup_verified': False,
                       'escaped_processes_excluded': False}
    try:
        return process.wait(timeout=max(1.0, grace)), group_signals_confirmed, observation
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('owned process termination could not be confirmed') from exc


def _timed_sample(sampler, pid: int, timeout: float):
    result = queue.Queue(maxsize=1)
    def collect():
        try:
            result.put((True, sampler(pid)))
        except Exception as exc:
            result.put((False, exc))
    threading.Thread(target=collect, daemon=True).start()
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty:
        return None
    return value if ok else None


def watch_owned_process(process: subprocess.Popen, *, baseline: dict, sampler,
                        timeout_seconds: float, interval_seconds: float = 1.0,
                        sample_timeout_seconds: float = 1.0,
                        terminate_grace_seconds: float = 1.0, health_sampler=None, resource_limits=None,
                        identity_sampler=None, identity_processor=None) -> dict:
    """Monitor a child that is its own session leader, stopping its process group.

    The function requires POSIX waitid/WNOWAIT to retain the child's identity
    until cleanup. It does not start the process, authenticate the sampler, or
    prove that processes which escaped the session have terminated.
    """
    numeric = (timeout_seconds, interval_seconds, sample_timeout_seconds, terminate_grace_seconds)
    if (not isinstance(process, subprocess.Popen) or process.returncode is not None or not callable(sampler) or
            (health_sampler is not None and not callable(health_sampler)) or
            (identity_sampler is not None and not callable(identity_sampler)) or
            (identity_processor is not None and (identity_sampler is None or not callable(identity_processor))) or
            any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in numeric) or
            interval_seconds > 5 or sample_timeout_seconds > 5 or terminate_grace_seconds > 5 or
            (resource_limits is not None and (not valid_resource_limits(resource_limits) or
                timeout_seconds > resource_limits['timeout_seconds'] or
                interval_seconds > resource_limits['sample_interval_seconds'] or
                sample_timeout_seconds > resource_limits['sample_interval_seconds'])) or
            not all(hasattr(os, name) for name in ('WEXITED', 'WNOHANG', 'WNOWAIT', 'P_PID', 'getsid'))):
        raise ValueError('invalid owned process monitor configuration')
    try:
        if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
            raise ValueError('process is not an owned session leader')
        _exited_unreaped(process)  # raises ChildProcessError when it is not our child
    except (OSError, ChildProcessError) as exc:
        raise ValueError('process ownership could not be verified') from exc
    start = time.monotonic()
    samples = []
    identity_observations = []
    stop_reason = None
    try:
        baseline_reason = training_resource_decision(baseline, [baseline], limits=resource_limits)
        if baseline_reason is not None:
            stop_reason = baseline_reason
        elif abs(start - baseline.get('monotonic_seconds', float('inf'))) > 5:
            stop_reason = 'monitor_evidence_invalid'
        while stop_reason is None:
            remaining = timeout_seconds - (time.monotonic() - start)
            if remaining <= 0:
                stop_reason = 'wall_clock_limit'
                break
            value = _timed_sample(sampler, process.pid, min(sample_timeout_seconds, remaining))
            sampled_at = time.monotonic()
            if not isinstance(value, dict) or any(key not in value for key in _FIELDS):
                stop_reason = 'monitor_unavailable'
                break
            state = {key: value[key] for key in _FIELDS}
            stamp = state['monotonic_seconds']
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or abs(sampled_at - stamp) > 5:
                stop_reason = 'monitor_evidence_invalid'
                break
            samples.append(state)
            stop_reason = training_resource_decision(baseline, samples, limits=resource_limits)
            if stop_reason is not None:
                break
            if identity_sampler is not None:
                remaining = timeout_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    stop_reason = 'wall_clock_limit'
                    break
                identity = _timed_sample(identity_sampler, process.pid, min(sample_timeout_seconds, remaining))
                if identity is not None and identity_processor is not None:
                    try:
                        identity = identity_processor(identity)
                    except Exception:
                        identity = None
                keys = ('live_observed_pids', 'zombie_observed_pids', 'missing_observed_pids', 'reused_pids')
                if (not isinstance(identity, dict) or
                        any(not isinstance(identity.get(key), list) or
                            any(type(pid) is not int or pid <= 0 for pid in identity[key]) for key in keys) or
                        identity.get('cleanup_verified') is not False or
                        identity.get('escaped_processes_excluded') is not False):
                    stop_reason = 'identity_monitor_unavailable'
                    break
                identity_observations.append({key: identity[key] for key in (*keys, 'cleanup_verified',
                                                                              'escaped_processes_excluded')})
                if identity['reused_pids']:
                    stop_reason = 'identity_reuse_ambiguous'
                    break
            if health_sampler is not None:
                remaining = timeout_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    stop_reason = 'wall_clock_limit'
                    break
                health = _timed_sample(health_sampler, process.pid, min(sample_timeout_seconds, remaining))
                if not isinstance(health, dict) or health.get('status') not in ('ok', 'nonfinite_loss'):
                    stop_reason = 'health_monitor_unavailable'
                    break
                if health['status'] == 'nonfinite_loss':
                    stop_reason = 'nonfinite_loss'
                    break
            if _exited_unreaped(process) is not None:
                stop_reason = 'process_exited'
                break
            remaining = timeout_seconds - (time.monotonic() - start)
            if remaining <= 0:
                stop_reason = 'wall_clock_limit'
                break
            time.sleep(min(interval_seconds, remaining))
    except BaseException:
        _stop_group(process, terminate_grace_seconds)
        raise
    exit_code, group_signals_confirmed, cleanup_observation = _stop_group(process, terminate_grace_seconds)
    return {'stop_reason': stop_reason, 'exit_code': exit_code, 'samples': samples,
            'identity_observations': identity_observations,
            'elapsed_seconds': time.monotonic() - start,
            'training_verified': False, 'group_signals_confirmed': group_signals_confirmed,
            'cleanup_observation': cleanup_observation,
            'escaped_processes_excluded': False}
