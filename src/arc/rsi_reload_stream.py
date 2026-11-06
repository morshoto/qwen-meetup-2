"""Bounded reload-process observation, without launching or authenticating a process."""
import io
import math
import os
import re
import select
import subprocess
import threading

from src.arc.rsi_process_monitor import _exited_unreaped, watch_owned_process
from src.arc.rsi_training_resources import training_resource_decision, valid_resource_limits
from src.arc.rsi_transport_audit import _json_object


def watch_reload_stream(process, *, baseline, sampler, timeout_seconds, interval_seconds=1.0,
                        sample_timeout_seconds=1.0, terminate_grace_seconds=1.0, resource_limits=None,
                        identity_sampler=None, identity_processor=None):
    if (not isinstance(process, subprocess.Popen) or process.returncode is not None or process.stdout is None or
            isinstance(process.stdout, io.TextIOBase) or process.stderr is not None):
        raise ValueError('reload process requires an owned binary output pipe')
    try:
        fd = process.stdout.fileno()
        if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
            raise ValueError('reload process is not an owned session leader')
        _exited_unreaped(process)
    except (OSError, ChildProcessError) as exc:
        raise ValueError('reload process ownership could not be verified') from exc
    stop, complete, failed = threading.Event(), threading.Event(), threading.Event()
    content = bytearray()

    def read_stream():
        try:
            while True:
                readable, _, _ = select.select([fd], [], [], 0.05)
                if not readable:
                    if stop.is_set(): return
                    continue
                chunk = os.read(fd, 8192)
                if not chunk:
                    complete.set()
                    return
                if len(content) + len(chunk) > 65536:
                    failed.set()
                    return
                content.extend(chunk)
        except (OSError, ValueError):
            failed.set()

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    try:
        result = watch_owned_process(process, baseline=baseline, sampler=sampler, timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds, sample_timeout_seconds=sample_timeout_seconds,
            terminate_grace_seconds=terminate_grace_seconds, resource_limits=resource_limits,
            health_sampler=lambda _: {'status': 'unavailable' if failed.is_set() else 'ok'},
            identity_sampler=identity_sampler, identity_processor=identity_processor)
    finally:
        stop.set()
        reader.join(timeout=1)
    if reader.is_alive():
        raise RuntimeError('reload stream reader did not stop')
    try:
        text = bytes(content).decode('utf-8') if complete.is_set() and not failed.is_set() else None
    except UnicodeDecodeError:
        text = None
    result.update({'stream_complete': text is not None, 'output': text, 'reload_verified': False,
                   'training_allowed': False, 'promotion_allowed': False})
    return result


def review_reload_observation(*, baseline, observation, model_key, adapter_sha256, resource_limits=None):
    def reject(reason):
        return {'status': 'reload_observation_rejected', 'reason': reason, 'reload_verified': False,
                'training_allowed': False, 'promotion_allowed': False}
    if (model_key not in ('14b', '32b') or not isinstance(adapter_sha256, str) or
            not re.fullmatch(r'[0-9a-f]{64}', adapter_sha256) or
            (resource_limits is not None and not valid_resource_limits(resource_limits))):
        return reject('invalid_registration')
    if (not isinstance(observation, dict) or observation.get('stop_reason') != 'process_exited' or
            type(observation.get('exit_code')) is not int or observation['exit_code'] != 0 or
            observation.get('stream_complete') is not True or observation.get('group_signals_confirmed') is not True or
            observation.get('reload_verified') is not False or observation.get('training_allowed') is not False):
        return reject('process_or_stream_incomplete')
    elapsed = observation.get('elapsed_seconds')
    timeout = resource_limits['timeout_seconds'] if resource_limits is not None else 300
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= timeout:
        return reject('invalid_elapsed_time')
    samples = observation.get('samples')
    reason = training_resource_decision(baseline, samples, limits=resource_limits)
    if reason is not None: return reject(reason)
    if samples[-1]['monotonic_seconds'] - baseline['monotonic_seconds'] > timeout:
        return reject('sample_window_exceeded')
    try:
        text = observation.get('output')
        if not isinstance(text, str) or len(text.encode('utf-8')) > 65536:
            return reject('invalid_probe_output')
        output = _json_object(text.encode('utf-8'))
    except (ValueError, UnicodeError):
        return reject('invalid_probe_output')
    expected = {'status': 'model_reload_observed_unverified', 'model_key': model_key,
                'matched_tensor_count': 28, 'adapter_sha256': adapter_sha256,
                'reload_verified': False, 'training_allowed': False, 'promotion_allowed': False,
                'evidence_type': 'caller_supplied_runtime_and_local_files'}
    if (output != expected or type(output.get('matched_tensor_count')) is not int or
            any(output.get(key) is not False for key in ('reload_verified', 'training_allowed', 'promotion_allowed'))):
        return reject('probe_output_differs')
    return {'status': 'reload_observation_consistent_unverified', 'reload_verified': False,
            'training_allowed': False, 'promotion_allowed': False,
            'remaining_gates': ['independent_execution_provenance', 'process_containment_proof', 'independent_review']}
