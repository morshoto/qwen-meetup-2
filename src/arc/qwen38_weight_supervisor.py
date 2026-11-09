"""Bound a single owned worker process group and retain private diagnostics."""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from src.cli.qwen38_canary import resource_limit


def system_resources():
    try:
        free = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True, timeout=3, check=False)
        swap = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    fm = re.search(r'free percentage:\s*(\d+)%', free.stdout)
    sm = re.search(r'used\s*=\s*([\d.]+)([KMG])', swap.stdout)
    value = int(float(sm.group(1)) * {'K': 1024, 'M': 1024**2, 'G': 1024**3}[sm.group(2)]) if sm else None
    return (int(fm.group(1)) if fm else None), value


def _group_exists(pid):
    try: os.killpg(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


def _signal_group(pid, sig):
    try: os.killpg(pid, sig)
    except ProcessLookupError: pass


def _private_write(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())


def run_supervised(command, request, directory, *, timeout_seconds, probe=system_resources,
                   poll_seconds=1, min_free=10, max_swap=16 * 1024**3, max_output_bytes=2 * 1024**2,
                   cwd=None):
    """Run once; an interrupted caller must reconcile rather than repeat the attempt."""
    if (not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command) or
        isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or not callable(probe) or
        isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)) or poll_seconds <= 0 or
        type(max_output_bytes) is not int or max_output_bytes <= 0):
        raise ValueError('invalid worker limits or command')
    directory = Path(directory)
    directory.mkdir(mode=0o700, exist_ok=False)
    request_bytes = (json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    _private_write(directory / 'request.json', request_bytes)
    child = None; reason = None; samples = []; start = time.monotonic()
    with (directory / 'request.json').open('rb') as stdin, \
         os.fdopen(os.open(directory / 'stdout.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stdout, \
         os.fdopen(os.open(directory / 'stderr.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stderr:
        try:
            free, swap = probe()
            samples.append({'seconds': 0.0, 'free_percent': free, 'swap_bytes': swap})
            reason = resource_limit(free, swap, min_free, max_swap)
            if reason is None:
                child = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr,
                                         cwd=cwd, start_new_session=True)
                while child.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed >= timeout_seconds:
                        reason = 'timeout'; break
                    if os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size > max_output_bytes:
                        reason = 'output_limit'; break
                    free, swap = probe()
                    samples.append({'seconds': elapsed, 'free_percent': free, 'swap_bytes': swap})
                    reason = resource_limit(free, swap, min_free, max_swap)
                    if reason: break
                    time.sleep(min(poll_seconds, max(0, timeout_seconds - (time.monotonic() - start))))
        except (OSError, ValueError, TypeError) as error:
            reason = f'supervisor_error:{type(error).__name__}'
        finally:
            if child is not None and child.poll() is None:
                _signal_group(child.pid, signal.SIGTERM)
                try: child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    _signal_group(child.pid, signal.SIGKILL); child.wait()
            if child is not None and _group_exists(child.pid):
                reason = reason or 'remaining_process_group'
                _signal_group(child.pid, signal.SIGTERM)
                deadline = time.monotonic() + 1
                while _group_exists(child.pid) and time.monotonic() < deadline:
                    time.sleep(.02)
                if _group_exists(child.pid): _signal_group(child.pid, signal.SIGKILL)
    gone = child is None or not _group_exists(child.pid)
    result = {'status': 'complete' if child is not None and child.returncode == 0 and reason is None and gone else 'error',
              'reason': reason if reason is not None else (None if child and child.returncode == 0 and gone else 'worker_failed'),
              'exit_code': child.returncode if child else None, 'process_group_gone': gone,
              'elapsed_seconds': time.monotonic() - start, 'samples': samples}
    _private_write(directory / 'execution.json', (json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode())
    return result
