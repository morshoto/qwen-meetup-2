"""Capture bounded trainer logs while monitoring an already-owned child.

This wrapper does not launch a process, establish isolation, attest samples or
prove a model update. It preserves the original RSI monitor and loss parser.
"""
import io
import os
import select
import subprocess
import threading

from src.arc.rsi_loss_watch import TrainingLossWatch
from src.arc.rsi_process_monitor import _exited_unreaped, watch_owned_process


def watch_phase_b_training(process, *, baseline, sampler, timeout_seconds, interval_seconds=1.0,
                           sample_timeout_seconds=1.0, terminate_grace_seconds=1.0,
                           resource_limits=None, identity_sampler=None, identity_processor=None,
                           max_log_bytes=4 * 1024 * 1024):
    if (not isinstance(process, subprocess.Popen) or process.returncode is not None or
            process.stdout is None or isinstance(process.stdout, io.TextIOBase) or process.stderr is not None or
            type(max_log_bytes) is not int or not 0 < max_log_bytes <= 4 * 1024 * 1024):
        raise ValueError('owned process and bounded binary stream required')
    try:
        fd = process.stdout.fileno()
        if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
            raise ValueError('process is not an owned session leader')
        _exited_unreaped(process)
    except (OSError, ChildProcessError) as exc:
        raise ValueError('process ownership could not be verified') from exc
    watch = TrainingLossWatch()
    stop, complete = threading.Event(), threading.Event()
    log = bytearray()

    def reader():
        try:
            while True:
                readable, _, _ = select.select([fd], [], [], 0.05)
                if not readable:
                    if stop.is_set(): return
                    continue
                chunk = os.read(fd, 8192)
                if not chunk:
                    watch.finish()
                    complete.set()
                    return
                available = max_log_bytes - len(log)
                log.extend(chunk[:available])
                if len(chunk) > available:
                    watch.feed(None)
                    return
                watch.feed(chunk)
        except (OSError, ValueError):
            watch.feed(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        result = watch_owned_process(process, baseline=baseline, sampler=sampler,
            timeout_seconds=timeout_seconds, interval_seconds=interval_seconds,
            sample_timeout_seconds=sample_timeout_seconds, terminate_grace_seconds=terminate_grace_seconds,
            resource_limits=resource_limits, health_sampler=lambda _: watch.snapshot(),
            identity_sampler=identity_sampler, identity_processor=identity_processor)
    finally:
        stop.set()
        thread.join(timeout=1)
    if thread.is_alive():
        raise RuntimeError('training log reader did not stop')
    try:
        text = bytes(log).decode('utf-8')
    except UnicodeDecodeError:
        text = bytes(log).decode('utf-8', 'replace')
        complete.clear()
    result.update({'loss': watch.snapshot(), 'log': text, 'stream_complete': complete.is_set(),
                   'training_verified': False})
    return result
