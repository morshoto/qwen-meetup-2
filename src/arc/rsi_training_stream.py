"""Observe a child stream and resources without launching or approving training."""
import io
import os
import select
import subprocess
import threading

from src.arc.rsi_loss_watch import TrainingLossWatch
from src.arc.rsi_process_monitor import _exited_unreaped, watch_owned_process


def watch_training_stream(process, *, baseline, sampler, timeout_seconds, interval_seconds=1.0,
                          sample_timeout_seconds=1.0, terminate_grace_seconds=1.0, resource_limits=None):
    if (not isinstance(process, subprocess.Popen) or process.returncode is not None or
            process.stdout is None or isinstance(process.stdout, io.TextIOBase) or process.stderr is not None):
        raise ValueError('owned process requires a binary stdout pipe and drained stderr')
    try:
        fd = process.stdout.fileno()
        if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
            raise ValueError('process is not an owned session leader')
        _exited_unreaped(process)
    except (OSError, ChildProcessError) as exc:
        raise ValueError('process ownership could not be verified') from exc
    watch = TrainingLossWatch()
    stop, complete = threading.Event(), threading.Event()

    def read_stream():
        total = 0
        try:
            while True:
                readable, _, _ = select.select([fd], [], [], 0.05)
                if not readable:
                    if stop.is_set():
                        return
                    continue
                chunk = os.read(fd, 8192)
                if not chunk:
                    watch.finish()
                    complete.set()
                    return
                total += len(chunk)
                if total > 4 * 1024 * 1024:
                    watch.feed(None)
                    return
                watch.feed(chunk)
        except (OSError, ValueError):
            watch.feed(None)

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    try:
        result = watch_owned_process(process, baseline=baseline, sampler=sampler,
            timeout_seconds=timeout_seconds, interval_seconds=interval_seconds,
            sample_timeout_seconds=sample_timeout_seconds, terminate_grace_seconds=terminate_grace_seconds,
            health_sampler=lambda _: watch.snapshot(), resource_limits=resource_limits)
    finally:
        stop.set()
        reader.join(timeout=1)
    if reader.is_alive():
        raise RuntimeError('training stream reader did not stop')
    result['loss'] = watch.snapshot()
    result['stream_complete'] = complete.is_set()
    result['training_verified'] = False
    return result
