"""Out-of-process deadline watchdog for the dedicated guarded model server.

The watchdog receives absolute monotonic deadlines over stdin. Closing stdin or
sending ``cancel`` disarms it. It only signals the process that spawned it.
"""
from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time


def main() -> None:
    parent = int(sys.argv[1])
    deadline = float(sys.argv[2])
    lock = threading.Lock()
    state = {'armed': True, 'deadline': deadline}
    timer: threading.Timer | None = None

    def expire() -> None:
        with lock:
            if state['armed'] and os.getppid() == parent:
                os.kill(parent, getattr(signal, 'SIGKILL', signal.SIGTERM))

    def arm() -> None:
        nonlocal timer
        delay = max(0.0, state['deadline'] - time.monotonic())
        timer = threading.Timer(delay, expire)
        timer.daemon = True
        timer.start()

    if parent <= 0 or not math.isfinite(deadline):
        raise ValueError('invalid watchdog arguments')
    arm()
    print('ready', flush=True)
    try:
        for line in sys.stdin:
            command = line.strip()
            if command == 'cancel':
                break
            new_deadline = float(command)
            if not math.isfinite(new_deadline):
                raise ValueError('invalid deadline')
            with lock:
                if new_deadline < state['deadline']:
                    state['deadline'] = new_deadline
                    if timer is not None:
                        timer.cancel()
                    arm()
    finally:
        with lock:
            state['armed'] = False
            if timer is not None:
                timer.cancel()

if __name__ == '__main__':
    main()
