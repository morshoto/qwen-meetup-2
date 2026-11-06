"""Supervise one owned task process; system services are outside this check."""
import os
import signal
import subprocess
import tempfile
import time


def run_worker(argv, *, timeout):
    if not isinstance(argv, list) or not argv or timeout <= 0:
        raise ValueError('invalid worker settings')
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
        started = time.monotonic()
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=error, start_new_session=True)
        status = 'complete'
        try:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                status = 'timeout'
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            if status == 'complete' and proc.returncode != 0:
                status = 'error'
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                group_absent = True
            else:
                group_absent = False
            output.seek(0); error.seek(0)
            return {'status': status, 'elapsed_seconds':time.monotonic()-started, 'returncode': proc.returncode, 'leader_reaped': proc.poll() is not None,
                    'group_absent': group_absent, 'stdout': output.read(65536).decode('utf-8', 'replace'),
                    'stderr': error.read(65536).decode('utf-8', 'replace')}
        finally:
            if proc.poll() is None:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait(timeout=10)
