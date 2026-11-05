"""Container sandbox boundary; FakeSandbox never interprets code."""
from __future__ import annotations
import json
import shutil
import subprocess
import threading
import uuid
from typing import Callable, Protocol
from src.arc.data import Grid, validate_grid
from src.arc.http_backend import BackendError


class Sandbox(Protocol):
    def run(self, code: str, grid: Grid) -> Grid: ...


IDENTITY_CODE = "def transform(grid):\n    return [row[:] for row in grid]"
ZERO_CODE = "def transform(grid):\n    return [[0 for _ in row] for row in grid]"
IDENTITY_CODE_VARIANTS = frozenset({
    IDENTITY_CODE,
    "def transform(grid):\n    return [list(row) for row in grid]",
    "def transform(grid):\n    result = []\n    for row in grid:\n        result.append(row[:])\n    return result",
})


class FakeSandbox:
    def run(self, code: str, grid: Grid) -> Grid:
        if code in IDENTITY_CODE_VARIANTS:
            return [row[:] for row in grid]
        if code == ZERO_CODE:
            return [[0 for _ in row] for row in grid]
        raise ValueError("unknown fake operation")


class DockerSandbox:
    def __init__(self, image: str = 'arc-sandbox:local', *, timeout: float = 2, memory_mb: int = 128,
                 cpus: float = 1, pids: int = 32, max_output_bytes: int = 65536,
                 max_code_bytes: int = 65536, reserve: Callable[[], None] | None = None,
                 remaining: Callable[[], float] | None = None):
        self.image, self.timeout, self.memory_mb, self.cpus, self.pids = image, timeout, memory_mb, cpus, pids
        self.max_output_bytes, self.max_code_bytes, self.reserve, self.remaining = max_output_bytes, max_code_bytes, reserve, remaining

    def run(self, code: str, grid: Grid) -> Grid:
        validate_grid(grid)
        if len(code.encode('utf-8')) > self.max_code_bytes: raise ValueError('code exceeds sandbox input limit')
        docker = shutil.which('docker')
        if not docker: raise BackendError('sandbox_unavailable', 'Docker unavailable; refusing to run code')
        if self.reserve: self.reserve()
        timeout = min(self.timeout, self.remaining()) if self.remaining else self.timeout
        name = 'arc-' + uuid.uuid4().hex
        command = [docker, 'run', '--rm', '--name', name, '--pull=never', '--network=none', '--read-only',
                   '--user=65534:65534', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                   f'--pids-limit={self.pids}', f'--memory={self.memory_mb}m', f'--memory-swap={self.memory_mb}m',
                   f'--cpus={self.cpus}', '--interactive', self.image]
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc: raise BackendError('sandbox_unavailable', 'Docker unavailable; refusing to run code') from exc
        def stop_container():
            try:
                subprocess.run([docker, 'kill', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        output, errors = bytearray(), bytearray()
        too_large = threading.Event()
        def read_bounded(stream, destination):
            while True:
                try: chunk = stream.read(4096)
                except (ValueError, OSError): break
                if not chunk: break
                remaining = self.max_output_bytes - len(destination)
                destination.extend(chunk[:max(0, remaining)])
                if len(chunk) > remaining:
                    too_large.set()
                    proc.kill()
                    break
        threads = [threading.Thread(target=read_bounded, args=(proc.stdout, output), daemon=True),
                   threading.Thread(target=read_bounded, args=(proc.stderr, errors), daemon=True)]
        for thread in threads: thread.start()
        input_error = []
        def write_input():
            try:
                assert proc.stdin is not None
                proc.stdin.write(json.dumps({'code': code, 'grid': grid}).encode())
                proc.stdin.close()
            except (BrokenPipeError, OSError) as exc:
                input_error.append(exc)
        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            stop_container()
            proc.kill(); proc.wait()
            raise TimeoutError('sandbox deadline exceeded') from exc
        finally:
            if too_large.is_set():
                stop_container()
            writer.join(timeout=1)
            for thread in threads: thread.join(timeout=1)
            for stream in (proc.stdout, proc.stderr):
                if stream: stream.close()
        if input_error:
            raise RuntimeError('Docker failed to accept input') from input_error[0]
        if too_large.is_set():
            raise RuntimeError('sandbox output limit exceeded')
        if proc.returncode in (125, 126, 127):
            raise BackendError('sandbox_unavailable', 'Docker could not start the isolated runner')
        if proc.returncode != 0:
            raise RuntimeError('sandbox execution failed: ' + errors.decode('utf-8', 'replace')[:1024])
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value: raise ValueError('duplicate JSON key')
                value[key] = item
            return value
        try:
            result = json.loads(output, object_pairs_hook=unique)
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            raise ValueError('sandbox returned invalid JSON') from exc
        if not isinstance(result, dict) or set(result) != {'grid'}:
            raise ValueError('sandbox returned unexpected JSON schema')
        try:
            return validate_grid(result['grid'])
        except (ValueError, TypeError) as exc:
            raise ValueError('sandbox returned invalid grid shape or values') from exc
