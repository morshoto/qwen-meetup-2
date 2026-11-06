"""LT-only wrapper checking the named Docker container after each execution."""
import subprocess
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from src.arc.http_backend import BackendError
from src.arc.sandbox import DockerSandbox


class CheckedSandbox:
    def __init__(self, **kwargs):
        self.inner = DockerSandbox(**kwargs)
        self.observations = []

    def run(self, code, grid):
        ident = uuid.uuid4()
        name = 'arc-' + ident.hex
        try:
            with patch('src.arc.sandbox.uuid', SimpleNamespace(uuid4=lambda: ident)):
                return self.inner.run(code, grid)
        finally:
            try:
                proc = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'],
                                      capture_output=True, text=True, timeout=10, check=True)
                absent = name not in proc.stdout.splitlines()
            except (OSError, subprocess.SubprocessError) as exc:
                self.observations.append({'name': name, 'cleanup': 'unverified'})
                raise BackendError('cleanup_unverified', 'container cleanup check failed') from exc
            self.observations.append({'name': name, 'cleanup': 'absent' if absent else 'present'})
            if not absent:
                raise BackendError('cleanup_unverified', 'container remains after execution')
