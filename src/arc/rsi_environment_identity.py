"""Observe local virtual-environment files without claiming full closure."""
import json
import os
import re
import subprocess
from pathlib import Path

from src.arc.rsi_model_identity import _hash_file

_PROBE = Path(__file__).resolve().parents[1] / 'cli' / 'rsi_environment_probe.py'
_HASH = re.compile(r'[0-9a-f]{64}')


def snapshot_environment(python_executable):
    python = Path(os.path.abspath(Path(python_executable).expanduser()))
    try:
        target = python.resolve(strict=True)
        env_root = python.parent.resolve(strict=True).parent
        if not target.is_file() or not os.access(python, os.X_OK):
            raise ValueError('runtime interpreter unavailable')
        before = _hash_file(target)
        result = subprocess.run([str(python), '-I', '-B', str(_PROBE)], capture_output=True, text=True,
                                timeout=120, check=False, env={'PATH': '/usr/bin:/bin'})
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('environment probe unavailable') from exc
    if result.returncode != 0 or len(result.stdout) > 65536:
        raise ValueError('environment probe failed')
    try:
        data = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise ValueError('environment probe output invalid') from exc
    if (not isinstance(data, dict) or set(data) != {'environment_root', 'interpreter_realpath',
            'interpreter_sha256', 'inventory_sha256', 'file_count', 'directory_count', 'total_bytes', 'scope'} or
            data['environment_root'] != str(env_root) or data['interpreter_realpath'] != str(target) or
            data['interpreter_sha256'] != before or data['scope'] != 'venv_files_only' or
            not isinstance(data['inventory_sha256'], str) or not _HASH.fullmatch(data['inventory_sha256']) or
            type(data['file_count']) is not int or not 0 < data['file_count'] <= 20000 or
            type(data['directory_count']) is not int or data['directory_count'] < 1 or
            type(data['total_bytes']) is not int or not 0 <= data['total_bytes'] <= 4 * 1024 ** 3):
        raise ValueError('environment identity differs from interpreter')
    if python.resolve(strict=True) != target or _hash_file(target) != before:
        raise ValueError('interpreter changed during environment probe')
    return data


def verify_environment(python_executable, expected):
    if not isinstance(expected, dict) or snapshot_environment(python_executable) != expected:
        raise ValueError('virtual environment differs from expected inventory')
    return {'status': 'local_environment_consistent', 'execution_ready': False,
            'evidence_type': 'local_venv_and_caller_supplied_manifest'}
