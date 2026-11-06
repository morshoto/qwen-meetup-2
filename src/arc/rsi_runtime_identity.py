"""Observe the proposed local MLX runtime without claiming complete closure."""
import json
import os
import re
import subprocess
from pathlib import Path

from src.arc.guarded_contract import AUDITED_GENERATE_STEP
from src.arc.rsi_model_identity import _hash_file

_PACKAGES = {'mlx', 'mlx-lm', 'transformers', 'tokenizers', 'numpy', 'huggingface-hub', 'jinja2'}
_HASH = re.compile(r'[0-9a-f]{64}')
_PROBE = Path(__file__).resolve().parents[1] / 'cli' / 'rsi_runtime_probe.py'


def snapshot_runtime_identity(python_executable):
    python = Path(os.path.abspath(Path(python_executable).expanduser()))
    try:
        target = python.resolve(strict=True)
        if not target.is_file() or not os.access(python, os.X_OK):
            raise ValueError('runtime interpreter unavailable')
        before = _hash_file(target)
        result = subprocess.run([str(python), '-I', '-B', str(_PROBE)], capture_output=True, text=True,
                                timeout=30, check=False, env={'PATH': '/usr/bin:/bin'})
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('runtime probe unavailable') from exc
    if result.returncode != 0 or len(result.stdout) > 65536:
        raise ValueError('runtime probe failed')
    try:
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError('duplicate runtime key')
                value[key] = item
            return value
        data = json.loads(result.stdout, object_pairs_hook=unique)
    except (ValueError, TypeError) as exc:
        raise ValueError('runtime probe output invalid') from exc
    if (not isinstance(data, dict) or set(data) != {'machine', 'python_version', 'interpreter_realpath',
            'interpreter_sha256', 'packages', 'generate_step_sha256', 'generate_module_sha256'} or
            data['machine'] != 'arm64' or not isinstance(data['python_version'], str) or
            not re.fullmatch(r'3\.\d+\.\d+', data['python_version']) or
            data['interpreter_realpath'] != str(target) or data['interpreter_sha256'] != before or
            not isinstance(data['packages'], dict) or set(data['packages']) != _PACKAGES or
            any(not isinstance(value, str) or not value for value in data['packages'].values()) or
            data['packages']['mlx-lm'] != '0.30.7' or
            data['generate_step_sha256'] != AUDITED_GENERATE_STEP['0.30.7'] or
            not isinstance(data['generate_module_sha256'], str) or not _HASH.fullmatch(data['generate_module_sha256'])):
        raise ValueError('runtime identity differs from proposed contract')
    if python.resolve(strict=True) != target or _hash_file(target) != before:
        raise ValueError('runtime interpreter changed during probe')
    return data


def verify_runtime_identity(python_executable, expected):
    if not isinstance(expected, dict) or snapshot_runtime_identity(python_executable) != expected:
        raise ValueError('local runtime differs from expected manifest')
    return {'status': 'local_runtime_consistent', 'execution_ready': False,
            'evidence_type': 'local_runtime_and_caller_supplied_manifest'}
