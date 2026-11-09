"""Single matched exploratory request with a hash-bound saved adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from src.arc.qwen38_local_runner import Backend, SEED
from src.cli.qwen38_weight_worker import run_request


def _sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def run_adapter_request(payload, model_path, adapter_path, identity, *, backend_factory=Backend):
    adapter = Path(adapter_path).resolve()
    if (not adapter.is_dir() or adapter.stat().st_mode & 0o077 or
        not isinstance(identity, dict) or set(identity) != {'adapter_sha256', 'adapter_config_sha256'} or
        any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None for v in identity.values())):
        raise ValueError('invalid private adapter identity')
    paths = {'adapter_sha256': adapter / 'adapters.safetensors', 'adapter_config_sha256': adapter / 'adapter_config.json'}
    if any(not path.is_file() or path.is_symlink() or _sha(path) != identity[key] for key, path in paths.items()):
        raise ValueError('adapter files changed')
    request = payload.get('request', {}) if isinstance(payload, dict) else {}
    index = request.get('candidate_index')
    if type(index) is not int or index not in (0, 1) or request.get('seed') != SEED + index or payload.get('max_tokens') != 4096:
        raise ValueError('evaluation settings differ from the baseline')
    result = run_request(payload, model_path, backend_factory=lambda model: backend_factory(model, adapter_path=adapter))
    if any(_sha(path) != identity[key] for key, path in paths.items()):
        raise ValueError('adapter files changed during inference')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    for flag in ('model-path', 'adapter-path', 'adapter-sha256', 'adapter-config-sha256', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv)
    raw = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('worker input too large')
    payload = json.loads(raw)
    result = run_adapter_request(payload, args.model_path, args.adapter_path,
                                 {'adapter_sha256': args.adapter_sha256, 'adapter_config_sha256': args.adapter_config_sha256})
    data = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
