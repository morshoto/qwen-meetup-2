"""Print local runtime identity without loading model weights or running inference."""
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
from pathlib import Path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    packages = ('mlx', 'mlx-lm', 'transformers', 'tokenizers', 'numpy', 'huggingface-hub', 'jinja2')
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_lm.generate import generate_step
    source = inspect.getsource(generate_step)
    module_file = inspect.getsourcefile(generate_step)
    if not module_file:
        raise ValueError('generation source unavailable')
    executable = os.path.realpath(sys.executable)
    result = {'machine': platform.machine(), 'python_version': platform.python_version(),
              'interpreter_realpath': executable, 'interpreter_sha256': _sha(executable),
              'packages': {name: importlib.metadata.version(name) for name in packages},
              'generate_step_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'generate_module_sha256': _sha(module_file)}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
