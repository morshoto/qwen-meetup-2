"""Check H100 runtime prerequisites before downloading or loading a model."""
import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def check_compiler():
    name = os.environ.get('CC') or 'cc'
    compiler = shutil.which(name)
    if not compiler:
        raise RuntimeError('C compiler not found; install build-essential or set CC to a compiler executable')
    try:
        with tempfile.TemporaryDirectory(prefix='h100-cc-') as tmp:
            source, binary = Path(tmp) / 'check.c', Path(tmp) / 'check'
            source.write_text('int main(void) { return 0; }\n')
            subprocess.run([compiler, str(source), '-o', str(binary)], check=True, capture_output=True, timeout=30)
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError('C compiler could not compile and run a test program') from exc
    return compiler


def check_version(expected):
    try:
        actual = importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('vLLM is not installed in this Python environment') from exc
    if actual != expected:
        raise RuntimeError(f'vLLM version differs: expected {expected}, found {actual}')
    return actual


def check_cuda(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable to PyTorch')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support BF16')
    return {'device': torch.cuda.get_device_name(0), 'cuda_runtime': torch.version.cuda}


def check(version):
    compiler = check_compiler()
    actual = check_version(version)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is not installed in this Python environment') from exc
    return {'compiler': compiler, 'vllm_version': actual, **check_cuda(torch)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-version', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.vllm_version), sort_keys=True))
    except RuntimeError as exc:
        parser.exit(1, f'preflight failed: {exc}\n')


if __name__ == '__main__':
    main()
