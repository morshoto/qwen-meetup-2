"""Run one public task using the configured single-model MLX snapshot."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_live_backend import MLXBackend
from src.arc.lt_public import load_public_export
from src.arc.lt_worker import execute_task


def make_backend(name, version, model_path):
    if name == 'vllm':
        if importlib.metadata.version('vllm') != version:
            raise ValueError('runtime version differs')
        from src.arc.lt_vllm_backend import VLLMBackend
        return VLLMBackend(model_path)
    if name != 'mlx' or importlib.metadata.version('mlx-lm') != '0.30.7' or importlib.metadata.version('mlx') != '0.32.4':
        raise ValueError('runtime version differs')
    return MLXBackend(model_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'receipt', 'output', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--backend', choices=('mlx', 'vllm'), default='mlx')
    parser.add_argument('--backend-version')
    args = parser.parse_args()
    tasks = load_public_export(json.loads(args.export.read_bytes()), json.loads(args.receipt.read_bytes()))
    if args.task not in tasks: raise ValueError('unregistered task')
    row = execute_task(tasks[args.task], make_backend(args.backend, args.backend_version, args.model_path), '14b')
    row['model'] = 'single'
    _create(args.output, row)


if __name__ == '__main__':
    main()
