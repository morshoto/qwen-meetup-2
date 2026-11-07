"""Run one public task using the configured single-model MLX snapshot."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_live_backend import MLXBackend
from src.arc.lt_public import load_public_export
from src.arc.lt_worker import execute_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'receipt', 'output', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    tasks = load_public_export(json.loads(args.export.read_bytes()), json.loads(args.receipt.read_bytes()))
    if args.task not in tasks: raise ValueError('unregistered task')
    if importlib.metadata.version('mlx-lm') != '0.30.7' or importlib.metadata.version('mlx') != '0.32.4':
        raise ValueError('runtime version differs')
    row = execute_task(tasks[args.task], MLXBackend(args.model_path), '14b')
    row['model'] = 'single'
    _create(args.output, row)


if __name__ == '__main__':
    main()
