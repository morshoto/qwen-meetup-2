"""Execute one public LT task in a dedicated native MLX process."""
import argparse
import json
import importlib.metadata
from pathlib import Path
from src.arc.lt_batch_journal import _create
from src.arc.lt_live_backend import MLXBackend
from src.arc.lt_public import load_public_export
from src.arc.lt_worker import execute_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'receipt', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', choices=('14b','32b'), required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    args = parser.parse_args()
    export = json.loads(args.export.read_bytes()); receipt = json.loads(args.receipt.read_bytes())
    tasks = load_public_export(export, receipt)
    if args.task not in tasks: raise ValueError('unregistered task')
    if importlib.metadata.version('mlx-lm') != '0.30.7' or importlib.metadata.version('mlx') != '0.32.4':
        raise ValueError('runtime version differs')
    backend = MLXBackend(args.model_path)
    row = execute_task(tasks[args.task], backend, args.model)
    _create(args.output, row)


if __name__ == '__main__': main()
