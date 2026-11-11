"""Pinned MLX-VLM LoRA worker for a source-replayed private training dataset."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

from src.arc.qwen38_local_runner import MODEL_REVISION


def _run_trainer(argv, initial):
    from src.cli import qwen38_canary_worker
    os.environ['QWEN38_CANARY_INITIAL'] = str(initial)
    sys.argv = ['mlx_vlm.lora', *argv]
    qwen38_canary_worker.main()


def run_training(model_path, dataset_dir, run_root, train_sha256, *, runner=_run_trainer):
    model_path, dataset_dir, run_root = Path(model_path).resolve(), Path(dataset_dir).resolve(), Path(run_root).resolve()
    if not model_path.is_dir() or model_path.name != MODEL_REVISION:
        raise ValueError('expected pinned local model snapshot')
    if (not dataset_dir.is_dir() or dataset_dir.stat().st_mode & 0o077 or
        {p.name for p in dataset_dir.iterdir()} != {'train.jsonl'} or
        not isinstance(train_sha256, str) or re.fullmatch('[0-9a-f]{64}', train_sha256) is None or
        hashlib.sha256((dataset_dir / 'train.jsonl').read_bytes()).hexdigest() != train_sha256):
        raise ValueError('private training dataset differs from its registration')
    if not run_root.is_dir() or run_root.stat().st_mode & 0o077 or (run_root / 'initial.safetensors').exists() or (run_root / 'adapter').exists():
        raise ValueError('training output already exists or is not private')
    argv = ['--model-path', str(model_path), '--dataset', str(dataset_dir), '--iters', '20', '--batch-size', '1',
            '--max-seq-length', '2048', '--grad-checkpoint', '--train-on-completions',
            '--learning-rate', '0.00002', '--lora-rank', '8', '--steps-per-report', '1',
            '--output-path', str(run_root / 'adapter')]
    runner(argv, run_root / 'initial.safetensors')


def main(argv=None):
    parser = argparse.ArgumentParser()
    for flag in ('model-path', 'dataset-dir', 'run-root', 'train-sha256'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv)
    run_training(args.model_path, args.dataset_dir, args.run_root, args.train_sha256)


if __name__ == '__main__':
    main()
