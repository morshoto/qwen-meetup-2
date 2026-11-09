"""Run the synthetic 27B trainer probe with bounded resources and retained logs.

This execution probe does not itself certify adapter changes or enable collection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

MODEL_REVISION = '10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
PACKAGES = {'mlx': '0.32.4', 'mlx-vlm': '0.6.8', 'datasets': '5.1.0'}


def resource_limit(free_percent, swap_bytes, minimum_free, maximum_swap):
    if free_percent is None or swap_bytes is None:
        return 'memory_probe_failed'
    if free_percent < minimum_free:
        return 'low_free_memory'
    if swap_bytes > maximum_swap:
        return 'swap_limit'
    return None


def resources():
    free = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True, check=False)
    swap = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True, check=False)
    fm = re.search(r'free percentage:\s*(\d+)%', free.stdout)
    sm = re.search(r'used\s*=\s*([\d.]+)([KMG])', swap.stdout)
    value = int(float(sm.group(1)) * {'K': 1024, 'M': 1024**2, 'G': 1024**3}[sm.group(2)]) if sm else None
    return (int(fm.group(1)) if fm else None), value


def hashes(model):
    files = sorted(model.glob('*.safetensors'))
    if not files:
        raise ValueError('no base weights')
    result = {}
    for path in files:
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(block)
        result[path.name] = sha.hexdigest()
    return result


def run(model, root, timeout=900, minimum_free=10, maximum_swap=16 * 1024**3):
    model, root = Path(model).resolve(), Path(root).resolve()
    if model.name != MODEL_REVISION or model.parent.parent.name != 'models--mlx-community--Qwen3.8-27B-4bit':
        raise ValueError('unexpected model snapshot')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != PACKAGES:
        raise ValueError(f'unexpected toolchain: {versions}')
    root.mkdir(parents=True, exist_ok=False)
    data = root / 'data'; data.mkdir()
    (data / 'train.jsonl').write_text(json.dumps({'messages': [{'role': 'user', 'content': 'Return one.'}, {'role': 'assistant', 'content': '1'}]}) + '\n')
    before = hashes(model)
    command = [sys.executable, '-u', '-m', 'mlx_vlm.lora', '--model-path', str(model), '--dataset', str(data),
               '--iters', '1', '--batch-size', '1', '--max-seq-length', '512', '--grad-checkpoint',
               '--train-on-completions', '--steps-per-report', '1', '--lora-rank', '8', '--learning-rate', '0.00002',
               '--output-path', str(root / 'adapter')]
    registration = {'model_revision': MODEL_REVISION, 'packages': versions, 'base_before': before, 'command': command,
                    'timeout_seconds': timeout, 'minimum_free_percent': minimum_free, 'maximum_swap_bytes': maximum_swap}
    (root / 'registration.json').write_text(json.dumps(registration, indent=2) + '\n')
    samples = []; reason = None; start = time.monotonic(); child = None
    with (root / 'train.log').open('x') as log:
        try:
            initial = resources(); samples.append({'seconds': 0, 'free_percent': initial[0], 'swap_bytes': initial[1]})
            reason = resource_limit(*initial, minimum_free, maximum_swap)
            if reason is None:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    elapsed = time.monotonic() - start
                    current = resources(); samples.append({'seconds': elapsed, 'free_percent': current[0], 'swap_bytes': current[1]})
                    reason = resource_limit(*current, minimum_free, maximum_swap)
                    if elapsed >= timeout: reason = 'timeout'
                    if reason: break
                    time.sleep(5)
        finally:
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
    group_gone = True
    if child:
        try: os.killpg(child.pid, 0); group_gone = False
        except ProcessLookupError: pass
    result = {'status': 'blocked' if reason or not child or child.returncode != 0 or not group_gone else 'execution_only_unverified',
              'reason': reason, 'exit_code': child.returncode if child else None, 'process_group_gone': group_gone,
              'elapsed_seconds': time.monotonic() - start, 'samples': samples, 'base_after': hashes(model)}
    (root / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model_path, args.run_dir), indent=2))


if __name__ == '__main__':
    main()
