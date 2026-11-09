"""Conditional, evidence-gated Qwen3.8 LoRA training and adapter verification."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from src.arc.qwen38_local_runner import Backend, atomic_json, prepare, run_tasks
from src.cli.qwen38_training_data import collect


def run_checked(command, log_path):
    with open(log_path, 'w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                time.sleep(10)
                probe = subprocess.run(['memory_pressure', '-Q'], text=True, capture_output=True, check=False)
                import re
                match = re.search(r'free percentage:\s*(\d+)%', probe.stdout)
                if match and int(match.group(1)) < 5:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    raise RuntimeError('training stopped at system free-memory threshold')
            if child.returncode:
                raise RuntimeError(f'training exited {child.returncode}; see {log_path}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()


def verify_adapter(model_path, adapter, output):
    import mlx.core as mx
    from mlx_vlm import load
    import gc
    weights = mx.load(str(adapter / 'adapters.safetensors'))
    nonzero = sum(int(float(mx.max(mx.abs(v)).item()) > 0) for k, v in weights.items() if 'lora' in k.lower())
    if nonzero == 0: raise ValueError('adapter contains no nonzero LoRA tensors')
    results = []
    for path in (None, str(adapter)):
        model, processor = load(str(model_path), adapter_path=path)
        tok = getattr(processor, 'tokenizer', processor)
        ids = tok.encode('The next number after one is', add_special_tokens=False)
        result = model.language_model(mx.array([ids]))
        logits = result.logits if hasattr(result, 'logits') else result
        mx.eval(logits)
        results.append(logits[0, -1, :].astype(mx.float32).tolist())
        del model, processor, result, logits
        gc.collect(); mx.clear_cache()
    max_delta = max(abs(a-b) for a, b in zip(*results))
    if max_delta <= 0: raise ValueError('adapter did not change logits')
    record = {'adapter_sha256': hashlib.sha256((adapter / 'adapters.safetensors').read_bytes()).hexdigest(),
              'nonzero_lora_tensors': nonzero, 'max_abs_logit_delta': max_delta}
    atomic_json(output, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    repo, root, model_path = Path(args.repo).resolve(), Path(args.run_dir).resolve(), Path(args.model_path).resolve()
    manifest, tasks = prepare(repo, root, model_path)
    data = collect(repo, root)
    if data['status'] != 'eligible_for_canary':
        from src.cli.qwen38_expand_training import expand
        try:
            data = expand(repo, root, model_path, manifest, tasks)
        except Exception as error:
            atomic_json(root / 'weights_blocked.json', {'error': 'example generation: ' + str(error), 'manifest_sha256': manifest['sha256']})
            raise
    train_dir = root / 'training'; train_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(train_dir / 'verified_data.json', data)
    if data['status'] != 'eligible_for_canary':
        atomic_json(root / 'weights_blocked.json', {'error': data['status'], 'manifest_sha256': manifest['sha256']})
        print('Weight arm stopped:', data['status'], flush=True); return
    # Training is deliberately guarded by the real-model one-step canary and adapter-logit verification.
    canary_data = train_dir / 'canary_data'; canary_data.mkdir(exist_ok=True)
    (canary_data / 'train.jsonl').write_text(json.dumps({'messages': [{'role':'user','content':'Return one.'}, {'role':'assistant','content':'1'}]}) + '\n')
    canary = train_dir / 'canary'
    seeded = "import random,runpy,numpy as np,mlx.core as mx;random.seed(20261111);np.random.seed(20261111);mx.random.seed(20261111);runpy.run_module('mlx_vlm.lora',run_name='__main__')"
    command = [sys.executable, '-c', seeded, '--model-path', str(model_path), '--dataset', str(canary_data),
               '--iters', '1', '--batch-size', '1', '--max-seq-length', '512', '--grad-checkpoint',
               '--train-on-completions', '--steps-per-report', '1', '--output-path', str(canary)]
    try:
        if not (canary / 'verification.json').exists():
            if canary.exists(): raise ValueError('unverified canary directory exists; preserve and inspect before retry')
            run_checked(command, train_dir / 'canary.log')
            verify_adapter(model_path, canary, canary / 'verification.json')
    except Exception as error:
        atomic_json(root / 'weights_blocked.json', {'error': 'canary: ' + str(error), 'manifest_sha256': manifest['sha256']})
        print('Weight arm canary failed:', error, flush=True); return
    from mlx_vlm.utils import load_processor
    processor = load_processor(model_path, True)
    tokenizer = getattr(processor, 'tokenizer', processor)
    clean = {}
    for split in ('train', 'valid'):
        clean[split] = []
        for item in data['splits'][split]:
            text = tokenizer.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=False, enable_thinking=False)
            if len(tokenizer.encode(text, add_special_tokens=False)) <= 2048:
                clean[split].append(item)
    if len(clean['train']) < 5 or len({x['task_id'] for x in clean['train']}) < 2 or not clean['valid']:
        atomic_json(root / 'weights_blocked.json', {'error': 'insufficient verified examples under 2048 tokens',
                                                     'manifest_sha256': manifest['sha256']})
        return
    dataset = train_dir / 'dataset'; dataset.mkdir(exist_ok=True)
    for split in ('train', 'valid'):
        path = dataset / (split + '.jsonl')
        text = ''.join(json.dumps({'messages': item['messages']}, ensure_ascii=False) + '\n' for item in clean[split])
        if path.exists() and path.read_text() != text: raise ValueError('training data changed')
        if not path.exists(): path.write_text(text)
    import importlib.metadata as metadata
    training_sources = ['src/cli/qwen38_local_train.py', 'src/cli/qwen38_training_data.py', 'src/cli/qwen38_expand_training.py']
    source_hashes = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in training_sources}
    config = {'manifest_sha256': manifest['sha256'], 'source_files_sha256': source_hashes,
              'training_packages': {name: metadata.version(name) for name in ('mlx', 'mlx-vlm', 'datasets')},
              'train_count': len(clean['train']), 'valid_count': len(clean['valid']),
              'train_sha256': hashlib.sha256((dataset / 'train.jsonl').read_bytes()).hexdigest(),
              'valid_sha256': hashlib.sha256((dataset / 'valid.jsonl').read_bytes()).hexdigest(),
              'steps': 20, 'batch_size': 1, 'max_seq_length': 2048, 'learning_rate': 2e-5,
              'rank': 8, 'gradient_checkpoint': True, 'train_on_completions': True, 'seed': 20261111}
    registration = train_dir / 'registration.json'
    if registration.exists() and json.loads(registration.read_text()) != config: raise ValueError('training registration changed')
    if not registration.exists(): atomic_json(registration, config)
    adapter = train_dir / 'adapter'
    training_cmd = [sys.executable, '-c', seeded, '--model-path', str(model_path), '--dataset', str(dataset),
                    '--iters', '20', '--batch-size', '1', '--max-seq-length', '2048', '--grad-checkpoint',
                    '--train-on-completions', '--learning-rate', '0.00002', '--lora-rank', '8',
                    '--steps-per-report', '1', '--output-path', str(adapter)]
    try:
        if not (adapter / 'verification.json').exists():
            if adapter.exists(): raise ValueError('unverified adapter directory exists; preserve and inspect before retry')
            run_checked(training_cmd, train_dir / 'train.log')
            verification = verify_adapter(model_path, adapter, adapter / 'verification.json')
        else:
            verification = json.loads((adapter / 'verification.json').read_text())
            if verification['adapter_sha256'] != hashlib.sha256((adapter / 'adapters.safetensors').read_bytes()).hexdigest():
                raise ValueError('adapter hash mismatch')
        backend = Backend(model_path, adapter_path=adapter)
        run_tasks(backend, root, manifest, tasks, 'evaluation/weights', manifest['allocation']['orders']['weights'],
                  __import__('src.arc.lt_live_backend', fromlist=['PROMPT']).PROMPT, variant_sha=verification['adapter_sha256'])
        print('Weight arm complete; score offline.', flush=True)
    except Exception as error:
        atomic_json(root / 'weights_blocked.json', {'error': 'training/evaluation: ' + str(error), 'manifest_sha256': manifest['sha256']})
        print('Weight arm stopped:', error, flush=True)


if __name__ == '__main__': main()
