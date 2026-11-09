"""Bounded extra self-generated development examples for the adapter arm."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_local_runner import Backend, SEED, atomic_json, parse_prediction, proxy_task
from src.cli.qwen38_training_data import collect


def expand(repo, root, model_path, manifest, tasks):
    root = Path(root)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    collector = Path(__file__).with_name('qwen38_training_data.py')
    registration = {'manifest_sha256': manifest['sha256'], 'generator_sha256': source_sha,
                    'collector_sha256': hashlib.sha256(collector.read_bytes()).hexdigest(),
                    'max_extra_attempts_per_task': 6, 'seed_base': SEED + 1000}
    path = root / 'training' / 'generation_registration.json'
    if path.exists() and json.loads(path.read_text()) != registration:
        raise ValueError('training generation registration changed')
    if not path.exists(): atomic_json(path, registration)
    backend = Backend(model_path)
    # Two baseline attempts already exist; allow six further seeds per parent.
    try:
        for attempt in range(6):
            for task_id in manifest['allocation']['development']:
                current = collect(repo, root)
                if current['status'] == 'eligible_for_canary':
                    return current
                if any(item['task_id'] == task_id for part in current['splits'].values() for item in part):
                    continue
                path = root / 'training' / 'generated' / task_id / f'{attempt:02}.json'
                if path.exists():
                    continue
                task, expected = proxy_task(tasks[task_id])
                content = json.dumps({'instruction': PROMPT, 'task': task}, sort_keys=True, separators=(',', ':'))
                print(f'training generation: task {task_id}, extra attempt {attempt + 1}/6', flush=True)
                response = backend.call(content, SEED + 1000 + attempt)
                outputs, mode = parse_prediction(response['raw'], 1)
                response.update({'candidate_index': attempt + 2, 'outputs': outputs,
                                 'valid': outputs is not None, 'parse_mode': mode, 'correct': outputs == expected})
                atomic_json(path, {'manifest_sha256': manifest['sha256'], 'task_id': task_id,
                                   'seed': SEED + 1000 + attempt, 'generator_sha256': source_sha, 'candidate': response})
        return collect(repo, root)
    finally:
        del backend
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
