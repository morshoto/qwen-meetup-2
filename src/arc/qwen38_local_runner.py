"""Resumable exploratory Qwen3.8 development and comparison inference.

This module never reads ARC test-solution files. Run scoring separately.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from src.arc.data import validate_grid
from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_rsi_comparison import allocate, digest

MODEL_ID = 'mlx-community/Qwen3.8-27B-4bit'
MODEL_REVISION = '10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
SEED = 20261111
MAX_TOKENS = 4096
MAX_CONTEXT = 18432
TEMPERATURE = 0.6


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def parse_prediction(raw, count):
    try:
        return normalize_direct(raw, count), 'strict'
    except (ValueError, TypeError):
        pass
    value = raw.strip()
    repairs = []
    if value.startswith('<think>') and value.count('</think>') == 1:
        value = value.split('</think>', 1)[1].strip()
        repairs.append('remove_thinking_block')
    if value.startswith('```') and value.endswith('```'):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in ('```', '```json') and lines[-1].strip() == '```':
            value = '\n'.join(lines[1:-1]).strip()
            repairs.append('remove_code_fence')
    try:
        return normalize_direct(value, count), '+'.join(repairs) or 'strict'
    except (ValueError, TypeError):
        if count == 1:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict) and set(parsed) == {'output'}:
                    parsed = parsed['output']
                    repairs.append('unwrap_output_key')
                return [validate_grid(parsed)], '+'.join(repairs + ['wrap_single_grid'])
            except (ValueError, TypeError, KeyError):
                pass
    return None, 'invalid'


def clean_task(source):
    return {'train': [{'input': validate_grid(x['input']), 'output': validate_grid(x['output'])} for x in source['train']],
            'test': [{'input': validate_grid(x['input'])} for x in source['test']]}


def proxy_task(source):
    task = clean_task(source)
    if len(task['train']) < 2:
        raise ValueError('proxy requires at least two examples')
    held = task['train'][-1]
    return {'train': task['train'][:-1], 'test': [{'input': held['input']}]}, [held['output']]


def prepare(repo, run_dir, model_path):
    repo, run_dir = Path(repo), Path(run_dir)
    challenge = (repo / 'data/raw/arc-agi_training_challenges.json').read_bytes()
    proposal = json.loads((repo / 'configs/lt.phase-a.comparison-proposal.json').read_text())
    if hashlib.sha256(challenge).hexdigest() != proposal['dataset_sha256']:
        raise ValueError('challenge hash mismatch')
    tasks = json.loads(challenge)
    allocation = allocate(proposal, tasks)
    ids = set(allocation['development']) | set(allocation['selection']) | set(allocation['evaluation'])
    if len(ids) != 70:
        raise ValueError('allocation splits overlap')
    split = json.loads((repo / 'research/rsi/provisional-split.json').read_text())
    prior = json.loads((repo / 'research/rsi/exploratory-split-proposal.json').read_text())
    exposure = json.loads((repo / 'research/rsi/extended-exposure-review.json').read_text())
    protected = set(prior['confirmatory_task_ids']) | set(prior['exploratory_task_ids'])
    if ids & protected or ids & set(exposure['matched_task_ids']):
        raise ValueError('allocation intersects current protected or known-exposed IDs')
    if set(allocation['development']) & set(proposal['training_exclusions']) or set(allocation['selection']) & set(proposal['training_exclusions']):
        raise ValueError('allocation intersects Phase B training pool')
    print('allocation', allocation['sha256'], 'exposure inventory bounded; independence unverified', flush=True)
    import importlib.metadata as metadata
    packages = {p: metadata.version(p) for p in ('mlx', 'mlx-vlm', 'mlx-lm', 'transformers')}
    model_path = Path(model_path)
    if model_path.name != MODEL_REVISION or not model_path.is_dir():
        raise ValueError('model path must be the pinned revision snapshot')
    file_hashes = {}
    for path in sorted(model_path.iterdir()):
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(block)
            file_hashes[path.name] = h.hexdigest()
    if not any(name.endswith('.safetensors') for name in file_hashes):
        raise ValueError('model snapshot has no weight files')
    source_files = ['src/arc/qwen38_local_runner.py', 'src/cli/qwen38_local_experiment.py',
                    'src/arc/qwen38_rsi_comparison.py', 'src/arc/lt_direct_format.py',
                    'src/arc/lt_live_backend.py', 'src/arc/data.py']
    source_hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in source_files}
    manifest = {'protocol': 'qwen38-local-exploratory-v1', 'allocation': allocation,
                'challenge_sha256': proposal['dataset_sha256'], 'proposal_sha256': digest(proposal),
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'model_files_sha256': file_hashes,
                'source_files_sha256': source_hashes, 'packages': packages,
                'max_tokens': MAX_TOKENS, 'max_context': MAX_CONTEXT, 'temperature': TEMPERATURE,
                'seed': SEED, 'candidate_count': 2, 'format_recovery': True,
                'platform': platform.platform(), 'machine': platform.machine(),
                'prior_exposure_independence': 'unverified', 'comparison_previously_evaluated': True,
                'provisional_split_sha256': digest(split), 'prior_split_sha256': digest(prior),
                'extended_exposure_sha256': digest(exposure)}
    manifest['sha256'] = digest(manifest)
    path = run_dir / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('registration changed; use a new run directory')
    else:
        atomic_json(path, manifest)
    return manifest, tasks


class Backend:
    def __init__(self, model_path, adapter_path=None):
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        self.mx, self.generate, self.template = mx, generate, apply_chat_template
        self.mx.set_memory_limit(48 * 1024**3)
        self.model, self.processor = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)

    def call(self, content, seed, max_tokens=MAX_TOKENS):
        prompt = self.template(self.processor, self.model.config, content, num_images=0, enable_thinking=False)
        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n + max_tokens > MAX_CONTEXT:
            raise ValueError(f'context overflow: {n}+{max_tokens}>{MAX_CONTEXT}')
        self.mx.random.seed(seed)
        started = time.monotonic()
        result = self.generate(self.model, self.processor, prompt=prompt, max_tokens=max_tokens,
                               temperature=TEMPERATURE, top_p=1.0, seed=seed, verbose=False)
        return {'raw': result.text, 'prompt_tokens': result.prompt_tokens,
                'completion_tokens': result.generation_tokens, 'finish_reason': result.finish_reason,
                'peak_memory_gb': result.peak_memory, 'seconds': round(time.monotonic() - started, 3)}


def run_tasks(backend, run_dir, manifest, tasks, label, ids, prompt, proxy=False, variant_sha=None):
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    completed = 0
    for task_id in ids:
        path = Path(run_dir) / label / (task_id + '.json')
        identity = digest({'manifest': manifest['sha256'], 'label': label, 'task_id': task_id, 'prompt_sha256': prompt_sha, 'variant_sha256': variant_sha})
        if path.exists():
            row = json.loads(path.read_text())
            if row.get('identity') != identity or row.get('status') != 'complete' or len(row.get('candidates', [])) != 2:
                raise ValueError(f'invalid checkpoint {path}')
            completed += 1
            continue
        task, expected = proxy_task(tasks[task_id]) if proxy else (clean_task(tasks[task_id]), None)
        content = json.dumps({'instruction': prompt, 'task': task}, sort_keys=True, separators=(',', ':'))
        candidates = []
        print(f'{label}: {completed}/{len(ids)} saved; task {task_id}', flush=True)
        for index in range(2):
            candidate = backend.call(content, SEED + index)
            outputs, mode = parse_prediction(candidate['raw'], len(task['test']))
            candidate.update({'candidate_index': index, 'outputs': outputs, 'valid': outputs is not None, 'parse_mode': mode})
            if proxy:
                candidate['correct'] = outputs == expected
            candidates.append(candidate)
        row = {'identity': identity, 'manifest_sha256': manifest['sha256'], 'task_id': task_id,
               'status': 'complete', 'prompt_sha256': prompt_sha, 'variant_sha256': variant_sha, 'candidates': candidates,
               'runtime_seconds': sum(c['seconds'] for c in candidates)}
        atomic_json(path, row)
        completed += 1
        print(f'{label}: {completed}/{len(ids)} saved, valid={sum(c["valid"] for c in candidates)}/2', flush=True)
    return [json.loads((Path(run_dir) / label / (t + '.json')).read_text()) for t in ids]
