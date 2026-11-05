"""CPU-first orchestration with atomic generation commits and strict resume identity."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any
import yaml
from src.arc.agent import FakeLLM, code_agent, direct
from src.arc.data import load_arc, split_tasks
from src.arc.sandbox import FakeSandbox
from src.arc.scoring import score
from src.arc.training import DryRunTrainer, make_record
from src.core.config import ExperimentConfig


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temp, path)


def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config: ExperimentConfig, *, resume: bool = False, run_dir: Path | None = None) -> Path:
    if config.rsi.trainer != 'dry_run':
        raise ValueError('real MLX-LoRA training is only available through the explicit P3-B CLI')
    if config.model.backend in {'openai_compatible', 'ollama_native', 'guarded_mlx'}:
        from src.pipelines.p3a import run_real
        return run_real(config, resume=resume, run_dir=run_dir)
    start = time.monotonic()
    tasks, verifier = load_arc(config.data.challenges, config.data.solutions)
    counts = {name: getattr(config.data, name) for name in ('exploration', 'promotion_gate', 'monitoring')}
    splits = split_tasks(list(tasks), config.data.split_seed, counts)
    identity = {'config': asdict(config), 'data_sha256': {'challenges': digest(config.data.challenges), 'solutions': digest(config.data.solutions)}, 'splits': splits}
    if resume:
        if run_dir is None or not (run_dir / 'identity.json').exists():
            raise ValueError('resume requires an existing run directory')
        if json.loads((run_dir / 'identity.json').read_text()) != identity:
            raise ValueError('resume config or data differs from original run')
    else:
        if run_dir is not None:
            raise ValueError('run-dir is only valid with resume')
        run_dir = Path(config.tracking.output_dir) / (config.experiment.name + '-' + uuid.uuid4().hex[:12])
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / 'identity.json', identity)
        (run_dir / 'config.snapshot.yml').write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        write_json(run_dir / 'data_split.json', splits)
        try:
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError): git = None
        write_json(run_dir / 'environment.json', {'python': platform.python_version(), 'platform': platform.platform(), 'git_commit': git, 'model_id': config.model.id, 'revision': config.model.revision, 'seed': config.experiment.seed, 'backend': config.model.backend})
    assert run_dir is not None
    generations = run_dir / 'generations'; generations.mkdir(exist_ok=True)
    existing = sorted(p.name for p in generations.glob('gen_[0-9][0-9][0-9]') if p.is_dir())
    expected = [f'gen_{i:03d}' for i in range(len(existing))]
    if existing != expected or len(existing) > config.experiment.generations + 1:
        raise ValueError('generation sequence is inconsistent')
    backend, sandbox, trainer = FakeLLM(), FakeSandbox(), DryRunTrainer()
    replay: dict[str, dict[str, Any]] = {}
    previous = 'base:' + config.model.id + '@' + config.model.revision
    all_metrics: list[dict[str, Any]] = []
    logs = []
    for generation in range(config.experiment.generations + 1):
        final = generations / f'gen_{generation:03d}'
        if final.exists():
            manifest_path = final / 'integrity.json'
            if not manifest_path.is_file(): raise ValueError('generation integrity manifest missing')
            manifest = json.loads(manifest_path.read_text())
            required = {'state.json', 'metrics.json', 'predictions.json', 'replay.json', 'training.jsonl'}
            if not isinstance(manifest, dict) or not required <= set(manifest) or any(Path(name).is_absolute() or '..' in Path(name).parts for name in manifest):
                raise ValueError('generation integrity manifest invalid')
            for relative, expected_hash in manifest.items():
                target = final / relative
                if not target.is_file() or digest(str(target)) != expected_hash:
                    raise ValueError(f'generation artifact corrupt: {relative}')
            state = json.loads((final / 'state.json').read_text())
            if state.get('status') != 'complete': raise ValueError('inconsistent generation state')
            replay = {r['id']: r for r in json.loads((final / 'replay.json').read_text())}
            previous = state['active_checkpoint']
            all_metrics.extend(json.loads((final / 'metrics.json').read_text()))
            logs.append(state)
            continue
        if time.monotonic() - start >= config.budget.max_seconds:
            break
        temp = generations / f'.gen_{generation:03d}.tmp'
        if temp.exists(): shutil.rmtree(temp)
        temp.mkdir()
        gen_start = time.monotonic()
        metrics = []
        predictions = []
        new_records = []
        # Exploration is used for collection. Evaluation is explicitly scheduled
        # and can use a separate monitoring split; those records never train.
        collection_ids = [splits['exploration'][(generation * config.rsi.exploration_per_generation + i) % len(splits['exploration'])]
                          for i in range(config.rsi.exploration_per_generation)]
        scheduled = generation in config.experiment.evaluation_generations
        evaluation_ids = (collection_ids if config.experiment.evaluation_split == 'exploration'
                          else splits['monitoring'])
        evaluation_conditions = (config.experiment.conditions if generation == 0
                                 else (['C'] if 'C' in config.experiment.conditions else []))
        jobs = []
        if scheduled:
            jobs.extend((condition, config.experiment.evaluation_split, evaluation_ids) for condition in evaluation_conditions)
        if 'C' in config.experiment.conditions and generation < config.experiment.generations:
            # Reuse the C evaluation only when it has exactly the same inputs.
            if not (scheduled and config.experiment.evaluation_split == 'exploration'):
                jobs.append(('C', 'exploration', collection_ids))
        exhausted = False
        for condition, split_name, ids in jobs:
            scores = []
            correct_outputs = total_outputs = 0
            job_start = time.monotonic()
            for task_id in ids:
                if time.monotonic() - start >= config.budget.max_seconds:
                    exhausted = True
                    break
                task = tasks[task_id]
                results = [(direct(task, backend, i) if condition == 'A' else code_agent(task, backend, sandbox, config.agent.max_repairs, i)) for i in range(config.agent.candidates)]
                outcome = score(verifier, task_id, ['timeout' if r.error_kind == 'timeout' else r.outputs for r in results])
                scores.append(outcome)
                correct_outputs += verifier.correct_outputs(task_id, results[0].outputs)
                total_outputs += len(task.test_inputs)
                predictions.append({'condition': condition, 'dataset_split': split_name, 'task_id': task_id, 'candidates': [asdict(r) for r in results], 'score': asdict(outcome)})
                if condition == 'C' and split_name == 'exploration' and generation < config.experiment.generations:
                    for result in results:
                        record = make_record(task, result, verifier, set(splits['exploration']), generation)
                        if record: new_records.append(record)
            if exhausted:
                break
            if scheduled and split_name == config.experiment.evaluation_split:
                n = len(scores)
                metrics.append({'run_id': run_dir.name, 'condition': condition, 'generation': generation, 'dataset_split': split_name, 'pass_at_1': sum(s.pass_at_1 for s in scores)/n if n else 0, 'pass_at_2': sum(s.pass_at_2 for s in scores)/n if n else 0, 'tasks': n, 'correct_outputs': correct_outputs, 'total_outputs': total_outputs, 'invalid': sum(s.invalid for s in scores), 'timeouts': sum(s.timeouts for s in scores), 'verified_solutions': 0, 'train_samples': 0, 'gpu_seconds': 0, 'inference_tokens': None, 'elapsed_seconds': time.monotonic() - job_start, 'checkpoint_id': previous})
        if exhausted or time.monotonic() - start >= config.budget.max_seconds:
            shutil.rmtree(temp)
            write_json(run_dir / 'run_state.json', {'status': 'budget_exhausted', 'next_generation': generation, 'elapsed_seconds': time.monotonic()-start})
            break
        unique_new = {r['id']: r for r in new_records if r['id'] not in replay}
        old_records = list(replay.values())
        replay.update(unique_new)
        # The configured ratio determines the share of old records when available.
        random.Random(config.experiment.seed + generation).shuffle(old_records)
        if config.rsi.replay_ratio == 1:
            replay_count = len(old_records)
        else:
            replay_count = min(len(old_records), int(len(unique_new) * config.rsi.replay_ratio / (1 - config.rsi.replay_ratio)))
        train_records = list(unique_new.values()) + old_records[:replay_count]
        buffer_records = list(replay.values())
        checkpoint = previous
        training_error = None
        training_status = 'not_scheduled' if generation == config.experiment.generations or 'C' not in config.experiment.conditions else 'insufficient_samples'
        training = {'mode': 'dry_run', 'trained': False, 'weights_updated': False, 'sample_count': len(train_records)}
        if generation < config.experiment.generations and 'C' in config.experiment.conditions and len(train_records) >= config.rsi.min_samples:
            try:
                training = trainer.train(train_records, temp / 'checkpoint', previous, asdict(config.rsi))
                checkpoint = str(final / 'checkpoint')
                training_status = 'dry_run_complete'
            except Exception as exc:
                training_error = f'{type(exc).__name__}: {exc}'
                training_status = 'failed'
        for row in metrics:
            if row['condition'] == 'C':
                row['verified_solutions'] = len(unique_new)
                row['train_samples'] = len(train_records)
        state = {'generation': generation, 'status': 'complete', 'active_checkpoint': checkpoint, 'previous_checkpoint': previous, 'promotion': 'not_applicable', 'training_status': training_status, 'training': training, 'error': training_error, 'elapsed_seconds': time.monotonic()-gen_start, 'new_verified_solutions': len(unique_new), 'seed': config.experiment.seed, 'model_id': config.model.id, 'revision': config.model.revision, 'backend': config.model.backend}
        write_json(temp / 'state.json', state)
        write_json(temp / 'metrics.json', metrics)
        write_json(temp / 'predictions.json', predictions)
        write_json(temp / 'replay.json', buffer_records)
        (temp / 'training.jsonl').write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in train_records))
        integrity = {str(p.relative_to(temp)): digest(str(p)) for p in sorted(temp.rglob('*')) if p.is_file()}
        write_json(temp / 'integrity.json', integrity)
        os.replace(temp, final)
        previous = checkpoint; all_metrics.extend(metrics); logs.append(state)
        write_json(run_dir / 'run_state.json', {'status': 'running', 'next_generation': generation + 1})
    if len(logs) == config.experiment.generations + 1:
        write_json(run_dir / 'run_state.json', {'status': 'complete', 'next_generation': None})
    elif not (run_dir / 'run_state.json').exists() or json.loads((run_dir / 'run_state.json').read_text()).get('status') != 'budget_exhausted':
        write_json(run_dir / 'run_state.json', {'status': 'budget_exhausted', 'next_generation': len(logs), 'elapsed_seconds': time.monotonic()-start})
    # Aggregate files are derived from committed generations, so resume cannot duplicate rows.
    (run_dir / 'training_logs.jsonl').write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in logs))
    if all_metrics:
        with (run_dir / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_metrics[0])); writer.writeheader(); writer.writerows(all_metrics)
    return run_dir
