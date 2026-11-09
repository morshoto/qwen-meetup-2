"""Prepare and run a matched exploratory adapter evaluation after source replay."""
from __future__ import annotations

import json
from pathlib import Path

from src.arc.qwen38_weight_allocation import freeze_registration
from src.arc.qwen38_weight_eval_registration import build_evaluation_descriptor
from src.arc.qwen38_weight_training_receipt import verify_training_evidence
from src.cli.qwen38_weight_prepare_sft import replay_bundle


def _context(**kwargs):
    comparison_dir = Path(kwargs['comparison_dir']).resolve()
    inputs = {k: v for k, v in kwargs.items() if k != 'comparison_dir'}
    root = Path(inputs['run_dir']).resolve()
    if inputs['manifest'].get('run_dir') != str(root):
        raise ValueError('evaluation run does not match frozen collection')
    selection, registration, replay = replay_bundle(**inputs)
    if (selection.get('status') != 'eligible_pending_source_replay' or selection['counts'].get('unique_parents', 0) < 5 or
        replay.get('status') != 'matches_supplied_replay'):
        raise ValueError('SFT source replay or gate failed')
    training = json.loads((root / 'training/training-result-v1.json').read_bytes())
    evidence = verify_training_evidence(training, root, inputs['model_path'], registration, replay['manifest_sha256'])
    if evidence.get('status') != 'source_evidence_matches':
        raise ValueError('training evidence replay failed')
    comparison = json.loads((comparison_dir / 'manifest.json').read_bytes())
    ids = comparison['allocation']['evaluation']; baseline_dir = comparison_dir / 'evaluation/baseline'
    if {p.stem for p in baseline_dir.glob('*.json')} != set(ids):
        raise ValueError('baseline checkpoint inventory differs')
    files = {}
    for task_id in ids:
        path = baseline_dir / (task_id + '.json')
        if path.is_symlink(): raise ValueError('baseline checkpoint cannot be a symlink')
        files[task_id] = path.read_bytes()
    descriptor = build_evaluation_descriptor(comparison, inputs['manifest'], training, evidence,
                                             inputs['challenges'], files, inputs['repo'])
    return descriptor, comparison, training, selection, files


def freeze_evaluation(**kwargs):
    descriptor, _, _, _, _ = _context(**kwargs)
    root = Path(kwargs['run_dir']).resolve() / 'evaluation'
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError('evaluation directory is not private')
    return freeze_registration(root / 'evaluation-v1.json', {k: v for k, v in descriptor.items() if k != 'sha256'})


def verify_evaluation_descriptor(**kwargs):
    descriptor, _, _, _, _ = _context(**kwargs)
    saved = json.loads((Path(kwargs['run_dir']).resolve() / 'evaluation/evaluation-v1.json').read_bytes())
    if saved != descriptor:
        raise ValueError('evaluation descriptor changed')
    return descriptor


import sys
from src.arc.qwen38_weight_eval_schedule import next_evaluation_action
from src.arc.qwen38_weight_generation import capture_response, create_request
from src.arc.qwen38_weight_journal import AttemptJournal
from src.arc.qwen38_weight_supervisor import run_supervised
from src.cli.qwen38_weight_collect import _artifact_hashes, _replay_artifacts, _run_start, _validate_worker_execution


def _worker_command(descriptor, root, model_path, index):
    adapter = root / 'training/adapter-run-v1/adapter'
    worker_dir = Path(descriptor['run_dir']) / 'workers' / f'{index:06}'
    return [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_eval_worker',
            '--model-path', str(Path(model_path).resolve()), '--adapter-path', str(adapter),
            '--adapter-sha256', descriptor['adapter_sha256'], '--adapter-config-sha256', descriptor['adapter_config_sha256'],
            '--output', str(worker_dir / 'response.json')]


def _replay_evaluation(rows, descriptor, root, model_path):
    workers = Path(descriptor['run_dir']) / 'workers'
    _replay_artifacts(rows, workers, descriptor['max_tokens'], descriptor['supervision'])
    for index, row in enumerate(rows):
        if row.get('worker_command') != _worker_command(descriptor, root, model_path, index):
            raise ValueError('evaluation worker command differs from registration')


def evaluate_step(**kwargs):
    descriptor = verify_evaluation_descriptor(**kwargs)
    root = Path(kwargs['run_dir']).resolve(); eval_root = Path(descriptor['run_dir'])
    if (descriptor.get('status') != 'evaluation_ready' or eval_root != root / 'evaluation/adapter-v1' or
        not eval_root.parent.is_dir() or eval_root.parent.stat().st_mode & 0o077):
        raise ValueError('evaluation destination or descriptor invalid')
    eval_root.mkdir(mode=0o700, exist_ok=True)
    workers = eval_root / 'workers'; workers.mkdir(mode=0o700, exist_ok=True)
    if any(p.is_symlink() or p.stat().st_mode & 0o077 for p in (eval_root, workers)):
        raise ValueError('evaluation evidence directories must be private')
    with AttemptJournal(eval_root / 'journal') as journal:
        wall_elapsed = _run_start(eval_root, descriptor['sha256'])
        rows = journal.read(); _replay_evaluation(rows, descriptor, root, kwargs['model_path'])
        action = next_evaluation_action(descriptor, rows)
        if action['status'] != 'ready': return action
        remaining = descriptor['wall_seconds_limit'] - wall_elapsed
        if remaining <= 0: return {'status': 'time_budget_exhausted'}
        tasks = json.loads(kwargs['challenges'])
        request = create_request(action['task_id'], tasks[action['task_id']],
                                 candidate_index=action['candidate_index'], seed=action['seed'])
        if request['prompt_sha256'] != descriptor['prompt_sha256']:
            raise ValueError('evaluation prompt differs from baseline')
        directory = workers / f'{len(rows):06}'
        if directory.exists(): raise ValueError('worker evidence exists without journal start')
        command = _worker_command(descriptor, root, kwargs['model_path'], len(rows))
        identity = {k: v for k, v in request.items() if k not in ('content', 'test_count')}
        identity.update({'manifest_sha256': descriptor['sha256'], 'model': kwargs['manifest']['model'], 'worker_command': command})
        journal.start(identity)
        limits = descriptor['supervision']
        execution = run_supervised(command, {'request': request, 'max_tokens': descriptor['max_tokens']}, directory,
                                   timeout_seconds=min(remaining, descriptor['per_attempt_seconds']),
                                   min_free=limits['minimum_free_percent'], max_swap=limits['maximum_swap_bytes'],
                                   max_output_bytes=limits['max_output_bytes'], cwd=kwargs['repo'])
        if json.loads((directory / 'execution.json').read_bytes()) != execution:
            raise ValueError('evaluation worker execution receipt mismatch')
        if execution.get('process_group_gone') is not True:
            return {'status': 'requires_reconciliation'}
        _validate_worker_execution(execution, command, limits)
        if verify_evaluation_descriptor(**kwargs) != descriptor:
            raise ValueError('evaluation postflight changed')
        artifacts = _artifact_hashes(directory)
        if execution['status'] == 'complete':
            try:
                response = json.loads((directory / 'response.json').read_bytes())
                capture = capture_response(request, response, descriptor['max_tokens'])
                if response != capture: raise ValueError('evaluation worker response mismatch')
                outcome = {k: v for k, v in capture.items() if k not in identity}
                outcome.update({'status': 'complete', 'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts})
            except (ValueError, KeyError, OSError, TypeError):
                outcome = {'status': 'error', 'error': 'worker_output_invalid', 'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts}
        else:
            outcome = {'status': 'error', 'error': execution.get('reason') or 'worker_failed',
                       'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts}
        journal.finish(len(rows), outcome)
        return {'status': 'attempt_recorded', 'task_id': action['task_id'], 'candidate_index': action['candidate_index'],
                'attempt_status': outcome['status']}
