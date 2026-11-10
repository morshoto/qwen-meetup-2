"""Guard a separately registered, supervised replication run."""
from __future__ import annotations

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_replication_freeze import verify_pinned_files, verify_frozen_outcome
from src.arc.qwen38_replication_audit import build_audit_receipt
from src.arc.qwen38_replication_registration import build_replication_registration, build_ready_registration
from src.arc.qwen38_weight_allocation import freeze_registration
from src.arc.qwen38_replication_schedule import next_replication_action
from src.arc.qwen38_weight_generation import create_request, capture_response
from src.arc.qwen38_weight_journal import AttemptJournal
from src.arc.qwen38_weight_supervisor import run_supervised
from src.cli.qwen38_weight_collect import _artifact_hashes, _replay_artifacts, _run_start, _validate_worker_execution


def verify_runtime(repo, model_path, execution):
    """Recheck pinned code, installed packages and model file contents."""
    verify_pinned_files(repo, execution['source_files_sha256'])
    if {name: metadata.version(name) for name in execution['packages']} != execution['packages']:
        raise ValueError('runtime package versions changed')
    model = Path(model_path)
    if not model.is_dir(): raise ValueError('model snapshot missing')
    files = {}
    for path in model.iterdir():
        if not path.is_file(): raise ValueError('unexpected model snapshot entry')
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                value.update(block)
        files[path.name] = value.hexdigest()
    if files != execution['model']['files_sha256']:
        raise ValueError('model snapshot contents changed')
    return {'status': 'runtime_matches'}


def rebuild_registration(*, repo, model_path, old_root, execution, proposal, history, snapshot, audit,
                         challenges, official, solutions, run_dir, report):
    """Replay the complete bounded source chain without decoding target answers."""
    load = lambda path: json.loads(Path(path).read_bytes())
    execution, proposal, history, snapshot = map(load, (execution, proposal, history, snapshot))
    old_root = Path(old_root).resolve()
    if snapshot.get('source_root') != str(old_root):
        raise ValueError('original run directory differs from snapshot')
    challenge_bytes = Path(challenges).read_bytes()
    official_bytes = Path(official).read_bytes()
    evaluation = load(old_root / 'evaluation/evaluation-v2.json')
    verify_runtime(repo, model_path, execution)
    verify_frozen_outcome((old_root / 'training/adapter-run-v1/adapter/adapters.safetensors').read_bytes(),
                          Path(report).read_bytes())
    replayed = build_audit_receipt(proposal, snapshot, history, challenge_bytes, official_bytes, solutions)
    if replayed != load(audit):
        raise ValueError('bounded exposure audit changed')
    sft = load(old_root / 'training/sft-v1/manifest.json')
    starts = sorted((old_root / 'journal').glob('*.start.json'))
    extras = {'actual_fresh_generation': sorted({load(path)['request']['task_id'] for path in starts}),
              'historical_fp8': history['phase_a_task_ids'] + history['dev_task_ids'],
              'sft_actual': [entry['task_id'] for entry in sft['parents']]}
    pending = build_replication_registration(execution, proposal, evaluation, snapshot, challenge_bytes, repo,
                                             additional_exclusions=extras, official=json.loads(official_bytes), run_dir=run_dir)
    return build_ready_registration(pending, replayed)


def _registration(root, rebuild):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077 or not callable(rebuild):
        raise ValueError('replication run directory must be private')
    descriptor = rebuild()
    if (not isinstance(descriptor, dict) or descriptor.get('status') != 'evaluation_ready' or
        descriptor.get('inference_authorized') is not True or descriptor.get('run_dir') != str(root.resolve()) or
        descriptor.get('sha256') != digest({k: v for k, v in descriptor.items() if k != 'sha256'})):
        raise ValueError('replayed replication registration is not ready')
    return descriptor


def freeze_replication(root, rebuild):
    descriptor = _registration(root, rebuild)
    path = Path(root) / 'registration.json'
    if path.is_symlink(): raise ValueError('registration cannot be linked')
    return freeze_registration(path, {k: v for k, v in descriptor.items() if k != 'sha256'})


def verify_replication(root, rebuild):
    descriptor = _registration(root, rebuild)
    path = Path(root) / 'registration.json'
    if path.is_symlink() or not path.is_file() or json.loads(path.read_bytes()) != descriptor:
        raise ValueError('registered replication sources changed')
    return descriptor


def _worker_command(descriptor, root, model_path, adapter_path, index, arm):
    return [sys.executable, '-u', '-m', 'src.cli.qwen38_replication_worker',
            '--registration', str(root / 'registration.json'), '--registration-sha256', descriptor['sha256'],
            '--arm', arm, '--model-path', str(Path(model_path).resolve()),
            '--adapter-path', str(Path(adapter_path).resolve()),
            '--output', str(root / 'workers' / f'{index:06}' / 'response.json')]


def _termination(root, descriptor, rows, status, elapsed):
    path = root / 'termination.json'
    if path.exists():
        if path.is_symlink(): raise ValueError('termination receipt cannot be linked')
        saved = json.loads(path.read_bytes())
        if (saved.get('sha256') != digest({k: v for k, v in saved.items() if k != 'sha256'}) or
            saved.get('registration_sha256') != descriptor['sha256'] or saved.get('journal_sha256') != digest(rows) or
            saved.get('status') != status):
            raise ValueError('replication termination evidence changed')
        return saved
    return freeze_registration(path, {'status': status, 'registration_sha256': descriptor['sha256'],
                                      'journal_sha256': digest(rows), 'wall_seconds': elapsed})


def run_replication_step(*, root, rebuild, challenges, model_path, adapter_path, repo):
    """Run one supervised attempt after replay; never pass answer data to a worker."""
    root = Path(root).resolve()
    descriptor = verify_replication(root, rebuild)
    if hashlib.sha256(challenges).hexdigest() != descriptor['challenge_sha256']:
        raise ValueError('replication challenges changed')
    tasks = json.loads(challenges)
    workers = root / 'workers'
    workers.mkdir(mode=0o700, exist_ok=True)
    if workers.is_symlink() or workers.stat().st_mode & 0o077:
        raise ValueError('worker evidence directory must be private')
    with AttemptJournal(root / 'journal') as journal:
        elapsed = _run_start(root, descriptor['sha256'])
        rows = journal.read()
        _replay_artifacts(rows, workers, descriptor['max_tokens'], descriptor['supervision'])
        for index, row in enumerate(rows):
            if row['status'] == 'started': continue
            request = create_request(row['task_id'], tasks[row['task_id']], candidate_index=row['candidate_index'], seed=row['seed'])
            if (row.get('manifest_sha256') != descriptor['sha256'] or row.get('model') != descriptor['model'] or
                row.get('worker_command') != _worker_command(descriptor, root, model_path, adapter_path, index, row['arm']) or
                any(row.get(k) != v for k, v in request.items() if k not in ('content', 'test_count'))):
                raise ValueError('replication attempt identity changed')
        action = next_replication_action(descriptor, rows)
        existing = root / 'termination.json'
        if existing.exists():
            saved = json.loads(existing.read_bytes())
            status = saved.get('status')
            if status != action['status'] and not (status == 'time_budget_exhausted' and action['status'] == 'ready'):
                raise ValueError('termination differs from schedule')
            _termination(root, descriptor, rows, status, elapsed)
            return {'status': status}
        if action['status'] != 'ready':
            if action['status'] != 'requires_reconciliation':
                _termination(root, descriptor, rows, action['status'], elapsed)
            return action
        remaining = descriptor['wall_seconds_limit'] - elapsed
        if remaining <= 0:
            _termination(root, descriptor, rows, 'time_budget_exhausted', elapsed)
            return {'status': 'time_budget_exhausted'}
        request = create_request(action['task_id'], tasks[action['task_id']],
                                 candidate_index=action['candidate_index'], seed=action['seed'])
        if request['prompt_sha256'] != descriptor['prompt_sha256']:
            raise ValueError('replication prompt differs from registration')
        directory = workers / f'{len(rows):06}'
        if directory.exists(): raise ValueError('worker evidence exists without journal start')
        command = _worker_command(descriptor, root, model_path, adapter_path, len(rows), action['arm'])
        identity = {k: v for k, v in request.items() if k not in ('content', 'test_count')}
        identity.update({'manifest_sha256': descriptor['sha256'], 'model': descriptor['model'],
                         'arm': action['arm'], 'worker_command': command})
        journal.start(identity)
        limits = descriptor['supervision']
        execution = run_supervised(command, {'request': request, 'max_tokens': descriptor['max_tokens']}, directory,
                                   timeout_seconds=min(remaining, descriptor['per_attempt_seconds']),
                                   min_free=limits['minimum_free_percent'], max_swap=limits['maximum_swap_bytes'],
                                   max_output_bytes=limits['max_output_bytes'], cwd=repo)
        if json.loads((directory / 'execution.json').read_bytes()) != execution:
            raise ValueError('worker execution receipt changed')
        if execution.get('process_group_gone') is not True:
            return {'status': 'requires_reconciliation'}
        _validate_worker_execution(execution, command, limits)
        if verify_replication(root, rebuild) != descriptor:
            raise ValueError('replication postflight changed')
        artifacts = _artifact_hashes(directory)
        if execution['status'] == 'complete':
            try:
                response = json.loads((directory / 'response.json').read_bytes())
                capture = capture_response(request, response, descriptor['max_tokens'])
                if response != capture: raise ValueError('worker response mismatch')
                outcome = {k: v for k, v in capture.items() if k not in identity}
                outcome.update({'status': 'complete', 'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts})
            except (ValueError, KeyError, OSError, TypeError):
                outcome = {'status': 'error', 'error': 'worker_output_invalid', 'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts}
        else:
            outcome = {'status': 'error', 'error': execution.get('reason') or 'worker_failed',
                       'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts}
        journal.finish(len(rows), outcome)
        return {'status': 'attempt_recorded', 'arm': action['arm'], 'task_id': action['task_id'],
                'candidate_index': action['candidate_index'], 'attempt_status': outcome['status']}
