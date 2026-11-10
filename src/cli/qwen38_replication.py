"""Guard a separately registered, supervised replication run."""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_replication_freeze import verify_pinned_files, verify_frozen_outcome
from src.arc.qwen38_replication_audit import build_audit_receipt
from src.arc.qwen38_replication_registration import build_replication_registration, build_ready_registration
from src.arc.qwen38_weight_allocation import freeze_registration


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
