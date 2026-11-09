"""Guarded training helpers for source-replayed exploratory LoRA data."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import re
from pathlib import Path


def materialize_trainer_dataset(bundle, training_root, expected_sha256):
    """Copy only the previously replayed JSONL into a new private dataset dir."""
    training_root, bundle = Path(training_root).resolve(), Path(bundle).resolve()
    source = bundle / 'train.jsonl'
    if (not training_root.is_dir() or training_root.stat().st_mode & 0o077 or
        bundle != training_root / 'sft-v1' or not bundle.is_dir() or bundle.stat().st_mode & 0o077 or source.is_symlink() or
        not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None):
        raise ValueError('invalid private SFT source')
    data = source.read_bytes()
    if not data or len(data) > 16 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('SFT dataset hash mismatch')
    target = training_root / 'trainer-data-v1'
    target.mkdir(mode=0o700, exist_ok=False)
    fd = os.open(target / 'train.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    dir_fd = os.open(target, os.O_RDONLY)
    try: os.fsync(dir_fd)
    finally: os.close(dir_fd)
    return target


from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_train import parse_training_losses
from src.cli.qwen38_weight_collect import _validate_worker_execution
from src.arc.qwen38_weight_supervisor import run_supervised
from src.cli.qwen38_weight_prepare_sft import replay_bundle


def _write_decision(path, value):
    import json
    payload = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _execute_training(**kwargs):
    root = Path(kwargs['run_dir']).resolve()
    manifest = kwargs['manifest']
    if manifest.get('run_dir') != str(root):
        raise ValueError('training run does not match the frozen collection')
    selection, registration, replay = replay_bundle(**kwargs)
    if replay.get('status') != 'matches_supplied_replay':
        raise ValueError('SFT source replay failed')
    training = root / 'training'
    if not training.is_dir() or training.is_symlink() or training.stat().st_mode & 0o077:
        raise ValueError('training directory is not private')
    if selection['status'] == 'insufficient_verified_data':
        result = {'status': 'insufficient_verified_data', 'mode': selection['mode'], 'counts': selection['counts'],
                  'execution_sha256': manifest['sha256'], 'sft_manifest_sha256': replay['manifest_sha256']}
        result['sha256'] = digest(result)
        _write_decision(training / 'training-decision-v1.json', result)
        return result
    if (selection.get('status') != 'eligible_pending_source_replay' or selection['counts'].get('unique_parents', 0) < 5 or
        registration.get('execution_sha256') != manifest['sha256']):
        raise ValueError('training gate or source identity failed')
    bundle = training / 'sft-v1'
    bundle_manifest = json.loads((bundle / 'manifest.json').read_bytes())
    if (bundle_manifest.get('status') != 'prepared_pending_source_replay' or
        bundle_manifest.get('sha256') != digest({k: v for k, v in bundle_manifest.items() if k != 'sha256'}) or
        bundle_manifest['sha256'] != replay['manifest_sha256']):
        raise ValueError('SFT bundle identity mismatch')
    train_sha = bundle_manifest['train_sha256']
    dataset = materialize_trainer_dataset(bundle, training, train_sha)
    train_root = training / 'adapter-run-v1'
    limits = {'minimum_free_percent': 10, 'maximum_swap_bytes': 16 * 1024**3, 'max_output_bytes': 2 * 1024**2}
    command = [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_train_worker',
               '--model-path', str(Path(kwargs['model_path']).resolve()), '--dataset-dir', str(dataset),
               '--run-root', str(train_root), '--train-sha256', train_sha]
    execution = run_supervised(command, {}, train_root, timeout_seconds=3600,
                               min_free=limits['minimum_free_percent'], max_swap=limits['maximum_swap_bytes'],
                               max_output_bytes=limits['max_output_bytes'], cwd=kwargs['repo'])
    if json.loads((train_root / 'execution.json').read_bytes()) != execution:
        raise ValueError('training execution receipt mismatch')
    _validate_worker_execution(execution, command, limits)
    if execution['status'] != 'complete':
        raise ValueError('training worker failed; preserve its evidence')
    log = (train_root / 'stdout.log').read_text() + '\n' + (train_root / 'stderr.log').read_text()
    losses = parse_training_losses(log, steps=20)
    initial = train_root / 'initial.safetensors'; adapter = train_root / 'adapter/adapters.safetensors'
    adapter_config = train_root / 'adapter/adapter_config.json'
    initial_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    adapter_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
    config_sha = hashlib.sha256(adapter_config.read_bytes()).hexdigest()
    # Replay source data and base identity again before loading the saved adapter.
    selection_after, registration_after, replay_after = replay_bundle(**kwargs)
    if selection_after != selection or registration_after != registration or replay_after != replay:
        raise ValueError('training source changed during execution')
    if hashlib.sha256((dataset / 'train.jsonl').read_bytes()).hexdigest() != train_sha:
        raise ValueError('trainer dataset changed')
    probe_dir = train_root / 'verification'
    probe_command = [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_probe',
                     '--model-path', str(Path(kwargs['model_path']).resolve()), '--run-dir', str(train_root)]
    probe_execution = run_supervised(probe_command, {}, probe_dir, timeout_seconds=900,
                                     min_free=limits['minimum_free_percent'], max_swap=limits['maximum_swap_bytes'],
                                     max_output_bytes=limits['max_output_bytes'], cwd=kwargs['repo'])
    if json.loads((probe_dir / 'execution.json').read_bytes()) != probe_execution:
        raise ValueError('adapter probe receipt mismatch')
    _validate_worker_execution(probe_execution, probe_command, limits)
    if probe_execution['status'] != 'complete':
        raise ValueError('adapter reload probe failed; preserve its evidence')
    probe = json.loads((train_root / 'tensor-probe.json').read_bytes())
    loaded = json.loads((train_root / 'adapter/probe.json').read_bytes())
    if ({k: v for k, v in probe.items() if k != 'changed_lora_tensors'} != loaded or
        probe.get('adapter_sha256') != adapter_sha or type(probe.get('changed_lora_tensors')) is not int or
        probe['changed_lora_tensors'] < 1 or type(probe.get('nonzero_lora_tensors')) is not int or
        probe['nonzero_lora_tensors'] < 1 or isinstance(probe.get('max_abs_logit_delta'), bool) or
        not isinstance(probe.get('max_abs_logit_delta'), (int, float)) or
        not math.isfinite(probe['max_abs_logit_delta']) or probe['max_abs_logit_delta'] <= 0):
        raise ValueError('adapter tensor or reload evidence failed')
    selection_final, registration_final, replay_final = replay_bundle(**kwargs)
    if selection_final != selection or registration_final != registration or replay_final != replay:
        raise ValueError('training source changed during adapter probe')
    result = {'status': 'trained', 'trained': True, 'weights_updated': True, 'mode': selection['mode'],
              'counts': selection['counts'], 'execution_sha256': manifest['sha256'],
              'sft_manifest_sha256': replay['manifest_sha256'], 'train_sha256': train_sha,
              'training_registration_sha256': digest(registration), 'training_settings': registration['training_settings'],
              'base_files_sha256': registration['model_files_sha256'], 'initial_sha256': initial_sha,
              'adapter_sha256': adapter_sha, 'adapter_config_sha256': config_sha,
              'changed_lora_tensors': probe['changed_lora_tensors'], 'max_abs_logit_delta': probe['max_abs_logit_delta'],
              'losses': losses, 'training_execution': execution, 'probe_execution': probe_execution,
              'interpretation': 'parameter update only; no ARC accuracy or recursive improvement claim'}
    result['sha256'] = digest(result)
    _write_decision(training / 'training-result-v1.json', result)
    return result


def execute_training(**kwargs):
    """Retain a score-free failure receipt if a real attempt has been prepared."""
    try:
        return _execute_training(**kwargs)
    except Exception as error:
        root = Path(kwargs['run_dir']).resolve()
        training = root / 'training'
        failed = training / 'training-failed-v1.json'
        if (kwargs['manifest'].get('run_dir') == str(root) and training.is_dir() and
            not training.is_symlink() and not training.stat().st_mode & 0o077 and
            ((training / 'trainer-data-v1').exists() or (training / 'adapter-run-v1').exists()) and
            not failed.exists() and not (training / 'training-result-v1.json').exists()):
            result = {'status': 'training_failed', 'execution_sha256': kwargs['manifest'].get('sha256'),
                      'error_type': type(error).__name__, 'evidence_directory': 'adapter-run-v1'}
            result['sha256'] = digest(result)
            try: _write_decision(failed, result)
            except OSError: pass  # Preserve the original failure and the underlying artifacts.
        raise
