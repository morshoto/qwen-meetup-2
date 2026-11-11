"""Replay saved training and adapter evidence before exploratory evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_train import parse_training_losses
from src.cli.qwen38_weight_collect import _validate_worker_execution


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_training_evidence(result, run_dir, model_path, registration, sft_manifest_sha256):
    """Recheck saved evidence; callers must separately replay the SFT sources."""
    root = Path(run_dir).resolve(); model = Path(model_path).resolve(); training = root / 'training'
    train_root = training / 'adapter-run-v1'; adapter = train_root / 'adapter'
    if (result.get('sha256') != digest({k: v for k, v in result.items() if k != 'sha256'}) or
        result.get('status') != 'trained' or result.get('trained') is not True or result.get('weights_updated') is not True or
        result.get('execution_sha256') != registration.get('execution_sha256') or
        result.get('sft_manifest_sha256') != sft_manifest_sha256 or
        result.get('training_registration_sha256') != digest(registration) or
        result.get('training_settings') != registration.get('training_settings') or
        result.get('base_files_sha256') != registration.get('model_files_sha256') or
        json.loads((training / 'training-result-v1.json').read_bytes()) != result):
        raise ValueError('training result identity mismatch')
    paths = {'train_sha256': training / 'trainer-data-v1/train.jsonl',
             'initial_sha256': train_root / 'initial.safetensors',
             'adapter_sha256': adapter / 'adapters.safetensors',
             'adapter_config_sha256': adapter / 'adapter_config.json'}
    if any(_sha(path) != result.get(key) for key, path in paths.items()):
        raise ValueError('training data or adapter artifact changed')
    train_command = [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_train_worker', '--model-path', str(model),
                     '--dataset-dir', str(training / 'trainer-data-v1'), '--run-root', str(train_root),
                     '--train-sha256', result['train_sha256']]
    probe_command = [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_probe', '--model-path', str(model), '--run-dir', str(train_root)]
    limits = {'minimum_free_percent': 10, 'maximum_swap_bytes': 16 * 1024**3}
    for name, command, path in [('training_execution', train_command, train_root / 'execution.json'),
                                ('probe_execution', probe_command, train_root / 'verification/execution.json')]:
        execution = json.loads(path.read_bytes())
        if execution != result.get(name):
            raise ValueError('training process evidence changed')
        _validate_worker_execution(execution, command, limits)
        if execution['status'] != 'complete':
            raise ValueError('training or probe process did not complete')
    log = (train_root / 'stdout.log').read_text() + '\n' + (train_root / 'stderr.log').read_text()
    if parse_training_losses(log, steps=20) != result.get('losses'):
        raise ValueError('training loss evidence changed')
    probe = json.loads((train_root / 'tensor-probe.json').read_bytes())
    loaded = json.loads((adapter / 'probe.json').read_bytes())
    value = probe.get('max_abs_logit_delta')
    if ({k: v for k, v in probe.items() if k != 'changed_lora_tensors'} != loaded or
        probe.get('adapter_sha256') != result['adapter_sha256'] or
        probe.get('changed_lora_tensors') != result.get('changed_lora_tensors') or
        type(probe.get('changed_lora_tensors')) is not int or probe['changed_lora_tensors'] < 1 or
        type(probe.get('nonzero_lora_tensors')) is not int or probe['nonzero_lora_tensors'] < 1 or
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or
        value != result.get('max_abs_logit_delta')):
        raise ValueError('adapter reload evidence changed')
    return {'status': 'source_evidence_matches', 'training_result_sha256': result['sha256'],
            'adapter_sha256': result['adapter_sha256']}
