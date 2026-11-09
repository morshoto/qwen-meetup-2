"""Replay pinned local synthetic canary evidence before a weight experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.arc.qwen38_weight_canary import parse_loss, validate_evidence
from src.cli.qwen38_canary import MODEL_REVISION, PACKAGES, hashes

ARTIFACTS = frozenset({'registration.json', 'execution.json', 'probe-execution.json', 'tensor-probe.json',
                       'train.log', 'probe.log', 'initial.safetensors', 'adapter/adapters.safetensors',
                       'adapter/adapter_config.json', 'data/train.jsonl'})


def _sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_canary_bundle(root, model_path, expected_artifacts, expected_base):
    """Validate artifacts against externally frozen hashes, not self-reported pins."""
    root, model_path = Path(root), Path(model_path)
    if set(expected_artifacts) != ARTIFACTS or any(_sha(root / name) != expected_artifacts[name] for name in ARTIFACTS):
        raise ValueError('canary artifact inventory mismatch')
    reg = json.loads((root / 'registration.json').read_bytes())
    run = json.loads((root / 'execution.json').read_bytes())
    probe_run = json.loads((root / 'probe-execution.json').read_bytes())
    probe = json.loads((root / 'tensor-probe.json').read_bytes())
    if (model_path.name != MODEL_REVISION or reg.get('model_revision') != MODEL_REVISION or
        reg.get('packages') != PACKAGES or reg.get('base_before') != expected_base or
        run.get('base_after') != expected_base or hashes(model_path) != expected_base):
        raise ValueError('canary model or toolchain mismatch')
    command = reg.get('command', [])
    required = {'--model-path': str(model_path), '--iters': '1', '--batch-size': '1', '--lora-rank': '8'}
    if ('src.cli.qwen38_canary_worker' not in command or any(command.count(flag) != 1 or command.index(flag) + 1 >= len(command) or command[command.index(flag) + 1] != value for flag, value in required.items()) or
        command.count('--grad-checkpoint') != 1 or command.count('--train-on-completions') != 1):
        raise ValueError('canary command mismatch')
    training = (root / 'data/train.jsonl').read_text().splitlines()
    if len(training) != 1 or json.loads(training[0]) != {'messages': [{'role': 'user', 'content': 'Return one.'}, {'role': 'assistant', 'content': '1'}]}:
        raise ValueError('canary is not the registered synthetic example')
    if (reg.get('timeout_seconds') != 900 or reg.get('minimum_free_percent') != 10 or
        reg.get('maximum_swap_bytes') != 16 * 1024**3 or run.get('status') != 'execution_only_unverified'):
        raise ValueError('canary limits mismatch')
    for execution in (run, probe_run):
        if (execution.get('exit_code') != 0 or execution.get('reason') is not None or
            execution.get('process_group_gone') is not True or not execution.get('samples')):
            raise ValueError('canary execution or cleanup incomplete')
        for sample in execution['samples']:
            free, swap = sample.get('free_percent'), sample.get('swap_bytes')
            if type(free) is not int or free < 10 or type(swap) is not int or not 0 <= swap <= 16 * 1024**3:
                raise ValueError('canary resource sample outside limits')
    if json.loads((root / 'probe.log').read_text()) != probe or probe.get('adapter_sha256') != expected_artifacts['adapter/adapters.safetensors']:
        raise ValueError('adapter probe does not match saved adapter')
    evidence = {'exit_code': run['exit_code'], 'loss': parse_loss((root / 'train.log').read_text()),
                'base_before': reg['base_before'], 'base_after': run['base_after'],
                'adapter_sha256': probe['adapter_sha256'], 'changed_lora_tensors': probe.get('changed_lora_tensors'),
                'max_abs_logit_delta': probe.get('max_abs_logit_delta'), 'timed_out': False,
                'memory_limit_exceeded': False, 'process_group_gone': run['process_group_gone']}
    validate_evidence(evidence)
    return {**evidence, 'status': 'passed', 'artifact_sha256': dict(expected_artifacts)}
