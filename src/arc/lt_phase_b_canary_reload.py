"""Reload a synthetic canary adapter and sample in a separate worker.

This is local consistency evidence only. The parent must independently review
training and reload supervision, resource observations and process cleanup.
The synthetic sample is not ARC evaluation or a Gen 1 promotion.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_adapter_artifact import inspect_adapter_artifact
from src.arc.lt_phase_b_backend import PhaseBBackend
from src.arc.lt_phase_b_canary_trainer import _private_file
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_tensor_evidence import compare_tensor_states
from src.arc.lt_source import verify_source_tree
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_model_identity import _json_file, verify_model_manifest
from src.arc.rsi_runtime_identity import verify_runtime_identity

_SAMPLE_TASK = {'train': [{'input': [[0]], 'output': [[1]]}], 'test': [{'input': [[2]]}]}


def run_canary_reload(*, canary_spec, expected_canary_sha256, collection_spec, expected_collection_sha256,
                      source_inventory, source_root, allocation_receipt, receipt, config_path,
                      training_summary, tensor_before, tensor_after, backend_factory=PhaseBBackend):
    _, model = validate_canary_identity(canary_spec=canary_spec, expected_canary_sha256=expected_canary_sha256,
        collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
        source_inventory=source_inventory, allocation_receipt=allocation_receipt, receipt=receipt)
    if not callable(backend_factory) or not isinstance(training_summary, dict):
        raise ValueError('invalid reload inputs')
    if (training_summary.get('status') != 'training_observed_reload_pending' or
            training_summary.get('training_verified') is not False or training_summary.get('promotion_allowed') is not False or
            training_summary.get('canary_sha256') != expected_canary_sha256 or
            training_summary.get('collection_sha256') != expected_collection_sha256):
        raise ValueError('canary training summary differs')
    checked = compare_tensor_states(tensor_before, tensor_after)
    if training_summary.get('tensor_evidence') != checked:
        raise ValueError('canary tensor evidence differs')
    config_path = Path(config_path)
    _private_file(config_path)
    config = _json_file(config_path, canary_spec['config_sha256'])
    try:
        adapter_path, model_path = Path(config['adapter_path']), Path(config['model'])
    except (KeyError, TypeError) as exc:
        raise ValueError('canary paths unavailable') from exc
    artifact = inspect_adapter_artifact(adapter_path, expected_tensors=tensor_after['adapter'])
    if (training_summary.get('adapter') != artifact or artifact.get('reload_verified') is not False or
            artifact.get('promotion_allowed') is not False):
        raise ValueError('saved canary adapter differs')
    verify_runtime_identity(sys.executable, canary_spec['runtime_manifest'])
    verify_environment(sys.executable, canary_spec['environment_manifest'])
    verify_model_manifest(model_path, model, model_key='14b')
    verify_source_tree(source_root, source_inventory)
    backend = backend_factory(model_path=model_path, model_manifest=model, adapter_state='on',
        adapter_path=adapter_path, adapter_sha256=artifact['adapter_sha256'], seed=20261111,
        max_tokens=64, temperature=0.6)
    if (getattr(backend, 'adapter_state', None) != 'on' or
            getattr(backend, 'adapter_sha256', None) != artifact['adapter_sha256']):
        raise ValueError('loaded adapter identity differs')
    payload = {'instruction': PROMPT, 'task': _SAMPLE_TASK}
    prompt = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    backend.generate(prompt, kind='direct', candidate=0)
    raw, usage = getattr(backend, 'raw', None), getattr(backend, 'stats', None)
    if (not isinstance(raw, str) or not raw or len(raw.encode()) > 32768 or not isinstance(usage, dict) or
            type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] < 0 or
            type(usage.get('completion_tokens')) is not int or not 0 < usage['completion_tokens'] <= 64 or
            type(usage.get('peak_memory_gb')) not in (int, float) or not math.isfinite(usage['peak_memory_gb']) or
            not 0 <= usage['peak_memory_gb'] <= (48 * 1024**3 / 1e9) or
            usage.get('finish_reason') not in ('stop', 'length')):
        raise ValueError('invalid reload inference observation')
    try:
        normalize_direct(raw, 1)
        format_valid = True
    except ValueError:
        format_valid = False
    verify_runtime_identity(sys.executable, canary_spec['runtime_manifest'])
    verify_environment(sys.executable, canary_spec['environment_manifest'])
    verify_model_manifest(model_path, model, model_key='14b')
    verify_source_tree(source_root, source_inventory)
    if inspect_adapter_artifact(adapter_path, expected_tensors=tensor_after['adapter']) != artifact:
        raise ValueError('adapter changed during reload')
    return {'status': 'reload_observed_unverified', 'reload_verified': False, 'promotion_allowed': False,
            'canary_sha256': expected_canary_sha256, 'collection_sha256': expected_collection_sha256,
            'adapter_sha256': artifact['adapter_sha256'], 'sample_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'sample_raw': raw, 'sample_format_valid': format_valid, 'usage': usage}
