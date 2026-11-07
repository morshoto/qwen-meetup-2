"""Run one registered public comparison task with adapter off or on.

The parent must independently validate allocation and public inputs, supervise
resources, retain observations and score offline. Local checks do not attest
process isolation or authorize evaluation.
"""
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_backend import PhaseBBackend
from src.arc.lt_phase_b_public_worker import execute_public_worker
from src.arc.lt_phase_b_real_reload_dispatch import review_real_reload_attempt
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_source import verify_source_tree
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_runtime_identity import verify_runtime_identity


def run_registered_comparison_task(*, comparison_spec, expected_comparison_sha256, collection_spec,
        expected_collection_sha256, training_spec, expected_training_sha256, canary_spec, expected_canary_sha256,
        source_inventory, source_root, run_dir, model_path, allocation_receipt, collection_receipt,
        comparison_receipt, export, task_id, adapter_state, backend_factory=PhaseBBackend):
    settings, model = validate_canary_identity(canary_spec=canary_spec, expected_canary_sha256=expected_canary_sha256,
        collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
        source_inventory=source_inventory, allocation_receipt=allocation_receipt, receipt=collection_receipt)
    keys = {'protocol', 'status', 'run_id', 'collection_sha256', 'training_sha256', 'canary_sha256',
            'source_code_sha256', 'prompt_sha256', 'selected_task_sha256', 'adapter_sha256', 'settings'}
    valid_hash = lambda x: isinstance(x, str) and bool(re.fullmatch(r'[0-9a-f]{64}', x))
    canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (not isinstance(comparison_spec, dict) or set(comparison_spec) != keys or
            not valid_hash(expected_comparison_sha256) or _digest(comparison_spec) != expected_comparison_sha256 or
            comparison_spec.get('protocol') != 'lt-phase-b-comparison-v1' or comparison_spec.get('status') != 'frozen' or
            comparison_spec.get('run_id') != collection_spec['run_id'] or
            comparison_spec.get('collection_sha256') != expected_collection_sha256 or
            comparison_spec.get('training_sha256') != expected_training_sha256 or
            comparison_spec.get('canary_sha256') != expected_canary_sha256 or
            comparison_spec.get('source_code_sha256') != collection_spec['source_code_sha256'] or
            comparison_spec.get('prompt_sha256') != hashlib.sha256(PROMPT.encode()).hexdigest() or
            canonical(comparison_spec.get('settings')) != canonical(settings) or
            not valid_hash(comparison_spec.get('adapter_sha256')) or adapter_state not in ('off', 'on') or
            not callable(backend_factory)):
        raise ValueError('comparison registration differs')
    tasks = comparison_spec['selected_task_sha256']
    if (not isinstance(tasks, dict) or len(tasks) != 50 or
            any(not isinstance(k, str) or not re.fullmatch(r'[0-9a-f]{8}', k) or not valid_hash(v) for k, v in tasks.items()) or
            tasks != allocation_receipt.get('comparison_task_sha256') or
            set(tasks) & set(collection_spec['selected_task_sha256']) or task_id not in tasks or
            comparison_receipt != {'comparison_sha256': expected_comparison_sha256, 'selected_task_sha256': tasks}):
        raise ValueError('comparison task allocation differs')
    def replay():
        return review_real_reload_attempt(run_dir, training=training_spec, canary=canary_spec,
            expected_training_sha256=expected_training_sha256, expected_canary_sha256=expected_canary_sha256,
            expected_collection_sha256=expected_collection_sha256)
    prior = replay()
    if (not isinstance(prior, dict) or prior.get('status') != 'real_adapter_reload_observed_unverified' or
            prior.get('reload_verified') is not False or prior.get('promotion_allowed') is not False or
            not isinstance(prior.get('reload'), dict) or
            prior['reload'].get('adapter_sha256') != comparison_spec['adapter_sha256']):
        raise ValueError('real adapter reload evidence differs')
    if (importlib.metadata.version('mlx-lm') != settings['mlx_lm_version'] or
            importlib.metadata.version('mlx') != settings['mlx_version']):
        raise ValueError('runtime version differs')
    def verify():
        verify_runtime_identity(sys.executable, collection_spec['runtime_manifest'])
        verify_environment(sys.executable, collection_spec['environment_manifest'])
        verify_source_tree(source_root, source_inventory)
    verify()
    adapter_sha = comparison_spec['adapter_sha256'] if adapter_state == 'on' else None
    adapter_kwargs = {'adapter_path': str(Path(run_dir) / 'gen1-adapter'), 'adapter_sha256': adapter_sha} if adapter_sha else {}
    row = execute_public_worker(export=export, receipt=comparison_receipt, task_id=task_id,
        expected_run_sha256=expected_comparison_sha256, source_code_sha256=comparison_spec['source_code_sha256'],
        adapter_state=adapter_state, expected_adapter_sha256=adapter_sha,
        backend_factory=lambda: backend_factory(model_path=model_path, model_manifest=model, adapter_state=adapter_state,
            seed=settings['seed'], max_tokens=settings['max_tokens'], temperature=settings['temperature'], **adapter_kwargs))
    verify()
    if replay() != prior:
        raise ValueError('adapter reload evidence changed')
    return {**row, 'run_id': comparison_spec['run_id']}
