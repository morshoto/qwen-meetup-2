"""Run one registered public training task in a supervised child process.

The descriptor and inventory are supplied by the parent. Their local
consistency does not establish preregistration, task allocation, process
isolation, or authorization. The parent must independently validate these and
supervise resource limits and cleanup before accepting a result.
"""
import hashlib
import importlib.metadata
import json
import re

from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_backend import PhaseBBackend
from src.arc.lt_phase_b_public_worker import execute_public_worker
from src.arc.lt_source import verify_source_tree

_SETTINGS = {'seed': 20261111, 'max_tokens': 4096, 'temperature': 0.6, 'thinking': 'off',
             'candidate_count': 2, 'mlx_memory_limit_bytes': 48 * 1024**3,
             'mlx_lm_version': '0.30.7', 'mlx_version': '0.32.4'}
_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value))


def validate_descriptor(*, spec, expected_run_sha256, source_inventory, receipt, allocation_receipt, task_id=None):
    keys = {'protocol', 'status', 'run_id', 'source_code_sha256', 'allocation_audit_sha256',
            'prompt_sha256', 'selected_task_sha256', 'model_manifest', 'settings'}
    if (not isinstance(spec, dict) or set(spec) != keys or spec.get('protocol') != 'lt-phase-b-collection-v1' or
            spec.get('status') != 'frozen' or not isinstance(spec.get('run_id'), str) or
            not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', spec['run_id']) or
            not _valid_hash(expected_run_sha256) or _digest(spec) != expected_run_sha256 or
            not _valid_hash(spec.get('source_code_sha256')) or not _valid_hash(spec.get('allocation_audit_sha256')) or
            spec.get('prompt_sha256') != hashlib.sha256(PROMPT.encode()).hexdigest() or
            not isinstance(source_inventory, dict) or source_inventory.get('code_sha256') != spec['source_code_sha256'] or
            not isinstance(allocation_receipt, dict) or
            _digest(allocation_receipt) != spec['allocation_audit_sha256'] or
            allocation_receipt.get('status') != 'allocation_consistent_unverified' or
            allocation_receipt.get('execution_ready') is not False):
        raise ValueError('invalid collection descriptor or source identity')
    settings, tasks, model = spec['settings'], spec['selected_task_sha256'], spec['model_manifest']
    if (not isinstance(settings, dict) or settings != _SETTINGS or
            any(type(settings[key]) is not type(value) for key, value in _SETTINGS.items()) or
            not isinstance(tasks, dict) or len(tasks) != 100 or
            any(not isinstance(key, str) or not re.fullmatch(r'[0-9a-f]{8}', key) or not _valid_hash(value)
                for key, value in tasks.items()) or (task_id is not None and task_id not in tasks) or
            allocation_receipt.get('training_task_sha256') != tasks or
            not isinstance(allocation_receipt.get('comparison_task_sha256'), dict) or
            len(allocation_receipt['comparison_task_sha256']) != 50 or
            set(tasks) & set(allocation_receipt['comparison_task_sha256']) or
            any(not isinstance(k, str) or not re.fullmatch(r'[0-9a-f]{8}', k) or not _valid_hash(v)
                for k, v in allocation_receipt['comparison_task_sha256'].items()) or
            not isinstance(model, dict) or model.get('model_id') != _MODEL or model.get('revision') != _REVISION or
            model.get('quantization') != {'bits': 4, 'group_size': 64} or
            not _valid_hash(model.get('chat_template_sha256')) or not isinstance(model.get('files_sha256'), dict) or
            not model['files_sha256'] or any(not isinstance(k, str) or not k or not _valid_hash(v)
                                            for k, v in model['files_sha256'].items()) or
            receipt != {'comparison_sha256': expected_run_sha256, 'selected_task_sha256': tasks}):
        raise ValueError('registered settings or tasks differ')
    return settings, model


def run_registered_task(*, spec, expected_run_sha256, source_inventory, source_root, model_path,
                        export, receipt, allocation_receipt, task_id, backend_factory=PhaseBBackend):
    settings, model = validate_descriptor(spec=spec, expected_run_sha256=expected_run_sha256,
        source_inventory=source_inventory, receipt=receipt, allocation_receipt=allocation_receipt, task_id=task_id)
    if not callable(backend_factory):
        raise ValueError('backend factory required')
    if (importlib.metadata.version('mlx-lm') != settings['mlx_lm_version'] or
            importlib.metadata.version('mlx') != settings['mlx_version']):
        raise ValueError('runtime version differs')
    verify_source_tree(source_root, source_inventory)
    row = execute_public_worker(export=export, receipt=receipt, task_id=task_id,
        expected_run_sha256=expected_run_sha256, source_code_sha256=spec['source_code_sha256'],
        adapter_state='off', backend_factory=lambda: backend_factory(model_path=model_path, model_manifest=model,
            adapter_state='off', seed=settings['seed'], max_tokens=settings['max_tokens'],
            temperature=settings['temperature']))
    verify_source_tree(source_root, source_inventory)
    return {**row, 'run_id': spec['run_id']}
