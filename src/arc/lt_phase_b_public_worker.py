"""Execute a public-only Phase B task with a caller-supplied backend.

Identity strings are caller claims, not source or process attestation. The
caller must bind them to a separately frozen run and supervise the process.
"""
import hashlib
import json
import re

from src.arc.lt_live_backend import PROMPT
from src.arc.lt_public import load_public_export
from src.arc.lt_worker import execute_task


def execute_public_worker(*, export, receipt, task_id, expected_run_sha256, source_code_sha256,
                          adapter_state, backend_factory, expected_adapter_sha256=None):
    valid_hash = lambda value: isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value))
    if (not valid_hash(expected_run_sha256) or not valid_hash(source_code_sha256) or
            not isinstance(receipt, dict) or receipt.get('comparison_sha256') != expected_run_sha256 or
            not isinstance(task_id, str) or not task_id or adapter_state not in ('off', 'on') or
            not callable(backend_factory) or
            (adapter_state == 'off' and expected_adapter_sha256 is not None) or
            (adapter_state == 'on' and not valid_hash(expected_adapter_sha256))):
        raise ValueError('invalid public worker identity')
    tasks = load_public_export(export, receipt)
    if task_id not in tasks:
        raise ValueError('task not registered')
    payload = json.dumps({'instruction': PROMPT, 'task': tasks[task_id].prompt_data()},
                         sort_keys=True, separators=(',', ':'))
    prompt_hash = hashlib.sha256(payload.encode()).hexdigest()
    backend = backend_factory()
    if (getattr(backend, 'adapter_state', None) != adapter_state or
            getattr(backend, 'adapter_sha256', None) != expected_adapter_sha256):
        raise ValueError('loaded adapter identity differs')
    row = execute_task(tasks[task_id], backend, '14b')
    return {**row, 'run_sha256': expected_run_sha256, 'source_code_sha256': source_code_sha256,
            'public_task_sha256': receipt['selected_task_sha256'][task_id],
            'prompt_sha256': prompt_hash, 'adapter_state': adapter_state,
            'adapter_sha256': expected_adapter_sha256}
