"""Build and validate a public-only task handoff; this is not isolation proof."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from src.arc.data import Task, validate_grid


def _receipt(receipt: dict[str, Any]) -> tuple[str, dict[str, str]]:
    if not isinstance(receipt, dict):
        raise ValueError('invalid receipt')
    comparison, hashes = receipt.get('comparison_sha256'), receipt.get('selected_task_sha256')
    valid_hash = lambda x: isinstance(x, str) and bool(re.fullmatch(r'[0-9a-f]{64}', x))
    if (not valid_hash(comparison) or not isinstance(hashes, dict) or not hashes or
            any(not isinstance(key, str) or not key or not valid_hash(value) for key, value in hashes.items())):
        raise ValueError('invalid task identities')
    return comparison, hashes


def _task(task_id: str, item: Any, expected: str) -> Task:
    if not isinstance(item, dict) or set(item) != {'train', 'test'}:
        raise ValueError('invalid public task schema')
    train, test = item['train'], item['test']
    if (not isinstance(train, list) or not train or not isinstance(test, list) or not test or
            any(not isinstance(pair, dict) or set(pair) != {'input', 'output'} for pair in train) or
            any(not isinstance(pair, dict) or set(pair) != {'input'} for pair in test)):
        raise ValueError('public tasks may not contain held-out outputs')
    raw = json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('public task content differs')
    return Task(task_id, tuple((validate_grid(p['input']), validate_grid(p['output'])) for p in train),
                tuple(validate_grid(p['input']) for p in test))


def build_public_export(challenges: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    comparison, hashes = _receipt(receipt)
    if not isinstance(challenges, dict) or not set(hashes) <= set(challenges):
        raise ValueError('selected tasks missing')
    tasks = {task_id: copy.deepcopy(challenges[task_id]) for task_id in hashes}
    result = {'comparison_sha256': comparison, 'tasks': tasks}
    load_public_export(result, receipt)
    return result


def load_public_export(export: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Task]:
    comparison, hashes = _receipt(receipt)
    if (not isinstance(export, dict) or set(export) != {'comparison_sha256', 'tasks'} or
            export['comparison_sha256'] != comparison or not isinstance(export['tasks'], dict) or
            set(export['tasks']) != set(hashes)):
        raise ValueError('public export identity or task set differs')
    return {task_id: _task(task_id, export['tasks'][task_id], sha)
            for task_id, sha in hashes.items()}
