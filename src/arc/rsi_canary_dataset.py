"""Construct a quarantined synthetic canary, without executing training."""
import hashlib
import json
import os

from src.arc.data import Task, validate_grid
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_sft_format import audit_sft_format
from src.arc.rsi_sft_materialize import _write_dataset
from src.arc.rsi_slot_audit import task_sha256
from src.arc.rsi_slot_lifecycle import _create, _directory

_CODE = 'def transform(grid):\n    return [row[::-1] for row in grid]'


def canary_tasks():
    tasks = {}
    for index in range(5):
        train = [[(index + cell) % 10 for cell in range(12)]]
        query = [[(index + 2 * cell + 1) % 10 for cell in range(13)]]
        key = f'preflight-canary-{index:02d}'
        tasks[key] = Task(key, ((train, [train[0][::-1]]),), (query,))
    return tasks


def _grid_key(grid):
    return json.dumps(validate_grid(grid), separators=(',', ':'))


def prepare_canary_dataset(*, root, forbidden_tasks, tokenizer, render):
    if (not isinstance(forbidden_tasks, dict) or not forbidden_tasks or
            any(not isinstance(key, str) or not isinstance(task, Task) or task.id != key or
                not task.train or not task.test_inputs for key, task in forbidden_tasks.items())):
        raise ValueError('invalid forbidden task inventory')
    try:
        forbidden = {_grid_key(grid) for task in forbidden_tasks.values()
                     for grid in [*[item for pair in task.train for item in pair], *task.test_inputs]}
        canaries = canary_tasks()
        canary_grids = {_grid_key(grid) for task in canaries.values()
                        for grid in [*[item for pair in task.train for item in pair], *task.test_inputs]}
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid task grid inventory') from exc
    if forbidden & canary_grids or set(canaries) & set(forbidden_tasks):
        raise ValueError('canary overlaps supplied task inventory')
    records = [{'task_id': key, 'source_sha256': hashlib.sha256(_CODE.encode()).hexdigest(),
                'request_id': f'synthetic-{key}',
                'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                             {'role': 'user', 'content': solve_prompt(task)},
                             {'role': 'assistant', 'content': _CODE}]}
               for key, task in canaries.items()]
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    plan = {'status': 'quarantined_not_authorized', 'training_allowed': False,
            'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
            'task_hashes': {key: task_sha256(task) for key, task in canaries.items()},
            'generation_training': {'mask_prompt': True, 'max_seq_length': 2048, 'records': records}}
    checked = audit_sft_format(plan, tasks=canaries, tokenizer=tokenizer, render=render)
    raw = checked['dataset_jsonl'].encode()
    audit = {key: value for key, value in checked.items() if key != 'dataset_jsonl'}
    audit.update({'status': 'canary_dataset_quarantined', 'training_allowed': False,
                  'source': 'fixed_synthetic_canary', 'task_hashes': plan['task_hashes'],
                  'forbidden_grid_set_sha256': hashlib.sha256(json.dumps(sorted(forbidden),
                      separators=(',', ':')).encode()).hexdigest()})
    with _directory(root) as parent:
        os.mkdir('preflight-canary.quarantine', 0o700, dir_fd=parent)
        os.fsync(parent)
        directory = os.open('preflight-canary.quarantine', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            os.fchmod(directory, 0o700)
            _write_dataset(directory, raw)
            _create(directory, 'audit.json', audit)
        finally:
            os.close(directory)
    return {'status': audit['status'], 'training_allowed': False, 'row_count': checked['row_count'],
            'dataset_jsonl_sha256': checked['dataset_jsonl_sha256']}
