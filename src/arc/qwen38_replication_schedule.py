"""Matched two-arm scheduling without access to evaluation answers."""
from __future__ import annotations

from src.arc.qwen38_weight_eval_schedule import next_evaluation_action


def next_replication_action(descriptor, journal):
    ids = descriptor.get('task_ids')
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or
        any(not isinstance(item, str) or not item for item in ids)):
        raise ValueError('invalid replication task inventory')
    order = [(arm, task) for arm in ('base', 'adapter') for task in ids]
    if len(journal) > len(order) * 2:
        raise ValueError('replication journal exceeds task inventory')
    synthetic = []
    for position, row in enumerate(journal):
        pair = position // 2
        if (row.get('arm'), row.get('task_id')) != order[pair]:
            raise ValueError('replication arm or task order changed')
        synthetic.append({**row, 'task_id': str(pair)})
    schedule = {**descriptor, 'task_ids': [str(i) for i in range(len(order))]}
    action = next_evaluation_action(schedule, synthetic)
    if action['status'] == 'ready':
        arm, task = order[int(action['task_id'])]
        action = {**action, 'arm': arm, 'task_id': task}
    return action
