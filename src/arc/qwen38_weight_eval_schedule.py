"""Schedule a fixed matched evaluation without feeding scores back into generation."""
from __future__ import annotations

import math

from src.arc.qwen38_local_runner import SEED


def next_evaluation_action(descriptor, journal):
    ids = descriptor['task_ids']; count = descriptor['candidate_count']; max_tokens = descriptor['max_tokens']
    token_limit = descriptor['reserved_completion_tokens']; wall_limit = descriptor['wall_seconds_limit']
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or
        any(not isinstance(task, str) or not task for task in ids) or count != 2 or
        type(max_tokens) is not int or max_tokens < 1 or type(token_limit) is not int or token_limit < 0 or
        isinstance(wall_limit, bool) or not isinstance(wall_limit, (int, float)) or
        not math.isfinite(wall_limit) or wall_limit <= 0):
        raise ValueError('invalid evaluation schedule')
    order = [(task_id, index) for task_id in ids for index in range(2)]
    if len(journal) > len(order) or len(journal) * max_tokens > token_limit:
        raise ValueError('evaluation journal exceeds registered budget')
    elapsed = 0.0; failed = False; resource_stop = False
    stop_reasons = {'low_free_memory', 'swap_limit', 'memory_probe_failed', 'remaining_process_group',
                    'output_limit', 'worker_failed', 'worker_output_invalid'}
    for position, row in enumerate(journal):
        seconds = row.get('elapsed_seconds'); status = row.get('status'); error = row.get('error')
        if (resource_stop or (row.get('task_id'), row.get('candidate_index')) != order[position] or
            status not in ('started', 'complete', 'error') or isinstance(seconds, bool) or
            not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0 or
            (status == 'started' and position != len(journal) - 1) or
            (status == 'error' and (not isinstance(error, str) or not error))):
            raise ValueError('invalid evaluation journal')
        elapsed += seconds
        if status == 'started': return {'status': 'requires_reconciliation'}
        if status == 'error':
            failed = True
            resource_stop = error in stop_reasons or error.startswith('supervisor_error:')
    if resource_stop: return {'status': 'resource_stop'}
    if len(journal) == len(order): return {'status': 'incomplete' if failed else 'complete'}
    if elapsed >= wall_limit: return {'status': 'time_budget_exhausted'}
    if (len(journal) + 1) * max_tokens > token_limit: return {'status': 'token_budget_exhausted'}
    task_id, index = order[len(journal)]
    return {'status': 'ready', 'task_id': task_id, 'candidate_index': index, 'seed': SEED + index}
