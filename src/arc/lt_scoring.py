"""Independent, offline scoring of recorded LT Phase A predictions.

The scorer receives predictions only after generation. It never returns held-out
answers, launches a model, or treats a partial result as a complete comparison.
"""
from __future__ import annotations

import math
from typing import Any

from src.arc.data import Verifier
from src.arc.scoring import score

_MODELS = ('14b', '32b')
_COSTS = ('tokens', 'runtime_seconds', 'peak_memory_bytes', 'peak_swap_bytes')


def _valid_cost(key: str, value: Any) -> bool:
    if value is None:
        return True
    if key == 'runtime_seconds':
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    return type(value) is int and value >= 0


def _output_matches(verifier: Verifier, task_id: str, candidate: Any) -> list[bool]:
    count = verifier.output_count(task_id)
    if not isinstance(candidate, list) or len(candidate) != count:
        return [False] * count
    matches = []
    for index, grid in enumerate(candidate):
        probe = [None] * count
        probe[index] = grid
        matches.append(verifier.correct_outputs(task_id, probe) == 1)
    return matches


def score_comparison(task_ids: list[str], rows: list[dict[str, Any]], verifier: Verifier) -> dict[str, Any]:
    if (not isinstance(task_ids, list) or not task_ids or
            any(not isinstance(x, str) or not x for x in task_ids) or len(set(task_ids)) != len(task_ids)):
        raise ValueError('invalid selected tasks')
    selected = set(task_ids)
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid result')
        model, task_id = row.get('model'), row.get('task_id')
        key = (model, task_id)
        candidates = row.get('candidates')
        if (model not in _MODELS or task_id not in selected or key in indexed or
                row.get('status') not in ('complete', 'error', 'timeout', 'aborted') or
                not isinstance(candidates, list) or len(candidates) > 2 or
                (row['status'] == 'complete' and len(candidates) != 2) or
                any(k not in row or not _valid_cost(k, row[k]) for k in _COSTS)):
            raise ValueError('invalid, duplicate or unselected result')
        indexed[key] = row

    scored = []
    summaries = {}
    for model in _MODELS:
        model_rows = []
        for task_id in task_ids:
            original = indexed.get((model, task_id))
            if original is None:
                model_rows.append({'model': model, 'task_id': task_id, 'status': 'not_attempted',
                                   'pass_at_1': None, 'pass_at_2': None,
                                   'output_pass_at_1': None, 'output_pass_at_2': None,
                                   'output_count': verifier.output_count(task_id),
                                   'invalid': 0, 'timeouts': 0, **{k: None for k in _COSTS}})
                continue
            candidates = original['candidates']
            result = score(verifier, task_id, candidates)
            matches = [_output_matches(verifier, task_id, c) for c in candidates]
            model_rows.append({'model': model, 'task_id': task_id, 'status': original['status'],
                'pass_at_1': result.pass_at_1 if candidates else None,
                'pass_at_2': result.pass_at_2 if len(candidates) == 2 else None,
                'output_pass_at_1': sum(matches[0]) if matches else None,
                'output_pass_at_2': sum(a or b for a, b in zip(*matches)) if len(matches) == 2 else None,
                'output_count': verifier.output_count(task_id), 'invalid': result.invalid,
                'timeouts': result.timeouts, **{k: original[k] for k in _COSTS}})
        attempted = [row for row in model_rows if row['status'] != 'not_attempted']
        summaries[model] = {
            'status': 'complete' if len(attempted) == len(task_ids) and
                all(row['status'] == 'complete' for row in attempted) else 'incomplete',
            'selected': len(task_ids), 'attempted': len(attempted),
            'pass_at_1': sum(row['pass_at_1'] is True for row in model_rows),
            'pass_at_1_denominator': sum(row['pass_at_1'] is not None for row in model_rows),
            'pass_at_2': sum(row['pass_at_2'] is True for row in model_rows),
            'pass_at_2_denominator': sum(row['pass_at_2'] is not None for row in model_rows),
            'output_pass_at_1': sum(row['output_pass_at_1'] or 0 for row in model_rows),
            'output_pass_at_1_denominator': sum(row['output_count'] for row in model_rows if row['output_pass_at_1'] is not None),
            'output_pass_at_2': sum(row['output_pass_at_2'] or 0 for row in model_rows),
            'output_pass_at_2_denominator': sum(row['output_count'] for row in model_rows if row['output_pass_at_2'] is not None),
            'errors': sum(row['status'] in ('error', 'timeout', 'aborted') for row in attempted),
            'invalid': sum(row['invalid'] for row in attempted),
            'timeouts': sum(row['timeouts'] for row in attempted),
            **{k: (sum(row[k] for row in attempted) if k in ('tokens', 'runtime_seconds') else max(row[k] for row in attempted))
               if attempted and all(row[k] is not None for row in attempted) else None for k in _COSTS}}
        scored.extend(model_rows)
    return {'rows': scored, 'models': summaries}
