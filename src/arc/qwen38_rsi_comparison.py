"""Allocation and offline reporting for the exploratory Qwen3.8 comparison."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from src.arc.scoring import score

SEED = 20261111
ARMS = ('prompt', 'agent', 'weights')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def allocate(proposal, tasks, seed=SEED):
    phase = proposal['task_ids']
    if len(phase) != 200 or len(set(phase)) != 200 or not set(phase) <= tasks.keys():
        raise ValueError('invalid Phase A task list')
    excluded = set(phase)
    for key in ('protected_exclusions', 'training_exclusions', 'known_exposure_exclusions', 'development_exclusions'):
        excluded.update(proposal[key])
    eligible = sorted(k for k, v in tasks.items() if k not in excluded and len(v['train']) >= 2)
    if len(eligible) < 20:
        raise ValueError('insufficient audited development and selection tasks')
    evaluation = list(phase)
    random.Random(seed).shuffle(evaluation)
    random.Random(seed + 1).shuffle(eligible)
    allocation = {'seed': seed, 'evaluation': evaluation[:50], 'development': eligible[:10], 'selection': eligible[10:20]}
    allocation['orders'] = {}
    for i, arm in enumerate(ARMS):
        order = allocation['evaluation'][:]
        random.Random(seed + 2 + i).shuffle(order)
        allocation['orders'][arm] = order
    allocation['sha256'] = digest(allocation)
    return allocation


def validate_allocation(allocation, proposal, tasks):
    if allocation != allocate(proposal, tasks, allocation['seed']):
        raise ValueError('allocation differs from deterministic audited selection')


def summarize(ids, rows, verifier):
    indexed = {}
    for row in rows:
        task_id = row.get('task_id')
        if task_id not in ids or task_id in indexed or row.get('status') != 'complete' or len(row.get('candidates', [])) != 2:
            raise ValueError('invalid or duplicate result')
        indexed[task_id] = row
    result = {}
    for task_id, row in indexed.items():
        candidates = [c.get('outputs') for c in row['candidates']]
        s = score(verifier, task_id, candidates)
        result[task_id] = {'pass1': s.pass_at_1, 'pass2': s.pass_at_2,
                           'valid': 2 - s.invalid - s.timeouts,
                           'seconds': row.get('runtime_seconds')}
    return result


def compare(baseline, other):
    common = baseline.keys() & other.keys()
    return {'paired': len(common),
            'gained': sum(not baseline[t]['pass2'] and other[t]['pass2'] for t in common),
            'lost': sum(baseline[t]['pass2'] and not other[t]['pass2'] for t in common)}
