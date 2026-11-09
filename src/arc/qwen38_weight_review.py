"""Independent bounded review of an exploratory weight allocation proposal."""
from __future__ import annotations

import hashlib
import json


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def review_selection(proposal, tasks, solutions, official):
    """Recheck IDs, challenge inputs, known lineage and solution fingerprints.

    Equal visible inputs are propagated transitively. No output grids are returned.
    This cannot establish semantic or model-pretraining independence.
    """
    allocation = proposal['allocation']; fresh = allocation['fresh']; validation = allocation['validation']
    selected = fresh + validation
    blocked = set().union(*(set(ids) for ids in proposal['exclusions'].values()))
    lineage = proposal['known_lineage']
    if (len(selected) != len(set(selected)) or not set(selected) <= tasks.keys() or
        not set(selected) <= solutions.keys() or not blocked <= tasks.keys() or
        not set(lineage) <= tasks.keys() or any(not isinstance(v, str) or not v for v in lineage.values()) or
        sorted(official) != proposal['official_evaluation_ids'] or set(selected) & blocked or
        set(selected) & official.keys()):
        raise ValueError('selection or exclusion IDs invalid')
    task_inputs = {key: {_hash(pair['input']) for pair in task['train'] + task['test']} for key, task in tasks.items()}
    official_inputs = set().union(*({_hash(pair['input']) for pair in t['train'] + t['test']} for t in official.values())) if official else set()
    blocked |= set(tasks) & set(official)
    blocked |= {key for key, values in task_inputs.items() if values & official_inputs}
    # Propagate all visible-input and registered-family connections from blocked tasks.
    while True:
        inputs = set().union(*(task_inputs[k] for k in blocked)) if blocked else set()
        families = {lineage[k] for k in blocked if k in lineage}
        expanded = blocked | {k for k, values in task_inputs.items() if values & inputs} | {
            k for k, family in lineage.items() if family in families}
        if expanded == blocked: break
        blocked = expanded
    if set(selected) & blocked:
        raise ValueError('selection intersects known blocked lineage')
    # Selected parents must also be disjoint through unselected intermediary tasks.
    for offset, parent in enumerate(selected):
        reachable = {parent}
        while True:
            inputs = set().union(*(task_inputs[k] for k in reachable))
            families = {lineage[k] for k in reachable if k in lineage}
            expanded = reachable | {k for k, values in task_inputs.items() if values & inputs} | {
                k for k, family in lineage.items() if family in families}
            if expanded == reachable: break
            reachable = expanded
        if reachable & set(selected[offset + 1:]):
            raise ValueError('selected parents share known input or lineage')
        task = tasks[parent]
        content = {'train': task['train'], 'test': [{'input': pair['input']} for pair in task['test']]}
        if (allocation['challenge_fingerprints'].get(parent) != _hash(content) or
            proposal['solution_fingerprints'].get(parent) != _hash(solutions[parent])):
            raise ValueError('selected content fingerprint mismatch')
    return {'fresh_parents': len(fresh), 'validation_parents': len(validation),
            'selected_parent_overlap': 0, 'selected_input_overlap': 0,
            'blocked_training_parents': len(blocked), 'official_challenges': len(official),
            'unresolved_semantic_exposure': True}
