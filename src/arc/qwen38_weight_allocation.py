"""Deterministic allocation for the new exploratory weight experiment.

Known lineage is supplied explicitly; unrecorded or pretraining exposure is not
resolved by this allocation. Historical manifests are never modified.
"""
from __future__ import annotations

import hashlib
import json
import random


def fingerprint(task):
    # Only challenge content is used: test answers never enter this allocation.
    value = {'train': task['train'], 'test': [{'input': t['input']} for t in task['test']]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def allocate_weight_pool(tasks, official_evaluation, exclusions, lineage, *, seed=20261111, pool_size=200, validation_size=25):
    if pool_size < 0 or validation_size < 1:
        raise ValueError('invalid allocation sizes')
    ids = set(tasks)
    reserved = set().union(*(set(v) for v in exclusions.values())) if exclusions else set()
    if not reserved <= ids or not set(lineage) <= ids:
        raise ValueError('unknown training reservation or lineage')
    if any(not isinstance(v, str) or not v for v in lineage.values()):
        raise ValueError('invalid lineage label')
    parent = {item: item for item in ids}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)

    fingerprints = {k: fingerprint(v) for k, v in tasks.items()}
    for mapping in (fingerprints, lineage):
        seen = {}
        for item, value in sorted(mapping.items()):
            if value in seen: union(item, seen[value])
            else: seen[value] = item
    eval_fingerprints = {fingerprint(t) for t in official_evaluation.values()}
    excluded = reserved | (ids & set(official_evaluation)) | {k for k, v in fingerprints.items() if v in eval_fingerprints}
    blocked_groups = {find(k) for k in excluded}
    groups = {}
    for item in sorted(ids):
        group = find(item)
        if group not in blocked_groups:
            groups.setdefault(group, []).append(item)
    representatives = sorted(group[0] for group in groups.values())
    random.Random(seed).shuffle(representatives)
    if len(representatives) < validation_size:
        raise ValueError('insufficient disjoint validation parents')
    validation = representatives[:validation_size]
    fresh = representatives[validation_size:validation_size + pool_size]
    return {'seed': seed, 'fresh': fresh, 'validation': validation, 'shortfall': pool_size - len(fresh),
            'requested_pool_size': pool_size, 'requested_validation_size': validation_size,
            'excluded_parent_count': sum(find(k) in blocked_groups for k in ids),
            'eligible_lineage_groups': len(representatives),
            'challenge_fingerprints': {k: fingerprints[k] for k in sorted(ids)}}
