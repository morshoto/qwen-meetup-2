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


def input_fingerprints(task):
    return {hashlib.sha256(json.dumps(pair['input'], separators=(',', ':')).encode()).hexdigest()
            for pair in task['train'] + task['test']}


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
    inputs = {k: input_fingerprints(v) for k, v in tasks.items()}
    seen_inputs = {}
    for item in sorted(ids):
        for value in sorted(inputs[item]):
            if value in seen_inputs: union(item, seen_inputs[value])
            else: seen_inputs[value] = item
    eval_fingerprints = {fingerprint(t) for t in official_evaluation.values()}
    eval_inputs = set().union(*(input_fingerprints(t) for t in official_evaluation.values())) if official_evaluation else set()
    excluded = reserved | (ids & set(official_evaluation)) | {k for k, v in fingerprints.items() if v in eval_fingerprints} | {k for k, v in inputs.items() if v & eval_inputs}
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


def freeze_registration(path, value):
    """Create an immutable registration or require the exact existing bytes."""
    import os
    from pathlib import Path
    if 'sha256' in value:
        raise ValueError('registration digest is derived')
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    registration = dict(value, sha256=hashlib.sha256(encoded).hexdigest())
    payload = json.dumps(registration, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_text(encoding='utf-8') != payload:
            raise ValueError('registration changed')
    return registration


def build_registration(tasks, official_evaluation, exclusions, lineage, *, input_hashes, comparison_sha256,
                       prompt, seed=20261111, pool_size=200, validation_size=25):
    from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION, MAX_CONTEXT, MAX_TOKENS, TEMPERATURE
    required = {'challenges', 'solutions'}
    if not required <= input_hashes.keys() or not all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in list(input_hashes.values()) + [comparison_sha256]):
        raise ValueError('invalid source hashes')
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('missing prompt')
    allocation = allocate_weight_pool(tasks, official_evaluation, exclusions, lineage, seed=seed,
                                      pool_size=pool_size, validation_size=validation_size)
    return {'protocol': 'qwen38-weight-exploratory-v1', 'comparison_sha256': comparison_sha256,
            'input_hashes': input_hashes, 'exclusions': exclusions, 'known_lineage': lineage,
            'official_evaluation_ids': sorted(official_evaluation), 'allocation': allocation,
            'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'quantization': '4-bit MLX'},
            'generation': {'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                           'temperature': TEMPERATURE, 'max_tokens_per_candidate': MAX_TOKENS,
                           'max_context': MAX_CONTEXT, 'initial_candidates': 2, 'additional_candidates': 6,
                           'order': 'breadth_first', 'extra_only_after_no_verified_success': True,
                           'seed_rule': f'{seed} + 10000 + fresh_index * 8 + candidate_index',
                           'wall_seconds_limit': 86400, 'reserved_token_limit': 6553600,
                           'stop_rule': 'all attempts completed or first budget exhaustion; retain incomplete tasks'},
            'validation_policy': 'reserved parents; no answers in generation or training targets',
            'exposure_limitations': 'Known registers checked; historical and model-pretraining exposure unresolved.'}
