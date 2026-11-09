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


PROPOSAL_SHA256 = '581d0250bf5aa1b9874094e12b16b151fdb9ea5dad8f9ef406279ab2b3f669de'
HISTORY_MANIFEST_SHA256 = '911763ed410c9bafc12e4259bd8c29a26c5d00a2e12a3d67c5cce045c0f9cc7b'
HISTORY_AGGREGATE_SHA256 = 'f07534b2a20704cda4eb19879b43d047bfb34d865e94e24028be552ace58f77e'
PHASE_B_HEAD = '4d6a9bc21e8756ce375ce493675bca71e49af221'


def bind_review(proposal, observed, source_hashes, review_source_sha256,
                history_manifest_sha256, history_aggregate_sha256, phase_b_head):
    """Bind a completed bounded audit; callers must actually replay the inputs."""
    import re
    if (proposal.get('sha256') != PROPOSAL_SHA256 or
        _hash({k: v for k, v in proposal.items() if k != 'sha256'}) != PROPOSAL_SHA256 or
        source_hashes != proposal['input_hashes'] or
        not isinstance(review_source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', review_source_sha256) or
        history_manifest_sha256 != HISTORY_MANIFEST_SHA256 or history_aggregate_sha256 != HISTORY_AGGREGATE_SHA256 or
        phase_b_head != PHASE_B_HEAD):
        raise ValueError('bounded review source mismatch')
    if (observed.get('fresh_parents') != len(proposal['allocation']['fresh']) or
        observed.get('validation_parents') != len(proposal['allocation']['validation']) or
        observed.get('selected_parent_overlap') != 0 or observed.get('selected_input_overlap') != 0 or
        observed.get('official_challenges') != 120 or observed.get('unresolved_semantic_exposure') is not True):
        raise ValueError('bounded selection review did not pass')
    result = {'status': 'reviewed_bounded', 'proposal_sha256': PROPOSAL_SHA256,
              'known_lineage_sha256': _hash(proposal['known_lineage']), 'unresolved_exposure': True,
              'selection': observed, 'input_hashes': source_hashes, 'review_source_sha256': review_source_sha256,
              'history_manifest_sha256': history_manifest_sha256, 'history_aggregate_sha256': history_aggregate_sha256,
              'phase_b_head': phase_b_head,
              'limitations': ['The supplied lineage register may omit unknown family relationships.',
                              'No complete semantic lineage or model-pretraining exposure inventory exists.',
                              'The Phase B comparison set is a proposal, not a frozen registration.']}
    result['sha256'] = _hash(result)
    return result
