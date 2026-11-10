"""Bounded, answer-free audit of reserved replication parents."""
from __future__ import annotations

from src.arc.qwen38_weight_allocation import fingerprint, input_fingerprints


def audit_reserved(tasks, selected, exclusions, lineage, answer_fingerprints, official):
    """Audit known connections; answer fingerprints must already be recorded.

    No target solutions are accepted. Unknown lineage and pretraining exposure are
    always disclosed, so this bounded check alone cannot establish confirmation.
    """
    ids = set(tasks)
    selected = list(selected)
    blocked = set().union(*(set(values) for values in exclusions.values())) if exclusions else set()
    if (not selected or len(set(selected)) != len(selected) or not set(selected) <= ids or
        not blocked <= ids or not set(lineage) <= ids or
        any(not isinstance(v, str) or not v for v in lineage.values()) or
        not (set(selected) | blocked) <= set(answer_fingerprints) or
        any(not isinstance(answer_fingerprints[k], str) or not answer_fingerprints[k]
            for k in set(selected) | blocked)):
        raise ValueError('invalid or incomplete audit inputs')
    task_inputs = {k: input_fingerprints(v) for k, v in tasks.items()}
    official_inputs = set().union(*(input_fingerprints(v) for v in official.values())) if official else set()
    official_challenges = {fingerprint(v) for v in official.values()}
    blocked |= ids & set(official)
    blocked |= {k for k in ids if task_inputs[k] & official_inputs or fingerprint(tasks[k]) in official_challenges}

    def connected(start):
        reached = set(start)
        while True:
            inputs = set().union(*(task_inputs[k] for k in reached)) if reached else set()
            families = {lineage[k] for k in reached if k in lineage}
            expanded = reached | {k for k, value in task_inputs.items() if value & inputs} | {
                k for k, family in lineage.items() if family in families}
            if expanded == reached:
                return reached
            reached = expanded

    blocked = connected(blocked)
    if blocked & set(selected):
        raise ValueError('reserved task has known exposure')
    for i, item in enumerate(selected):
        if connected({item}) & set(selected[i + 1:]):
            raise ValueError('reserved tasks have shared lineage')
    # Require fingerprints for all transitively blocked tasks, too.
    if (not blocked <= set(answer_fingerprints) or
        any(not isinstance(answer_fingerprints[k], str) or not answer_fingerprints[k] for k in blocked)):
        raise ValueError('missing blocked answer fingerprints')
    protected_answers = {answer_fingerprints[k] for k in blocked}
    selected_answers = [answer_fingerprints[k] for k in selected]
    if set(selected_answers) & protected_answers or len(set(selected_answers)) != len(selected_answers):
        raise ValueError('answer fingerprint overlap')
    return {'status': 'bounded_disjoint', 'designation': 'exploratory', 'selected_count': len(selected),
            'blocked_count': len(blocked), 'unresolved_exposure': True,
            'challenge_fingerprints': {k: fingerprint(tasks[k]) for k in selected}}
