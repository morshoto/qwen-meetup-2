"""Offline utilities for the separate LT exploratory ARC comparison.

A manifest is a proposal until the operator freezes and verifies all runtime
identities; these utilities do not authorize or start inference.
"""
from __future__ import annotations

import random
import re
from typing import Any

from src.arc.data import Verifier
from src.arc.scoring import score


def make_manifest(ids: list[str], *, seed: int, development: set[str],
                  training: set[str], exposures: set[str], dataset_sha256: str) -> dict[str, Any]:
    if (type(seed) is not int or not isinstance(ids, list) or
            any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)) or
            not isinstance(dataset_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', dataset_sha256)):
        raise ValueError('invalid task IDs, seed, or dataset digest')
    universe = set(ids)
    if any(not isinstance(group, set) or not group <= universe for group in (development, training, exposures)):
        raise ValueError('invalid exclusions')
    if development & training:
        raise ValueError('development and training exclusions overlap')
    eligible = sorted(universe - development - training - exposures)
    if len(eligible) < 200:
        raise ValueError('fewer than 200 eligible tasks')
    random.Random(seed).shuffle(eligible)
    return {'protocol': 'lt-phase-a-exploratory', 'status': 'proposal', 'seed': seed,
            'dataset_sha256': dataset_sha256, 'task_ids': sorted(eligible[:200]),
            'development_exclusions': sorted(development), 'training_exclusions': sorted(training),
            'known_exposure_exclusions': sorted(exposures),
            'unknown_pretraining_exposure': True, 'unknown_project_exposure': True}


def summarize(task_ids: list[str], rows: list[dict[str, Any]], verifier: Verifier) -> dict[str, Any]:
    if len(task_ids) != len(set(task_ids)):
        raise ValueError('duplicate selected task')
    selected = set(task_ids)
    seen: set[str] = set()
    pass1 = pass2 = correct_outputs = output_count = 0
    for row in rows:
        task_id = row['task_id']
        if task_id not in selected or task_id in seen:
            raise ValueError('duplicate or unselected result')
        seen.add(task_id)
        candidates = row['candidates']
        if not isinstance(candidates, list) or len(candidates) > 2:
            raise ValueError('invalid candidate list')
        result = score(verifier, task_id, candidates)
        pass1 += result.pass_at_1
        pass2 += result.pass_at_2
        output_count += verifier.output_count(task_id)
        if candidates:
            correct_outputs += verifier.correct_outputs(task_id, candidates[0])
    return {'status': 'complete' if len(seen) == len(selected) else 'incomplete',
            'selected': len(selected), 'attempted': len(seen), 'pass_at_1': pass1,
            'pass_at_2': pass2, 'first_candidate_correct_outputs': correct_outputs,
            'test_outputs_attempted': output_count}
