"""Offline, full-task eligibility for supervised Phase B candidates.

A caller must independently bind the journal to the preregistered model and
source. This module does not establish provenance or dispatch inference.
"""
import hashlib
import json

from src.arc.data import validate_grid


def assess_candidates(row, verifier):
    if (not isinstance(row, dict) or not isinstance(row.get('task_id'), str) or
            not row['task_id'] or row.get('model') != '14b' or row.get('status') != 'complete' or
            not isinstance(row.get('candidates'), list) or len(row['candidates']) != 2 or
            not isinstance(row.get('cleanup'), dict) or
            row['cleanup'].get('leader_reaped') is not True or
            row['cleanup'].get('group_absent') is not True):
        raise ValueError('unusable worker result')
    task_id = row['task_id']
    try:
        count = verifier.output_count(task_id)
    except (KeyError, TypeError) as exc:
        raise ValueError('unknown task') from exc
    if type(count) is not int or count <= 0:
        raise ValueError('invalid solution count')
    lineage = hashlib.sha256(json.dumps({'task_id': task_id}, sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest()
    records, accepted = [], False
    for index, candidate in enumerate(row['candidates']):
        valid = isinstance(candidate, list) and len(candidate) == count
        if valid:
            try:
                for grid in candidate:
                    validate_grid(grid)
            except (TypeError, ValueError):
                valid = False
        exact = valid and verifier.matches(task_id, candidate)
        status = ('duplicate_task' if accepted else 'eligible') if exact else ('incorrect' if valid else 'invalid')
        accepted |= exact
        records.append({'task_id': task_id, 'candidate_index': index, 'prediction': candidate,
                        'verification_status': status, 'task_exact_match': bool(exact),
                        'solution_lineage_sha256': lineage})
    return records
