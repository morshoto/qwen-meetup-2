"""Offline audit of archived FP8 candidates; this does not authorize training."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from src.arc.qwen38_local_runner import parse_prediction
from src.arc.qwen38_weight_allocation import input_fingerprints
from src.arc.qwen38_rsi_comparison import digest


def historical_blocked_ids(tasks, official, excluded_ids, lineage):
    """Propagate known exclusions through identical inputs and declared families."""
    if not set(excluded_ids) <= tasks.keys() or not set(lineage) <= tasks.keys():
        raise ValueError('unknown blocked task or lineage')
    inputs = {key: input_fingerprints(task) for key, task in tasks.items()}
    blocked = set(excluded_ids) | (set(tasks) & set(official))
    official_inputs = set().union(*(input_fingerprints(t) for t in official.values())) if official else set()
    blocked.update(key for key, values in inputs.items() if values & official_inputs)
    while True:
        shared_inputs = set().union(*(inputs[key] for key in blocked)) if blocked else set()
        shared_families = {lineage[key] for key in blocked if key in lineage}
        expanded = blocked | {key for key, values in inputs.items() if values & shared_inputs} | {
            key for key, family in lineage.items() if family in shared_families}
        if expanded == blocked:
            return blocked
        blocked = expanded


def audit_history(*, tasks, solutions, manifest, aggregate_bytes, checkpoints, blocked_ids,
                  expected_aggregate_sha256, expected_manifest_sha256, expected_challenge_sha256,
                  expected_prompt_sha256):
    if hashlib.sha256(aggregate_bytes).hexdigest() != expected_aggregate_sha256:
        raise ValueError('historical aggregate hash mismatch')
    if (manifest.get('sha256') != expected_manifest_sha256 or
        digest({k: v for k, v in manifest.items() if k != 'sha256'}) != expected_manifest_sha256 or
        manifest.get('protocol') != 'qwen38-quick-exploratory-v1' or
        manifest.get('model_id') != 'Qwen/Qwen3.8-27B-FP8' or
        manifest.get('model_revision') != 'c3a462cc3d38cf351f3b26cbd763469eeb74ef81' or
        manifest.get('precision') != 'fp8' or manifest.get('candidate_count') != 2 or
        manifest.get('challenge_sha256') != expected_challenge_sha256 or
        manifest.get('prompt_sha256') != expected_prompt_sha256):
        raise ValueError('historical manifest identity mismatch')
    aggregate = json.loads(aggregate_bytes)
    rows = aggregate['task_results']
    ids = manifest['phase_a_task_ids']
    if (aggregate.get('manifest') != manifest or aggregate.get('complete') is not True or
        len(ids) != len(set(ids)) or set(rows) != set(ids) or set(checkpoints) != set(ids) or
        not set(ids) <= tasks.keys() or not set(ids) <= solutions.keys() or not isinstance(blocked_ids, set)):
        raise ValueError('incomplete historical source inventory')
    counts = Counter(); eligible = []; seen_answers = set()
    for task_id in ids:
        row = rows[task_id]
        if (row != checkpoints[task_id] or row.get('status') != 'complete' or row.get('task_id') != task_id or
            row.get('manifest_sha256') != expected_manifest_sha256 or row.get('test_count') != len(tasks[task_id]['test']) or
            len(row.get('candidates', [])) != 2):
            raise ValueError('historical checkpoint mismatch')
        counts['parents'] += 1
        blocked = task_id in blocked_ids
        counts['blocked_parents'] += int(blocked)
        chosen = False; accepted = False
        for index, candidate in enumerate(row['candidates']):
            counts['candidates'] += 1
            raw = candidate.get('raw')
            if not isinstance(raw, str) or candidate.get('candidate_index') != index:
                raise ValueError('missing original candidate response')
            parsed, mode = parse_prediction(raw, row['test_count'])
            if parsed != candidate.get('outputs') or (parsed is not None) != candidate.get('valid') or mode != candidate.get('parse_mode'):
                raise ValueError('saved parse does not match original response')
            if blocked:
                counts['blocked_candidates'] += 1
                continue
            if parsed is None:
                counts['invalid_candidates'] += 1
            elif parsed != solutions[task_id]:
                counts['incorrect_candidates'] += 1
            else:
                counts['verified_correct_candidates'] += 1
                if chosen:
                    counts['duplicate_parent_candidates'] += 1
                else:
                    chosen = True
                    answer_lineage = digest(parsed)
                    if answer_lineage in seen_answers:
                        counts['duplicate_answer_lineage'] += 1
                        continue
                    seen_answers.add(answer_lineage)
                    accepted = True
                    eligible.append({'source': 'historical_fp8', 'task_id': task_id,
                                     'candidate_index': index, 'raw': raw,
                                     'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                                     'checkpoint_sha256': digest(row), 'manifest_sha256': expected_manifest_sha256})
        if accepted:
            counts['eligible_unique_parents'] += 1
    for key in ('parents', 'blocked_parents', 'candidates', 'blocked_candidates', 'invalid_candidates',
                'incorrect_candidates', 'verified_correct_candidates', 'duplicate_parent_candidates', 'duplicate_answer_lineage', 'eligible_unique_parents'):
        counts.setdefault(key, 0)
    return {'source': 'historical_fp8', 'counts': dict(counts), 'eligible': eligible,
            'limitations': ['FP8 cross-run distillation; not current 4-bit self-generation',
                            'Caller must bind and audit all blocked IDs and lineage before use.']}
