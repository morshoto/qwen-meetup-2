"""Offline exact-match verification of saved full-task generation attempts.

Callers must first authenticate the frozen manifest and replay the journal.
No held-out answer or raw response is returned in the summary.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION
from src.arc.qwen38_weight_generation import capture_response, create_request


def verify_attempts(manifest, challenges_bytes, solutions_bytes, rows):
    if (hashlib.sha256(challenges_bytes).hexdigest() != manifest['input_hashes']['challenges'] or
        hashlib.sha256(solutions_bytes).hexdigest() != manifest['input_hashes']['solutions']):
        raise ValueError('training dataset hash mismatch')
    tasks, solutions = json.loads(challenges_bytes), json.loads(solutions_bytes)
    ids = manifest['allocation']['fresh']; validation = manifest['allocation']['validation']
    model = {'id': MODEL_ID, 'revision': MODEL_REVISION, 'quantization': '4-bit MLX'}
    if (any(manifest['model'].get(k) != v for k, v in model.items()) or len(ids) != len(set(ids)) or set(ids) & set(validation) or
        not set(ids) <= tasks.keys() or not set(ids) <= solutions.keys()):
        raise ValueError('invalid registered model or allocation')
    max_tokens = manifest['generation']['max_tokens_per_candidate']
    seed = manifest['allocation']['seed']; seen = set(); outcomes = {}; counts = Counter()
    for row in rows:
        task_id, index = row.get('task_id'), row.get('candidate_index')
        if (task_id not in ids or type(index) is not int or not 0 <= index < 8 or (task_id, index) in seen or
            row.get('manifest_sha256') != manifest['sha256'] or row.get('model') != manifest['model'] or
            row.get('seed') != seed + 10000 + ids.index(task_id) * 8 + index):
            raise ValueError('attempt identity mismatch')
        seen.add((task_id, index)); counts['attempts'] += 1
        if row.get('status') in ('started', 'error'):
            counts[row['status']] += 1
            continue
        if row.get('status') != 'complete':
            raise ValueError('invalid attempt status')
        request = create_request(task_id, tasks[task_id], candidate_index=index, seed=row['seed'])
        reconstructed = capture_response(request, row, max_tokens)
        if any(row.get(key) != value for key, value in reconstructed.items()):
            raise ValueError('attempt raw response or prompt identity mismatch')
        correct = reconstructed['valid'] and reconstructed['outputs'] == solutions[task_id]
        outcomes[(task_id, index)] = bool(correct)
        counts['verified_correct' if correct else 'invalid' if not reconstructed['valid'] else 'incorrect'] += 1
    for key in ('attempts', 'started', 'error', 'verified_correct', 'invalid', 'incorrect'):
        counts.setdefault(key, 0)
    return {'outcomes': outcomes, 'counts': dict(counts)}
