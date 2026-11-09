"""Pure, non-authorizing scheduling for the new full-task collection protocol."""
from __future__ import annotations

import math


def next_action(manifest, journal, verifications):
    """Replay a bounded journal and return the next action without dispatching a model.

    Callers must authenticate the registration, journal and offline verifier results.
    A started attempt is never retried until its process and outcome are reconciled.
    """
    ids = manifest['allocation']['fresh']
    settings = manifest['generation']; seed = manifest['allocation']['seed']
    initial = settings['initial_candidates']; extra = settings['additional_candidates']
    max_tokens = settings['max_tokens_per_candidate']; token_limit = settings['reserved_token_limit']
    time_limit = settings['wall_seconds_limit']
    if (not isinstance(ids, list) or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids) or
        type(seed) is not int or initial != 2 or extra != 6 or type(max_tokens) is not int or max_tokens <= 0 or
        type(token_limit) is not int or token_limit < 0 or not isinstance(time_limit, (int, float)) or
        not math.isfinite(time_limit) or time_limit <= 0 or not isinstance(verifications, dict)):
        raise ValueError('invalid schedule registration')
    seen = {}; elapsed = 0.0
    for record in journal:
        task_id, index, status = record.get('task_id'), record.get('candidate_index'), record.get('status')
        seconds = record.get('elapsed_seconds')
        if (task_id not in ids or type(index) is not int or not 0 <= index < initial + extra or
            status not in ('started', 'complete', 'error') or isinstance(seconds, bool) or
            not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0 or
            (task_id, index) in seen):
            raise ValueError('invalid collection journal')
        seen[(task_id, index)] = record
        elapsed += seconds
    for key, value in verifications.items():
        if key not in seen or seen[key]['status'] != 'complete' or type(value) is not bool:
            raise ValueError('invalid offline verification')

    def decide(prefix):
        if any(r['status'] == 'started' for r in prefix.values()):
            return {'status': 'requires_reconciliation'}
        for index in range(initial):
            for task_id in ids:
                if (task_id, index) not in prefix:
                    return {'status': 'ready', 'task_id': task_id, 'candidate_index': index}
        for index in range(initial, initial + extra):
            for task_id in ids:
                previous = [prefix.get((task_id, i)) for i in range(index)]
                for i, record in enumerate(previous):
                    if record is not None and record['status'] == 'complete' and (task_id, i) not in verifications:
                        return {'status': 'awaiting_verification'}
                if any(verifications.get((task_id, i)) is True for i in range(index)):
                    continue
                if (task_id, index) not in prefix:
                    return {'status': 'ready', 'task_id': task_id, 'candidate_index': index}
        # Even a final candidate's outcome must be independently accounted for.
        if any(r['status'] == 'complete' and key not in verifications for key, r in prefix.items()):
            return {'status': 'awaiting_verification'}
        return {'status': 'complete'}

    prefix = {}
    for record in journal:
        expected = decide(prefix)
        if expected['status'] != 'ready' or (record['task_id'], record['candidate_index']) != (expected['task_id'], expected['candidate_index']):
            raise ValueError('journal violates registered schedule')
        prefix[(record['task_id'], record['candidate_index'])] = record
    result = decide(prefix)
    if result['status'] != 'ready':
        return result
    if elapsed >= time_limit:
        return {'status': 'time_budget_exhausted'}
    if (len(journal) + 1) * max_tokens > token_limit:
        return {'status': 'token_budget_exhausted'}
    result['seed'] = seed + 10000 + ids.index(result['task_id']) * 8 + result['candidate_index']
    return result
