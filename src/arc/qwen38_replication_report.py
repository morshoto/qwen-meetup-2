"""Offline reporting for a separately registered exploratory replication."""
import hashlib
import math

from src.arc.qwen38_replication_statistics import paired_sign_test
from src.arc.qwen38_replication_schedule import next_replication_action
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_paired import paired_report


def score_replication(descriptor, baseline_rows, adapter_rows, verifier, *, journal=None, termination=None):
    """Score authenticated rows; callers must replay source and journal evidence."""
    if (descriptor.get('status') != 'evaluation_ready' or
        descriptor.get('sha256') != digest({k: v for k, v in descriptor.items() if k != 'sha256'})):
        raise ValueError('replication registration is not ready or changed')
    pairs = paired_report(descriptor['task_ids'], baseline_rows, adapter_rows, verifier,
                          manifest_sha256=descriptor['sha256'], prompt_sha256=descriptor['prompt_sha256'],
                          adapter_sha256=descriptor['adapter_sha256'])
    pairs['comparison'] = 'new exploratory replication tasks'
    counts = pairs['paired']
    terminated = False
    if (journal is None) != (termination is None):
        raise ValueError('journal and termination must be supplied together')
    if termination is not None:
        schedule = next_replication_action(descriptor, journal)
        wall = termination.get('wall_seconds')
        wall_stop = (termination.get('status') == 'time_budget_exhausted' and schedule['status'] == 'ready' and
                     not isinstance(wall, bool) and isinstance(wall, (int, float)) and math.isfinite(wall) and
                     wall >= descriptor['wall_seconds_limit'])
        if (termination.get('sha256') != digest({k: v for k, v in termination.items() if k != 'sha256'}) or
            termination.get('registration_sha256') != descriptor['sha256'] or
            termination.get('journal_sha256') != digest(journal) or
            (termination.get('status') != schedule['status'] and not wall_stop) or
            (schedule['status'] in ('ready', 'requires_reconciliation') and not wall_stop)):
            raise ValueError('replication termination evidence mismatch')
        recorded = {(r['arm'], r['task_id'], r['candidate_index']): r for r in journal}
        for arm, rows in (('base', baseline_rows), ('adapter', adapter_rows)):
            for row in rows:
                if row.get('status') != 'complete':
                    continue
                candidates = row.get('candidates', [])
                if len(candidates) != 2:
                    continue
                attempts = [recorded.get((arm, row['task_id'], i)) for i in (0, 1)]
                if (any(not attempt or attempt.get('status') != 'complete' for attempt in attempts) or
                    any(attempt.get('raw_sha256') != hashlib.sha256(candidate['raw'].encode()).hexdigest()
                        for attempt, candidate in zip(attempts, candidates)) or
                    any(not isinstance(attempt.get('seconds'), (int, float)) or isinstance(attempt.get('seconds'), bool)
                        for attempt in attempts) or
                    row.get('runtime_seconds') != sum(attempt['seconds'] for attempt in attempts)):
                    raise ValueError('replication rows differ from journal')
        terminated = termination['status'] == 'complete'
    complete = terminated and counts['complete'] == len(descriptor['task_ids'])
    result = {'status': 'complete' if complete else 'incomplete', 'paired': pairs,
              'uncertainty': paired_sign_test(counts['gained'], counts['lost']) if counts['complete'] else None,
              'uncertainty_scope': 'complete paired tasks only',
              'limitations': ['Exploratory tasks; unknown semantic and pretraining exposure.',
                              'Recorded candidates are not assumed independent.',
                              'Mixed-source distillation is not demonstrated recursive improvement.']}
    result['sha256'] = digest(result)
    return result
