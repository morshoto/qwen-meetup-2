"""Replay offline verification before deriving provisional fresh SFT records."""
from __future__ import annotations

import math

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_schedule import next_action
from src.arc.qwen38_weight_verification_receipt import verify_receipt


def extract_fresh_records(manifest, challenges, solutions, rows, receipt, *, wall_elapsed):
    """Derive records only after a terminal, fully replayed collection.

    The caller must authenticate the run-start timestamp, manifest and worker
    artifacts. No record returned here alone authorizes training.
    """
    if isinstance(wall_elapsed, bool) or not isinstance(wall_elapsed, (int, float)) or not math.isfinite(wall_elapsed) or wall_elapsed < 0:
        raise ValueError('invalid collection wall time')
    outcomes = verify_receipt(receipt, manifest, challenges, solutions, rows)
    action = next_action(manifest, rows, outcomes)
    status = action['status']
    if status == 'ready' and wall_elapsed >= manifest['generation']['wall_seconds_limit']:
        status = 'time_budget_exhausted'
    if status not in ('complete', 'time_budget_exhausted', 'token_budget_exhausted'):
        raise ValueError('collection is incomplete or requires reconciliation')
    records = []
    for row in rows:
        key = (row['task_id'], row['candidate_index'])
        if outcomes.get(key) is True:
            records.append({'source': 'current_gen0_self_generated', 'task_id': row['task_id'],
                            'candidate_index': row['candidate_index'], 'seed': row['seed'],
                            'raw': row['raw'], 'raw_sha256': row['raw_sha256'],
                            'manifest_sha256': manifest['sha256'], 'checkpoint_sha256': digest(row)})
    return {'collection_status': status, 'journal_sha256': digest(rows),
            'verification_sha256': receipt['sha256'], 'counts': receipt['counts'], 'records': records}
