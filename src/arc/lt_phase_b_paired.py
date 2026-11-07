"""Offline paired scoring of recorded 14B adapter-off/on predictions.

This reuses the Phase A scorer. It does not attest the generation settings,
model, adapter or journal; callers must verify those before making comparisons.
"""
from src.arc.lt_scoring import score_comparison


def score_paired(task_ids, rows, verifier):
    if not isinstance(rows, list):
        raise ValueError('invalid result rows')
    aliases = {'off': '14b', 'on': '32b'}
    mapped, originals = [], {}
    for row in rows:
        if not isinstance(row, dict) or row.get('adapter_state') not in aliases or row.get('model') != '14b':
            raise ValueError('invalid adapter state or model')
        state = row['adapter_state']
        key = (row.get('task_id'), state)
        if key in originals:
            raise ValueError('duplicate state result')
        originals[key] = row
        mapped.append({**row, 'model': aliases[state]})
    scored = score_comparison(task_ids, mapped, verifier)
    indexed = {(r['task_id'], 'off' if r['model'] == '14b' else 'on'): r for r in scored['rows']}
    names = ('improved', 'regressed', 'unchanged_right', 'unchanged_wrong')
    paired = {metric: {'denominator': 0, **dict.fromkeys(names, 0)}
              for metric in ('pass_at_1', 'two_recorded_attempts')}
    result_rows = []
    for task_id in task_ids:
        off, on = indexed[task_id, 'off'], indexed[task_id, 'on']
        complete = off['status'] == on['status'] == 'complete'
        result = {'task_id': task_id, 'pair_status': 'complete' if complete else 'incomplete'}
        for state, current in (('off', off), ('on', on)):
            result[state] = {k: v for k, v in current.items() if k not in ('task_id', 'model')}
            original = originals.get((task_id, state))
            result[state]['candidates'] = original['candidates'] if original else None
        for metric, field in (('pass_at_1', 'pass_at_1'), ('two_recorded_attempts', 'pass_at_2')):
            if not complete:
                result[metric + '_change'] = None
                continue
            before, after = off[field], on[field]
            category = ('unchanged_right' if before else 'improved') if after else (
                'regressed' if before else 'unchanged_wrong')
            paired[metric]['denominator'] += 1
            paired[metric][category] += 1
            result[metric + '_change'] = category
        result_rows.append(result)
    identical = {state: sum(row['status'] == 'complete' and len(row['candidates']) == 2 and
                            row['candidates'][0] == row['candidates'][1]
                            for (task, s), row in originals.items() if s == state)
                 for state in aliases}
    return {'rows': result_rows, 'paired': paired,
            'states': {state: scored['models'][alias] for state, alias in aliases.items()},
            'identical_candidates': identical}
