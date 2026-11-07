"""Offline paired scoring of recorded 14B adapter-off/on predictions.

This reuses the Phase A scorer. It does not attest the generation settings,
model, adapter or journal; callers must verify those before making comparisons.
"""
import math
import re

from src.arc.lt_scoring import score_comparison


def score_paired(task_ids, rows, verifier, *, registration):
    valid_hash = lambda value: isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value))
    if (not isinstance(task_ids, list) or not task_ids or
            any(not isinstance(x, str) or not x for x in task_ids) or len(set(task_ids)) != len(task_ids) or
            not isinstance(rows, list) or not isinstance(registration, dict) or
            set(registration) != {'run_sha256', 'source_code_sha256', 'adapter_sha256',
                                  'selected_task_sha256', 'selected_prompt_sha256'} or
            any(not valid_hash(registration[key]) for key in ('run_sha256', 'source_code_sha256', 'adapter_sha256')) or
            not isinstance(registration['selected_task_sha256'], dict) or
            set(registration['selected_task_sha256']) != set(task_ids) or
            any(not valid_hash(x) for x in registration['selected_task_sha256'].values()) or
            not isinstance(registration['selected_prompt_sha256'], dict) or
            set(registration['selected_prompt_sha256']) != set(task_ids) or
            any(not valid_hash(x) for x in registration['selected_prompt_sha256'].values())):
        raise ValueError('invalid comparison registration')
    aliases = {'off': '14b', 'on': '32b'}
    mapped, originals = [], {}
    for row in rows:
        if not isinstance(row, dict) or row.get('adapter_state') not in aliases or row.get('model') != '14b':
            raise ValueError('invalid adapter state or model')
        state = row['adapter_state']
        task_id = row.get('task_id')
        if not isinstance(task_id, str) or task_id not in registration['selected_task_sha256']:
            raise ValueError('unregistered task')
        if row.get('status') == 'complete':
            if (row.get('run_sha256') != registration['run_sha256'] or
                    row.get('source_code_sha256') != registration['source_code_sha256'] or
                    row.get('public_task_sha256') != registration['selected_task_sha256'][task_id] or
                    row.get('prompt_sha256') != registration['selected_prompt_sha256'][task_id] or
                    'adapter_sha256' not in row or
                    row['adapter_sha256'] != (registration['adapter_sha256'] if state == 'on' else None)):
                raise ValueError('comparison row identity differs')
        elif row.get('candidates') != []:
            raise ValueError('failed row may not contribute candidates')
        key = (row.get('task_id'), state)
        if key in originals:
            raise ValueError('duplicate state result')
        originals[key] = row
        mapped.append({**row, 'model': aliases[state]})
    for task_id in task_ids:
        off, on = originals.get((task_id, 'off')), originals.get((task_id, 'on'))
        if (off is not None and on is not None and off.get('status') == on.get('status') == 'complete' and
                off['prompt_sha256'] != on['prompt_sha256']):
            raise ValueError('paired prompts differ')
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
    uncertainty = {}
    for metric, counts in paired.items():
        n = counts['denominator']
        difference = (counts['improved'] - counts['regressed']) / n if n else None
        radius = math.sqrt(2 * math.log(40) / n) if n else None
        interval = [max(-1., difference - radius), min(1., difference + radius)] if n else None
        uncertainty[metric] = {'difference': difference, 'interval_95': interval,
            'method': 'Hoeffding; assumes independent paired task observations'}
    return {'rows': result_rows, 'paired': paired, 'uncertainty': uncertainty,
            'states': {state: scored['models'][alias] for state, alias in aliases.items()},
            'identical_candidates': identical}
