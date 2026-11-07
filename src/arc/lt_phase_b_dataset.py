"""Offline preparation of verified-prediction chat examples.

All returned data is private: verified model predictions can reproduce protected
answers. Source fields are identities supplied by the caller, not attestation.
"""
import hashlib
import json
import re

from src.arc.data import Task, validate_grid
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_phase_b_eligibility import assess_candidates

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _task_content(task):
    try:
        if not task.train or not task.test_inputs:
            raise ValueError('empty task')
        train = sorted(_json({'input': validate_grid(i), 'output': validate_grid(o)}) for i, o in task.train)
        test = sorted(_json(validate_grid(i)) for i in task.test_inputs)
        return _json({'train': train, 'test': test})
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid visible task content') from exc


def _check_task_sets(training, comparison):
    groups = (training, comparison)
    if (any(not isinstance(group, dict) or not group or
            any(not isinstance(k, str) or not k or not isinstance(v, Task) or v.id != k
                for k, v in group.items()) for group in groups) or set(training) & set(comparison)):
        raise ValueError('invalid or overlapping task sets')
    content = [set() for _ in groups]
    for index, group in enumerate(groups):
        for task in group.values():
            item = _task_content(task)
            if item in content[index]:
                raise ValueError('repeated visible task content')
            content[index].add(item)
    if content[0] & content[1]:
        raise ValueError('visible task content overlaps')


def build_dataset(tasks, rows, verifier, *, source, comparison_tasks):
    _check_task_sets(tasks, comparison_tasks)
    if not isinstance(rows, list):
        raise ValueError('invalid result rows')
    if (not isinstance(source, dict) or source.get('model_id') != _MODEL or
            source.get('model_revision') != _REVISION or
            not isinstance(source.get('source_code_sha256'), str) or
            not re.fullmatch(r'[0-9a-f]{64}', source['source_code_sha256']) or
            not isinstance(source.get('run_sha256'), str) or
            not re.fullmatch(r'[0-9a-f]{64}', source['run_sha256']) or
            not isinstance(source.get('run_id'), str) or not source['run_id']):
        raise ValueError('invalid source identity')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('task_id') not in tasks or row['task_id'] in indexed:
            raise ValueError('unknown or duplicate task result')
        calls = row.get('calls')
        if row.get('status') == 'complete':
            if not isinstance(calls, list) or len(calls) != 2 or any(not isinstance(call, dict) for call in calls):
                raise ValueError('invalid call records')
        elif (row.get('status') not in ('timeout', 'error') or row.get('model') != '14b' or
              row.get('candidates') != [] or calls != [] or not isinstance(row.get('cleanup'), dict) or
              row['cleanup'].get('leader_reaped') is not True or
              row['cleanup'].get('group_absent') is not True):
            raise ValueError('invalid failed worker record')
        indexed[row['task_id']] = row
    journal, eligible, failed = [], {}, 0
    for task_id in sorted(indexed):
        row = indexed[task_id]
        prompt = _json({'instruction': PROMPT, 'task': tasks[task_id].prompt_data()})
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        task_sha = hashlib.sha256(_json(tasks[task_id].prompt_data()).encode()).hexdigest()
        if row['status'] == 'complete' and (row.get('run_sha256') != source['run_sha256'] or
                row.get('source_code_sha256') != source['source_code_sha256'] or
                row.get('public_task_sha256') != task_sha or row.get('prompt_sha256') != prompt_sha or
                row.get('adapter_state') != 'off'):
            raise ValueError('successful worker identity differs')
        if row['status'] != 'complete':
            failed += 1
            journal.append({'task_id': task_id, 'candidate_index': None, 'prediction': None,
                'verification_status': 'worker_' + row['status'], 'task_exact_match': None,
                'solution_lineage_sha256': hashlib.sha256(_json({'task_id': task_id}).encode()).hexdigest(),
                'source_model_id': _MODEL, 'source_model_revision': _REVISION, 'source_generation': 0,
                'prompt_sha256': prompt_sha, 'source_split': 'phase_b_training_pool',
                'source_code_sha256': source['source_code_sha256'], 'run_id': source['run_id'],
                'usage': None, 'error': row['status'], 'worker_diagnostics': row.get('worker_diagnostics')})
            continue
        for record in assess_candidates(row, verifier):
            call = row['calls'][record['candidate_index']]
            if call.get('candidate') != record['candidate_index']:
                raise ValueError('call index differs')
            if record['verification_status'] in ('eligible', 'duplicate_task', 'incorrect'):
                try:
                    raw_prediction = normalize_direct(call['raw'], len(tasks[task_id].test_inputs))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError('invalid recorded response') from exc
                if call.get('error') is not None or raw_prediction != record['prediction']:
                    raise ValueError('prediction differs from successful response')
            journal.append({**record, 'source_model_id': _MODEL, 'source_model_revision': _REVISION,
                'source_generation': 0, 'prompt_sha256': prompt_sha,
                'source_split': 'phase_b_training_pool', 'source_code_sha256': source['source_code_sha256'],
                'run_id': source['run_id'], 'run_sha256': source['run_sha256'], 'public_task_sha256': task_sha,
                'usage': call.get('usage'), 'error': call.get('error')})
            if record['verification_status'] == 'eligible':
                eligible[task_id] = {'messages': [{'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': _json(record['prediction'])}]}
    ids = sorted(eligible)
    if len(indexed) != len(tasks):
        status, train, valid = 'collection_incomplete', [], []
    elif failed:
        status, train, valid = 'collection_failed', [], []
    elif not ids:
        status, train, valid = 'insufficient_verified_self_solutions', [], []
    elif len(ids) < 3:
        status, train, valid = 'overfitting_only', [eligible[x] for x in ids], []
    else:
        nvalid = max(1, len(ids) // 5)
        status, train, valid = 'disjoint_split', [eligible[x] for x in ids[nvalid:]], [eligible[x] for x in ids[:nvalid]]
    split_ids = {'train': [], 'valid': []}
    if status == 'overfitting_only':
        split_ids['train'] = ids
    elif status == 'disjoint_split':
        split_ids = {'train': ids[nvalid:], 'valid': ids[:nvalid]}
    return {'yield_status': status, 'selected_task_count': len(tasks), 'attempted_task_count': len(indexed),
            'eligible_task_count': len(ids), 'failed_task_count': failed, 'split_task_ids': split_ids,
            'train': train, 'valid': valid, 'journal': journal}
