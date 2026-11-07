"""Offline preparation of verified-prediction chat examples.

All returned data is private: verified model predictions can reproduce protected
answers. Source fields are identities supplied by the caller, not attestation.
"""
import hashlib
import json
import re

from src.arc.data import Task
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_phase_b_eligibility import assess_candidates

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def build_dataset(tasks, rows, verifier, *, source, comparison_ids):
    if (not isinstance(tasks, dict) or not tasks or
            any(not isinstance(k, str) or not k or not isinstance(v, Task) or v.id != k
                for k, v in tasks.items()) or not isinstance(rows, list) or
            not isinstance(comparison_ids, set) or not comparison_ids or
            any(not isinstance(x, str) or not x for x in comparison_ids) or
            set(tasks) & comparison_ids):
        raise ValueError('invalid or overlapping task sets')
    if (not isinstance(source, dict) or source.get('model_id') != _MODEL or
            source.get('model_revision') != _REVISION or
            not isinstance(source.get('source_code_sha256'), str) or
            not re.fullmatch(r'[0-9a-f]{64}', source['source_code_sha256']) or
            not isinstance(source.get('run_id'), str) or not source['run_id']):
        raise ValueError('invalid source identity')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('task_id') not in tasks or row['task_id'] in indexed:
            raise ValueError('unknown or duplicate task result')
        if not isinstance(row.get('calls'), list) or len(row['calls']) != 2 or any(
                not isinstance(call, dict) for call in row['calls']):
            raise ValueError('invalid call records')
        indexed[row['task_id']] = row
    journal, eligible = [], {}
    for task_id in sorted(indexed):
        row = indexed[task_id]
        prompt = _json({'instruction': PROMPT, 'task': tasks[task_id].prompt_data()})
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
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
                'run_id': source['run_id'], 'usage': call.get('usage'), 'error': call.get('error')})
            if record['verification_status'] == 'eligible':
                eligible[task_id] = {'messages': [{'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': _json(record['prediction'])}]}
    ids = sorted(eligible)
    if len(indexed) != len(tasks):
        status, train, valid = 'collection_incomplete', [], []
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
            'eligible_task_count': len(ids), 'split_task_ids': split_ids,
            'train': train, 'valid': valid, 'journal': journal}
