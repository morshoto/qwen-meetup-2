"""Construct provisional SFT examples from independently replayed raw outputs.

This pure transformation does not authenticate records or authorize training.
The caller must replay and bind every source journal and exclusion beforehand.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_local_runner import clean_task, parse_prediction
from src.arc.qwen38_rsi_comparison import digest


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def prepare_examples(tasks, solutions, records, *, blocked_ids, forbidden_answer_fingerprints, token_counter, max_tokens=2048):
    if not isinstance(blocked_ids, set) or not isinstance(forbidden_answer_fingerprints, set) or not callable(token_counter) or max_tokens < 1:
        raise ValueError('invalid preparation constraints')
    counts = Counter(); examples = []; parents = set(); answers = set()
    for record in records:
        counts['records'] += 1
        task_id = record.get('task_id')
        if task_id in blocked_ids:
            counts['excluded_blocked'] += 1
            continue
        if task_id not in tasks or task_id not in solutions:
            raise ValueError('unknown or unverifiable parent')
        raw = record.get('raw')
        if (not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != record.get('raw_sha256') or
            record.get('source') not in ('historical_fp8', 'current_gen0_self_generated') or
            not _hash(record.get('checkpoint_sha256')) or not _hash(record.get('manifest_sha256')) or
            type(record.get('candidate_index')) is not int or record['candidate_index'] < 0):
            raise ValueError('unverifiable original response or provenance')
        task = clean_task(tasks[task_id])
        parsed, _ = parse_prediction(raw, len(task['test']))
        if parsed is None or parsed != solutions[task_id]:
            raise ValueError('response does not independently match permitted solution')
        if task_id in parents:
            counts['duplicate_parent'] += 1
            continue
        answer_hash = digest(parsed)
        if answer_hash in forbidden_answer_fingerprints:
            counts['protected_answer_lineage'] += 1
            continue
        if answer_hash in answers:
            counts['duplicate_answer_lineage'] += 1
            continue
        prompt = json.dumps({'instruction': PROMPT, 'task': task}, sort_keys=True, separators=(',', ':'))
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': raw}]
        length = token_counter(messages)
        if type(length) is not int or length < 0:
            raise ValueError('invalid token count')
        if length > max_tokens:
            counts['length_excluded'] += 1
            continue
        parents.add(task_id); answers.add(answer_hash)
        counts[record['source']] += 1
        examples.append({'task_id': task_id, 'source': record['source'], 'messages': messages,
                         'raw_sha256': record['raw_sha256'], 'manifest_sha256': record['manifest_sha256'],
                         'checkpoint_sha256': record['checkpoint_sha256'], 'candidate_index': record['candidate_index'],
                         'token_count': length})
    for key in ('records', 'excluded_blocked', 'duplicate_parent', 'duplicate_answer_lineage',
                'length_excluded', 'protected_answer_lineage', 'historical_fp8', 'current_gen0_self_generated'):
        counts.setdefault(key, 0)
    counts['unique_parents'] = len(parents)
    return {'status': 'requires_source_replay' if len(parents) >= 5 else 'insufficient_verified_data',
            'counts': dict(counts), 'examples': examples}


def select_training_examples(tasks, solutions, fresh_records, historical_records, *, blocked_ids,
                             forbidden_answer_fingerprints, token_counter, max_tokens=2048):
    """Apply the exploratory five-parent gate without authorizing training.

    Inputs must be independently replayed against their frozen sources by the
    caller. The return value remains provisional until a source-bound receipt.
    """
    if (any(record.get('source') != 'current_gen0_self_generated' for record in fresh_records) or
        any(record.get('source') != 'historical_fp8' for record in historical_records)):
        raise ValueError('sample source label mismatch')
    options = {'blocked_ids': blocked_ids, 'forbidden_answer_fingerprints': forbidden_answer_fingerprints,
               'token_counter': token_counter, 'max_tokens': max_tokens}
    fresh = prepare_examples(tasks, solutions, fresh_records, **options)
    selected = fresh if fresh['counts']['unique_parents'] >= 5 else prepare_examples(
        tasks, solutions, list(fresh_records) + list(historical_records), **options)
    counts = selected['counts']
    if counts['historical_fp8'] and counts['current_gen0_self_generated']:
        mode = 'mixed_source'
    elif counts['historical_fp8']:
        mode = 'historical_fp8_distillation'
    elif counts['current_gen0_self_generated']:
        mode = 'fresh_only'
    else:
        mode = 'none'
    return {'status': 'eligible_pending_source_replay' if counts['unique_parents'] >= 5 else 'insufficient_verified_data',
            'mode': mode, 'fresh_only_counts': fresh['counts'], 'counts': counts, 'examples': selected['examples']}
