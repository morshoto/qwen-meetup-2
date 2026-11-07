"""Audit verified chat examples against native trainer tokenization.

Caller must separately verify journal provenance, task allocation, runtime and
canary gates. This check does not authorize training or expose answer content.
"""
import hashlib
import json
import re

from src.arc.data import Task, validate_grid
from src.arc.lt_live_backend import PROMPT


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def audit_sft_dataset(*, dataset, tasks, tokenizer, dataset_factory):
    if (not isinstance(dataset, dict) or not isinstance(tasks, dict) or not tasks or
            any(not isinstance(k, str) or not isinstance(v, Task) or v.id != k for k, v in tasks.items()) or
            not callable(getattr(tokenizer, 'apply_chat_template', None)) or not callable(dataset_factory) or
            dataset.get('yield_status') not in ('disjoint_split', 'overfitting_only') or
            dataset.get('selected_task_count') != len(tasks) or dataset.get('attempted_task_count') != len(tasks) or
            dataset.get('failed_task_count') != 0 or not isinstance(dataset.get('journal'), list)):
        raise ValueError('dataset collection is incomplete or invalid')
    eligible = {}
    for record in dataset['journal']:
        if not isinstance(record, dict):
            raise ValueError('invalid eligibility record')
        if record.get('verification_status') != 'eligible':
            continue
        task_id, lineage = record.get('task_id'), record.get('solution_lineage_sha256')
        if (not isinstance(task_id, str) or task_id not in tasks or task_id in eligible or
                record.get('task_exact_match') is not True or type(record.get('candidate_index')) is not int or
                record['candidate_index'] not in (0, 1) or not isinstance(lineage, str) or
                not re.fullmatch(r'[0-9a-f]{64}', lineage) or lineage != _sha(_json({'task_id': task_id}))):
            raise ValueError('invalid eligible task lineage')
        eligible[task_id] = record
    ids = sorted(eligible)
    if dataset.get('eligible_task_count') != len(ids) or not ids:
        raise ValueError('eligible task count differs')
    if len(ids) < 3:
        if dataset['yield_status'] != 'overfitting_only':
            raise ValueError('small yield must be overfitting only')
        split = {'train': ids, 'valid': []}
    else:
        if dataset['yield_status'] != 'disjoint_split':
            raise ValueError('validation split required')
        nvalid = max(1, len(ids) // 5)
        split = {'train': ids[nvalid:], 'valid': ids[:nvalid]}
    if dataset.get('split_task_ids') != split:
        raise ValueError('task split differs')
    lineages = [eligible[task]['solution_lineage_sha256'] for task in ids]
    if len(set(lineages)) != len(lineages):
        raise ValueError('solution lineages overlap')
    hashes, max_tokens = {}, 0
    for name in ('train', 'valid'):
        examples = dataset.get(name)
        if not isinstance(examples, list) or len(examples) != len(split[name]):
            raise ValueError('dataset rows differ from split')
        for task_id, example in zip(split[name], examples):
            record = eligible[task_id]
            prediction = record.get('prediction')
            if not isinstance(prediction, list) or len(prediction) != len(tasks[task_id].test_inputs):
                raise ValueError('prediction count differs')
            for grid in prediction:
                validate_grid(grid)
            prompt = _json({'instruction': PROMPT, 'task': tasks[task_id].prompt_data()})
            expected = {'messages': [{'role': 'user', 'content': prompt},
                                     {'role': 'assistant', 'content': _json(prediction)}]}
            if example != expected or record.get('prompt_sha256') != _sha(prompt):
                raise ValueError('example differs from verified record')
            messages = example['messages']
            native = dataset_factory(example, tokenizer)
            tokens, offset = native.process(example)
            prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True,
                enable_thinking=False, return_dict=False)
            if (not isinstance(tokens, list) or not isinstance(prefix, list) or
                    any(type(x) is not int or x < 0 for x in tokens + prefix) or
                    type(offset) is not int or offset != len(prefix) or not 0 < offset < len(tokens) or
                    tokens[:offset] != prefix or len(tokens) > 512):
                raise ValueError('trainer prefix, mask or sequence limit differs')
            max_tokens = max(max_tokens, len(tokens))
        payload = ''.join(_json(example) + '\n' for example in examples)
        hashes[name] = _sha(payload)
    return {'status': 'dataset_tokens_consistent_unverified', 'training_allowed': False,
            'train_count': len(split['train']), 'valid_count': len(split['valid']), 'maximum_tokens': max_tokens,
            'train_jsonl_sha256': hashes['train'], 'valid_jsonl_sha256': hashes['valid'],
            'evidence_type': 'caller_supplied_verified_records_and_tokenizer'}
