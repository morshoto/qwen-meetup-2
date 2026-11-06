"""Check RSI chat rows and loss masks with supplied tokenizer components.

No model training or tokenizer loading is performed here. The caller must
independently bind the supplied components to the registered MLX runtime.
"""
import hashlib
import json

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_slot_audit import task_sha256


def _tokens(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
        raise ValueError('token sequence is invalid')
    return value


def audit_sft_format(plan, *, tasks, tokenizer, render):
    if (not isinstance(plan, dict) or plan.get('status') != 'quarantined_not_authorized' or
            plan.get('training_allowed') is not False or not isinstance(plan.get('generation_training'), dict) or
            not isinstance(tasks, dict) or not callable(render)):
        raise ValueError('invalid training preparation')
    training = plan['generation_training']
    rows = training.get('records')
    if (training.get('mask_prompt') is not True or training.get('max_seq_length') != 2048 or
            not isinstance(rows, list) or len(rows) < 5):
        raise ValueError('invalid prepared dataset or mask settings')
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    if hashlib.sha256(encoded).hexdigest() != plan.get('dataset_sha256'):
        raise ValueError('prepared dataset digest differs')
    checked, seen_tasks, seen_requests = [], set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'task_id', 'messages', 'source_sha256', 'request_id'}:
            raise ValueError('invalid prepared row')
        task_id, request_id, messages = row['task_id'], row['request_id'], row['messages']
        if (not isinstance(task_id, str) or task_id in seen_tasks or task_id not in tasks or
                not isinstance(tasks[task_id], Task) or tasks[task_id].id != task_id or
                not isinstance(request_id, str) or not request_id or request_id in seen_requests or
                not isinstance(plan.get('task_hashes'), dict) or
                plan['task_hashes'].get(task_id) != task_sha256(tasks[task_id]) or
                not isinstance(messages, list) or len(messages) != 3 or
                any(not isinstance(item, dict) or set(item) != {'role', 'content'} for item in messages) or
                [item['role'] for item in messages] != ['system', 'user', 'assistant'] or
                messages[0]['content'] != FINAL_ANSWER_SYSTEM_PROMPT or
                messages[1]['content'] != solve_prompt(tasks[task_id]) or
                not isinstance(messages[2]['content'], str) or not messages[2]['content'].strip() or
                hashlib.sha256(messages[2]['content'].strip().encode()).hexdigest() != row['source_sha256']):
            raise ValueError('prepared row identity or content differs')
        seen_tasks.add(task_id)
        seen_requests.add(request_id)
        checked.append({'messages': messages})
    if set(plan['task_hashes']) != seen_tasks:
        raise ValueError('task hash set differs from dataset')
    offsets = []
    for row in checked:
        token_ids, offset = render(row)
        token_ids = _tokens(token_ids)
        prefix = _tokens(tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                         add_generation_prompt=True, return_dict=False, enable_thinking=False))
        if (type(offset) is not int or offset != len(prefix) or token_ids[:offset] != prefix or
                offset < 1 or len(token_ids) <= offset or len(token_ids) > 2048):
            raise ValueError('training mask differs from inference prefix or sequence limit')
        offsets.append({'prefix_tokens': offset, 'completion_tokens': len(token_ids) - offset})
    jsonl = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in checked)
    return {'status': 'format_consistent_unverified', 'training_allowed': False, 'row_count': len(checked),
            'dataset_jsonl': jsonl, 'dataset_jsonl_sha256': hashlib.sha256(jsonl.encode()).hexdigest(),
            'token_spans': offsets, 'evidence_type': 'caller_supplied_tokenizer_and_renderer'}
