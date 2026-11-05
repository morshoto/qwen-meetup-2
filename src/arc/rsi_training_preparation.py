"""Prepare quarantined training inputs; never authorize or invoke training."""
import hashlib
import json
import re

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_batch_precheck import precheck_batch
from src.arc.rsi_slot_audit import task_sha256
from src.arc.rsi_slot_lifecycle import _create, _directory

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
_BLOCKERS = ['execution_registration', 'independent_batch_audit', 'preflight_canary_registration',
             'trainer_data_format_audit', 'process_containment_proof', 'training_authorization']


def prepare_training_inputs(*, root, drafts, tasks, generation, parent_adapter, registration_sha256, seed):
    if (not isinstance(registration_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', registration_sha256)
            or type(seed) is not int or seed < 0 or not isinstance(tasks, dict) or not tasks
            or any(not isinstance(key, str) or not isinstance(task, Task) or task.id != key
                   for key, task in tasks.items())):
        raise ValueError('invalid training preparation identity')
    precheck = precheck_batch(drafts, generation=generation, parent_adapter=parent_adapter,
                              allowed_tasks=set(tasks))
    for draft in drafts:
        messages = draft['messages']
        if (messages[0]['content'] != FINAL_ANSWER_SYSTEM_PROMPT or
                messages[1]['content'] != solve_prompt(tasks[draft['parent_task_id']]) or
                len(messages[2]['content'].encode('utf-8')) > 65536):
            raise ValueError('draft messages differ from canonical task or model prompt')
    if precheck['status'] == 'insufficient_records':
        return {'status': 'insufficient_records', 'training_allowed': False,
                'structurally_valid_count': precheck['structurally_valid_count']}
    rows = [{'task_id': item['parent_task_id'], 'messages': item['messages'],
             'source_sha256': item['source_sha256'], 'request_id': item['provenance']['request_id']}
            for item in drafts]
    dataset_bytes = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    if len(dataset_bytes) > 4 * 1024 * 1024:
        raise ValueError('training preparation exceeds size limit')
    parameters = {'batch_size': 1, 'learning_rate': 0.0001, 'lora_rank': 8, 'lora_alpha': 16,
                  'max_seq_length': 2048, 'num_layers': 2, 'mask_prompt': True}
    plan = {'status': 'quarantined_not_authorized', 'training_allowed': False,
            'registration_sha256': registration_sha256, 'generation': generation,
            'parent_adapter': parent_adapter, 'model_id': _MODEL, 'model_revision': _REVISION,
            'seed': seed, 'batch_sha256': precheck['draft_batch_sha256'],
            'dataset_sha256': hashlib.sha256(dataset_bytes).hexdigest(),
            'task_hashes': {item['parent_task_id']: task_sha256(tasks[item['parent_task_id']]) for item in drafts},
            'preflight': {**parameters, 'max_steps': 1, 'timeout_seconds': 300,
                          'data_source': 'separate_synthetic_canary_not_registered'},
            'generation_training': {**parameters, 'max_steps': 20, 'timeout_seconds': 1800,
                                    'records': rows},
            'blockers': _BLOCKERS}
    with _directory(root) as directory:
        _create(directory, 'training.preparation.json', plan)
    return {'status': 'prepared_quarantined', 'training_allowed': False, 'blockers': list(_BLOCKERS),
            'batch_sha256': precheck['draft_batch_sha256'], 'dataset_sha256': plan['dataset_sha256']}
