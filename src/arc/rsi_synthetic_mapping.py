"""Propose synthetic chat rows without authenticating or authorizing them."""
import hashlib
import json

from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.data import Task, validate_grid
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_slot_audit import task_sha256


def map_synthetic_row(*, task, draft, generated):
    if not isinstance(task, Task) or not task.train or not task.test_inputs:
        raise ValueError('invalid source task')
    if not isinstance(draft, dict) or not isinstance(generated, dict):
        raise ValueError('invalid draft or generated examples')
    try:
        messages, provenance = draft['messages'], draft['provenance']
        code = messages[2]['content']
        source_hash = hashlib.sha256(code.strip().encode()).hexdigest()
        if (draft.get('status') != 'pending_independent_audit' or draft.get('eligible_for_training') is not False or
                draft.get('source_type') != 'generator_distillation' or draft.get('parent_task_id') != task.id or
                len(messages) != 3 or [item['role'] for item in messages] != ['system', 'user', 'assistant'] or
                messages[0]['content'] != FINAL_ANSWER_SYSTEM_PROMPT or messages[1]['content'] != solve_prompt(task) or
                draft['source_sha256'] != source_hash or draft['ast_sha256'] != stable_ast_sha256(code) or
                provenance.get('execution_mode') != 'unverified_capture' or
                generated.get('status') != 'synthetic_examples_unverified' or
                generated.get('training_allowed') is not False or
                generated.get('label_origin') != 'model_generated_program' or
                generated.get('parent_task_id') != task.id or generated.get('parent_task_sha256') != task_sha256(task) or
                generated.get('source_sha256') != source_hash or
                generated.get('request_id') != provenance.get('request_id') or not provenance.get('request_id') or
                generated.get('slot_evidence_sha256') != provenance.get('slot_evidence_sha256') or
                not provenance.get('slot_evidence_sha256')):
            raise ValueError('synthetic source binding differs')
        examples = generated['examples']
        if not isinstance(examples, list) or len(examples) != len(task.test_inputs):
            raise ValueError('generated example count differs')
        pairs = []
        for item, input_grid in zip(examples, task.test_inputs):
            if not isinstance(item, dict) or set(item) != {'input', 'output'} or item['input'] != input_grid:
                raise ValueError('generated input differs')
            pairs.append((validate_grid(item['input']), validate_grid(item['output'])))
        for input_grid, output_grid in task.train:
            validate_grid(input_grid)
            validate_grid(output_grid)
        inputs = [item[0] for item in task.train] + [item[0] for item in pairs]
        fingerprints = [json.dumps(grid, separators=(',', ':')) for grid in inputs]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError('repeated inputs could expose held-out query')
        synthetic_task = Task(task.id, tuple(task.train[1:]) + tuple(pairs), (task.train[0][0],))
    except (KeyError, TypeError, IndexError, AttributeError, SyntaxError, RecursionError) as exc:
        raise ValueError('invalid synthetic row inputs') from exc
    return {'mapping': 'generated_demos_holdout_first_visible_v1', 'training_allowed': False,
            'task_id': task.id, 'source_sha256': source_hash, 'request_id': provenance['request_id'],
            'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                         {'role': 'user', 'content': solve_prompt(synthetic_task)},
                         {'role': 'assistant', 'content': code}]}
