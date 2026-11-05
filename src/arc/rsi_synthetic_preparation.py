"""Build a quarantined preparation from unverified synthetic generation rows."""
import hashlib
import json
import re

from src.arc.data import Task
from src.arc.rsi_slot_audit import task_sha256
from src.arc.rsi_slot_lifecycle import _create, _directory
from src.arc.rsi_synthetic_mapping import RepeatedSyntheticInput, map_synthetic_row
from src.arc.rsi_training_preparation import _BLOCKERS, _MODEL, _REVISION
from src.arc.rsi_transport_audit import _json_object


def prepare_synthetic_training(*, root, stage_bytes, expected_stage_sha256, tasks, seed):
    if (not isinstance(stage_bytes, bytes) or not 1 <= len(stage_bytes) <= 4 * 1024 * 1024 or
            not isinstance(expected_stage_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_stage_sha256) or
            hashlib.sha256(stage_bytes).hexdigest() != expected_stage_sha256 or
            not isinstance(tasks, dict) or type(seed) is not int or seed < 0):
        raise ValueError('invalid synthetic preparation inputs')
    stage = _json_object(stage_bytes)
    generation, parent = stage.get('generation'), stage.get('parent_adapter')
    registration = stage.get('registration_sha256')
    drafts, examples = stage.get('drafts'), stage.get('synthetic_examples')
    if (stage.get('status') != 'generation_drafts_quarantined' or stage.get('training_allowed') is not False or
            type(generation) is not int or generation not in (0, 1) or
            (generation == 0 and parent != 'none') or
            (generation == 1 and (not isinstance(parent, str) or not re.fullmatch(r'[0-9a-f]{64}', parent))) or
            not isinstance(registration, str) or not re.fullmatch(r'[0-9a-f]{64}', registration) or
            not isinstance(drafts, list) or not isinstance(examples, list) or len(drafts) != len(examples) or
            type(stage.get('draft_count')) is not int or stage['draft_count'] != len(drafts) or len(drafts) > 10):
        raise ValueError('invalid synthetic generation artifact')
    rows, rejections, synthetic_tasks = [], [], {}
    seen = {key: set() for key in ('task', 'request', 'source', 'ast', 'behavior')}
    for draft, generated in zip(drafts, examples):
        if not isinstance(draft, dict) or not isinstance(generated, dict):
            raise ValueError('invalid synthetic artifact row')
        try:
            task_id, provenance = draft['parent_task_id'], draft['provenance']
            task = tasks[task_id]
            identities = {'task': task_id, 'request': provenance['request_id'], 'source': draft['source_sha256'],
                          'ast': draft['ast_sha256'], 'behavior': draft['behavior_sha256']}
            if (not isinstance(task, Task) or task.id != task_id or draft.get('generation') != generation or
                    draft.get('parent_adapter') != parent or provenance.get('model_id') != _MODEL or
                    provenance.get('model_revision') != _REVISION):
                raise ValueError('synthetic candidate identity differs')
            for key, value in identities.items():
                if not isinstance(value, str) or not value or value in seen[key]:
                    raise ValueError('duplicate synthetic candidate identity')
                if key in ('source', 'ast', 'behavior') and not re.fullmatch(r'[0-9a-f]{64}', value):
                    raise ValueError('invalid synthetic fingerprint')
                seen[key].add(value)
            try:
                row = map_synthetic_row(task=task, draft=draft, generated=generated)
            except RepeatedSyntheticInput:
                rejections.append({'task_id': task_id, 'request_id': identities['request'], 'reason': 'repeated_input'})
                continue
            rows.append(row)
            pairs = tuple((item['input'], item['output']) for item in generated['examples'])
            synthetic_tasks[task_id] = Task(task_id, tuple(task.train[1:]) + pairs, (task.train[0][0],))
        except (KeyError, TypeError) as exc:
            raise ValueError('invalid synthetic candidate') from exc
    if stage.get('proposed_sft_rows') != rows or stage.get('mapping_rejections') != rejections:
        raise ValueError('synthetic mapping differs from generation artifact')
    if len(rows) < 5:
        return {'status': 'insufficient_mapped_records', 'training_allowed': False, 'row_count': len(rows)}
    records = [{key: row[key] for key in ('task_id', 'messages', 'source_sha256', 'request_id')} for row in rows]
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    if len(encoded) > 4 * 1024 * 1024:
        raise ValueError('synthetic preparation exceeds size limit')
    parameters = {'batch_size': 1, 'learning_rate': 0.0001, 'lora_rank': 8, 'lora_alpha': 16,
                  'max_seq_length': 2048, 'num_layers': 2, 'mask_prompt': True}
    blockers = list(_BLOCKERS) + ['synthetic_mapping_registration']
    plan = {'status': 'quarantined_not_authorized', 'training_allowed': False, 'stage_sha256': expected_stage_sha256,
            'registration_sha256': registration, 'generation': generation, 'parent_adapter': parent,
            'model_id': _MODEL, 'model_revision': _REVISION, 'seed': seed,
            'mapping': 'generated_demos_holdout_first_visible_v1',
            'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
            'task_hashes': {key: task_sha256(value) for key, value in synthetic_tasks.items()},
            'source_task_hashes': {key: task_sha256(tasks[key]) for key in synthetic_tasks},
            'preflight': {**parameters, 'max_steps': 1, 'timeout_seconds': 300,
                          'data_source': 'separate_synthetic_canary_not_registered'},
            'generation_training': {**parameters, 'max_steps': 20, 'timeout_seconds': 1800, 'records': records},
            'blockers': blockers}
    with _directory(root) as directory:
        _create(directory, 'synthetic.training.preparation.json', plan)
    return {'status': 'synthetic_preparation_quarantined', 'training_allowed': False,
            'row_count': len(rows), 'synthetic_tasks': synthetic_tasks, 'blockers': blockers}
