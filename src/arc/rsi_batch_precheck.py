"""Structural precheck for quarantined drafts; it never authorizes training."""
import hashlib
import json
import re

from src.arc.ast_fingerprint import stable_ast_sha256

from src.arc.rsi_model_identity import _MODELS

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
_HASH = re.compile(r'^[0-9a-f]{64}$')
_FORBIDDEN = {'_verifier_only', 'hidden_labels', 'private_probes', 'query_outputs'}


def _has_private_payload(value) -> bool:
    if isinstance(value, dict):
        return bool(_FORBIDDEN & value.keys()) or any(_has_private_payload(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_private_payload(v) for v in value)
    return False


def precheck_batch(drafts: list[dict], *, generation: int, parent_adapter: str,
                   allowed_tasks: set[str], model_key: str = '14b') -> dict:
    """Reject structural conflicts; all valid drafts still require independent audit."""
    if not isinstance(model_key, str) or model_key not in _MODELS:
        raise ValueError('unsupported model identity')
    model, revision = _MODELS[model_key]
    if (not isinstance(drafts, list) or type(generation) is not int or generation not in (0, 1) or
            (generation == 0 and parent_adapter != 'none') or
            (generation == 1 and (not isinstance(parent_adapter, str) or not _HASH.fullmatch(parent_adapter))) or
            not isinstance(allowed_tasks, set) or not allowed_tasks or
            any(not isinstance(task, str) or not task for task in allowed_tasks)):
        raise ValueError('invalid batch registration')
    seen = {key: set() for key in ('task', 'ast', 'behavior', 'request')}
    probe_set = None
    for draft in drafts:
        if not isinstance(draft, dict) or _has_private_payload(draft):
            raise ValueError('invalid draft or private verifier payload')
        task = draft.get('parent_task_id')
        provenance = draft.get('provenance')
        messages = draft.get('messages')
        if (draft.get('status') != 'pending_independent_audit' or
                draft.get('eligible_for_training') is not False or
                draft.get('source_type') != 'generator_distillation' or
                type(draft.get('generation')) is not int or draft['generation'] != generation or
                draft.get('parent_adapter') != parent_adapter or
                not isinstance(task, str) or task not in allowed_tasks or
                draft.get('mask_prompt') is not True or draft.get('loss_target') != 'assistant_completion_only' or
                not isinstance(provenance, dict) or provenance.get('source_type') != 'generator_distillation' or
                provenance.get('model_id') != model or provenance.get('model_revision') != revision or
                provenance.get('execution_mode') != 'live_model' or
                not isinstance(provenance.get('request_id'), str) or not provenance['request_id'] or
                not isinstance(messages, list) or len(messages) != 3 or
                any(not isinstance(message, dict) or not isinstance(message.get('content'), str) or
                    not message['content'] for message in messages) or
                [message.get('role') for message in messages] != ['system', 'user', 'assistant']):
            raise ValueError('draft identity, source, or message contract mismatch')
        for key in ('source_sha256', 'ast_sha256', 'behavior_sha256', 'probe_set_sha256', 'reference_pool_sha256'):
            if not isinstance(draft.get(key), str) or not _HASH.fullmatch(draft[key]):
                raise ValueError('invalid draft fingerprint')
        if (not isinstance(provenance.get('raw_response_sha256'), str) or
                not _HASH.fullmatch(provenance['raw_response_sha256'])):
            raise ValueError('invalid response fingerprint')
        code = messages[-1]['content']
        try:
            if (hashlib.sha256(code.strip().encode()).hexdigest() != draft['source_sha256'] or
                    stable_ast_sha256(code) != draft['ast_sha256']):
                raise ValueError('draft source binding mismatch')
        except (SyntaxError, RecursionError) as exc:
            raise ValueError('invalid draft source') from exc
        if probe_set is not None and draft['probe_set_sha256'] != probe_set:
            raise ValueError('drafts use different probe sets')
        probe_set = draft['probe_set_sha256']
        values = {'task': task, 'ast': draft['ast_sha256'], 'behavior': draft['behavior_sha256'],
                  'request': provenance['request_id']}
        for key, value in values.items():
            if value in seen[key]:
                raise ValueError('duplicate draft ' + key)
            seen[key].add(value)
    body = json.dumps(drafts, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return {'status': 'insufficient_records' if len(drafts) < 5 else 'pending_independent_audit',
            'structurally_valid_count': len(drafts), 'draft_batch_sha256': hashlib.sha256(body).hexdigest(),
            'training_allowed': False}
