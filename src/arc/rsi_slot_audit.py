"""Offline first-attempt binding; this is not an independent run audit."""
import hashlib
import json

from src.arc.data import Task
from src.arc.rsi_task_generation import generation_slots, task_prompt
from src.arc.rsi_transport_audit import audit_transport


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def manifest_sha256(task_ids: list[str]) -> str:
    return _digest(task_ids)


def task_sha256(task: Task) -> str:
    if not isinstance(task, Task) or not isinstance(task.id, str) or not task.id:
        raise ValueError('invalid source task')
    return _digest({'task_id': task.id, 'prompt_data': task.prompt_data()})


def audit_scheduled_attempt(task_ids: list[str], task: Task, *, generation: int, slot: int,
                            attempt: int, request_body: bytes, response_body: bytes, ledger: dict,
                            expected_code: str, expected_manifest_sha256: str,
                            expected_task_sha256: str) -> dict:
    """Check attempt zero only; repair lineage requires a separate full audit."""
    if type(attempt) is not int or attempt != 0:
        raise ValueError('repair attempts require a separately verified lineage')
    slots = generation_slots(task_ids, generation)
    if type(slot) is not int or not 0 <= slot < len(slots):
        raise ValueError('invalid scheduled slot')
    selected = slots[slot]
    if (manifest_sha256(task_ids) != expected_manifest_sha256 or
            task_sha256(task) != expected_task_sha256 or task.id != selected['task_id']):
        raise ValueError('task or schedule differs from frozen registration')
    seed = selected['attempt_seeds'][attempt]
    transport = audit_transport(request_body, response_body, ledger,
                                expected_prompt=task_prompt(task), expected_code=expected_code,
                                expected_seed=seed)
    return {'status': 'slot_transport_consistent', 'task_id': task.id, 'generation': generation,
            'slot': slot, 'attempt': attempt, 'seed_requested': seed,
            'request_id': transport['request_id'], 'eligible_for_training': False}
