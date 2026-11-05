"""Replay conditioned repair evidence without certifying an independent run.

The supplied sandbox and transport records are not authenticated here. This
precheck cannot authorize training or establish sandbox provenance.
"""
import hashlib

from src.arc.rsi_candidate import inspect_candidate
from src.arc.data import validate_grid
from src.arc.e6_curriculum import canonical
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_task_generation import _REPAIRABLE, generation_slots, task_prompt, task_repair_prompt
from src.arc.rsi_transport_audit import audit_transport


def probes_sha256(probes: list) -> str:
    if not isinstance(probes, list) or len(probes) < 5:
        raise ValueError('probe set must contain at least five grids')
    checked = [validate_grid(grid) for grid in probes]
    return hashlib.sha256(canonical(checked).encode()).hexdigest()


def inspect_lineage(task_ids: list[str], task, *, generation: int, slot: int, attempts: list[dict],
                    expected_manifest_sha256: str, expected_task_sha256: str,
                    public_probes: list, private_probes: list,
                    expected_public_probes_sha256: str, expected_private_probes_sha256: str,
                    sandbox: object, reference_pool: list[dict] | None = None, model_key: str = '14b') -> dict:
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
        raise ValueError('lineage must contain one to three attempts')
    slots = generation_slots(task_ids, generation)
    if type(slot) is not int or not 0 <= slot < len(slots):
        raise ValueError('invalid scheduled slot')
    selected = slots[slot]
    if (manifest_sha256(task_ids) != expected_manifest_sha256 or
            task_sha256(task) != expected_task_sha256 or task.id != selected['task_id']):
        raise ValueError('task or schedule differs from registration')
    if (probes_sha256(public_probes) != expected_public_probes_sha256 or
            probes_sha256(private_probes) != expected_private_probes_sha256):
        raise ValueError('probe set differs from registration')
    seen_requests, seen_sources = set(), set()
    statuses = []
    prompt = task_prompt(task)
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or not isinstance(attempt.get('ledger'), dict):
            raise ValueError('invalid attempt evidence')
        code = attempt.get('code')
        invalid = attempt.get('invalid_model_output', False)
        if type(invalid) is not bool:
            raise ValueError('invalid completion marker')
        transport = audit_transport(attempt.get('request_body'), attempt.get('response_body'),
                                    attempt['ledger'], expected_prompt=prompt, expected_code=code,
                                    expected_seed=selected['attempt_seeds'][index], expected_invalid=invalid, model_key=model_key)
        request_id = transport['request_id']
        if request_id in seen_requests:
            raise ValueError('duplicate request identifier')
        seen_requests.add(request_id)
        if invalid:
            status = 'invalid_model_output'
        else:
            source_hash = hashlib.sha256(code.strip().encode()).hexdigest()
            if source_hash in seen_sources:
                status = 'repeated_code'
            else:
                seen_sources.add(source_hash)
                inspected = inspect_candidate(task, code, public_probes=public_probes,
                                              private_probes=private_probes, sandbox=sandbox)
                status = inspected['status']
                if status == 'verified_candidate' and reference_pool is not None:
                    status = screen_novelty(code, inspected['private_behavior_sha256'],
                                            inspected['private_probe_set_sha256'], reference_pool)['status']
        statuses.append(status)
        if index < len(attempts) - 1:
            if status not in _REPAIRABLE:
                raise ValueError('attempt follows after terminal or unauditable outcome')
            prompt = task_repair_prompt(task, code, status)
    return {'status': 'lineage_consistent', 'task_id': task.id, 'generation': generation,
            'slot': slot, 'attempt_statuses': statuses, 'eligible_for_training': False,
            'evidence_type': 'caller_supplied_sandbox_and_transport'}
