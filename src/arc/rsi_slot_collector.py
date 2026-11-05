"""Conditioned slot orchestration with caller-supplied, unauthenticated components.

This function does not establish registration, transport or sandbox authenticity
and must not be treated as an authorization to run a live experiment.
"""
import hashlib
import json
import re
from pathlib import Path

from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_budget_adapters import BudgetedSandbox
from src.arc.rsi_call_reservations import reserve_call
from src.arc.rsi_candidate import inspect_candidate
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_slot_lifecycle import _create, _directory, claim_slot, complete_slot
from src.arc.rsi_task_generation import _REPAIRABLE, generation_slots, task_prompt, task_repair_prompt


def collect_slot(*, root, registration_sha256, generation, slot, task_ids, task,
                 expected_manifest_sha256, expected_task_sha256, public_probes, private_probes,
                 expected_public_probes_sha256, expected_private_probes_sha256, reference_pool,
                 transport, sandbox, budget):
    schedule = generation_slots(task_ids, generation)
    if type(slot) is not int or not 0 <= slot < len(schedule):
        raise ValueError('invalid scheduled slot')
    selected = schedule[slot]
    if (task.id != selected['task_id'] or manifest_sha256(task_ids) != expected_manifest_sha256 or
            task_sha256(task) != expected_task_sha256 or
            probes_sha256(public_probes) != expected_public_probes_sha256 or
            probes_sha256(private_probes) != expected_private_probes_sha256):
        raise ValueError('slot inputs differ from expected registration')
    if (budget.identity != {'registration_sha256': registration_sha256, 'generation': generation} or
            Path(budget.root).resolve() != Path(root).resolve()):
        raise ValueError('budget differs from slot identity')
    claim = claim_slot(root, registration_sha256=registration_sha256, generation=generation, slot=slot)
    metered_sandbox = BudgetedSandbox(sandbox, budget)
    prompt = task_prompt(task)
    attempts, seen, request_ids = [], set(), set()
    novelty = None
    for attempt, seed in enumerate(selected['attempt_seeds']):
        reserve_call(root, registration_sha256=registration_sha256, generation=generation, slot=slot,
                     attempt=attempt, claim_token=claim['claim_token'])
        messages = [{'role': 'system', 'content': system_prompt_for_kind('e5_generator')},
                    {'role': 'user', 'content': prompt}]
        tokens = 512 + len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
        budget.reserve_model(tokens, kind='generation')
        response = transport(prompt, seed=seed, timeout=budget.remaining())
        request_id = None
        if isinstance(response, dict):
            if set(response) != {'code', 'request_id'}:
                raise ValueError('invalid transport response metadata')
            code, request_id = response['code'], response['request_id']
            if (not isinstance(request_id, str) or not re.fullmatch(r'[0-9a-f]{32}', request_id)
                    or request_id in request_ids):
                raise ValueError('invalid or duplicate transport request identifier')
            request_ids.add(request_id)
        else:
            code = response
        if not isinstance(code, str) or len(code.encode('utf-8')) > 65536:
            raise ValueError('transport response is not bounded source text')
        source_hash = hashlib.sha256(code.strip().encode()).hexdigest()
        if source_hash in seen:
            status = 'repeated_code'
        else:
            seen.add(source_hash)
            candidate = inspect_candidate(task, code, public_probes=public_probes,
                                          private_probes=private_probes, sandbox=metered_sandbox)
            status = candidate['status']
            if status == 'verified_candidate':
                novelty = screen_novelty(code, candidate['private_behavior_sha256'],
                                         candidate['private_probe_set_sha256'], reference_pool)
                status = novelty['status']
        attempts.append({'seed': seed, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                         'source': code, 'status': status, 'request_id': request_id})
        if status not in _REPAIRABLE or attempt == 2:
            break
        prompt = task_repair_prompt(task, code, status)
    evidence = {'status': status, 'eligible_for_training': False, 'generation': generation,
                'slot': slot, 'task_id': task.id, 'attempts': attempts, 'novelty': novelty,
                'evidence_type': 'caller_supplied_components_unverified'}
    raw = (json.dumps(evidence, sort_keys=True, separators=(',', ':')) + '\n').encode()
    if len(raw) > 1024 * 1024:
        raise ValueError('collection evidence exceeds size limit')
    digest = hashlib.sha256(raw).hexdigest()
    with _directory(root) as directory:
        _create(directory, f'g{generation}-slot-{slot:02d}.collection.json', evidence)
    complete_slot(root, registration_sha256=registration_sha256, generation=generation, slot=slot,
                  claim_token=claim['claim_token'], evidence_sha256=digest)
    return {'status': status, 'eligible_for_training': False, 'evidence_sha256': digest}
