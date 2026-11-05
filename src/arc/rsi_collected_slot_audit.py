"""Replay locally captured slot evidence; does not authenticate its origin."""
import base64
import binascii
import hashlib
import json
import re

from src.arc.rsi_candidate import inspect_candidate
from src.arc.rsi_lineage_audit import inspect_lineage
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_task_generation import generation_slots, task_prompt, task_repair_prompt


def _index(records):
    if not isinstance(records, list):
        raise ValueError('capture records must be a list')
    result = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get('request_id'), str):
            raise ValueError('capture record has no request ID')
        key = item['request_id']
        if not re.fullmatch(r'[0-9a-f]{32}', key) or key in result:
            raise ValueError('invalid or duplicated request ID')
        result[key] = item
    return result


def _decode(value):
    try:
        return base64.b64decode(value, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise ValueError('invalid captured bytes') from exc


def audit_collected_slot(*, evidence_bytes, evidence_sha256, request_records, response_records,
                         ledger_records, task_ids, task, generation, slot, expected_manifest_sha256,
                         expected_task_sha256, public_probes, private_probes,
                         expected_public_probes_sha256, expected_private_probes_sha256,
                         reference_pool, sandbox, model_key='14b'):
    if (not isinstance(evidence_bytes, bytes) or len(evidence_bytes) > 1024 * 1024 or
            hashlib.sha256(evidence_bytes).hexdigest() != evidence_sha256):
        raise ValueError('slot evidence digest differs')
    try:
        evidence = json.loads(evidence_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid slot evidence') from exc
    if (not isinstance(evidence, dict) or evidence.get('generation') != generation or
            evidence.get('slot') != slot or evidence.get('task_id') != task.id or
            evidence.get('eligible_for_training') is not False or
            evidence.get('evidence_type') != 'caller_supplied_components_unverified'):
        raise ValueError('slot evidence identity differs')
    attempts = evidence.get('attempts')
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
        raise ValueError('invalid slot attempts')
    requests, responses, ledger = map(_index, (request_records, response_records, ledger_records))
    selected = generation_slots(task_ids, generation)
    if type(slot) is not int or not 0 <= slot < len(selected):
        raise ValueError('invalid scheduled slot')
    prompt = task_prompt(task)
    replay, seen = [], set()
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise ValueError('invalid attempt')
        request_id = attempt.get('request_id')
        if request_id in seen or request_id not in requests or request_id not in responses or request_id not in ledger:
            raise ValueError('slot request capture missing or repeated')
        seen.add(request_id)
        if (attempt.get('seed') != selected[slot]['attempt_seeds'][index] or
                attempt.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest()):
            raise ValueError('slot prompt or seed differs')
        request, response = requests[request_id], responses[request_id]
        request_body = _decode(request.get('body_base64'))
        response_body = _decode(response.get('raw_http_body_base64'))
        if request.get('body_sha256') != hashlib.sha256(request_body).hexdigest():
            raise ValueError('request capture hash differs')
        replay.append({'code': attempt.get('source'), 'request_body': request_body,
                       'response_body': response_body, 'ledger': ledger[request_id],
                       'invalid_model_output': attempt.get('status') == 'invalid_model_output'})
        if index < len(attempts) - 1:
            prompt = task_repair_prompt(task, attempt.get('source'), attempt.get('status'))
    checked = inspect_lineage(task_ids, task, generation=generation, slot=slot, attempts=replay,
        expected_manifest_sha256=expected_manifest_sha256, expected_task_sha256=expected_task_sha256,
        public_probes=public_probes, private_probes=private_probes,
        expected_public_probes_sha256=expected_public_probes_sha256,
        expected_private_probes_sha256=expected_private_probes_sha256, sandbox=sandbox, reference_pool=reference_pool,
        model_key=model_key)
    statuses = checked['attempt_statuses']
    for attempt, status in zip(attempts[:-1], statuses[:-1]):
        if attempt.get('status') != status:
            raise ValueError('attempt status differs from replay')
    final_status = statuses[-1]
    if final_status in {'novel_unverified', 'duplicate_source', 'duplicate_ast', 'sampled_behavior_match'}:
        code = attempts[-1]['source']
        candidate = inspect_candidate(task, code, public_probes=public_probes,
                                      private_probes=private_probes, sandbox=sandbox)
        if candidate.get('status') != 'verified_candidate':
            raise ValueError('candidate replay changed')
        novelty = screen_novelty(code, candidate['private_behavior_sha256'],
                                 candidate['private_probe_set_sha256'], reference_pool)
        if evidence.get('novelty') != novelty:
            raise ValueError('novelty evidence differs from replay')
        if final_status != novelty['status']:
            raise ValueError('novelty status changed during replay')
    elif evidence.get('novelty') is not None:
        raise ValueError('terminal failure contains novelty evidence')
    if attempts[-1].get('status') != final_status or evidence.get('status') != final_status:
        raise ValueError('final status differs from replay')
    return {'status': 'captured_slot_consistent', 'eligible_for_training': False,
            'evidence_type': 'caller_supplied_capture_and_sandbox'}
