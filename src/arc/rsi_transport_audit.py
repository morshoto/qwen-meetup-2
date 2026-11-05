"""Check captured transport bytes against a proposed slot, without certifying origin.

A matching pair can be fabricated. A separately sealed run and independent
server, adapter, budget, task, probe, and sandbox audits are still required.
"""
import hashlib
import json
import math

from src.arc.http_backend import BackendError, final_answer
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


def _json_object(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid_constant(_):
        raise ValueError('non-finite JSON number')
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid captured JSON') from exc
    if not isinstance(value, dict):
        raise ValueError('captured body must be an object')
    return value


def audit_transport(request_body: bytes, response_body: bytes, ledger: dict, *,
                    expected_prompt: str, expected_code: str, expected_seed: int) -> dict:
    if (not isinstance(request_body, bytes) or not 1 <= len(request_body) <= 1024 * 1024 or
            not isinstance(response_body, bytes) or not 1 <= len(response_body) <= 4 * 1024 * 1024 or
            not isinstance(ledger, dict) or not isinstance(expected_prompt, str) or not expected_prompt or
            not isinstance(expected_code, str) or not expected_code.strip() or type(expected_seed) is not int):
        raise ValueError('missing or invalid transport evidence')
    req_hash = hashlib.sha256(request_body).hexdigest()
    resp_hash = hashlib.sha256(response_body).hexdigest()
    if (ledger.get('request_body_sha256') != req_hash or ledger.get('response_sha256') != resp_hash or
            ledger.get('model') != _MODEL or ledger.get('revision') != _REVISION or
            ledger.get('seed_requested') != expected_seed or ledger.get('thinking_mode_effective') != 'off' or
            ledger.get('error') is not None or not isinstance(ledger.get('request_id'), str) or
            not ledger['request_id']):
        raise ValueError('transport ledger does not match captured evidence')
    request = _json_object(request_body)
    allowed = {'model', 'messages', 'temperature', 'max_tokens', 'seed', 'stream',
               'thinking_mode', 'deadline_monotonic'}
    deadline = request.get('deadline_monotonic')
    if (set(request) != allowed or request.get('model') != _MODEL or
            request.get('messages') != [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                                        {'role': 'user', 'content': expected_prompt}] or
            type(request.get('temperature')) not in (int, float) or request['temperature'] != 0.2 or
            type(request.get('max_tokens')) is not int or request['max_tokens'] != 512 or
            type(request.get('seed')) is not int or request['seed'] != expected_seed or
            request.get('stream') is not False or request.get('thinking_mode') != 'off' or
            type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError('captured request differs from slot configuration')
    response = _json_object(response_body)
    if (response.get('model') != _MODEL or response.get('thinking_mode') != 'off' or
            response.get('thinking_control') != 'tokenizer_template_canary_passed'):
        raise ValueError('captured response model or thinking control differs')
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('captured response must contain one choice')
    message = choices[0].get('message')
    if (not isinstance(message, dict) or message.get('role') != 'assistant' or
            not isinstance(message.get('content'), str) or choices[0].get('finish_reason') != 'stop'):
        raise ValueError('captured response is incomplete')
    try:
        extracted = final_answer(message['content'])
    except BackendError as exc:
        raise ValueError('captured response has invalid final content') from exc
    if extracted.strip() != expected_code.strip():
        raise ValueError('captured response differs from candidate source')
    return {'status': 'transport_consistent', 'request_id': ledger['request_id'],
            'request_body_sha256': req_hash, 'response_sha256': resp_hash,
            'eligible_for_training': False}
