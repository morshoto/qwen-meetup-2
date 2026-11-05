"""Summarize server-reported generation tokens without asserting provenance."""
from src.arc.rsi_collected_slot_audit import _decode, _index
from src.arc.rsi_comparison_usage import _summarize_usage
from src.arc.rsi_generation_audit import audit_generation
from src.arc.rsi_transport_audit import _json_object


def audit_generation_usage(**inputs):
    replay = audit_generation(**inputs)
    ledger = _index(inputs['ledger_records'])
    responses = _index(inputs['response_records'])
    if set(ledger) != set(responses) or len(ledger) != replay['budget_usage']['model_calls']:
        raise ValueError('generation usage coverage differs')
    for request_id, event in ledger.items():
        body = _json_object(_decode(responses[request_id].get('raw_http_body_base64')))
        if event.get('usage') != body.get('usage'):
            raise ValueError('generation ledger usage differs from captured response')
    reported = _summarize_usage(list(ledger.values()))
    if sum(event['token_reservation'] for event in ledger.values()) != replay['budget_usage']['reserved_tokens']:
        raise ValueError('generation token reservations differ')
    audit_generation(**inputs)
    return {'status': 'generation_reported_usage_consistent_unverified', 'reported': reported,
            'reserved_tokens': replay['budget_usage']['reserved_tokens'], 'monetary_cost': None,
            'independently_verified': False, 'evidence_type': 'server_reported_usage_and_caller_supplied_captures'}
