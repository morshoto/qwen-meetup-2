"""Summarize server-reported generation tokens without asserting provenance."""
from src.arc.rsi_collected_slot_audit import _decode, _index
from src.arc.rsi_comparison_usage import _summarize_usage
from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_generation_audit import audit_generation
from src.arc.rsi_generation_stage import _budget_delta
from src.arc.rsi_slot_lifecycle import _directory
from src.arc.rsi_transport_audit import _json_object


def audit_generation_usage(**inputs):
    replay = audit_generation(**inputs)
    with _directory(inputs['root']) as directory:
        start_raw, started = _read(directory, f"g{inputs['generation']}.stage.started.json")
        end_raw, completed = _read(directory, f"g{inputs['generation']}.stage.completed.json")
    before, after = started.get('budget_before'), completed.get('budget_after')
    if _budget_delta(before, after) != replay['budget_usage']:
        raise ValueError('generation stage reservation counters differ')
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
    with _directory(inputs['root']) as directory:
        start_after, _ = _read(directory, f"g{inputs['generation']}.stage.started.json")
        end_after, _ = _read(directory, f"g{inputs['generation']}.stage.completed.json")
    if start_raw != start_after or end_raw != end_after:
        raise ValueError('generation stage accounting changed during audit')
    return {'status': 'generation_reported_usage_consistent_unverified', 'reported': reported,
            'reserved_tokens': replay['budget_usage']['reserved_tokens'], 'monetary_cost': None,
            'reported_scope': 'captured_slot_attempts_only', 'setup_reservations': before,
            'stage_reservations_total': after,
            'stage_reported_total_tokens_if_complete': (reported['total_tokens_if_complete']
                if before['model_calls'] == 0 and before['reserved_tokens'] == 0 else None),
            'independently_verified': False, 'evidence_type': 'server_reported_usage_and_caller_supplied_captures'}
