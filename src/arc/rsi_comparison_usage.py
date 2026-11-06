"""Summarize server-reported comparison token counts, not monetary cost."""
from pathlib import Path

from src.arc.rsi_collected_slot_audit import _decode, _index
from src.arc.rsi_comparison_audit import _read_jsonl, audit_comparison_captures
from src.arc.rsi_slot_lifecycle import _directory
from src.arc.rsi_transport_audit import _json_object

_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')


def _summarize_usage(events):
    sums = {key: 0 for key in _KEYS}
    missing = {key: 0 for key in _KEYS}
    complete = 0
    for event in events:
        reservation, usage = event.get('token_reservation'), event.get('usage')
        if type(reservation) is not int or reservation <= 0 or (usage is not None and not isinstance(usage, dict)):
            raise ValueError('invalid reported usage or reservation')
        counts = {} if usage is None else {key: usage[key] for key in _KEYS if key in usage}
        if (any(type(value) is not int or value < 0 for value in counts.values()) or
                counts.get('completion_tokens', 0) > 512 or
                max(counts.get('total_tokens', 0), counts.get('prompt_tokens', 0) + counts.get('completion_tokens', 0)) > reservation or
                ('total_tokens' in counts and any(value > counts['total_tokens'] for key, value in counts.items() if key != 'total_tokens')) or
                (len(counts) == 3 and counts['prompt_tokens'] + counts['completion_tokens'] != counts['total_tokens'])):
            raise ValueError('invalid or over-budget reported tokens')
        complete += len(counts) == 3
        for key in _KEYS:
            if key in counts:
                sums[key] += counts[key]
            else:
                missing[key] += 1
    return {'request_count': len(events), 'complete_requests': complete,
            **{key + '_reported': sums[key] for key in _KEYS},
            **{key + '_missing_requests': missing[key] for key in _KEYS},
            'total_tokens_if_complete': sums['total_tokens'] if complete == len(events) and events else None}


def audit_comparison_usage(root, *, expected_sha256, tasks, sandbox, verifier):
    replay = audit_comparison_captures(root, expected_sha256=expected_sha256, tasks=tasks,
                                       sandbox=sandbox, verifier=verifier)
    summaries = {}
    for arm in ('parent', 'candidate'):
        with _directory(Path(root) / f'{arm}.responses') as directory:
            events = _read_jsonl(directory, 'requests.jsonl')
            responses = _index(_read_jsonl(directory, 'llm_responses.jsonl'))
        seen = set()
        for event in events:
            request_id = event.get('request_id')
            if request_id in seen or request_id not in responses:
                raise ValueError('reported usage response coverage differs')
            seen.add(request_id)
            body = _json_object(_decode(responses[request_id].get('raw_http_body_base64')))
            if event.get('usage') != body.get('usage'):
                raise ValueError('ledger usage differs from captured response')
        if len(events) != 400 or set(responses) != seen:
            raise ValueError('reported usage request coverage differs')
        summaries[arm] = _summarize_usage(events)
    audit_comparison_captures(root, expected_sha256=expected_sha256, tasks=tasks, sandbox=sandbox, verifier=verifier)
    return {'status': 'comparison_reported_usage_consistent_unverified', 'arms': summaries,
            'reserved_tokens': replay['budget_usage']['reserved_tokens'], 'independently_verified': False,
            'monetary_cost': None, 'evidence_type': 'server_reported_usage_and_caller_supplied_captures'}
