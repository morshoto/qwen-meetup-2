"""Report replayed generation outcomes without asserting independently audited Oracles."""
from src.arc.rsi_generation_audit import audit_generation

_NEW = {'novel_unverified'}
_DUPLICATE = {'duplicate_source', 'duplicate_ast', 'sampled_behavior_match', 'repeated_code'}
_INVALID = {'syntax_error', 'parse_or_contract_error', 'prohibited_operation', 'static_safety_rejected',
            'runtime_exception', 'shape_or_color_violation', 'visible_demonstration_mismatch',
            'nondeterministic_output_rejected', 'identity_only_rejected', 'public_validation_rejected',
            'private_acceptance_rejected', 'query_execution_rejected', 'invalid_model_output'}
_ALLOWED = _NEW | _DUPLICATE | _INVALID


def _totals(counts):
    if (not isinstance(counts, dict) or any(not isinstance(k, str) or k not in _ALLOWED or
            type(v) is not int or v <= 0 for k, v in counts.items())):
        raise ValueError('generation outcome counts invalid')
    return {'provisional_new': sum(counts.get(k, 0) for k in _NEW),
            'duplicate_or_behavior_match': sum(counts.get(k, 0) for k in _DUPLICATE),
            'invalid': sum(counts.get(k, 0) for k in _INVALID)}


def summarize_generation_outcomes(*, model_key, **inputs):
    if model_key not in ('14b', '32b'):
        raise ValueError('unsupported generation model')
    replay = audit_generation(model_key=model_key, **inputs)
    if (not isinstance(replay, dict) or replay.get('status') != 'generation_replay_consistent' or
            type(replay.get('checked_slots')) is not int or replay['checked_slots'] != 10 or
            replay.get('training_allowed') is not False):
        raise ValueError('complete generation replay required')
    counts, attempts = replay.get('counts'), replay.get('attempt_counts')
    slot_totals, attempt_totals = _totals(counts), _totals(attempts)
    if (sum(counts.values()) != 10 or not 10 <= sum(attempts.values()) <= 30 or
            any(count > attempts.get(status, 0) for status, count in counts.items())):
        raise ValueError('generation outcome totals differ from schedule')
    return {'status': 'generation_outcome_summary_unverified', 'model_key': model_key,
            'slot_outcomes': slot_totals, 'attempt_outcomes': attempt_totals,
            'status_counts': dict(counts), 'attempt_status_counts': dict(attempts),
            'independently_verified_new_oracles': None, 'training_allowed': False,
            'next_step': ('stop_before_training' if slot_totals['provisional_new'] < 5
                          else 'independent_audit_required'),
            'limitations': ['behavior matches are sampled, not proof of semantic duplication',
                            'local replay and provisional novelty do not prove independent provenance']}


def summarize_incomplete_generation(*, model_key, report):
    """Summarize a caller-supplied partial record; never certify its provenance."""
    import re
    from collections import Counter
    from src.arc.rsi_generation_stage import _BUDGET_LIMITS

    if (model_key not in ('14b', '32b') or not isinstance(report, dict) or
            report.get('status') != 'generation_incomplete' or report.get('training_allowed') is not False or
            report.get('evidence_type') != 'caller_supplied_components_unverified' or
            type(report.get('completed_slots')) is not int or not 0 <= report['completed_slots'] <= 10 or
            not isinstance(report.get('slots'), list) or len(report['slots']) != report['completed_slots'] or
            not isinstance(report.get('failure_type'), str) or
            not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,127}', report['failure_type'])):
        raise ValueError('invalid incomplete generation report')
    seen = set()
    for index, slot in enumerate(report['slots']):
        if (not isinstance(slot, dict) or set(slot) != {'slot', 'task_id', 'status', 'evidence_sha256'} or
                type(slot['slot']) is not int or slot['slot'] != index or not isinstance(slot['task_id'], str) or
                not slot['task_id'] or slot['task_id'] in seen or slot['status'] not in _ALLOWED or
                not isinstance(slot['evidence_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', slot['evidence_sha256'])):
            raise ValueError('invalid incomplete slot evidence')
        seen.add(slot['task_id'])
    counts, attempts = report.get('counts'), report.get('attempt_counts')
    slot_totals, attempt_totals = _totals(counts), _totals(attempts)
    if (dict(Counter(slot['status'] for slot in report['slots'])) != counts or
            not report['completed_slots'] <= sum(attempts.values()) <= 3 * report['completed_slots'] or
            any(count > attempts.get(status, 0) for status, count in counts.items())):
        raise ValueError('incomplete outcome totals differ')
    after, usage = report.get('budget_after'), report.get('budget_usage')
    if (after is None) != (usage is None):
        raise ValueError('incomplete budget evidence differs')
    unattributed = None
    if after is not None:
        for item in (after, usage):
            if (not isinstance(item, dict) or set(item) != set(_BUDGET_LIMITS) or
                    any(type(item[key]) is not int or not 0 <= item[key] <= limit
                        for key, limit in _BUDGET_LIMITS.items())):
                raise ValueError('invalid incomplete budget counters')
        if any(usage[key] > after[key] for key in _BUDGET_LIMITS) or usage['model_calls'] < sum(attempts.values()):
            raise ValueError('incomplete budget counters differ')
        unattributed = usage['model_calls'] - sum(attempts.values())
    return {'status':'incomplete_generation_summary_unverified', 'model_key':model_key,
            'completed_slots':report['completed_slots'], 'uncompleted_slots':10-report['completed_slots'],
            'slot_outcomes':slot_totals, 'attempt_outcomes':attempt_totals,
            'unattributed_model_reservations':unattributed, 'failure_type':report['failure_type'],
            'independently_verified':False, 'training_allowed':False, 'next_step':'stop_before_training',
            'limitations':['caller-supplied records have not been independently audited',
                           'interrupted attempts may not appear in completed-slot counts',
                           'reservations do not establish dispatched model calls']}
