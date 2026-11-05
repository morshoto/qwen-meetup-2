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
