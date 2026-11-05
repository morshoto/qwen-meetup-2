"""Prepare a draft-bound local budget without authorizing execution.

Hashes establish only consistency of caller-supplied bytes, not independent
review, provenance, or permission to run a model or sandbox.
"""
import hashlib
import re

from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_model_identity import _MODELS
from src.arc.rsi_transport_audit import _json_object


def prepare_draft_generation_budget(*, root, registration_bytes, expected_registration_sha256, reference_bytes):
    if (not isinstance(registration_bytes, bytes) or not 1 <= len(registration_bytes) <= 262144 or
            not isinstance(reference_bytes, bytes) or not 1 <= len(reference_bytes) <= 65536 or
            not isinstance(expected_registration_sha256, str) or
            not re.fullmatch(r'[0-9a-f]{64}', expected_registration_sha256) or
            hashlib.sha256(registration_bytes).hexdigest() != expected_registration_sha256):
        raise ValueError('draft registration bytes or digest differ')
    draft, reference = _json_object(registration_bytes), _json_object(reference_bytes)
    key = draft.get('model_key')
    if not isinstance(key, str) or key not in _MODELS:
        raise ValueError('unsupported draft model')
    model, revision = _MODELS[key]
    limits = {'attempts_per_slot': 3, 'model_calls_including_setup': 32,
              'reserved_tokens': 200000, 'sandbox_starts': 300, 'slots': 10, 'wall_seconds': 3600}
    expected_index = f'research/rsi/audits/historical-candidates/{key}-behavior-reference.json'
    reference_digest = hashlib.sha256(reference_bytes).hexdigest()
    provisional = draft.get('provisional_behavior_reference')
    source_hashes = draft.get('source_sha256')
    probes = draft.get('provisional_probe_candidates')
    if (draft.get('experiment') != 'task_conditioned_oracle_pilot_v1' or
            draft.get('status') != 'draft_generation_registration_unverified' or
            draft.get('distinct_from') != 'E9' or type(draft.get('generation')) is not int or
            draft['generation'] != 0 or draft.get('model_id') != model or draft.get('model_revision') != revision or
            draft.get('parent_adapter') != 'none' or draft.get('execution_ready') is not False or
            draft.get('independent_review_completed') is not False or draft.get('training_allowed') is not False or
            draft.get('gate_policy') != 'five_independently_audited_new_same_model_oracles' or
            draft.get('limits') != limits or any(type(v) is not int for v in draft['limits'].values()) or
            not isinstance(source_hashes, dict) or source_hashes.get(expected_index) != reference_digest or
            not isinstance(provisional, dict) or provisional.get('review_index') != expected_index or
            type(provisional.get('sandbox_starts_in_preparation')) is not int or
            provisional['sandbox_starts_in_preparation'] != 10 or
            provisional.get('independently_verified') is not False or provisional.get('eligible_for_training') is not False or
            reference.get('model_key') != key or provisional.get('pool_sha256') != reference.get('pool_sha256') or
            provisional.get('local_result_sha256') != reference.get('local_result_sha256') or
            not isinstance(probes, dict) or not isinstance(probes.get('private'), dict) or
            probes['private'].get('canonical_probe_sha256') != reference.get('probe_set_sha256')):
        raise ValueError('draft and reference accounting identity differs')
    budget = CollectionBudget.create(root, registration_sha256=expected_registration_sha256, generation=0,
        reference_setup_bytes=reference_bytes, expected_reference_setup_sha256=reference_digest, model_key=key)
    return {'budget': budget, 'reference_setup_sha256': reference_digest, 'model_key': key,
            'execution_authorized': False, 'training_allowed': False,
            'status': 'draft_budget_consistent_unverified'}
