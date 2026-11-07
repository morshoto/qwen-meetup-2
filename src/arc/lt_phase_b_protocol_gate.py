"""Reject execution until the source-bound Phase B protocol is frozen.

This validates local registration consistency, not independent attestation,
complete historical exposure, or authorization to run a model.
"""
import hashlib
import json
import re
from pathlib import Path

from src.arc.lt_attestation import _object
from src.arc.lt_phase_b_registered_worker import _digest


def validate_frozen_protocol(*, source_root, spec, source_inventory, allocation_receipt, challenge_bytes):
    root = Path(source_root)
    protocol = _object((root / 'configs/lt.phase-b.protocol.json').read_bytes())
    data, evaluation = protocol.get('data'), protocol.get('evaluation')
    registration, inference = protocol.get('registration'), protocol.get('inference')
    training, canary = protocol.get('training'), protocol.get('canary')
    if (protocol.get('protocol') != 'lt-phase-b-exploratory' or protocol.get('status') != 'frozen' or
            not all(isinstance(x, dict) for x in (data, evaluation, registration, inference, training, canary)) or
            not isinstance(protocol.get('open_gates'), list) or
            any(not isinstance(gate, str) or gate != 'real_14b_canary_and_reload' for gate in protocol['open_gates']) or
            len(protocol['open_gates']) != len(set(protocol['open_gates']))):
        raise ValueError('Phase B protocol is not frozen or has open execution gates')
    settings = spec['settings']
    if (data.get('training_pool_status') != 'frozen' or data.get('training_pool_count') != 100 or
            data.get('training_pool_file') != 'configs/lt.phase-b.training-pool.json' or
            data.get('challenge_sha256') != hashlib.sha256(challenge_bytes).hexdigest() or
            not isinstance(data.get('solution_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', data['solution_sha256']) or
            evaluation.get('comparison_status') != 'frozen' or evaluation.get('comparison_count') != 50 or
            evaluation.get('comparison_proposal_file') != 'configs/lt.phase-b.comparison-proposal.json' or
            evaluation.get('official_public_evaluation') != 'reserved_for_phase_c' or
            registration.get('actual_evaluation_manifest') != 'frozen' or
            registration.get('actual_training_settings') != 'frozen' or
            registration.get('allocation') != _digest(allocation_receipt) or
            registration.get('source_closure') != source_inventory.get('code_sha256') or
            registration.get('source_closure') != spec['source_code_sha256'] or
            registration.get('runtime_and_environment') != 'bound_by_collection_descriptor' or
            registration.get('protocol_hash') != 'bound_by_source_inventory' or
            json.dumps(protocol.get('model'), sort_keys=True, separators=(',', ':')) !=
            json.dumps(spec['model_manifest'], sort_keys=True, separators=(',', ':'))):
        raise ValueError('frozen protocol inputs or registration differ')
    expected_inference = {'agent': 'Phase A direct-grid', 'candidate_count': settings['candidate_count'],
        'max_tokens_per_candidate': settings['max_tokens'], 'mlx_memory_limit_bytes': settings['mlx_memory_limit_bytes'],
        'prompt_sha256': spec['prompt_sha256'], 'seed_base': settings['seed'],
        'task_timeout_seconds_proposed': settings['task_timeout_seconds'], 'temperature': settings['temperature'],
        'thinking': settings['thinking']}
    expected_training = {'adapted_layers': 8, 'batch_size': 1, 'learning_rate': 0.0001, 'mask_prompt': True,
        'max_sequence_length': 512, 'maximum_optimizer_steps': 20, 'optimizer': 'adam', 'rank': 8, 'scale': 2.0,
        'seed': 20261111, 'thinking': 'off'}
    expected_canary = {'adapted_layers': 8, 'batch_size': 1, 'learning_rate': 0.0001, 'mask_prompt': True,
        'max_sequence_length': 512, 'optimizer_steps': 1, 'rank': 8, 'seed': 20261111, 'thinking': 'off',
        'purpose': 'synthetic_technical_only', 'promotion_allowed': False}
    for actual, expected in ((inference, expected_inference), (training, expected_training), (canary, expected_canary)):
        if any(type(actual.get(k)) is not type(v) or actual[k] != v for k, v in expected.items()):
            raise ValueError('frozen protocol settings differ')
    metrics = {'pass_at_1', 'two_recorded_attempts', 'format_validity', 'paired_improvements_regressions_ties',
               'tokens', 'latency', 'raw_counts', 'uncertainty'}
    canary_records = {'finite_loss', 'base_unchanged', 'adapter_tensor_change', 'saved_adapter_hash',
                      'new_process_reload_and_inference', 'resource_observations', 'owned_process_cleanup'}
    stops = {'failed_canary', 'unverified_or_insufficient_yield', 'sequence_truncation', 'nonfinite_loss',
             'resource_limit', 'runtime_identity_drift', 'cleanup_unverified'}
    for value, required in ((evaluation.get('metrics'), metrics), (canary.get('must_record'), canary_records),
                            (training.get('stop_conditions'), stops)):
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value) or set(value) != required or len(value) != len(required):
            raise ValueError('frozen protocol evidence policy differs')
    if (training.get('selection') != 'Final step only; no validation-score checkpoint cherry-picking.' or
            training.get('yield_policy') != {'one_or_two': 'overfitting_only', 'three_or_more': 'task_disjoint_train_validation',
                                             'zero': 'insufficient_verified_self_solutions'} or
            evaluation.get('paired_policy') != 'same base, public tasks, prompt, parser, tokenizer, seed policy, budget and scorer; only adapter state differs'):
        raise ValueError('frozen protocol comparison or selection policy differs')
    if (type(training.get('maximum_wall_seconds')) is not int or
            not settings['task_timeout_seconds'] <= training['maximum_wall_seconds'] <= 14400):
        raise ValueError('frozen training time ceiling differs')
    try:
        pool = json.loads((root / data['training_pool_file']).read_bytes())
        comparison = _object((root / evaluation['comparison_proposal_file']).read_bytes())['task_ids']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError('frozen task manifests unavailable') from exc
    if (not isinstance(pool, list) or len(pool) != 100 or any(not isinstance(x, str) for x in pool) or
            len(set(pool)) != 100 or set(pool) != set(spec['selected_task_sha256']) or
            not isinstance(comparison, list) or len(comparison) != 50 or any(not isinstance(x, str) for x in comparison) or
            len(set(comparison)) != 50 or set(comparison) != set(allocation_receipt['comparison_task_sha256']) or
            set(pool) & set(comparison)):
        raise ValueError('frozen task manifests differ from allocation')
    return {'status': 'frozen_protocol_consistent_unverified', 'execution_ready': False}
