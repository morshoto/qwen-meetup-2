"""Audit inputs to the new weight experiment without exposing held-out answers."""
from __future__ import annotations

import hashlib
import json
import re

from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION, MAX_CONTEXT, MAX_TOKENS, TEMPERATURE
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import build_registration


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def audited_registration(*, challenges_bytes, solutions_bytes, official, baseline, proposal,
                         phase_b, prior, exposure, lineage, prompt, source_hashes, expected_comparison_sha256):
    """Return a proposed registration; caller must authenticate source files too.

    The solution file is used solely for identity and per-task fingerprints. No
    solution grids are returned or sent to an inference process.
    """
    if not isinstance(source_hashes, dict) or not source_hashes or not all(_valid_hash(v) for v in source_hashes.values()):
        raise ValueError('invalid source inventory')
    tasks, solutions = json.loads(challenges_bytes), json.loads(solutions_bytes)
    if not isinstance(tasks, dict) or not isinstance(solutions, dict) or set(tasks) != set(solutions):
        raise ValueError('training challenge and solution IDs differ')
    if any(not isinstance(solutions[k], list) or len(solutions[k]) != len(task['test']) for k, task in tasks.items()):
        raise ValueError('training solution count differs from test inputs')
    if not isinstance(official, dict) or len(official) != 120 or source_hashes.get('official_challenges') != digest(official):
        raise ValueError('expected 120 official evaluation challenges')
    challenge_hash = _sha(challenges_bytes)
    if challenge_hash != proposal['dataset_sha256'] or challenge_hash != baseline['challenge_sha256']:
        raise ValueError('training challenges differ from frozen comparison')
    if baseline.get('sha256') != expected_comparison_sha256 or not _valid_hash(expected_comparison_sha256) or digest({k: v for k, v in baseline.items() if k != 'sha256'}) != baseline['sha256']:
        raise ValueError('comparison manifest digest mismatch')
    allocation = baseline['allocation']
    if allocation.get('sha256') != digest({k: v for k, v in allocation.items() if k != 'sha256'}):
        raise ValueError('comparison allocation digest mismatch')
    evaluation = allocation['evaluation']
    if len(evaluation) != 50 or len(set(evaluation)) != 50 or not set(evaluation) <= tasks.keys():
        raise ValueError('invalid protected comparison set')
    phase_a = proposal['task_ids']
    if len(phase_a) != 200 or len(set(phase_a)) != 200 or not set(evaluation) <= set(phase_a):
        raise ValueError('invalid Phase A comparison allocation')
    if len(set(phase_b)) != len(phase_b) or set(phase_b) != set(proposal['training_exclusions']):
        raise ValueError('Phase B reservations disagree')
    if (baseline['protocol'] != 'qwen38-local-exploratory-v1' or baseline['model_id'] != MODEL_ID or
        baseline['model_revision'] != MODEL_REVISION or baseline['prompt_sha256'] != _sha(prompt.encode()) or
        baseline['candidate_count'] != 2 or baseline['temperature'] != TEMPERATURE or
        baseline['max_tokens'] != MAX_TOKENS or baseline['max_context'] != MAX_CONTEXT or
        baseline['format_recovery'] is not True):
        raise ValueError('comparison model or generation settings mismatch')
    weights = baseline.get('model_files_sha256', {})
    if not weights or not any(k.endswith('.safetensors') for k in weights) or not all(_valid_hash(v) for v in weights.values()):
        raise ValueError('invalid comparison model inventory')
    exclusions = {'prior_phase_a': phase_a, 'phase_b': phase_b,
                  'protected_rsi': prior['confirmatory_task_ids'] + prior['exploratory_task_ids'],
                  'proposal_protected': proposal['protected_exclusions'],
                  'known_exposure': sorted(set(proposal['known_exposure_exclusions']) | set(exposure['matched_task_ids'])),
                  'prior_development': proposal['development_exclusions'],
                  'local_evaluation': evaluation, 'local_development': allocation['development'],
                  'local_selection': allocation['selection']}
    inputs = dict(source_hashes, challenges=challenge_hash, solutions=_sha(solutions_bytes))
    result = build_registration(tasks, official, exclusions, lineage, input_hashes=inputs,
                                comparison_sha256=baseline['sha256'], prompt=prompt)
    result['model']['files_sha256'] = weights
    selected = result['allocation']['fresh'] + result['allocation']['validation']
    result['solution_fingerprints'] = {k: digest(solutions[k]) for k in selected}
    result['training_gate'] = {'minimum_unique_verified_parents': 5, 'validation_success_required': False,
                               'validation_targets_allowed': False, 'fallback': 'audit historical verified Qwen outputs if fresh parents under five',
                               'insufficient_status': 'insufficient_verified_data'}
    return result
