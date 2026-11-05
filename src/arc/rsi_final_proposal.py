"""Freeze an unregistered final diagnostic proposal without executing it."""
import hashlib
import json

from src.arc.rsi_comparison_freeze import _hash
from src.arc.rsi_slot_lifecycle import _create, _directory
from src.arc.rsi_transition_screen import screen_captured_rsi_chain


def prepare_final_diagnostic_proposal(*, root, first, second=None, tasks, sandbox, verifier,
                                      diagnostic_task_hashes, generation_task_ids,
                                      harness_sha256, settings_sha256, budget):
    ceilings = {'model_calls': 800, 'reserved_tokens': 16000000, 'sandbox_runs': 8000, 'wall_seconds': 36000}
    if (not _hash(harness_sha256) or not _hash(settings_sha256) or not isinstance(tasks, dict) or
            not isinstance(generation_task_ids, list) or len(generation_task_ids) != 800 or
            any(not isinstance(key, str) or not key or len(key) > 200 for key in generation_task_ids) or
            len(set(generation_task_ids)) != 800 or not isinstance(diagnostic_task_hashes, dict) or
            not 1 <= len(diagnostic_task_hashes) <= 400 or
            any(not isinstance(key, str) or not key or len(key) > 200 or not _hash(value)
                for key, value in diagnostic_task_hashes.items()) or
            not isinstance(budget, dict) or set(budget) != set(ceilings) or
            any(type(budget[key]) is not int or not 1 <= budget[key] <= limit for key, limit in ceilings.items())):
        raise ValueError('invalid diagnostic proposal inputs')
    generation = set(generation_task_ids)
    development, diagnostic = set(tasks), set(diagnostic_task_hashes)
    if generation & development or diagnostic & (generation | development):
        raise ValueError('diagnostic, development and generation tasks must be disjoint')
    screen = screen_captured_rsi_chain(first=first, second=second, tasks=tasks, sandbox=sandbox, verifier=verifier)
    if screen['provisional_selected_adapter'] == 'none':
        raise ValueError('no candidate passed development screening')
    encoded_generation = json.dumps(sorted(generation), separators=(',', ':'), ensure_ascii=False).encode()
    record = {'status': 'final_diagnostic_proposed_unverified', 'execution_ready': False,
              'selected_adapter': screen['provisional_selected_adapter'],
              'comparison_sha256': [item['comparison_sha256'] for item in screen['transitions']],
              'generation_task_ids_sha256': hashlib.sha256(encoded_generation).hexdigest(),
              'diagnostic_task_hashes': diagnostic_task_hashes, 'harness_sha256': harness_sha256,
              'settings_sha256': settings_sha256, 'budget': budget, 'competitive_target_percent': 42.4,
              'target_split': 'arc_agi_2_semi_private', 'diagnostic_split': 'arc_agi_2_public_evaluation',
              'evidence_type': 'caller_supplied_screen_and_unwitnessed_proposal'}
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with _directory(root) as directory:
        _create(directory, 'final.diagnostic.proposal.json', record)
    return {'status': record['status'], 'execution_ready': False, 'selected_adapter': record['selected_adapter'],
            'sha256': hashlib.sha256(raw).hexdigest(), 'independently_verified': False}
