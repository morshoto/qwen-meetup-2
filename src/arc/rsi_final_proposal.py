"""Freeze an unregistered final diagnostic proposal without executing it."""
import hashlib
import json

from src.arc.rsi_comparison_freeze import _hash
from src.arc.data import Task, validate_grid
from src.arc.rsi_slot_audit import task_sha256
from src.arc.rsi_slot_lifecycle import _create, _directory
from src.arc.rsi_transition_screen import screen_captured_rsi_chain


def _task_content(task):
    if not isinstance(task, Task) or not task.train or not task.test_inputs:
        raise ValueError('task must have visible demonstrations and query inputs')
    try:
        for source, target in task.train:
            validate_grid(source)
            validate_grid(target)
        for source in task.test_inputs:
            validate_grid(source)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid task grid') from exc
    raw = json.dumps(task.prompt_data(), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def prepare_final_diagnostic_proposal(*, root, first, second=None, tasks, sandbox, verifier,
                                      diagnostic_tasks, generation_tasks,
                                      harness_sha256, settings_sha256, budget):
    ceilings = {'model_calls': 800, 'reserved_tokens': 16000000, 'sandbox_runs': 8000, 'wall_seconds': 36000}
    if (not _hash(harness_sha256) or not _hash(settings_sha256) or not isinstance(tasks, dict) or
            not isinstance(generation_tasks, dict) or len(generation_tasks) != 800 or
            not isinstance(diagnostic_tasks, dict) or not 1 <= len(diagnostic_tasks) <= 400 or
            any(not isinstance(key, str) or not key or len(key) > 200 or not isinstance(task, Task) or task.id != key
                for group in (tasks, generation_tasks, diagnostic_tasks) for key, task in group.items()) or
            not isinstance(budget, dict) or set(budget) != set(ceilings) or
            any(type(budget[key]) is not int or not 1 <= budget[key] <= limit for key, limit in ceilings.items())):
        raise ValueError('invalid diagnostic proposal inputs')
    generation = set(generation_tasks)
    development, diagnostic = set(tasks), set(diagnostic_tasks)
    if generation & development or diagnostic & (generation | development):
        raise ValueError('diagnostic, development and generation tasks must be disjoint')
    generation_content = {key: _task_content(task) for key, task in generation_tasks.items()}
    development_content = {key: _task_content(task) for key, task in tasks.items()}
    diagnostic_content = {key: _task_content(task) for key, task in diagnostic_tasks.items()}
    if (set(generation_content.values()) & set(development_content.values()) or
            set(diagnostic_content.values()) & (set(generation_content.values()) | set(development_content.values())) or
            len(set(diagnostic_content.values())) != len(diagnostic_content)):
        raise ValueError('task prompt content overlaps across partitions')
    screen = screen_captured_rsi_chain(first=first, second=second, tasks=tasks, sandbox=sandbox, verifier=verifier)
    if screen['provisional_selected_adapter'] == 'none':
        raise ValueError('no candidate passed development screening')
    encoded_generation = json.dumps(generation_content, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    record = {'status': 'final_diagnostic_proposed_unverified', 'execution_ready': False,
              'selected_adapter': screen['provisional_selected_adapter'],
              'comparison_sha256': [item['comparison_sha256'] for item in screen['transitions']],
              'generation_task_content_manifest_sha256': hashlib.sha256(encoded_generation).hexdigest(),
              'diagnostic_task_hashes': {key: task_sha256(value) for key, value in diagnostic_tasks.items()},
              'diagnostic_content_hashes': diagnostic_content, 'harness_sha256': harness_sha256,
              'settings_sha256': settings_sha256, 'budget': budget, 'competitive_target_percent': 42.4,
              'target_split': 'arc_agi_2_semi_private', 'diagnostic_split': 'arc_agi_2_public_evaluation',
              'evidence_type': 'caller_supplied_screen_and_unwitnessed_proposal'}
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with _directory(root) as directory:
        _create(directory, 'final.diagnostic.proposal.json', record)
    return {'status': record['status'], 'execution_ready': False, 'selected_adapter': record['selected_adapter'],
            'sha256': hashlib.sha256(raw).hexdigest(), 'independently_verified': False}
