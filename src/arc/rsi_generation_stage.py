"""Run scheduled collection slots with local, unauthenticated evidence.

The caller must separately establish registration, model, sandbox and server
identity and obtain any required authorization before live dispatch.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from src.arc.data import Task, validate_grid
from src.arc.e6_curriculum import canonical
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_slot_collector import collect_slot
from src.arc.rsi_slot_lifecycle import _create, _directory
from src.arc.rsi_task_generation import generation_slots


def _pool_sha(pool):
    return hashlib.sha256(json.dumps(pool, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


_BUDGET_LIMITS = {'model_calls': 32, 'reserved_tokens': 200000, 'sandbox_runs': 300}


def _counters(state):
    return {key: state[key] for key in _BUDGET_LIMITS}


def _budget_delta(before, after):
    for item in (before, after):
        if (not isinstance(item, dict) or set(item) != set(_BUDGET_LIMITS) or
                any(type(item[key]) is not int or not 0 <= item[key] <= limit
                    for key, limit in _BUDGET_LIMITS.items())):
            raise ValueError('invalid generation budget counters')
    delta = {key: after[key] - before[key] for key in _BUDGET_LIMITS}
    if any(value < 0 for value in delta.values()):
        raise ValueError('generation budget counters regressed')
    return delta


def collect_generation(*, root, registration_sha256, generation, parent_adapter, task_ids, tasks,
                       expected_manifest_sha256, expected_task_hashes, public_probes, private_probes,
                       expected_public_probes_sha256, expected_private_probes_sha256, reference_pool,
                       expected_reference_pool_sha256, transport, sandbox, budget,
                       model_key='14b', expected_reference_setup_sha256=None):
    slots = generation_slots(task_ids, generation)
    selected = [item['task_id'] for item in slots]
    if ((generation == 0 and parent_adapter != 'none') or
            (generation == 1 and (not isinstance(parent_adapter, str) or not re.fullmatch(r'[0-9a-f]{64}', parent_adapter))) or
            type(budget) is not CollectionBudget or
            budget.identity != {'registration_sha256': registration_sha256, 'generation': generation} or
            Path(budget.root).resolve() != Path(root).resolve() or
            manifest_sha256(task_ids) != expected_manifest_sha256 or
            not isinstance(tasks, dict) or not isinstance(expected_task_hashes, dict) or
            set(expected_task_hashes) != set(selected) or
            any(not isinstance(tasks.get(key), Task) or tasks[key].id != key or
                task_sha256(tasks[key]) != expected_task_hashes[key] for key in selected)):
        raise ValueError('generation task, parent or budget identity differs')
    if (probes_sha256(public_probes) != expected_public_probes_sha256 or
            probes_sha256(private_probes) != expected_private_probes_sha256):
        raise ValueError('generation probes differ')
    public = {canonical(grid) for grid in public_probes}
    private = {canonical(grid) for grid in private_probes}
    if len(public) != len(public_probes) or len(private) != len(private_probes) or public & private:
        raise ValueError('generation probes must be distinct')
    for key in selected:
        task = tasks[key]
        if not task.train or not task.test_inputs:
            raise ValueError('generation task is empty')
        for grid, target in task.train:
            if canonical(validate_grid(grid)) in public | private:
                raise ValueError('generation input overlaps probes')
            validate_grid(target)
        for grid in task.test_inputs:
            if canonical(validate_grid(grid)) in public | private:
                raise ValueError('generation input overlaps probes')
    if not isinstance(reference_pool, list) or _pool_sha(reference_pool) != expected_reference_pool_sha256:
        raise ValueError('reference pool differs')
    screen_novelty('def transform(grid): return grid', '0' * 64, expected_private_probes_sha256, reference_pool)
    pool = [dict(item) for item in reference_pool]
    state = budget.snapshot()
    if model_key not in ('14b', '32b'):
        raise ValueError('invalid model identity')
    reference_digest = state.get('reference_setup_sha256')
    if (reference_digest != expected_reference_setup_sha256 or
            (reference_digest is not None and
             (state.get('reference_model_key') != model_key or
              state.get('reference_pool_sha256') != expected_reference_pool_sha256))):
        raise ValueError('reference setup budget identity differs')
    budget_before = _counters(state)
    start = {'registration_sha256': registration_sha256, 'generation': generation,
             'parent_adapter': parent_adapter, 'manifest_sha256': expected_manifest_sha256,
             'task_hashes': expected_task_hashes, 'public_probe_sha256': expected_public_probes_sha256,
             'private_probe_sha256': expected_private_probes_sha256,
             'initial_reference_pool_sha256': expected_reference_pool_sha256, 'budget_before': budget_before}
    if reference_digest is not None:
        start.update(reference_setup_sha256=reference_digest, reference_model_key=model_key)
    with _directory(root) as directory:
        _create(directory, f'g{generation}.stage.started.json', start)
    completed, counts, attempt_counts = [], Counter(), Counter()
    try:
        for item in slots:
            index, key = item['slot'], item['task_id']
            result = collect_slot(root=root, registration_sha256=registration_sha256, generation=generation,
                slot=index, task_ids=task_ids, task=tasks[key], expected_manifest_sha256=expected_manifest_sha256,
                expected_task_sha256=expected_task_hashes[key], public_probes=public_probes,
                private_probes=private_probes, expected_public_probes_sha256=expected_public_probes_sha256,
                expected_private_probes_sha256=expected_private_probes_sha256, reference_pool=pool,
                transport=transport, sandbox=sandbox, budget=budget)
            with _directory(root) as directory:
                raw, value = _read(directory, f'g{generation}-slot-{index:02d}.collection.json')
            if hashlib.sha256(raw).hexdigest() != result['evidence_sha256'] or value.get('status') != result['status']:
                raise ValueError('completed slot evidence differs')
            if result['status'] == 'novel_unverified':
                novelty = value['novelty']
                pool.append({name: novelty[name] for name in
                             ('source_sha256', 'ast_sha256', 'behavior_sha256', 'probe_set_sha256')})
            counts[result['status']] += 1
            attempt_counts.update(attempt['status'] for attempt in value['attempts'])
            completed.append({'slot': index, 'task_id': key, 'status': result['status'],
                              'evidence_sha256': result['evidence_sha256']})
        budget_after = _counters(budget.snapshot())
        usage = _budget_delta(budget_before, budget_after)
        report = {'status': 'generation_complete_unverified', 'completed_slots': len(completed),
                  'slots': completed, 'counts': dict(counts), 'attempt_counts': dict(attempt_counts),
                  'final_reference_pool_sha256': _pool_sha(pool),
                  'budget_after': budget_after, 'budget_usage': usage,
                  'training_allowed': False, 'evidence_type': 'caller_supplied_components_unverified'}
        with _directory(root) as directory:
            _create(directory, f'g{generation}.stage.completed.json', report)
        return report
    except Exception as exc:
        try:
            budget_after = _counters(budget.snapshot())
            usage = _budget_delta(budget_before, budget_after)
        except Exception:
            budget_after, usage = None, None
        report = {'status': 'generation_incomplete', 'completed_slots': len(completed), 'slots': completed,
                  'counts': dict(counts), 'attempt_counts': dict(attempt_counts),
                  'failure_type': type(exc).__name__, 'training_allowed': False,
                  'budget_after': budget_after, 'budget_usage': usage,
                  'evidence_type': 'caller_supplied_components_unverified'}
        with _directory(root) as directory:
            _create(directory, f'g{generation}.stage.incomplete.json', report)
        return report
