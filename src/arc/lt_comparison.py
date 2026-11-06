"""Offline utilities for the separate LT exploratory ARC comparison.

A manifest is a proposal until the operator freezes and verifies all runtime
identities; these utilities do not authorize or start inference.
"""
from __future__ import annotations

import random
import copy
import hashlib
import json
import math
import re
from typing import Any

from src.arc.agent import code_agent
from src.arc.http_backend import BackendError
from src.arc.data import Verifier
from src.arc.scoring import score


def make_manifest(ids: list[str], *, seed: int, development: set[str],
                  training: set[str], exposures: set[str], dataset_sha256: str) -> dict[str, Any]:
    if (type(seed) is not int or not isinstance(ids, list) or
            any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)) or
            not isinstance(dataset_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', dataset_sha256)):
        raise ValueError('invalid task IDs, seed, or dataset digest')
    universe = set(ids)
    if any(not isinstance(group, set) or not group <= universe for group in (development, training, exposures)):
        raise ValueError('invalid exclusions')
    if development & training:
        raise ValueError('development and training exclusions overlap')
    eligible = sorted(universe - development - training - exposures)
    if len(eligible) < 200:
        raise ValueError('fewer than 200 eligible tasks')
    random.Random(seed).shuffle(eligible)
    return {'protocol': 'lt-phase-a-exploratory', 'status': 'proposal', 'seed': seed,
            'dataset_sha256': dataset_sha256, 'task_ids': sorted(eligible[:200]),
            'development_exclusions': sorted(development), 'training_exclusions': sorted(training),
            'known_exposure_exclusions': sorted(exposures),
            'unknown_pretraining_exposure': True, 'unknown_project_exposure': True}


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def freeze_manifest(proposal: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    required = {'prompt_sha256', 'code_sha256', 'models', 'thinking', 'temperature',
                'candidates', 'max_tokens', 'timeout_seconds'}
    ids = proposal.get('task_ids')
    if (not isinstance(ids, list) or len(ids) != 200 or
            any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != 200 or
            any(set(ids) & set(proposal.get(k, [])) for k in
                ('development_exclusions', 'training_exclusions', 'known_exposure_exclusions'))):
        raise ValueError('invalid comparison task selection')
    if set(config) != required or proposal.get('status') != 'proposal':
        raise ValueError('invalid comparison proposal or configuration')
    if (any(not isinstance(config[k], str) or not re.fullmatch(r'[0-9a-f]{64}', config[k])
            for k in ('prompt_sha256', 'code_sha256')) or
            not isinstance(config['models'], dict) or set(config['models']) != {'14b', '32b'} or
            any(not isinstance(v, str) or not v for v in config['models'].values()) or
            config['thinking'] not in {'on', 'off'} or config['candidates'] != 2 or
            type(config['max_tokens']) is not int or config['max_tokens'] <= 0 or
            type(config['timeout_seconds']) not in (int, float) or not math.isfinite(config['timeout_seconds']) or config['timeout_seconds'] <= 0 or
            type(config['temperature']) not in (int, float) or not math.isfinite(config['temperature']) or config['temperature'] <= 0):
        raise ValueError('invalid comparison configuration')
    value = {'manifest': copy.deepcopy(proposal), 'config': copy.deepcopy(config)}
    return {**value, 'sha256': _digest(value)}


def verify_manifest(frozen: dict[str, Any]) -> None:
    try:
        expected = freeze_manifest(frozen['manifest'], frozen['config'])
        if frozen != expected:
            raise ValueError('comparison manifest digest differs')
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid frozen comparison') from exc


def run_task(task: Any, backend: Any, sandbox: Any, *, max_repairs: int) -> dict[str, Any]:
    """Exercise the existing solver; caller supplies the isolated execution boundary."""
    if type(max_repairs) is not int or max_repairs < 0:
        raise ValueError('invalid repair limit')
    candidates: list[Any] = []
    for candidate in range(2):
        try:
            result = code_agent(task, backend, sandbox, max_repairs, candidate)
        except BackendError as exc:
            return {'task_id': task.id, 'status': 'error', 'candidates': candidates,
                    'error': str(exc)}
        if result.error:
            candidates.append('timeout' if result.error_kind == 'timeout' else None)
        else:
            candidates.append(result.outputs)
    return {'task_id': task.id, 'status': 'complete', 'candidates': candidates}


def run_bounded_task(task: Any, backend: Any, sandbox: Any, budget: Any, *,
                     max_tokens: int, max_repairs: int) -> dict[str, Any]:
    """Apply existing in-process reservations; this does not prove containment."""
    from src.arc.search import BudgetedBackend, BudgetedSandbox, SearchBudget
    if (not isinstance(budget, SearchBudget) or type(max_tokens) is not int or max_tokens <= 0 or
            (hasattr(backend, 'max_tokens') and backend.max_tokens != max_tokens) or
            getattr(backend, 'reserve', None) is not None or
            getattr(sandbox, 'reserve', None) is not None):
        raise ValueError('invalid or conflicting budget configuration')
    scope = task.id
    return run_task(task, BudgetedBackend(backend, budget, scope, max_tokens=max_tokens),
                    BudgetedSandbox(sandbox, budget, scope), max_repairs=max_repairs)


def summarize(task_ids: list[str], rows: list[dict[str, Any]], verifier: Verifier) -> dict[str, Any]:
    if len(task_ids) != len(set(task_ids)):
        raise ValueError('duplicate selected task')
    selected = set(task_ids)
    seen: set[str] = set()
    pass1 = pass2 = correct_outputs = output_count = errors = 0
    for row in rows:
        task_id = row['task_id']
        if task_id not in selected or task_id in seen:
            raise ValueError('duplicate or unselected result')
        seen.add(task_id)
        candidates = row['candidates']
        if not isinstance(candidates, list) or len(candidates) > 2:
            raise ValueError('invalid candidate list')
        if row['status'] != 'complete' or len(candidates) != 2:
            errors += 1
        result = score(verifier, task_id, candidates)
        pass1 += result.pass_at_1
        pass2 += result.pass_at_2
        output_count += verifier.output_count(task_id)
        if candidates:
            correct_outputs += verifier.correct_outputs(task_id, candidates[0])
    return {'status': 'complete' if len(seen) == len(selected) and not errors else 'incomplete',
            'selected': len(selected), 'attempted': len(seen), 'errors': errors, 'pass_at_1': pass1,
            'pass_at_2': pass2, 'first_candidate_correct_outputs': correct_outputs,
            'test_outputs_attempted': output_count}
