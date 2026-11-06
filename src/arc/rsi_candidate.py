"""Inspect task-conditioned generator output without asserting training eligibility."""
from __future__ import annotations

import hashlib

from src.arc.data import Task, validate_grid
from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.e6_curriculum import canonical, verify_oracle
from src.arc.e6f_repair import _static_failure, classify_failure
from src.arc.http_backend import BackendError


def _failure(status: str) -> dict:
    return {'status': status, 'eligible_for_training': False}


def inspect_candidate(task: Task, code: str, *, public_probes: list, private_probes: list,
                      sandbox: object) -> dict:
    """Run only through the supplied sandbox; private verifier details stay private.

    A passing return is not provenance, novelty, or training-gate evidence.
    Infrastructure and resource exceptions propagate to stop the caller.
    """
    if not isinstance(task, Task) or not task.train or not task.test_inputs:
        raise ValueError('task must contain demonstrations and test inputs')
    public = [validate_grid(grid) for grid in public_probes]
    private = [validate_grid(grid) for grid in private_probes]
    source = [validate_grid(grid) for grid, _ in task.train] + [validate_grid(grid) for grid in task.test_inputs]
    public_set, private_set, source_set = ({canonical(grid) for grid in group}
                                            for group in (public, private, source))
    if public_set & private_set or source_set & (public_set | private_set):
        raise ValueError('source and verification inputs overlap')
    if not isinstance(code, str):
        return _failure('static_safety_rejected')
    static_failure = _static_failure(code)
    if static_failure:
        return _failure(static_failure)
    synthetic_train = []
    try:
        for grid, target in task.train:
            first = validate_grid(sandbox.run(code, grid))
            second = validate_grid(sandbox.run(code, grid))
            if first != second:
                return _failure('nondeterministic_output_rejected')
            if first != validate_grid(target):
                return _failure('visible_demonstration_mismatch')
            synthetic_train.append({'input': grid, 'output': first})
        public_outputs = verify_oracle(code, public, sandbox=sandbox)
        if public_outputs == public:
            return _failure('identity_only_rejected')
    except (BackendError, TimeoutError):
        raise
    except (ValueError, TypeError, RuntimeError) as exc:
        category, _ = classify_failure(exc)
        if category in {'runtime_exception', 'shape_or_color_violation'}:
            return _failure(category)
        if category == 'nondeterministic_output':
            return _failure('nondeterministic_output_rejected')
        return _failure('public_validation_rejected')
    try:
        hidden_outputs = verify_oracle(code, private, sandbox=sandbox)
        if hidden_outputs == private:
            return _failure('private_acceptance_rejected')
    except (BackendError, TimeoutError):
        raise
    except (ValueError, TypeError, RuntimeError):
        return _failure('private_acceptance_rejected')
    try:
        query_outputs = []
        for grid in task.test_inputs:
            first = validate_grid(sandbox.run(code, grid))
            if first != validate_grid(sandbox.run(code, grid)):
                return _failure('nondeterministic_output_rejected')
            query_outputs.append(first)
    except (BackendError, TimeoutError):
        raise
    except (ValueError, TypeError, RuntimeError):
        return _failure('query_execution_rejected')
    return {'status': 'verified_candidate', 'eligible_for_training': False,
            'source_type': 'generator_distillation',
            'source_sha256': hashlib.sha256(code.strip().encode()).hexdigest(),
            'ast_sha256': stable_ast_sha256(code),
            'synthetic_task': {'parent_task_id': task.id, 'train': synthetic_train,
                               'query_inputs': list(task.test_inputs), 'query_outputs': query_outputs},
            'private_behavior_sha256': hashlib.sha256(canonical(hidden_outputs).encode()).hexdigest(),
            'private_probe_set_sha256': hashlib.sha256(canonical(private).encode()).hexdigest()}
