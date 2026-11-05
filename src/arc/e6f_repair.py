"""Bounded model-authored oracle repair. Generated code runs only in a sandbox."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any, Callable

from src.arc.data import Task
from src.arc.e5_selfplay import _generator_prompt, _safe_transform_code
from src.arc.e6_curriculum import _quality_grids, make_distillation_record, verify_oracle
from src.arc.http_backend import BackendError


def _hash(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def repair_prompt(code: str, category: str, exception_type: str | None) -> str:
    """Only the model's prior code and an allowlisted error class leave validation."""
    safe_types = {'IndexError', 'TypeError', 'ValueError', 'KeyError', 'ZeroDivisionError',
                  'NameError', 'AttributeError', 'OverflowError'}
    error = exception_type if exception_type in safe_types else category
    return (
        'Repair your previous Python function. Return only one complete Python function '
        'named transform(grid), with one argument. The input and output must each be '
        'a nonempty rectangular list of lists of integers from 0 through 9; output '
        'dimensions must be at most 30 by 30. The function must be deterministic, '
        'use its input, and work on rectangular inputs. Do not import modules, use '
        'files, network, dynamic execution, or external dependencies.\n'
        f'The function failed validation with: {error}.\n'
        'Your immediately previous function:\n' + code
    )


def _static_failure(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return 'syntax_error'
    if _safe_transform_code(code):
        return None
    forbidden_names = {'eval', 'exec', 'compile', 'open', '__import__', 'input', 'globals',
                       'locals', 'getattr', 'setattr', 'delattr', 'vars', 'dir'}
    if any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef)) or
           (isinstance(n, ast.Name) and (n.id in forbidden_names or n.id.startswith('__'))) or
           (isinstance(n, ast.Attribute) and n.attr.startswith('__')) for n in ast.walk(tree)):
        return 'prohibited_operation'
    # Reject other static-filter failures conservatively; only an invalid function
    # signature or missing function is eligible for a generic contract repair.
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != 'transform':
        return 'parse_or_contract_error'
    f = functions[0]
    if ([a.arg for a in f.args.args] != ['grid'] or f.args.vararg or f.args.kwarg or
            f.args.kwonlyargs or f.args.posonlyargs or f.decorator_list):
        return 'parse_or_contract_error'
    return 'static_safety_rejected'


def classify_failure(exc: Exception) -> tuple[str, str | None]:
    if isinstance(exc, BackendError):
        return 'sandbox_infrastructure_error', None
    if isinstance(exc, TimeoutError):
        if 'budget exhausted' in str(exc) or 'budget' in str(exc).lower():
            return 'budget_exhausted', None
        return 'timeout_or_resource_limit', None
    text = str(exc)
    if text.startswith('sandbox execution failed:'):
        match = re.search(r'(?m)^([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception))(?::|$)', text)
        return 'runtime_exception', match.group(1) if match else None
    if 'nondeterministic' in text:
        return 'nondeterministic_output', None
    if any(term in text for term in ('grid', 'output_size', 'shape', 'values', 'cells')):
        return 'shape_or_color_violation', None
    return 'validation_failure', None


REPAIRABLE = {'syntax_error', 'parse_or_contract_error', 'runtime_exception',
              'shape_or_color_violation'}


def run_candidate(*, index: int, max_repairs: int, generate: Callable, sandbox: Any,
                  emit: Callable[[dict[str, Any]], None], counters: Callable[[], dict[str, Any]],
                  probe_inputs: list, generation: int = 0) -> dict[str, Any]:
    """Generate, validate, and optionally repair one lineage; hidden failures reject only."""
    seen: set[str] = set()
    previous_code = ''
    category, exception_type = '', None
    parent = None
    baseline_pass = False
    for attempt in range(max_repairs + 1):
        trial_id = f'candidate-{index}-attempt-{attempt}'
        before = dict(counters())
        prompt = _generator_prompt() if attempt == 0 else repair_prompt(previous_code, category, exception_type)
        emit({'event': 'attempt_started', 'trial_id': trial_id, 'parent_trial_id': parent,
              'candidate_index': index, 'attempt': attempt})
        try:
            code, provenance = generate(prompt, index, attempt)
        except Exception as exc:
            category = 'budget_exhausted' if isinstance(exc, TimeoutError) else 'model_request_failed'
            emit({'event': 'attempt_finished', 'trial_id': trial_id, 'category': category,
                  'candidate_index': index, 'attempt': attempt, 'parent_trial_id': parent,
                  'cost_before': before, 'cost_after': dict(counters())})
            return {'status': category, 'baseline_pass': False, 'repair_pass': False}
        if not isinstance(code, str) or not code.strip():
            emit({'event': 'attempt_finished', 'trial_id': trial_id, 'category': 'invalid_model_response',
                  'candidate_index': index, 'attempt': attempt, 'parent_trial_id': parent,
                  'cost_before': before, 'cost_after': dict(counters())})
            return {'status': 'invalid_model_response', 'baseline_pass': False, 'repair_pass': False}
        code = code.strip()
        code_hash = _hash(code)
        if code_hash in seen:
            emit({'event': 'attempt_finished', 'trial_id': trial_id, 'category': 'repeated_code',
                  'candidate_index': index, 'attempt': attempt, 'code_sha256': code_hash, 'parent_trial_id': parent,
                  'cost_before': before, 'cost_after': dict(counters())})
            return {'status': 'repeated_code', 'baseline_pass': False, 'repair_pass': False}
        seen.add(code_hash)
        category = _static_failure(code) or ''
        exception_type = None
        stage = 'static'
        public_inputs = _quality_grids(71000 + index * 100 + attempt, 5)
        if {_hash(grid) for grid in public_inputs} & {_hash(grid) for grid in probe_inputs}:
            raise ValueError('public and private verification inputs overlap')
        public_outputs = None
        if not category:
            try:
                stage = 'public_validation'
                public_outputs = verify_oracle(code, public_inputs, sandbox=sandbox)
                if public_outputs == public_inputs:
                    category = 'identity_only_rejected'
            except Exception as exc:
                category, exception_type = classify_failure(exc)
        if not category:
            try:
                stage = 'private_acceptance'
                private_outputs = verify_oracle(code, probe_inputs, sandbox=sandbox)
                if private_outputs == probe_inputs:
                    category = 'identity_only_rejected'
            except Exception as exc:
                category, exception_type = classify_failure(exc)
            if category and category not in {'budget_exhausted', 'timeout_or_resource_limit',
                                              'sandbox_infrastructure_error'}:
                category = 'private_acceptance_rejected'
        passed = not category
        if attempt == 0:
            baseline_pass = passed
        emit({'event': 'attempt_finished', 'trial_id': trial_id, 'parent_trial_id': parent,
              'candidate_index': index, 'attempt': attempt, 'code_sha256': code_hash,
              'category': 'passed' if passed else category, 'exception_type': exception_type,
              'validation_stage': stage, 'cost_before': before, 'cost_after': dict(counters()),
              'request_id': provenance.get('request_id') if isinstance(provenance, dict) else None})
        if passed:
            task = Task('e6f-' + _hash({'candidate': index, 'code_sha256': code_hash,
                                       'inputs': public_inputs})[:24],
                        tuple((public_inputs[i], public_outputs[i]) for i in range(3)),
                        (public_inputs[3],))
            record = make_distillation_record(task, code, provenance=provenance,
                heldout_inputs=probe_inputs, heldout_outputs=private_outputs, generation=generation)
            return {'status': 'passed', 'record': record, 'behavior_sha256': _hash(private_outputs),
                    'baseline_pass': baseline_pass, 'repair_pass': attempt > 0,
                    'attempts': attempt + 1}
        if stage == 'private_acceptance' or category not in REPAIRABLE:
            return {'status': category, 'baseline_pass': baseline_pass, 'repair_pass': False}
        previous_code, parent = code, trial_id
    return {'status': 'repair_limit', 'baseline_pass': baseline_pass, 'repair_pass': False}
