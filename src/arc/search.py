"""Train-only candidate search strategies for the P3-C Exploration studies.

This module never imports the held-out verifier. Strategy selection sees only a
task's demonstrations, generated code, and sandbox feedback on those examples.
"""
from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from src.arc.agent import (AgentResult, LLMBackend, _code_analysis_prompt,
                           _code_history_text, _code_task_text, code_agent)
from src.arc.data import Task, validate_grid
from src.arc.http_backend import BackendError
from src.arc.sandbox import Sandbox


RECOVERABLE_SEARCH_ERRORS = {'malformed_thinking', 'invalid_response',
                             'missing_response', 'invalid_json',
                             'code_extraction_error'}


class SearchBudgetError(BackendError):
    """A pre-dispatch or pre-sandbox budget rejected work safely."""

    def __init__(self, message: str):
        super().__init__('budget_exhausted', message)


@dataclass
class ScopeUsage:
    calls: int = 0
    reserved_tokens: int = 0
    actual_tokens: int | None = 0
    actual_token_calls: int = 0
    sandbox_starts: int = 0
    elapsed_seconds: float = 0.0


class SearchBudget:
    """One run-wide ledger plus independent per-task/per-strategy ceilings."""

    def __init__(self, *, max_seconds: float, max_calls: int, max_tokens: int,
                 max_sandbox_starts: int, max_calls_per_task: int,
                 max_tokens_per_task: int, max_sandbox_per_task: int,
                 max_seconds_per_task: float,
                 max_candidates: int | None = None,
                 state: dict[str, Any] | None = None,
                 on_change: Callable[[dict[str, Any]], None] | None = None):
        self.limits = {'seconds': max_seconds, 'calls': max_calls, 'tokens': max_tokens,
                       'sandbox_starts': max_sandbox_starts,
                       'candidates': max_calls if max_candidates is None else max_candidates}
        self.task_limits = {'calls': max_calls_per_task, 'tokens': max_tokens_per_task,
                            'sandbox_starts': max_sandbox_per_task,
                            'seconds': max_seconds_per_task}
        self.state = state if state is not None else {
            'calls': 0, 'reserved_tokens': 0, 'actual_tokens': 0,
            'actual_token_calls': 0, 'sandbox_starts': 0,
            'candidate_calls': 0, 'elapsed_seconds': 0.0, 'scopes': {}}
        self.state.setdefault('candidate_calls', self.state.get('calls', 0))
        self.started = time.monotonic()
        self.elapsed_before = float(self.state.get('elapsed_seconds', 0.0))
        self.scope_started: dict[str, float] = {}
        self.on_change = on_change

    def _scope(self, key: str) -> dict[str, Any]:
        self.scope_started.setdefault(key, time.monotonic())
        return self.state.setdefault('scopes', {}).setdefault(key, asdict(ScopeUsage()))

    def _elapsed(self) -> float:
        return self.elapsed_before + time.monotonic() - self.started

    def remaining(self, key: str | None = None) -> float:
        left = self.limits['seconds'] - self._elapsed()
        if key is not None:
            scope = self._scope(key)
            task_elapsed = float(scope.get('elapsed_seconds', 0.0)) + time.monotonic() - self.scope_started[key]
            left = min(left, self.task_limits['seconds'] - task_elapsed)
        if left <= 0:
            raise SearchBudgetError('run or task wall-time budget exhausted')
        return left

    def _save(self) -> None:
        self.state['elapsed_seconds'] = self._elapsed()
        for key, started in self.scope_started.items():
            scope = self.state['scopes'][key]
            scope['elapsed_seconds'] = float(scope.get('elapsed_seconds', 0.0)) + time.monotonic() - started
            self.scope_started[key] = time.monotonic()
        if self.on_change:
            self.on_change(self.state)

    def reserve_call(self, key: str, amount: int) -> None:
        self.remaining(key)
        scope = self._scope(key)
        if (self.state['calls'] + 1 > self.limits['calls'] or
                self.state['candidate_calls'] + 1 > self.limits['candidates'] or
                self.state['reserved_tokens'] + amount > self.limits['tokens'] or
                scope['calls'] + 1 > self.task_limits['calls'] or
                scope['reserved_tokens'] + amount > self.task_limits['tokens']):
            raise SearchBudgetError('LLM call or reserved-token budget exhausted before dispatch')
        self.state['calls'] += 1
        self.state['candidate_calls'] += 1
        self.state['reserved_tokens'] += amount
        scope['calls'] += 1
        scope['reserved_tokens'] += amount
        self._save()

    def reserve_sandbox(self, key: str) -> None:
        self.remaining(key)
        scope = self._scope(key)
        if (self.state['sandbox_starts'] + 1 > self.limits['sandbox_starts'] or
                scope['sandbox_starts'] + 1 > self.task_limits['sandbox_starts']):
            raise SearchBudgetError('sandbox-start budget exhausted before execution')
        self.state['sandbox_starts'] += 1
        scope['sandbox_starts'] += 1
        self._save()

    def record_actual(self, key: str, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        value = usage.get('total_tokens')
        if type(value) is not int or value < 0:
            prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
            value = prompt + completion if type(prompt) is int and type(completion) is int else None
        if type(value) is int:
            self.state['actual_tokens'] = int(self.state.get('actual_tokens', 0)) + value
            self.state['actual_token_calls'] = int(self.state.get('actual_token_calls', 0)) + 1
            scope = self._scope(key)
            scope['actual_tokens'] = int(scope.get('actual_tokens') or 0) + value
            scope['actual_token_calls'] = int(scope.get('actual_token_calls', 0)) + 1
            self._save()

    def usage(self, key: str) -> dict[str, Any]:
        self.remaining(key)
        return dict(self._scope(key))

    def finish(self) -> dict[str, Any]:
        self.state['elapsed_seconds'] = self._elapsed()
        for key, started in self.scope_started.items():
            scope = self.state['scopes'][key]
            scope['elapsed_seconds'] = float(scope.get('elapsed_seconds', 0.0)) + time.monotonic() - started
            self.scope_started[key] = time.monotonic()
        return dict(self.state)


class BudgetedBackend:
    """Reserve task and run budgets before delegating any request."""
    def __init__(self, backend: LLMBackend, budget: SearchBudget, scope: str,
                 *, max_tokens: int, prompt_events: list[dict[str, Any]] | None = None,
                 before_dispatch: Callable[[dict[str, Any]], None] | None = None,
                 on_response: Callable[[dict[str, Any]], None] | None = None,
                 analysis_prompt_mode: bool = False):
        self.backend, self.budget, self.scope, self.max_tokens = backend, budget, scope, max_tokens
        self.prompt_events = prompt_events
        self.before_dispatch, self.on_response = before_dispatch, on_response
        self.analysis_prompt_mode = analysis_prompt_mode

    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        from src.arc.prompting import system_prompt_for_kind
        messages = [{'role': 'system', 'content': system_prompt_for_kind(
                        kind, analysis_mode=self.analysis_prompt_mode)},
                    {'role': 'user', 'content': prompt}]
        amount = self.max_tokens + len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
        self.budget.reserve_call(self.scope, amount)
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        if self.prompt_events is not None:
            self.prompt_events.append({'scope': self.scope, 'kind': kind,
                                       'candidate': candidate,
                                       'prompt_sha256': prompt_hash})
        if self.before_dispatch:
            self.before_dispatch({'scope': self.scope, 'kind': kind,
                                  'candidate': candidate, 'prompt_sha256': prompt_hash,
                                  'reserved_tokens': amount, 'status': 'reserved_before_dispatch'})
        if hasattr(self.backend, 'timeout'):
            self.backend.timeout = min(float(self.backend.timeout), self.budget.remaining(self.scope))
        try:
            response = self.backend.generate(prompt, kind=kind, candidate=candidate)
        except BaseException as exc:
            if self.on_response:
                self.on_response({'scope': self.scope, 'kind': kind, 'candidate': candidate,
                                  'prompt_sha256': prompt_hash,
                                  'error_kind': getattr(exc, 'kind', type(exc).__name__),
                                  'status': 'request_ended_with_error'})
            raise
        if self.on_response:
            self.on_response({'scope': self.scope, 'kind': kind, 'candidate': candidate,
                              'prompt_sha256': prompt_hash,
                              'response_sha256': hashlib.sha256(response.encode('utf-8')).hexdigest(),
                              'status': 'response_received'})
        return response


class BudgetedSandbox:
    def __init__(self, sandbox: Sandbox, budget: SearchBudget, scope: str):
        self.sandbox, self.budget, self.scope = sandbox, budget, scope

    def run(self, code: str, grid: list[list[int]]) -> list[list[int]]:
        self.budget.reserve_sandbox(self.scope)
        if hasattr(self.sandbox, 'remaining'):
            self.sandbox.remaining = lambda: self.budget.remaining(self.scope)
        if hasattr(self.sandbox, 'timeout'):
            self.sandbox.timeout = min(float(self.sandbox.timeout), self.budget.remaining(self.scope))
        return self.sandbox.run(code, grid)


@dataclass
class StrategyRun:
    result: AgentResult
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_index: int | None = None
    strategy: str = ''


def _strip_code_fence(text: str) -> str:
    import re
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)\n```', text, flags=re.DOTALL | re.IGNORECASE)
    return blocks[0].strip() if len(blocks) == 1 else text.strip()


def static_code_check(code: str) -> tuple[bool, list[str], int | None]:
    """Conservative static checks; this is a heuristic, never a proof of novelty."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False, ['invalid_python'], None
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'transform']
    if len(funcs) != 1:
        return False, ['missing_transform'], None
    reasons: list[str] = []
    forbidden_names = {'open', 'exec', 'eval', 'compile', '__import__', 'input', 'getattr', 'globals',
                       'locals', 'vars', 'breakpoint'}
    forbidden_modules = {'os', 'sys', 'pathlib', 'socket', 'requests', 'urllib', 'subprocess',
                         'http', 'ftplib', 'shutil'}
    constant_grid_names: set[str] = set()

    def is_grid_literal(node: ast.AST) -> bool:
        if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
            return False
        return all(isinstance(row, (ast.List, ast.Tuple)) and bool(row.elts) and
                   all(isinstance(cell, ast.Constant) and type(cell.value) is int
                       for cell in row.elts) for row in node.elts)

    for node in ast.walk(funcs[0]):
        if isinstance(node, ast.Assign) and is_grid_literal(node.value):
            constant_grid_names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            reasons.append('imports_forbidden')
        if isinstance(node, ast.Name):
            if node.id in forbidden_names:
                reasons.append('dynamic_or_external_access')
            if node.id.lower() in {'task_id', 'taskid'}:
                reasons.append('task_identifier_reference')
            if node.id.startswith('__') and node.id.endswith('__'):
                reasons.append('dunder_access')
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in forbidden_modules:
            reasons.append('external_access')
        if isinstance(node, ast.Attribute) and node.attr.startswith('__') and node.attr.endswith('__'):
            reasons.append('dunder_access')
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.lower() in {'task_id', 'taskid'}:
            reasons.append('task_identifier_reference')
    returned: list[ast.AST] = []
    for node in ast.walk(funcs[0]):
        if isinstance(node, ast.Return):
            returned.append(node.value or ast.Constant(None))
    if returned and any(is_grid_literal(node) or isinstance(node, ast.Name) and node.id in constant_grid_names
                        for node in returned):
        reasons.append('constant_output')
    if any(isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == 'grid' and
           any(is_grid_literal(value) for value in node.comparators)
           for node in ast.walk(tree)):
        reasons.append('input_keyed_hardcode')
    complexity = sum(1 for _ in ast.walk(tree))
    return not reasons, sorted(set(reasons)), complexity


def _feedback_for_candidate(task: Task, code: str, sandbox: Sandbox) -> dict[str, Any]:
    valid, reasons, complexity = static_code_check(code)
    examples: list[dict[str, Any]] = []
    execution_errors = 0
    exact = 0
    if valid:
        for index, (input_grid, expected) in enumerate(task.train):
            try:
                actual = validate_grid(sandbox.run(code, input_grid))
                matched = actual == expected
                exact += int(matched)
                examples.append({'example': index, 'matched': matched, 'actual': actual,
                                 'expected': expected, 'error': None})
            except BackendError:
                raise
            except (ValueError, RuntimeError, TimeoutError) as exc:
                execution_errors += 1
                examples.append({'example': index, 'matched': False, 'actual': None,
                                 'expected': expected, 'error_kind': 'timeout' if isinstance(exc, TimeoutError) else 'sandbox_error'})
    return {'train_match_count': exact, 'train_example_count': len(task.train),
            'execution_errors': execution_errors, 'static_filter_pass': valid,
            'static_filter_reasons': reasons, 'code_complexity': complexity,
            'examples': examples, 'complete_train_match': bool(valid and execution_errors == 0 and exact == len(task.train))}


def _independent_candidates(task: Task, backend: LLMBackend, sandbox: Sandbox, *,
                            candidate_count: int, base_candidate: int, strategy: str) -> StrategyRun:
    candidates: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index in range(candidate_count):
        instruction = (
            'Derive a transformation from the public training pairs and check it against every training pair. '
            'Write a complete Python '
            'function named transform(grid) returning rectangular integer rows. Do not import modules or use task IDs, '
            'external state, fixed grids, or exact test answers. Return only source code. This is an independent '
            f'candidate {index + 1} of {candidate_count}; solve from the training pairs without copying another response.'
        )
        prompt = instruction + '\n\n' + _code_task_text(task)
        try:
            code = _strip_code_fence(backend.generate(prompt, kind='code', candidate=base_candidate + index))
        except BackendError as exc:
            if exc.kind not in RECOVERABLE_SEARCH_ERRORS:
                raise
            candidates.append({'candidate_index': index, 'code': None,
                               'code_sha256': None, 'parse_status': 'error',
                               'error_kind': exc.kind, 'train_match_count': 0,
                               'train_example_count': len(task.train),
                               'execution_errors': 0, 'static_filter_pass': False,
                               'static_filter_reasons': [exc.kind],
                               'complete_train_match': False})
            continue
        except (ValueError, RuntimeError, TimeoutError) as exc:
            candidates.append({'candidate_index': index, 'code': None,
                               'code_sha256': None, 'parse_status': 'error',
                               'error_kind': 'timeout' if isinstance(exc, TimeoutError) else 'generation_error',
                               'train_match_count': 0, 'train_example_count': len(task.train),
                               'execution_errors': 0, 'static_filter_pass': False,
                               'static_filter_reasons': ['generation_error'],
                               'complete_train_match': False})
            continue
        digest = hashlib.sha256(code.encode()).hexdigest()
        if digest in seen:
            candidates.append({'candidate_index': index, 'code': code, 'code_sha256': digest,
                               'parse_status': 'duplicate', 'duplicate_of': seen[digest],
                               'train_match_count': 0, 'execution_errors': 0,
                               'static_filter_pass': False, 'complete_train_match': False})
            continue
        seen[digest] = index
        try:
            ast.parse(code)
            parsed = True
        except (SyntaxError, ValueError, RecursionError):
            parsed = False
        feedback = _feedback_for_candidate(task, code, sandbox) if parsed else {
            'train_match_count': 0, 'train_example_count': len(task.train),
            'execution_errors': 0, 'static_filter_pass': False,
            'static_filter_reasons': ['invalid_python'], 'code_complexity': None,
            'examples': [], 'complete_train_match': False}
        candidates.append({'candidate_index': index, 'code': code, 'code_sha256': digest,
                           'parse_status': 'parsed' if parsed else 'error', **feedback})
    eligible = [row for row in candidates if row.get('parse_status') == 'parsed' and
                row.get('code') is not None and row.get('static_filter_pass') is True]
    if strategy == 'S1':
        # S1 is a first-runnable baseline; train match scores are logged but
        # deliberately do not affect which independent candidate is selected.
        eligible = [row for row in eligible if row.get('execution_errors') == 0]
        key = lambda row: row['candidate_index']
    else:
        # S3 selects only by public train execution, then deterministic index.
        key = lambda row: (-row['train_match_count'], row['execution_errors'],
                           row['candidate_index'])
    selected = min(eligible, key=key) if eligible else None
    if selected is None or not selected['static_filter_pass']:
        error = ('no candidate passed static filter, parsing, and train execution'
                 if strategy == 'S1' else 'no candidate passed static filter and Python parsing')
        return StrategyRun(AgentResult([], None, [], error, 'invalid'),
                           candidates, None, strategy)
    code = selected['code']
    outputs: list[Any] = []
    try:
        outputs = [validate_grid(sandbox.run(code, grid)) for grid in task.test_inputs]
    except BackendError:
        raise
    except (ValueError, RuntimeError, TimeoutError) as exc:
        return StrategyRun(AgentResult([], None, [], str(exc),
                                       'timeout' if isinstance(exc, TimeoutError) else 'invalid',
                                       ), candidates,
                           selected['candidate_index'], strategy)
    history = [{'code': row.get('code') or '',
                'feedback': [item for item in row.get('examples', []) if not item.get('matched')],
                'candidate_index': row['candidate_index'],
                'duplicate_of': row.get('duplicate_of')}
               for row in candidates]
    return StrategyRun(AgentResult(outputs, code, history), candidates,
                       selected['candidate_index'], strategy)


def run_strategy(task: Task, strategy: str, backend: LLMBackend, sandbox: Sandbox, *,
                 max_repairs: int, independent_candidates: int = 3,
                 base_candidate: int = 0) -> StrategyRun:
    """Run one pre-registered strategy; no verifier is accepted by this API."""
    if strategy not in {'S0', 'S1', 'S2', 'S3'}:
        raise ValueError('unsupported search strategy')
    if strategy in {'S0', 'S2'}:
        result = code_agent(task, backend, sandbox, max_repairs, base_candidate)
        return StrategyRun(result, strategy=strategy)
    return _independent_candidates(task, backend, sandbox,
                                   candidate_count=independent_candidates,
                                   base_candidate=base_candidate, strategy=strategy)

