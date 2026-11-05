"""Restricted C2 search-plugin proposal interface (Fake-only in E5 v1)."""
from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


PLUGIN_FUNCTION = 'search_strategy'
_NAME = re.compile(r'^[a-z][a-z0-9_]{0,47}$')
_FORBIDDEN_CALLS = {'open', 'eval', 'exec', 'compile', '__import__', 'input', 'globals', 'locals'}


@dataclass(frozen=True)
class PluginDecision:
    name: str
    path: str
    sha256: str
    syntax_valid: bool
    sandbox_status: str
    accepted: bool
    reason: str


class PluginEvaluator(Protocol):
    def evaluate(self, source: str) -> tuple[bool, str]: ...


def validate_plugin_source(source: str) -> tuple[bool, str]:
    if not isinstance(source, str) or not source.strip() or len(source.encode('utf-8')) > 32768:
        return False, 'plugin source is empty or oversized'
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return False, 'plugin source is not valid Python'
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == PLUGIN_FUNCTION]
    if (len(functions) != 1 or len(tree.body) != 1 or
            functions[0].decorator_list or functions[0].args.vararg or functions[0].args.kwarg or
            [arg.arg for arg in functions[0].args.args] != ['task_context']):
        return False, 'plugin must define only search_strategy(task_context)'
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                             ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False, 'plugin uses a forbidden Python construct'
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            return False, 'plugin calls a forbidden capability'
        if ((isinstance(node, ast.Attribute) and node.attr.startswith('__')) or
                (isinstance(node, ast.Name) and node.id.startswith('__'))):
            return False, 'plugin uses a forbidden reflective attribute'
    return True, 'static contract accepted'


class FakePluginEvaluator:
    """Deterministic contract fixture; it does not execute candidate code."""
    def evaluate(self, source: str) -> tuple[bool, str]:
        valid, reason = validate_plugin_source(source)
        return (valid, 'fake sandbox accepted contract') if valid else (False, reason)


def evaluate_and_store(root: str | Path, name: str, source: str,
                       evaluator: PluginEvaluator) -> PluginDecision:
    """Write proposals only below plugins/proposals, then run the supplied evaluator."""
    if not _NAME.fullmatch(name):
        raise ValueError('plugin proposal name is invalid')
    root_path = Path(root).expanduser().resolve()
    proposal_root = (root_path / 'plugins' / 'proposals').resolve()
    if not proposal_root.is_relative_to(root_path):
        raise ValueError('plugin proposal path escaped the E5 run root')
    proposal_root.mkdir(parents=True, exist_ok=True)
    target = (proposal_root / f'{name}.py').resolve()
    if not target.is_relative_to(proposal_root) or target.exists():
        raise ValueError('plugin proposal path is outside its allowlist or already exists')
    valid, reason = validate_plugin_source(source)
    if not valid:
        sandbox_ok, sandbox_reason = False, reason
    else:
        sandbox_ok, sandbox_reason = evaluator.evaluate(source)
    if target.is_symlink():
        raise ValueError('plugin proposal target cannot be a symlink')
    temporary = target.with_suffix('.py.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(source)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return PluginDecision(name=name, path=str(target), sha256=hashlib.sha256(source.encode()).hexdigest(),
                          syntax_valid=valid, sandbox_status='fake' if isinstance(evaluator, FakePluginEvaluator) else 'external',
                          accepted=bool(valid and sandbox_ok), reason=sandbox_reason)
