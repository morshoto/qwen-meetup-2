"""Python-version-stable AST fingerprints for sealed provenance records."""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any


_VERSION_SPECIFIC_FIELDS = {'type_params'}


def _node_value(value: Any) -> Any:
    if isinstance(value, ast.AST):
        return [type(value).__name__, [
            [field, _node_value(item)] for field, item in ast.iter_fields(value)
            if field not in _VERSION_SPECIFIC_FIELDS
        ]]
    if isinstance(value, list):
        return [_node_value(item) for item in value]
    if isinstance(value, tuple):
        return [_node_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    return type(value).__name__


def stable_ast_sha256(code_or_tree: str | ast.AST) -> str:
    tree = ast.parse(code_or_tree) if isinstance(code_or_tree, str) else code_or_tree
    canonical = json.dumps(_node_value(tree), sort_keys=False, separators=(',', ':'),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class _ShapeFingerprint(ast.NodeTransformer):
    """Ignore local names and literal values to catch superficial rewrites."""
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in {'grid', 'transform'}:
            node.id = 'VARIABLE'
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg != 'grid':
            node.arg = 'VARIABLE'
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        node.value = type(node.value).__name__
        return node


def stable_shape_sha256(code: str) -> str:
    tree = ast.parse(code)
    shape = _ShapeFingerprint().visit(tree)
    ast.fix_missing_locations(shape)
    return stable_ast_sha256(shape)

