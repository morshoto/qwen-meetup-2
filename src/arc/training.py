"""Verified exploration-only records and replaceable trainer interface."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol
from src.arc.agent import AgentResult
from src.arc.data import Task, Verifier
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT


def acceptable_code(code: str) -> bool:
    """Reject malformed, oversized and obvious fixed-grid solutions without execution."""
    if not code.strip() or len(code.encode('utf-8')) > 65536:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'transform']
    if len(functions) != 1:
        return False
    for node in ast.walk(functions[0]):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.List):
            if all(isinstance(row, ast.List) and all(isinstance(cell, ast.Constant) for cell in row.elts) for row in node.value.elts):
                return False
    return True


def make_record(task: Task, result: AgentResult, verifier: Verifier, exploration: set[str], generation: int) -> dict[str, Any] | None:
    if task.id not in exploration:
        raise ValueError('only exploration tasks can enter training')
    if not result.code or not acceptable_code(result.code) or result.error or not verifier.matches(task.id, result.outputs):
        return None
    # Prompt never contains held-out test solutions or verifier feedback.
    prompt = json.dumps(task.prompt_data(), sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256((prompt + '\0' + result.code).encode()).hexdigest()
    return {'id': digest, 'task_id': task.id, 'split': 'exploration', 'generation': generation,
            'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                         {'role': 'user', 'content': prompt},
                         {'role': 'assistant', 'content': result.code}],
            'repair_history': result.history}


class TrainerBackend(Protocol):
    def train(self, records: list[dict[str, Any]], output: Path, previous: str, parameters: dict[str, Any]) -> dict[str, Any]: ...


class DryRunTrainer:
    def train(self, records: list[dict[str, Any]], output: Path, previous: str, parameters: dict[str, Any]) -> dict[str, Any]:
        if not records or len({r['id'] for r in records}) != len(records):
            raise ValueError('training records must be nonempty and unique')
        for record in records:
            messages = record.get('messages', [])
            if (record.get('split') != 'exploration' or not record.get('task_id') or len(messages) != 3 or
                    [message.get('role') for message in messages] != ['system', 'user', 'assistant'] or
                    not messages[-1].get('content')):
                raise ValueError('invalid training record')
        output.mkdir(parents=True, exist_ok=True)
        metadata = {'mode': 'dry_run', 'trained': False, 'weights_updated': False,
                    'sample_count': len(records), 'previous_checkpoint': previous, 'parameters': parameters}
        (output / 'checkpoint.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata
