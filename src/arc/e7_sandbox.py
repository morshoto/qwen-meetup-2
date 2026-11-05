"""Metered Sandbox with privacy-preserving execution traces."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Callable


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class TracedSandbox:
    """Log only digests and error types, never grid values or sandbox output."""

    def __init__(self, sandbox: Any, budget: Any, image: str, emit: Callable[[dict], None]):
        self.sandbox, self.budget, self.image, self.emit = sandbox, budget, image, emit
        self.candidate_id = ''
        self.attempt = 0
        self.public_inputs: set[str] = set()
        self.seen: dict[tuple[str, str], int] = {}

    def run(self, code: str, grid: list[list[int]]) -> list[list[int]]:
        self.budget.reserve_sandbox()
        code_hash = hashlib.sha256(code.strip().encode()).hexdigest()
        input_hash = _digest(grid)
        key = (code_hash, input_hash)
        repeat = self.seen.get(key, 0)
        self.seen[key] = repeat + 1
        trace_id = uuid.uuid4().hex
        event = {'event': 'sandbox_dispatch', 'trace_id': trace_id, 'candidate_id': self.candidate_id,
                 'attempt': self.attempt, 'code_sha256': code_hash, 'input_digest': input_hash,
                 'image_digest': self.image, 'timestamp': time.time(),
                 'phase': 'public_validation' if input_hash in self.public_inputs else 'private_acceptance',
                 'trigger': 'initial_execution' if repeat == 0 else 'determinism_or_repeated_validation',
                 'prior_executions_same_code_input': repeat}
        self.emit(event)
        try:
            result = self.sandbox.run(code, grid)
        except Exception as exc:
            self.emit({'event': 'sandbox_finished', 'trace_id': trace_id, 'candidate_id': self.candidate_id,
                       'attempt': self.attempt, 'status': 'error', 'error_type': type(exc).__name__})
            raise
        self.emit({'event': 'sandbox_finished', 'trace_id': trace_id, 'candidate_id': self.candidate_id,
                   'attempt': self.attempt, 'status': 'ok'})
        return result
