"""Conservative candidate, arm and experiment reservation limits for E7."""
from __future__ import annotations

from typing import Any

from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.pipelines.e5 import _canonical


class LayeredBudget:
    """Reserve before dispatch at every level; partial writes fail conservatively."""

    def __init__(self, candidate: Any, arm: Any, total: Any):
        self.candidate, self.arm, self.total = candidate, arm, total
        self.layers = (candidate, arm, total)
        self.last_exhausted: str | None = None

    @property
    def state(self) -> dict:
        return self.candidate.state

    def remaining(self) -> float:
        remaining = []
        for name, layer in zip(('candidate', 'arm', 'total'), self.layers):
            try:
                remaining.append(layer.remaining())
            except TimeoutError:
                self.last_exhausted = name
                raise TimeoutError(f'{name}_budget_exhausted: wall time') from None
        return min(remaining)

    def reserve_sandbox(self) -> None:
        self.remaining()
        for name, layer in zip(('candidate', 'arm', 'total'), self.layers):
            c, state = layer.config, layer.state
            if state['sandbox_runs'] + 1 > min(c.e5.max_sandbox_runs, c.budget.max_container_starts):
                self.last_exhausted = name
                raise TimeoutError(f'{name}_budget_exhausted: sandbox')
        for layer in self.layers:
            layer.reserve_sandbox()

    def reserve_model(self, prompt: str, kind: str) -> None:
        self.remaining()
        messages = [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}]
        for name, layer in zip(('candidate', 'arm', 'total'), self.layers):
            c, state = layer.config, layer.state
            reserved = c.model.max_tokens + len(_canonical(messages).encode('utf-8')) + 1024
            if (state['model_calls'] + 1 > min(c.e5.max_model_calls, c.budget.max_llm_calls) or
                    state['reserved_tokens'] + reserved > min(c.e5.max_reserved_tokens, c.budget.max_tokens)):
                self.last_exhausted = name
                raise TimeoutError(f'{name}_budget_exhausted: model or reserved tokens')
        for layer in self.layers:
            layer.reserve_model(prompt, kind)
