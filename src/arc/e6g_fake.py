"""Deterministic fixture-only paired comparison; never used by Live."""
from __future__ import annotations

from src.arc.e6f_fake import PROGRAMS, FakeSandbox
from src.arc.e6g_novelty import FEEDBACK_INSTRUCTION


class FakeBackend:
    def __init__(self, condition: str):
        self.condition = condition

    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        index, _ = divmod(candidate, 3)
        if self.condition == 'G1_novelty_feedback' and index > 0 and FEEDBACK_INSTRUCTION in prompt:
            return PROGRAMS[index + 1]
        return PROGRAMS[0]
