"""Scripted fixtures for CPU-only E6-F protocol tests; never used by Live."""
from __future__ import annotations

from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox

BAD = 'def transform(grid):\n    return grid[999]'
SHIFT = 'def transform(grid):\n    return [row[1:] + row[:1] for row in grid]'
PROGRAMS = (*FAKE_PROGRAMS[1:], SHIFT)


class FakeBackend:
    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        index, attempt = divmod(candidate, 3)
        if index == 0 and attempt == 0:
            return BAD
        return PROGRAMS[index % len(PROGRAMS)]


class FakeSandbox:
    def __init__(self):
        self.delegate = FakeE6Sandbox()

    def run(self, code, grid):
        if code == BAD:
            raise RuntimeError('sandbox execution failed: fixture\nIndexError: fixture')
        if code == SHIFT:
            return [row[1:] + row[:1] for row in grid]
        return self.delegate.run(code, grid)
