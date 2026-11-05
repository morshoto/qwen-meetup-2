"""CPU-only E6 protocol fixtures. Never imported by live E6 execution."""
from __future__ import annotations

from src.arc.data import Grid


FAKE_PROGRAMS = (
    'def transform(grid):\n    return [row[:] for row in grid]',
    'def transform(grid):\n    return [row[::-1] for row in grid]',
    'def transform(grid):\n    return grid[::-1]',
    'def transform(grid):\n    return [list(row) for row in zip(*grid)]',
    'def transform(grid):\n    return [[(value + 1) % 10 for value in row] for row in grid]',
)


class FakeE6Backend:
    def __init__(self, programs: tuple[str, ...] = FAKE_PROGRAMS):
        self.programs = programs
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        self.calls.append({'prompt': prompt, 'kind': kind, 'candidate': candidate})
        if kind == 'e5_generator':
            return self.programs[candidate % len(self.programs)]
        if kind == 'e5_self_solve':
            return self.programs[(candidate + 1) % len(self.programs)]
        if kind == 'direct':
            return '[]'
        return ''


class FakeE6Sandbox:
    """Fixture dispatcher; it maps known source strings to fixed operations."""

    def run(self, code: str, grid: Grid) -> Grid:
        if code == FAKE_PROGRAMS[0]:
            return [row[:] for row in grid]
        if code == FAKE_PROGRAMS[1]:
            return [row[::-1] for row in grid]
        if code == FAKE_PROGRAMS[2]:
            return grid[::-1]
        if code == FAKE_PROGRAMS[3]:
            return [list(row) for row in zip(*grid)]
        if code == FAKE_PROGRAMS[4]:
            return [[(value + 1) % 10 for value in row] for row in grid]
        raise ValueError('unknown fake E6 fixture')


class FakeE6Trainer:
    def __init__(self, *, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def train(self, records, *, steps: int, prior_adapter_hash: str = 'none'):
        self.calls += 1
        if self.fail:
            return {'status': 'fake_failed', 'trained': False, 'weights_updated': False,
                    'training_invoked': True, 'simulated': True,
                    'candidate_adapter_hash': None, 'active_adapter_hash': prior_adapter_hash,
                    'reason': 'fake_training_failure'}
        return {'status': 'fake_only', 'trained': False, 'weights_updated': False,
                'training_invoked': True, 'simulated': True,
                'candidate_adapter_hash': None, 'active_adapter_hash': prior_adapter_hash,
                'record_count': len(records), 'steps_requested': steps,
                'reason': 'fake_mode_does_not_create_model_weights'}

