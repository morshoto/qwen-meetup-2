"""Train-only hypothesis search over a small, explicit grid-operation grammar."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.arc.data import Grid, Task, validate_grid


SPATIAL_OPERATIONS = ('identity', 'flip_h', 'flip_v', 'rotate180',
                      'rotate90', 'rotate270', 'transpose', 'anti_transpose')
SHIFT_RANGE = range(-2, 3)


@dataclass(frozen=True)
class GridHypothesis:
    spatial: str
    dx: int
    dy: int
    background: int
    color_map: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {'spatial': self.spatial, 'dx': self.dx, 'dy': self.dy,
                'background': self.background,
                'color_map': {str(source): target for source, target in self.color_map}}


def _spatial(grid: Grid, operation: str) -> Grid:
    if operation == 'identity':
        return [row[:] for row in grid]
    if operation == 'flip_h':
        return [list(reversed(row)) for row in grid]
    if operation == 'flip_v':
        return [row[:] for row in reversed(grid)]
    if operation == 'rotate180':
        return [list(reversed(row)) for row in reversed(grid)]
    if operation == 'rotate90':
        return [list(row) for row in zip(*reversed(grid))]
    if operation == 'rotate270':
        return [list(row) for row in zip(*grid)][::-1]
    if operation == 'transpose':
        return [list(row) for row in zip(*grid)]
    if operation == 'anti_transpose':
        rotated = [list(reversed(row)) for row in reversed(grid)]
        return [list(row) for row in zip(*rotated)]
    raise ValueError(f'unsupported spatial operation: {operation}')


def apply_hypothesis(grid: Grid, hypothesis: GridHypothesis) -> Grid:
    """Apply one orientation, bounded content translation, and color mapping."""
    oriented = _spatial(grid, hypothesis.spatial)
    height, width = len(oriented), len(oriented[0])
    shifted = [[hypothesis.background for _ in range(width)] for _ in range(height)]
    for row, line in enumerate(oriented):
        for col, value in enumerate(line):
            target_row, target_col = row + hypothesis.dy, col + hypothesis.dx
            if 0 <= target_row < height and 0 <= target_col < width:
                shifted[target_row][target_col] = value
    mapping = dict(hypothesis.color_map)
    return [[mapping.get(value, value) for value in line] for line in shifted]


def _task_backgrounds(task: Task) -> tuple[int, ...]:
    modes: set[int] = set()
    for grid, _ in task.train:
        counts: dict[int, int] = {}
        for row in grid:
            for value in row:
                counts[value] = counts.get(value, 0) + 1
        maximum = max(counts.values())
        modes.update(value for value, count in counts.items() if count == maximum)
    return tuple(sorted(modes))


def _learn_color_map(pairs: tuple[tuple[Grid, Grid], ...], *, spatial: str,
                     dx: int, dy: int, background: int) -> tuple[tuple[int, int], ...] | None:
    mapping: dict[int, int] = {}
    for input_grid, expected in pairs:
        transformed = _spatial(input_grid, spatial)
        height, width = len(transformed), len(transformed[0])
        if len(expected) != height or any(len(row) != width for row in expected):
            return None
        shifted = [[background for _ in range(width)] for _ in range(height)]
        for row, line in enumerate(transformed):
            for col, value in enumerate(line):
                target_row, target_col = row + dy, col + dx
                if 0 <= target_row < height and 0 <= target_col < width:
                    shifted[target_row][target_col] = value
        for source_row, target_row in zip(shifted, expected):
            for source, target in zip(source_row, target_row):
                previous = mapping.get(source)
                if previous is not None and previous != target:
                    return None
                mapping[source] = target
    return tuple(sorted((source, target) for source, target in mapping.items()
                        if source != target))


def _hypothesis_cost(hypothesis: GridHypothesis) -> tuple[int, int, int, int, str, int, int]:
    spatial_cost = int(hypothesis.spatial != 'identity')
    shift_cost = abs(hypothesis.dx) + abs(hypothesis.dy)
    color_cost = len(hypothesis.color_map)
    priority = SPATIAL_OPERATIONS.index(hypothesis.spatial)
    return (spatial_cost + shift_cost + color_cost, spatial_cost,
            shift_cost, color_cost, hypothesis.spatial, priority,
            hypothesis.background)


def search_hypotheses(task: Task) -> list[GridHypothesis]:
    """Return train-consistent hypotheses ordered by a fixed simplicity rule."""
    hypotheses: list[GridHypothesis] = []
    seen: set[GridHypothesis] = set()
    backgrounds = _task_backgrounds(task)
    for spatial in SPATIAL_OPERATIONS:
        for background in backgrounds:
            for dy in SHIFT_RANGE:
                for dx in SHIFT_RANGE:
                    color_map = _learn_color_map(task.train, spatial=spatial,
                                                 dx=dx, dy=dy,
                                                 background=background)
                    if color_map is None:
                        continue
                    hypothesis = GridHypothesis(spatial, dx, dy, background, color_map)
                    if hypothesis in seen:
                        continue
                    if all(apply_hypothesis(input_grid, hypothesis) == expected
                           for input_grid, expected in task.train):
                        seen.add(hypothesis)
                        hypotheses.append(hypothesis)
    return sorted(hypotheses, key=_hypothesis_cost)


def solve_task(task: Task) -> dict[str, Any]:
    """Select the simplest train-consistent hypothesis and predict test grids."""
    hypotheses = search_hypotheses(task)
    if not hypotheses:
        return {'task_id': task.id, 'status': 'unsupported',
                'hypothesis': None, 'train_match_count': 0,
                'train_example_count': len(task.train),
                'train_match_rate': 0.0, 'complete_train_match': False,
                'consistent_hypothesis_count': 0, 'test_outputs': []}
    selected = hypotheses[0]
    outputs = [validate_grid(apply_hypothesis(grid, selected))
               for grid in task.test_inputs]
    return {'task_id': task.id, 'status': 'train_consistent',
            'hypothesis': selected.as_dict(),
            'train_match_count': len(task.train),
            'train_example_count': len(task.train),
            'train_match_rate': 1.0, 'complete_train_match': True,
            'consistent_hypothesis_count': len(hypotheses),
            'test_outputs': outputs}
