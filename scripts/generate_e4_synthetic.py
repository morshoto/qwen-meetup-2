"""Generate a small, deterministic ARC-style benchmark with hidden labels split out."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'


def make_grid(seed: int, height: int, width: int, palette: list[int], *,
              interior: bool = False) -> list[list[int]]:
    rng = random.Random(seed)
    grid = [[0 for _ in range(width)] for _ in range(height)]
    rows = list(range(1, height - 1)) if interior else list(range(height))
    cols = list(range(1, width - 1)) if interior else list(range(width))
    positions = [(row, col) for row in rows for col in cols]
    rng.shuffle(positions)
    # Include every mapped color in each demonstration and evaluation input.
    for color, (row, col) in zip(palette, positions):
        grid[row][col] = color
    extra = rng.randint(2, min(5, len(positions) - len(palette)))
    for row, col in positions[len(palette):len(palette) + extra]:
        grid[row][col] = rng.choice(palette)
    return grid


def color_map(grid: list[list[int]], mapping: dict[int, int]) -> list[list[int]]:
    return [[mapping.get(cell, cell) for cell in row] for row in grid]


def flip_h(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in grid]


def flip_v(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in reversed(grid)]


def rotate90(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*reversed(grid))]


def rotate180(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in reversed(grid)]


def shift(grid: list[list[int]], dx: int, dy: int) -> list[list[int]]:
    height, width = len(grid), len(grid[0])
    result = [[0 for _ in range(width)] for _ in range(height)]
    for row, line in enumerate(grid):
        for col, value in enumerate(line):
            target_row, target_col = row + dy, col + dx
            if value and 0 <= target_row < height and 0 <= target_col < width:
                result[target_row][target_col] = value
    return result


def generate() -> None:
    specifications: list[tuple[str, str, tuple[int, int], list[int], dict[int, int], tuple[int, int]]] = [
        ('color-01', 'color_map', (6, 6), [1, 2, 3], {1: 5, 2: 7, 3: 8}, (0, 0)),
        ('color-02', 'color_map', (5, 7), [1, 4, 6], {1: 9, 4: 2, 6: 5}, (0, 0)),
        ('color-03', 'color_map', (7, 5), [2, 5, 8], {2: 6, 5: 1, 8: 4}, (0, 0)),
        ('reflect-h-01', 'flip_h', (6, 7), [2, 4, 7], {}, (0, 0)),
        ('reflect-v-01', 'flip_v', (5, 6), [1, 3, 8], {}, (0, 0)),
        ('rotate-90-01', 'rotate90', (5, 7), [2, 5, 9], {}, (0, 0)),
        ('rotate-180-01', 'rotate180', (6, 6), [1, 4, 7], {}, (0, 0)),
        ('translate-right-01', 'translate', (6, 7), [2, 5, 8], {}, (1, 0)),
        ('translate-down-01', 'translate', (7, 6), [1, 3, 6], {}, (0, 1)),
        ('compose-reflect-color-01', 'flip_h+color_map', (6, 7), [1, 4, 8], {1: 7, 4: 2, 8: 5}, (0, 0)),
    ]
    challenges: dict[str, object] = {}
    solutions: dict[str, object] = {}
    for index, (name, operation, (height, width), palette, mapping, delta) in enumerate(specifications):
        task_id = f'e4-{name}'
        interior = operation == 'translate'

        def apply(grid: list[list[int]]) -> list[list[int]]:
            if operation == 'color_map':
                return color_map(grid, mapping)
            if operation == 'flip_h':
                return flip_h(grid)
            if operation == 'flip_v':
                return flip_v(grid)
            if operation == 'rotate90':
                return rotate90(grid)
            if operation == 'rotate180':
                return rotate180(grid)
            if operation == 'translate':
                return shift(grid, *delta)
            if operation == 'flip_h+color_map':
                return color_map(flip_h(grid), mapping)
            raise ValueError(operation)

        samples = [make_grid(4100 + index * 20 + sample, height, width, palette,
                             interior=interior) for sample in range(5)]
        challenges[task_id] = {
            'train': [{'input': grid, 'output': apply(grid)} for grid in samples[:3]],
            'test': [{'input': grid} for grid in samples[3:]],
        }
        solutions[task_id] = [apply(grid) for grid in samples[3:]]
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / 'e4_synthetic_challenges.v1.json').write_text(
        json.dumps(challenges, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8')
    (FIXTURES / 'e4_synthetic_solutions.v1.json').write_text(
        json.dumps(solutions, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8')


if __name__ == '__main__':
    generate()
