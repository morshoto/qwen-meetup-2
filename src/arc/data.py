"""ARC dataset loading, label isolation, and reproducible splits."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Grid = list[list[int]]


def validate_grid(value: Any) -> Grid:
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        raise ValueError("grid height must be 1..30")
    if not all(isinstance(row, list) and 1 <= len(row) <= 30 for row in value):
        raise ValueError("grid rows must have width 1..30")
    if len({len(row) for row in value}) != 1:
        raise ValueError("grid must be rectangular")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in value for cell in row):
        raise ValueError("grid cells must be integers in 0..9")
    return value


@dataclass(frozen=True)
class Task:
    id: str
    train: tuple[tuple[Grid, Grid], ...]
    test_inputs: tuple[Grid, ...]

    def prompt_data(self) -> dict[str, Any]:
        return {"train": [{"input": i, "output": o} for i, o in self.train],
                "test": [{"input": i} for i in self.test_inputs]}


class Verifier:
    """Sole owner of held-out test outputs; never passed to model or agent."""
    def __init__(self, solutions: dict[str, list[Grid]]) -> None:
        self.__solutions = solutions

    def output_count(self, task_id: str) -> int:
        return len(self.__solutions[task_id])

    def correct_outputs(self, task_id: str, candidate: Any) -> int:
        expected = self.__solutions[task_id]
        if not isinstance(candidate, list) or len(candidate) != len(expected):
            return 0
        count = 0
        for actual, target in zip(candidate, expected):
            try:
                count += validate_grid(actual) == target
            except (ValueError, TypeError):
                pass
        return count

    def matches(self, task_id: str, candidate: Any) -> bool:
        try:
            if not isinstance(candidate, list):
                return False
            expected = self.__solutions[task_id]
            return len(candidate) == len(expected) and all(
                validate_grid(actual) == target for actual, target in zip(candidate, expected)
            )
        except (ValueError, KeyError, TypeError):
            return False


def load_arc(challenges_path: str | Path, solutions_path: str | Path) -> tuple[dict[str, Task], Verifier]:
    def read(path: str | Path) -> Any:
        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result
        with open(path, encoding="utf-8") as f:
            return json.load(f, object_pairs_hook=unique)
    challenges, solutions = read(challenges_path), read(solutions_path)
    if not isinstance(challenges, dict) or not challenges or not isinstance(solutions, dict) or challenges.keys() != solutions.keys():
        raise ValueError("challenge and solution task IDs must match and be nonempty")
    tasks = {}
    checked_solutions = {}
    for task_id, item in challenges.items():
        if not isinstance(item, dict) or not isinstance(item.get("train"), list) or not item["train"] or not isinstance(item.get("test"), list) or not item["test"]:
            raise ValueError(f"{task_id}: missing train or test examples")
        try:
            train = tuple((validate_grid(ex["input"]), validate_grid(ex["output"])) for ex in item["train"])
            inputs = tuple(validate_grid(ex["input"]) for ex in item["test"])
            answers = solutions[task_id]
            if not isinstance(answers, list) or len(answers) != len(inputs):
                raise ValueError("solution count differs from test count")
            checked_solutions[task_id] = [validate_grid(x) for x in answers]
            tasks[task_id] = Task(task_id, train, inputs)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{task_id}: invalid ARC data: {exc}") from exc
    return tasks, Verifier(checked_solutions)


def split_tasks(ids: list[str], seed: int, counts: dict[str, int]) -> dict[str, list[str]]:
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task IDs")
    if set(counts) != {"exploration", "promotion_gate", "monitoring"} or any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) != len(ids):
        raise ValueError("split counts must cover the dataset exactly")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    result = {}
    offset = 0
    for name in ("exploration", "promotion_gate", "monitoring"):
        result[name] = shuffled[offset:offset + counts[name]]
        offset += counts[name]
    return result


def exploration_window(splits: dict[str, list[str]], offset: int, count: int) -> list[str]:
    """Select a bounded, reproducible window from the Exploration split."""
    ids = splits.get('exploration')
    if (type(offset) is not int or offset < 0 or type(count) is not int or count <= 0 or
            not isinstance(ids, list) or any(type(task_id) is not str for task_id in ids)):
        raise ValueError('invalid Exploration window')
    if offset + count > len(ids):
        raise ValueError('Exploration window exceeds split size')
    return ids[offset:offset + count]
