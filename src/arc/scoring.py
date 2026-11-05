"""Task-level exact match for up to two complete multi-test candidates."""
from dataclasses import dataclass
from typing import Any
from src.arc.data import Verifier, validate_grid


@dataclass(frozen=True)
class Score:
    pass_at_1: bool
    pass_at_2: bool
    invalid: int
    timeouts: int


def score(verifier: Verifier, task_id: str, candidates: list[Any]) -> Score:
    matches = []
    invalid = timeouts = 0
    for candidate in candidates[:2]:
        if candidate == "timeout":
            timeouts += 1
            matches.append(False)
            continue
        try:
            if not isinstance(candidate, list) or len(candidate) != verifier.output_count(task_id):
                raise ValueError("output count must match test inputs")
            for grid in candidate:
                validate_grid(grid)
            matches.append(verifier.matches(task_id, candidate))
        except (TypeError, ValueError):
            invalid += 1
            matches.append(False)
    return Score(bool(matches and matches[0]), any(matches), invalid, timeouts)
