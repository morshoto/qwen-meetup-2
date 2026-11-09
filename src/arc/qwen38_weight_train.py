"""Validation helpers for a real exploratory LoRA training run."""
from __future__ import annotations

import math
import re


def parse_training_losses(log, *, steps):
    if type(steps) is not int or steps < 1:
        raise ValueError('invalid step count')
    clean = re.sub(r'\x1b\[[0-9;]*m', '', log)
    matches = re.findall(r'Iter\s+(\d+):\s*Train loss\s+([^,\s]+)', clean)
    if len(matches) != steps or [int(step) for step, _ in matches] != list(range(1, steps + 1)):
        raise ValueError('training loss log does not cover every step exactly once')
    try:
        losses = [float(value) for _, value in matches]
    except ValueError as error:
        raise ValueError('invalid training loss') from error
    if not all(math.isfinite(value) for value in losses):
        raise ValueError('nonfinite training loss')
    return losses
