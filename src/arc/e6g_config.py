"""Strict, symmetric G0/G1 budget configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from src.core.config import _settings, load_experiment_config


@dataclass(frozen=True)
class E6GSettings:
    profile: str
    base_config: str
    candidates_per_condition: int
    max_repairs: int
    per_condition_calls: int
    per_condition_reserved_tokens: int
    per_condition_sandbox: int
    per_condition_wall_seconds: int
    total_calls: int
    total_reserved_tokens: int
    total_sandbox: int
    total_wall_seconds: int
    seed: int = 42
    min_verified_tasks: int = 5
    pool_path: str = 'artifacts/e6g/seed-pool-v2'


def load_e6g_config(path: str | Path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != {'e6g'}:
        raise ValueError('E6-G requires exactly one e6g configuration mapping')
    settings = _settings(E6GSettings, value['e6g'])
    maxima = {'cpu_fake': (3, 30, 100000, 240, 600),
              'live_smoke': (2, 16, 50000, 40, 900),
              'live_compare': (10, 60, 280000, 200, 2400)}
    if settings.profile not in maxima:
        raise ValueError('invalid E6-G profile')
    values = (settings.candidates_per_condition, settings.total_calls,
              settings.total_reserved_tokens, settings.total_sandbox, settings.total_wall_seconds)
    if any(x <= 0 for x in values) or any(x > cap for x, cap in zip(values, maxima[settings.profile])):
        raise ValueError('E6-G total limits must be positive and within the profile caps')
    sub = (settings.per_condition_calls, settings.per_condition_reserved_tokens,
           settings.per_condition_sandbox, settings.per_condition_wall_seconds)
    total = values[1:]
    if any(x <= 0 for x in sub) or any(2 * x > y for x, y in zip(sub, total)):
        raise ValueError('E6-G requires equal positive condition caps within the total budget')
    if not 0 <= settings.max_repairs <= 2 or settings.min_verified_tasks != 5:
        raise ValueError('E6-G allows at most two repairs and retains the five-task gate')
    if settings.candidates_per_condition > settings.per_condition_calls:
        raise ValueError('condition candidates exceed its call budget')
    if settings.profile != 'cpu_fake' and settings.pool_path != 'artifacts/e6g/seed-pool-v2':
        raise ValueError('Live E6-G requires the registered seed-pool path')
    project = Path(__file__).resolve().parents[2]
    base = load_experiment_config(project / settings.base_config)
    if settings.profile == 'cpu_fake':
        if base.model.backend != 'fake' or base.agent.sandbox != 'fake':
            raise ValueError('Fake comparison requires fake model and sandbox')
    elif (base.model.backend != 'guarded_mlx' or base.agent.sandbox != 'docker' or
          base.model.thinking != 'off' or base.model.adapter_sha256 != 'none'):
        raise ValueError('Live comparison requires guarded base Qwen, thinking off, and Docker')
    return settings, base
