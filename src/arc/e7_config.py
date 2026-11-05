"""E7 comparison configuration with equal per-arm limits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from src.core.config import _settings, load_experiment_config


@dataclass(frozen=True)
class E7Settings:
    profile: str
    base_config: str
    slots: int
    candidate_sandbox: int
    arm_sandbox: int
    total_sandbox: int
    candidate_calls: int
    arm_calls: int
    total_calls: int
    candidate_tokens: int
    arm_tokens: int
    total_tokens: int
    candidate_seconds: int
    arm_seconds: int
    total_seconds: int
    max_repairs: int = 2
    seed: int = 42
    pool_path: str = 'artifacts/e6g/seed-pool-v2'


def load_e7_config(path: str | Path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != {'e7'}:
        raise ValueError('expected only e7 configuration')
    s = _settings(E7Settings, value['e7'])
    caps = {'cpu_fake': (3, 180, 30, 120000, 900),
            'live_smoke': (2, 80, 20, 100000, 900),
            'live_full': (6, 240, 48, 300000, 2400)}
    if s.profile not in caps:
        raise ValueError('invalid E7 profile')
    values = (s.slots, s.total_sandbox, s.total_calls, s.total_tokens, s.total_seconds)
    if any(type(v) is not int or v <= 0 for v in values) or any(v > cap for v, cap in zip(values, caps[s.profile])):
        raise ValueError('E7 profile limit exceeded')
    for candidate, arm, total in ((s.candidate_sandbox, s.arm_sandbox, s.total_sandbox),
                                   (s.candidate_calls, s.arm_calls, s.total_calls),
                                   (s.candidate_tokens, s.arm_tokens, s.total_tokens),
                                   (s.candidate_seconds, s.arm_seconds, s.total_seconds)):
        if any(type(v) is not int or v <= 0 for v in (candidate, arm, total)) or candidate > arm or 2 * arm > total:
            raise ValueError('E7 requires positive candidate and equal arm caps within total')
    if s.candidate_sandbox < 20:
        raise ValueError('a complete verification requires 20 sandbox starts')
    if not 0 <= s.max_repairs <= 2:
        raise ValueError('at most two repairs')
    base = load_experiment_config(Path(s.base_config))
    if s.profile == 'cpu_fake':
        if base.model.backend != 'fake' or base.agent.sandbox != 'fake':
            raise ValueError('fake profile requires fake backend and sandbox')
    elif (base.model.backend != 'guarded_mlx' or base.agent.sandbox != 'docker' or
          base.model.thinking != 'off' or base.model.adapter_sha256 != 'none' or
          s.pool_path != 'artifacts/e6g/seed-pool-v2'):
        raise ValueError('live profile requires the pinned base model, Docker and seed pool')
    return s, base
