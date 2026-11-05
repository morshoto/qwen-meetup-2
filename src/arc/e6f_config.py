"""Typed, bounded configuration for collection without training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.core.config import ExperimentConfig, _settings, load_experiment_config


@dataclass(frozen=True)
class E6FSettings:
    profile: str
    base_config: str
    initial_candidates: int
    max_repair_attempts: int
    target_unique_oracles: int
    reuse_existing: bool
    max_model_calls: int
    max_reserved_tokens: int
    max_sandbox_runs: int
    max_wall_seconds: int
    source_artifact: str = 'artifacts/e6/e6-local-bootstrap-e526bae72f'
    smoke_artifact: str = 'artifacts/e6f/live-smoke'


def load_e6f_config(path: str | Path) -> tuple[E6FSettings, ExperimentConfig]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or set(value) != {'e6f'}:
        raise ValueError('E6-F config must contain exactly the e6f mapping')
    settings = _settings(E6FSettings, value['e6f'])
    limits = {'cpu_fake': (100, 100, 100000, 1000, 300),
              'live_smoke': (2, 12, 100000, 60, 900),
              'live_collection': (20, 60, 400000, 200, 2400)}
    if settings.profile not in limits:
        raise ValueError('invalid E6-F profile')
    values = (settings.initial_candidates, settings.max_model_calls, settings.max_reserved_tokens,
              settings.max_sandbox_runs, settings.max_wall_seconds)
    if any(value <= 0 for value in values) or any(v > cap for v, cap in zip(values, limits[settings.profile])):
        raise ValueError('E6-F limits must be positive and within the profile caps')
    if not 0 <= settings.max_repair_attempts <= 2 or settings.target_unique_oracles < 5:
        raise ValueError('E6-F permits at most two repairs and a minimum target of five')
    if settings.initial_candidates > settings.max_model_calls:
        raise ValueError('candidate count exceeds model-call budget')
    if settings.profile != 'cpu_fake' and (
            settings.source_artifact != 'artifacts/e6/e6-local-bootstrap-e526bae72f' or
            settings.smoke_artifact != 'artifacts/e6f/live-smoke'):
        raise ValueError('Live E6-F requires the registered source and smoke artifact paths')
    project = Path(__file__).resolve().parents[2]
    base = load_experiment_config(project / settings.base_config)
    if settings.profile == 'cpu_fake':
        if base.model.backend != 'fake' or base.agent.sandbox != 'fake':
            raise ValueError('Fake E6-F requires fake inference and sandbox')
    elif (base.model.backend != 'guarded_mlx' or base.agent.sandbox != 'docker' or
          base.model.thinking != 'off' or base.model.adapter_sha256 != 'none'):
        raise ValueError('Live E6-F requires the base guarded Qwen model, thinking off, and Docker')
    return settings, base
