"""Pinned dual-model settings for the E9 generation bootstrap."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.arc.e8_config import MODELS as E8_MODELS
from src.core.config import _settings

MODELS = {key: E8_MODELS[key] for key in ('14b', '32b')}
ORDER = ('14b', '32b')


@dataclass(frozen=True)
class E9Settings:
    profile: str
    models: list[str]
    base_config: str
    slots: int
    slot_offset: int
    candidate_sandbox: int
    sandbox: int
    calls: int
    reserved_tokens: int
    wall_seconds: int
    candidate_seconds: int
    max_repairs: int
    seed: int
    min_free_disk_gib: int
    min_memory_free_percent: int
    max_swap_increase_mib: int


def load_e9_config(path: str | Path) -> E9Settings:
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or set(raw) != {'e9'} or not isinstance(raw['e9'], dict):
        raise ValueError('expected one e9 mapping')
    data = dict(raw['e9'])
    models = data.get('models')
    if not isinstance(models, list) or not models or any(x not in ORDER for x in models):
        raise ValueError('models must contain 14b and/or 32b')
    if len(set(models)) != len(models):
        raise ValueError('models must be unique')
    settings = _settings(E9Settings, data)
    if settings.profile not in {'fake', 'live'}:
        raise ValueError('unknown E9 profile')
    if settings.profile == 'live' and len(settings.models) != 1:
        raise ValueError('each E9 live config must name exactly one model')
    if (settings.slots <= 0 or settings.slot_offset < 0 or
            settings.slot_offset + settings.slots > 40):
        raise ValueError('E9 supports slots 0 through 39')
    if (settings.candidate_sandbox < 20 or settings.sandbox < settings.candidate_sandbox or
            settings.calls < 2 or settings.reserved_tokens <= 0 or
            settings.wall_seconds <= 0 or settings.candidate_seconds <= 0 or
            settings.candidate_seconds > settings.wall_seconds or
            not 0 <= settings.max_repairs <= 2 or settings.min_free_disk_gib < 5 or
            not 1 <= settings.min_memory_free_percent <= 100 or
            settings.max_swap_increase_mib <= 0):
        raise ValueError('invalid E9 budget or verifier settings')
    if settings.profile == 'fake':
        if settings.slots > 10 or settings.wall_seconds > 900 or settings.calls > 60:
            raise ValueError('fake E9 settings exceed fixture limits')
    else:
        if settings.slots > 10 and settings.slot_offset == 0:
            raise ValueError('initial live runs are limited to ten slots')
        if settings.wall_seconds > 3600 or settings.calls > 200 or settings.sandbox > 1200:
            raise ValueError('live E9 settings exceed hard limits')
        if settings.slot_offset > 0 and settings.wall_seconds > 1200:
            raise ValueError('additional E9 batches are limited to 1200 seconds')
    if settings.slot_offset == 0 and settings.slots > 10:
        raise ValueError('initial live runs are limited to ten slots')
    return settings
