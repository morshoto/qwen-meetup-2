"""Fixed model identities and bounded settings for E8."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
from src.core.config import _settings

MODELS = {
    '8b': ('mlx-community/Qwen3-8B-4bit', '545dc4251c05440727734bcd94334791f6ab0192', 4623784971),
    '14b': ('mlx-community/Qwen3-14B-4bit', 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4', 8323870520),
    '32b': ('mlx-community/Qwen3-32B-4bit', 'bcaaf7f538adf166c1080a2befdb4f6019f66639', 18445874457),
}
ORDER = tuple(MODELS)

@dataclass(frozen=True)
class E8Settings:
    profile: str
    base_config: str
    slots: int
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
    slot_offset: int = 0


def load_e8_config(path: str | Path) -> E8Settings:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) != {'e8'}:
        raise ValueError('expected one e8 mapping')
    s = _settings(E8Settings, raw['e8'])
    limits = {'fake': (2, 60, 8, 60000, 900),
              'smoke': (2, 60, 8, 60000, 900),
              'full': (10, 300, 32, 200000, 3600),
              'fake_full': (10, 300, 32, 200000, 3600)}
    if s.profile not in limits:
        raise ValueError('unknown E8 profile')
    values = (s.slots, s.sandbox, s.calls, s.reserved_tokens, s.wall_seconds)
    if any(type(v) is not int or v <= 0 for v in values) or any(x > y for x, y in zip(values, limits[s.profile])):
        raise ValueError('invalid profile limits')
    if (s.candidate_sandbox < 20 or s.candidate_sandbox > s.sandbox or
        s.calls < 2 or s.candidate_seconds <= 0 or s.candidate_seconds > s.wall_seconds or
        not 0 <= s.max_repairs <= 2 or s.min_free_disk_gib < 5 or
        not 1 <= s.min_memory_free_percent <= 100 or s.max_swap_increase_mib <= 0):
        raise ValueError('invalid verifier or resource limit')
    if type(s.slot_offset) is not int or s.slot_offset < 0 or (s.profile in ('full', 'fake_full') and s.slot_offset < 2):
        raise ValueError('full comparison requires unused slot IDs')
    return s
