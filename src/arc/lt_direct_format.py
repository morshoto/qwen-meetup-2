"""Strict format-only normalization for LT direct-grid responses."""
import json
from src.arc.data import validate_grid


def normalize_direct(raw: str, count: int) -> list:
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj: raise ValueError('duplicate key')
            obj[key] = value
        return obj
    value = json.loads(raw, object_pairs_hook=unique)
    if isinstance(value, dict):
        if set(value) != {'outputs'}: raise ValueError('ambiguous output object')
        value = value['outputs']
    if not isinstance(value, list) or len(value) != count:
        raise ValueError('output count differs')
    grids = []
    for item in value:
        if isinstance(item, dict):
            if set(item) != {'output'}: raise ValueError('ambiguous grid object')
            item = item['output']
        grids.append(validate_grid(item))
    return grids


def extract_final(raw: str, thinking: bool) -> str:
    """Extract a single explicit Qwen thinking block; preserve all other text."""
    value = raw.strip()
    if (thinking and value.startswith('<think>') and value.count('<think>') == 1 and
            value.count('</think>') == 1):
        return value.split('</think>', 1)[1].strip()
    return raw
