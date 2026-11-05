"""Inventory retained task-use evidence before proposing an RSI split.

This checks a defined set of retained files; it cannot prove missing history never
existed or that model pretraining did not expose an evaluation task.
"""
import hashlib
import json
from pathlib import Path

import yaml

from src.arc.exploration_registry import build_registry

_JSON_NAMES = {'results.json', 'holdout_manifest.json', 'collection.json',
               'source_manifest.json', 'selection_manifest.json',
               'comparison_identity.json', 'collection_manifest.json',
               'collection_summary.json', 'comparison.json', 'task_manifest.json',
               'identity.json'}
_JSONL_NAMES = {'task_diagnostics.jsonl', 'candidate_traces.jsonl',
                'attempted_tasks.jsonl', 'verified_solutions.jsonl',
                'accepted_solutions.jsonl'}


def _inventory(root: Path, required: list[str]) -> list[dict[str, str]]:
    paths = set()
    for relative in required:
        path = root / relative
        if path.is_absolute() and not path.resolve().is_relative_to(root):
            raise ValueError('required source is outside audit root')
        if not path.is_file():
            raise ValueError(f'required source missing: {relative}')
        paths.add(path)
    artifacts = root / 'artifacts'
    if artifacts.exists():
        paths.update(p for p in artifacts.rglob('*') if p.is_file() and
                     ((p.suffix == '.json' and p.name in _JSON_NAMES) or
                      (p.suffix == '.jsonl' and p.name in _JSONL_NAMES)))
    configs = root / 'configs'
    if configs.exists():
        paths.update(p for p in configs.glob('*.yml') if p.is_file())
    rows = []
    for path in sorted(paths):
        try:
            raw = path.read_bytes()
            text = raw.decode('utf-8')
            if path.suffix == '.json':
                json.loads(text)
            elif path.suffix == '.jsonl':
                for line in text.splitlines():
                    if line.strip():
                        json.loads(line)
            elif path.suffix in ('.yml', '.yaml'):
                yaml.safe_load(text)
            else:
                raise ValueError('unsupported source format')
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f'unreadable or invalid source: {path}') from exc
        rows.append({'path': path.relative_to(root).as_posix(),
                     'sha256': hashlib.sha256(raw).hexdigest()})
    return rows


def audit_history(root: str | Path, *, required_paths: list[str]) -> dict:
    root = Path(root).resolve()
    before = _inventory(root, required_paths)
    registry = build_registry(root)
    after = _inventory(root, required_paths)
    if before != after:
        raise ValueError('history inventory changed during audit')
    return {'task_ids': registry['task_ids'], 'files': before}
