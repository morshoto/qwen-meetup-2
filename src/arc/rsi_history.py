"""Inventory retained task-use evidence before proposing an RSI split.

This checks a defined set of retained files; it cannot prove missing history never
existed or that model pretraining did not expose an evaluation task.
"""
import hashlib
import json
from pathlib import Path

import yaml

from src.arc.exploration_registry import _walk_known_ids, build_registry, validate_registry
from src.arc.rsi_split import make_split

_JSON_NAMES = {'results.json', 'holdout_manifest.json', 'collection.json',
               'source_manifest.json', 'selection_manifest.json',
               'comparison_identity.json', 'collection_manifest.json',
               'collection_summary.json', 'comparison.json', 'task_manifest.json',
               'identity.json', 'generation.json'}
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
    extra_ids: dict[str, set[str]] = {}
    for row in before:
        if Path(row['path']).name == 'generation.json':
            raw = (root / row['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('history inventory changed during audit')
            _walk_known_ids(json.loads(raw), extra_ids, row['path'])
    after = _inventory(root, required_paths)
    if before != after:
        raise ValueError('history inventory changed during audit')
    return {'task_ids': sorted(set(registry['task_ids']) | set(extra_ids)), 'files': before}


def provisional_snapshot(root: str | Path, *, task_path: str, registry_path: str,
                         required_paths: list[str], development_size: int) -> dict:
    root = Path(root).resolve()
    audit = audit_history(root, required_paths=[*required_paths, task_path, registry_path])
    hashes = {row['path']: row['sha256'] for row in audit['files']}

    def read_checked(relative: str):
        raw = (root / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != hashes[relative]:
            raise ValueError('history inventory changed after audit')
        return json.loads(raw)

    tasks = read_checked(task_path)
    if not isinstance(tasks, dict):
        raise ValueError('task source must be a JSON object')
    old = validate_registry(read_checked(registry_path))
    exposed = set(audit['task_ids']) | set(old['task_ids'])
    return {'status': 'provisional_not_preregistered', 'independence_established': False,
            'known_exposed_in_dataset': len(set(tasks) & exposed),
            'partitions': make_split(tasks, exposed, development_size=development_size),
            'task_source': task_path, 'prior_registry': registry_path,
            'required_paths': sorted(set(required_paths)), 'inspected_files': audit['files']}
