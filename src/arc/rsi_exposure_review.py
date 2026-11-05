"""Review additional retained JSON formats for known task identifiers.

This is a bounded metadata review, not proof that missing files, unknown schemas,
model prompts, or pretraining lack exposure.
"""
import hashlib
import json
from pathlib import Path

from src.arc.exploration_registry import _walk_known_ids
from src.arc.rsi_history import _JSON_NAMES, _JSONL_NAMES


def audit_extra_artifacts(root: str | Path, dataset_ids: set[str]) -> dict:
    root = Path(root).resolve()
    artifacts = root / 'artifacts'
    if not artifacts.is_dir():
        raise ValueError('artifact history missing')
    if any(not isinstance(task, str) or not task for task in dataset_ids):
        raise ValueError('invalid dataset task identifiers')

    def paths():
        return sorted(p for p in artifacts.rglob('*') if p.is_file() and
                      p.suffix in ('.json', '.jsonl') and
                      p.name not in (_JSON_NAMES | _JSONL_NAMES))

    found: dict[str, set[str]] = {}
    inventory = []
    for path in paths():
        try:
            raw = path.read_bytes()
            contents = raw.decode('utf-8')
            values = ([json.loads(line) for line in contents.splitlines() if line.strip()]
                      if path.suffix == '.jsonl' else [json.loads(contents)])
            relative = path.relative_to(root).as_posix()
            for value in values:
                _walk_known_ids(value, found, relative)
            inventory.append((relative, hashlib.sha256(raw).hexdigest()))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f'unreadable or invalid artifact: {path}') from exc
    try:
        after = [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in paths()]
    except OSError as exc:
        raise ValueError('artifact inventory changed during review') from exc
    if inventory != after:
        raise ValueError('artifact inventory changed during review')
    encoded = json.dumps(inventory, separators=(',', ':'), ensure_ascii=True).encode()
    return {'file_count': len(inventory), 'inventory_sha256': hashlib.sha256(encoded).hexdigest(),
            'task_ids': sorted(set(found) & dataset_ids)}
