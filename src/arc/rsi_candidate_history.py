"""Review candidate reservations without asserting model dispatch."""
import hashlib
import json
import re
from pathlib import Path


def audit_candidate_journals(root, dataset_ids):
    root = Path(root).resolve()
    artifacts = root / 'artifacts'
    if not artifacts.is_dir() or not isinstance(dataset_ids, set) or any(
            not isinstance(x, str) or not x for x in dataset_ids):
        raise ValueError('invalid artifact root or dataset IDs')
    def paths():
        return sorted(p for p in artifacts.rglob('budget.json') if p.is_file())
    inventory, found = [], set()
    journals = reservations = 0
    for path in paths():
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('invalid budget')
            if 'candidate_keys' in value:
                keys = value['candidate_keys']
                if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
                    raise ValueError('invalid candidate keys')
                for key in keys:
                    match = re.fullmatch(r'[ABC]:([0-9a-f]{8}):(0|[1-9][0-9]*)', key)
                    if not match:
                        raise ValueError('invalid candidate key')
                    found.add(match.group(1))
                journals += 1
                reservations += len(keys)
            inventory.append((path.relative_to(root).as_posix(), hashlib.sha256(raw).hexdigest()))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f'invalid budget journal: {path}') from exc
    try:
        after = [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths()]
    except OSError as exc:
        raise ValueError('budget inventory changed during review') from exc
    if inventory != after:
        raise ValueError('budget inventory changed during review')
    digest = hashlib.sha256(json.dumps(inventory, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    return {'budget_file_count': len(inventory), 'journal_count': journals, 'reservation_count': reservations,
            'inventory_sha256': digest, 'task_ids': sorted(found & dataset_ids)}
