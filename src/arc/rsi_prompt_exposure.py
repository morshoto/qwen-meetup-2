"""Bounded exact-string review; matches do not prove model dispatch."""
import hashlib
import json
from pathlib import Path


def audit_prompt_strings(root, prompts):
    root = Path(root).resolve()
    artifacts = root / 'artifacts'
    if not artifacts.is_dir():
        raise ValueError('artifact history missing')
    if (not isinstance(prompts, dict) or not prompts or
            any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in prompts.items())):
        raise ValueError('invalid prompts')

    def paths():
        return sorted(p for p in artifacts.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl'))

    matches = {}
    inventory = []
    for path in paths():
        relative = path.relative_to(root).as_posix()
        try:
            raw = path.read_bytes()
            content = raw.decode('utf-8')
            records = ([json.loads(line) for line in content.splitlines() if line.strip()]
                       if path.suffix == '.jsonl' else [json.loads(content)])
            pending = list(records)
            found = set()
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
                elif isinstance(value, str):
                    found.update(key for key, prompt in prompts.items() if prompt in value)
            for key in found:
                matches.setdefault(key, []).append(relative)
            inventory.append((relative, hashlib.sha256(raw).hexdigest()))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise ValueError(f'unreadable or invalid artifact: {relative}') from exc
    try:
        after = [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths()]
    except OSError as exc:
        raise ValueError('artifact inventory changed during review') from exc
    if inventory != after:
        raise ValueError('artifact inventory changed during review')
    encoded = json.dumps(inventory, separators=(',', ':'), ensure_ascii=True).encode()
    return {'file_count': len(inventory), 'inventory_sha256': hashlib.sha256(encoded).hexdigest(),
            'matches': dict(sorted(matches.items()))}
