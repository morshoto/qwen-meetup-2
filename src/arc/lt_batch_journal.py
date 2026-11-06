"""Durable per-task journal for exploratory batches; no inference or scoring."""
import json
import os
import re
from pathlib import Path


def _create(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
    except BaseException:
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


class BatchJournal:
    def __init__(self, root: Path, ids: list[str], identity: str):
        if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', x) for x in ids):
            raise ValueError('invalid task ids')
        self.root, self.ids, self.identity = root, ids, identity
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        marker = root / 'identity.json'
        expected = {'identity': identity, 'task_ids': ids}
        try: _create(marker, expected)
        except FileExistsError:
            try: actual = json.loads(marker.read_text())
            except (OSError, ValueError) as exc: raise ValueError('invalid journal identity') from exc
            if actual != expected: raise ValueError('journal identity differs')

    def pending(self, limit: int) -> list[str]:
        if type(limit) is not int or limit <= 0: raise ValueError('invalid limit')
        remaining = []
        for task in self.ids:
            start, result = self.root / (task + '.started.json'), self.root / (task + '.result.json')
            if result.exists():
                try: row = json.loads(result.read_text())
                except (OSError, ValueError) as exc: raise ValueError('partial result') from exc
                if not start.exists() or row.get('task_id') != task or row.get('identity') != self.identity:
                    raise ValueError('result identity differs')
            elif start.exists(): raise ValueError('interrupted task requires reconciliation')
            else: remaining.append(task)
        return remaining[:limit]

    def start(self, task: str):
        if task not in self.ids or (self.root / (task + '.result.json')).exists(): raise ValueError('invalid or completed task')
        try: _create(self.root / (task + '.started.json'), {'task_id': task, 'identity': self.identity})
        except FileExistsError as exc: raise ValueError('task already started') from exc

    def finish(self, task: str, row: dict):
        if task not in self.ids or row.get('task_id') != task or not (self.root / (task + '.started.json')).exists():
            raise ValueError('task was not started')
        _create(self.root / (task + '.result.json'), {**row, 'identity': self.identity})
