"""Exclusive, append-only attempt evidence for resumable weight collection."""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest


class AttemptJournal:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = None

    def __enter__(self):
        if self.root.is_symlink():
            raise ValueError('journal root cannot be a symlink')
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.root.stat().st_mode & 0o077:
            raise ValueError('journal directory must be private')
        self.lock = (self.root / '.lock').open('a+')
        os.chmod(self.root / '.lock', 0o600)
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close(); self.lock = None
            raise RuntimeError('journal already in use') from error
        return self

    def __exit__(self, *args):
        if self.lock is not None:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close(); self.lock = None

    def _locked(self):
        if self.lock is None:
            raise RuntimeError('journal lock required')

    def _write(self, name, value):
        payload = (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()
        fd = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        dir_fd = os.open(self.root, os.O_RDONLY)
        try: os.fsync(dir_fd)
        finally: os.close(dir_fd)

    def read(self):
        self._locked()
        names = {p.name for p in self.root.iterdir()} - {'.lock'}
        if any(not re.fullmatch(r'\d{6}\.(start|result)\.json', n) for n in names):
            raise ValueError('unknown journal entry')
        starts = sorted(n for n in names if n.endswith('.start.json'))
        result = []
        for index, name in enumerate(starts):
            if name != f'{index:06}.start.json':
                raise ValueError('noncontiguous journal')
            try:
                start = json.loads((self.root / name).read_bytes())
                if start.get('sequence') != index or start.get('sha256') != digest({k: v for k, v in start.items() if k != 'sha256'}):
                    raise ValueError('invalid start identity')
                result_name = f'{index:06}.result.json'
                if result_name in names:
                    end = json.loads((self.root / result_name).read_bytes())
                    if (end.get('start_sha256') != start['sha256'] or end.get('sha256') != digest({k: v for k, v in end.items() if k != 'sha256'}) or
                        end.get('status') not in ('complete', 'error') or
                        bool((set(end) - {'sha256', 'start_sha256'}) & set(start['request']))):
                        raise ValueError('invalid result identity')
                    result.append({**start['request'], **{k: v for k, v in end.items() if k not in ('sha256', 'start_sha256')}})
                    names.remove(result_name)
                else:
                    if index != len(starts) - 1:
                        raise ValueError('unresolved earlier attempt')
                    result.append({**start['request'], 'status': 'started', 'elapsed_seconds': 0})
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError('corrupt journal entry') from error
            names.remove(name)
        if names:
            raise ValueError('orphan result entry')
        return result

    def start(self, request):
        self._locked()
        rows = self.read()
        if rows and rows[-1]['status'] == 'started':
            raise ValueError('previous attempt requires reconciliation')
        if not isinstance(request, dict) or not {'task_id', 'candidate_index', 'seed', 'manifest_sha256'} <= request.keys():
            raise ValueError('invalid attempt identity')
        value = {'sequence': len(rows), 'request': request}
        value['sha256'] = digest(value)
        self._write(f'{len(rows):06}.start.json', value)

    def finish(self, index, outcome):
        self._locked()
        rows = self.read()
        if type(index) is not int or index != len(rows) - 1 or not rows or rows[index]['status'] != 'started':
            raise ValueError('no matching unresolved attempt')
        if not isinstance(outcome, dict) or outcome.get('status') not in ('complete', 'error') or set(outcome) & ((set(rows[index]) - {'status', 'elapsed_seconds'}) | {'sha256', 'start_sha256'}):
            raise ValueError('invalid attempt outcome')
        seconds = outcome.get('elapsed_seconds')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('invalid attempt elapsed time')
        start = json.loads((self.root / f'{index:06}.start.json').read_bytes())
        value = {**outcome, 'start_sha256': start['sha256']}
        value['sha256'] = digest(value)
        self._write(f'{index:06}.result.json', value)
