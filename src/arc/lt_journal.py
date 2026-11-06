"""Fixture-only model-run journal; caller-supplied components are not verified.

Never starts a server or proves containment, identity, seed support, cleanup,
resource sampling, or permission for a live run. A started run cannot be resumed.
"""
from __future__ import annotations

import json
import hashlib
import stat
import os
import time
from pathlib import Path
from typing import Any

from src.arc.lt_comparison import budget_for_model, run_bounded_task, verify_manifest
from src.arc.lt_public import load_public_export
from src.arc.http_backend import BackendError
from src.arc.sandbox import FakeSandbox, IDENTITY_CODE
from src.arc.rsi_slot_lifecycle import _create, _directory, _read


class FixtureBackend:
    """Deterministic synthetic replies; never invokes a model or transport."""
    def __init__(self, *, fail_at: int | None = None, interrupt_at: int | None = None,
                 tokens_per_call: int | None = None):
        if (any(value is not None and (type(value) is not int or value <= 0)
                for value in (fail_at, interrupt_at)) or
                (tokens_per_call is not None and (type(tokens_per_call) is not int or tokens_per_call < 0))):
            raise ValueError('invalid fixture settings')
        self.fail_at, self.interrupt_at, self.tokens_per_call = fail_at, interrupt_at, tokens_per_call
        self.calls = 0
        self.event_callback = None

    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        self.calls += 1
        if self.event_callback:
            usage = None if self.tokens_per_call is None else {'total_tokens': self.tokens_per_call}
            self.event_callback({'usage': usage})
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt()
        if self.calls == self.fail_at:
            raise BackendError('fixture_unavailable', 'synthetic failure')
        return 'copy input' if kind == 'analysis' else IDENTITY_CODE


def _usage_total(events: list, reserved_calls: int) -> int | None:
    if type(reserved_calls) is not int or reserved_calls <= 0 or len(events) != reserved_calls:
        return None
    total = 0
    for event in events:
        usage = event.get('usage') if isinstance(event, dict) else None
        if not isinstance(usage, dict):
            return None
        counts = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens') if k in usage}
        if any(type(v) is not int or v < 0 for v in counts.values()):
            return None
        if 'total_tokens' in counts:
            if counts['total_tokens'] < max(counts.get('prompt_tokens', 0), counts.get('completion_tokens', 0)):
                return None
            if ('prompt_tokens' in counts and 'completion_tokens' in counts and
                    counts['total_tokens'] != counts['prompt_tokens'] + counts['completion_tokens']):
                return None
            value = counts['total_tokens']
        elif 'prompt_tokens' in counts and 'completion_tokens' in counts:
            value = counts['prompt_tokens'] + counts['completion_tokens']
        else:
            return None
        total += value
    return total


def run_fixture_model(root: Path, frozen: dict, receipt: dict, public_export: dict,
                      model: str, backend: Any, sandbox: Any) -> dict:
    if type(backend) is not FixtureBackend or type(sandbox) is not FakeSandbox:
        raise ValueError('fixture journal requires synthetic components')
    verify_manifest(frozen)
    if model not in ('14b', '32b') or receipt.get('comparison_sha256') != frozen['sha256']:
        raise ValueError('model or comparison identity differs')
    tasks = load_public_export(public_export, receipt)
    task_ids = frozen['manifest']['task_ids']
    if set(tasks) != set(task_ids):
        raise ValueError('registered task set differs')
    budget = budget_for_model(frozen, model)
    config = frozen['config']
    identity = {'model': model, 'comparison_sha256': frozen['sha256'],
                'evidence_type': 'caller_supplied_fixture_components',
                'run_identity_independently_verified': False,
                'model_revision': config['models'][model],
                'model_files_sha256': config['model_files_sha256'][model],
                'runtime_sha256': config['runtime_sha256'],
                'task_hashes_sha256': _task_hash(receipt)}
    with _directory(root) as directory:
        _create(directory, f'{model}.started.json', identity)
        fd = os.open(f'{model}.results.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        attempted = 0
        result_hash = hashlib.sha256()
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'wb', closefd=False) as stream:
                for index, task_id in enumerate(task_ids):
                    _create(directory, f'{model}.task-{index:03d}.started.json',
                            {'model': model, 'comparison_sha256': frozen['sha256'], 'task_id': task_id})
                    started = time.monotonic()
                    calls_before = budget.state['calls']
                    events = []
                    has_callback = hasattr(backend, 'event_callback')
                    previous_callback = getattr(backend, 'event_callback', None)
                    if has_callback:
                        def capture(event):
                            events.append(event)
                            if previous_callback:
                                previous_callback(event)
                        backend.event_callback = capture
                    try:
                        row = run_bounded_task(tasks[task_id], backend, sandbox, budget,
                                               max_tokens=config['max_tokens'], max_repairs=config['max_repairs'])
                    finally:
                        if has_callback:
                            backend.event_callback = previous_callback
                    tokens = _usage_total(events, budget.state['calls'] - calls_before)
                    row.update(model=model, comparison_sha256=frozen['sha256'], tokens=tokens,
                               runtime_seconds=time.monotonic() - started, peak_memory_bytes=None,
                               peak_swap_bytes=None)
                    line = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                    stream.write(line)
                    result_hash.update(line)
                    stream.flush()
                    os.fsync(fd)
                    attempted += 1
                    if row['status'] != 'complete':
                        break
        except BaseException as exc:
            _create(directory, f'{model}.incomplete.json', {**identity, 'status': 'incomplete',
                    'attempted': attempted, 'failure_type': type(exc).__name__})
            raise
        finally:
            os.close(fd)
        result = {**identity, 'status': 'complete' if attempted == len(task_ids) and row['status'] == 'complete'
                  else 'incomplete', 'attempted': attempted, 'results_sha256': result_hash.hexdigest()}
        _create(directory, f'{model}.terminal.json', result)
        return result


def _task_hash(receipt: dict) -> str:
    return hashlib.sha256(json.dumps(receipt['selected_task_sha256'], sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def audit_fixture_run(root: Path, frozen: dict, receipt: dict, model: str) -> dict:
    """Check journal consistency only; mutable local files are not authenticated."""
    from src.arc.lt_attestation import _object
    verify_manifest(frozen)
    if (model not in ('14b', '32b') or receipt.get('comparison_sha256') != frozen['sha256'] or
            set(receipt.get('selected_task_sha256', {})) != set(frozen['manifest']['task_ids'])):
        raise ValueError('comparison or task identity differs')
    config = frozen['config']
    identity = {'model': model, 'comparison_sha256': frozen['sha256'],
                'evidence_type': 'caller_supplied_fixture_components',
                'run_identity_independently_verified': False, 'model_revision': config['models'][model],
                'model_files_sha256': config['model_files_sha256'][model], 'runtime_sha256': config['runtime_sha256'],
                'task_hashes_sha256': _task_hash(receipt)}
    with _directory(root) as directory:
        started = _read(directory, f'{model}.started.json')
        terminal = _read(directory, f'{model}.terminal.json')
        if started != identity or any(terminal.get(k) != v for k, v in identity.items()):
            raise ValueError('run identity differs')
        fd = os.open(f'{model}.results.jsonl', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError('results must be a private regular file')
            raw = stream.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != terminal.get('results_sha256'):
            raise ValueError('result journal differs')
        rows = [_object(line) for line in raw.splitlines()]
        ids = frozen['manifest']['task_ids']
        if len(rows) != terminal.get('attempted') or len(rows) > len(ids):
            raise ValueError('result coverage differs')
        for index, row in enumerate(rows):
            marker = _read(directory, f'{model}.task-{index:03d}.started.json')
            expected_marker = {'model': model, 'comparison_sha256': frozen['sha256'], 'task_id': ids[index]}
            if (marker != expected_marker or row.get('model') != model or row.get('task_id') != ids[index] or
                    row.get('comparison_sha256') != frozen['sha256'] or row.get('status') not in ('complete', 'error') or
                    not isinstance(row.get('candidates'), list) or len(row['candidates']) > 2 or
                    (row['status'] == 'complete' and len(row['candidates']) != 2) or
                    (index < len(rows) - 1 and row['status'] != 'complete')):
                raise ValueError('task journal differs')
        expected_status = 'complete' if len(rows) == len(ids) and all(r['status'] == 'complete' for r in rows) else 'incomplete'
        if terminal.get('status') != expected_status:
            raise ValueError('terminal status differs')
    return {'status': 'consistent_unverified', 'model': model, 'attempted': len(rows),
            'run_status': expected_status, 'run_identity_independently_verified': False}
