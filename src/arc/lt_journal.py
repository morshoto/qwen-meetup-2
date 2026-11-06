"""Fixture-only model-run journal; caller-supplied components are not verified.

Never starts a server or proves containment, identity, seed support, cleanup,
resource sampling, or permission for a live run. A started run cannot be resumed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from src.arc.lt_comparison import budget_for_model, run_bounded_task, verify_manifest
from src.arc.lt_public import load_public_export
from src.arc.rsi_slot_lifecycle import _create, _directory


def run_fixture_model(root: Path, frozen: dict, receipt: dict, public_export: dict,
                      model: str, backend: Any, sandbox: Any) -> dict:
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
                'run_identity_independently_verified': False}
    with _directory(root) as directory:
        _create(directory, f'{model}.started.json', identity)
        fd = os.open(f'{model}.results.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        attempted = 0
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'wb', closefd=False) as stream:
                for index, task_id in enumerate(task_ids):
                    _create(directory, f'{model}.task-{index:03d}.started.json',
                            {'model': model, 'comparison_sha256': frozen['sha256'], 'task_id': task_id})
                    started = time.monotonic()
                    row = run_bounded_task(tasks[task_id], backend, sandbox, budget,
                                           max_tokens=config['max_tokens'], max_repairs=config['max_repairs'])
                    row.update(model=model, comparison_sha256=frozen['sha256'], tokens=None,
                               runtime_seconds=time.monotonic() - started, peak_memory_bytes=None,
                               peak_swap_bytes=None)
                    stream.write((json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
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
                  else 'incomplete', 'attempted': attempted}
        _create(directory, f'{model}.terminal.json', result)
        return result
