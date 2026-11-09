"""Replay a finished fresh collection before constructing private SFT data."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_data_audit import extract_fresh_records
from src.arc.qwen38_weight_journal import AttemptJournal
from src.arc.qwen38_weight_preflight import verify_preflight
from src.arc.qwen38_weight_verification_receipt import verify_receipt
from src.cli.qwen38_weight_collect import _replay_artifacts


def replay_fresh(manifest, proposal, review, repo, model_path, challenges, solutions, canary_dir, run_dir):
    """Return private candidate evidence; callers must enforce the final SFT gate."""
    root = Path(run_dir).resolve(); model_path = Path(model_path).resolve()
    if manifest.get('run_dir') != str(root) or not root.is_dir() or not (root / 'journal').is_dir():
        raise ValueError('registered collection run missing or mismatched')
    if verify_preflight(manifest, proposal, review, repo, model_path, challenges, solutions, canary_dir=canary_dir).get('status') != 'passed':
        raise ValueError('collection source preflight failed')
    with AttemptJournal(root / 'journal') as journal:
        start = json.loads((root / 'run-start.json').read_bytes())
        if (start.get('manifest_sha256') != manifest['sha256'] or
            start.get('sha256') != digest({k: v for k, v in start.items() if k != 'sha256'}) or
            isinstance(start.get('started_epoch'), bool) or not isinstance(start.get('started_epoch'), (int, float)) or
            not math.isfinite(start['started_epoch'])):
            raise ValueError('invalid collection start')
        elapsed = time.time() - start['started_epoch']
        if elapsed < 0:
            raise ValueError('system clock precedes collection start')
        rows = journal.read()
        _replay_artifacts(rows, root / 'workers', manifest['generation']['max_tokens_per_candidate'], manifest['supervision'])
        for index, row in enumerate(rows):
            command = [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_worker', '--model-path', str(model_path),
                       '--output', str(root / 'workers' / f'{index:06}' / 'response.json')]
            if row.get('worker_command') != command:
                raise ValueError('worker command differs from frozen collection')
        receipt_path = root / 'receipts' / (digest(rows) + '.json')
        receipt = json.loads(receipt_path.read_bytes())
        verify_receipt(receipt, manifest, challenges, solutions, rows)
        return extract_fresh_records(manifest, challenges, solutions, rows, receipt, wall_elapsed=elapsed)
