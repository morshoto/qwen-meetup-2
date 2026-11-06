"""Score saved predictions offline; never starts inference or runs model code."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.arc.data import load_arc
from src.arc.lt_attestation import _object, verify_inputs
from src.arc.lt_scoring import score_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'receipt', 'challenges', 'solutions', 'sources', 'prompt', 'results'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--format', choices=('json', 'jsonl'), default='json')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--source-inventory', type=Path)
    args = parser.parse_args(argv)
    if (args.source_root is None) != (args.source_inventory is None):
        parser.error('--source-root and --source-inventory must be provided together')
    try:
        inventory = _object(args.source_inventory.read_bytes()) if args.source_inventory else None
        frozen = _object(args.frozen.read_bytes())
        receipt = _object(args.receipt.read_bytes())
        sources = _object(args.sources.read_bytes())
        verify_inputs(receipt, frozen, args.challenges, args.solutions, sources, args.prompt,
                      source_root=args.source_root, source_inventory=inventory)
        tasks, verifier = load_arc(args.challenges, args.solutions)
        raw = args.results.read_bytes()
        rows = [_object(line) for line in raw.splitlines() if line.strip()]
        if any(row.get('comparison_sha256') != frozen['sha256'] for row in rows):
            raise ValueError('result comparison identity differs')
        if not set(frozen['manifest']['task_ids']) <= set(tasks):
            raise ValueError('missing selected tasks')
        report = score_comparison(frozen['manifest']['task_ids'], rows, verifier)
        verify_inputs(receipt, frozen, args.challenges, args.solutions, sources, args.prompt,
                      source_root=args.source_root, source_inventory=inventory)
        report['evidence'] = {'comparison_sha256': frozen['sha256'],
            'results_sha256': hashlib.sha256(raw).hexdigest(),
            'solutions_sha256': receipt['solutions_sha256'],
            'run_identity_independently_verified': False}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f'scoring failed: {type(exc).__name__}', file=sys.stderr)
        return 1
    if args.format == 'jsonl':
        for row in report['rows']:
            print(json.dumps({'record_type': 'task', **row}, sort_keys=True, allow_nan=False))
        print(json.dumps({'record_type': 'summary', 'models': report['models'],
                          'evidence': report['evidence']}, sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
