"""Prepare reviewable offline comparison documents without launching a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.arc.lt_attestation import _object, attest_inputs
from src.arc.lt_comparison import freeze_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proposal', 'config', 'challenges', 'solutions', 'sources', 'prompt'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--part', required=True, choices=('frozen', 'receipt'))
    parser.add_argument('--fixture-only', action='store_true', help='use unchecked synthetic fixture inputs only')
    for name in ('registry', 'split', 'development', 'training'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--source-inventory', type=Path)
    args = parser.parse_args(argv)
    if (args.source_root is None) != (args.source_inventory is None):
        parser.error('--source-root and --source-inventory must be provided together')
    evidence_paths = (args.registry, args.split, args.development, args.training)
    if args.fixture_only and any(evidence_paths):
        parser.error('fixture-only cannot be combined with allocation evidence')
    if not args.fixture_only and (not all(evidence_paths) or args.source_root is None):
        parser.error('protected allocation and complete source inventory are required')
    try:
        inventory = _object(args.source_inventory.read_bytes()) if args.source_inventory else None
        allocation = None if args.fixture_only else {'registry_bytes': args.registry.read_bytes(),
            'split_bytes': args.split.read_bytes(), 'development': _object(args.development.read_bytes()),
            'training_task_ids': json.loads(args.training.read_bytes())}
        frozen = freeze_manifest(_object(args.proposal.read_bytes()), _object(args.config.read_bytes()))
        sources = _object(args.sources.read_bytes())
        receipt = attest_inputs(frozen, args.challenges, args.solutions, sources, args.prompt,
                                source_root=args.source_root, source_inventory=inventory, allocation=allocation)
        document = frozen if args.part == 'frozen' else receipt
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f'preparation failed: {type(exc).__name__}', file=sys.stderr)
        return 1
    print(json.dumps(document, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
