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
    args = parser.parse_args(argv)
    try:
        frozen = freeze_manifest(_object(args.proposal.read_bytes()), _object(args.config.read_bytes()))
        sources = _object(args.sources.read_bytes())
        receipt = attest_inputs(frozen, args.challenges, args.solutions, sources, args.prompt)
        document = frozen if args.part == 'frozen' else receipt
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f'preparation failed: {type(exc).__name__}', file=sys.stderr)
        return 1
    print(json.dumps(document, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
