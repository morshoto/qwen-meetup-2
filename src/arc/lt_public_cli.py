"""Export only registered public tasks to stdout; no model is started."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.arc.lt_attestation import _object
from src.arc.lt_comparison import verify_manifest
from src.arc.lt_public import build_public_export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'receipt', 'challenges'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        frozen, receipt = _object(args.frozen.read_bytes()), _object(args.receipt.read_bytes())
        verify_manifest(frozen)
        raw = args.challenges.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if (receipt.get('comparison_sha256') != frozen['sha256'] or
                digest != frozen['manifest']['dataset_sha256'] or digest != receipt.get('challenges_sha256') or
                set(receipt['selected_task_sha256']) != set(frozen['manifest']['task_ids'])):
            raise ValueError('registered dataset or selection differs')
        result = build_public_export(_object(raw), receipt)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f'public export failed: {type(exc).__name__}', file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
