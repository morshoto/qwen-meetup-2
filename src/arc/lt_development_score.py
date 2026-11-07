"""Score recorded development predictions offline; never starts a model."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.arc.data import load_arc
from src.arc.lt_attestation import _object
from src.arc.lt_development_export import build_development_export
from src.arc.lt_scoring import score_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('challenges', 'exposure', 'manifest', 'receipt', 'solutions', 'results'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        challenges, exposure = args.challenges.read_bytes(), args.exposure.read_bytes()
        manifest, receipt = _object(args.manifest.read_bytes()), _object(args.receipt.read_bytes())
        _, expected_receipt = build_development_export(challenges, exposure, manifest)
        if receipt != expected_receipt:
            raise ValueError('development receipt differs')
        solutions, results = args.solutions.read_bytes(), args.results.read_bytes()
        tasks, verifier = load_arc(args.challenges, args.solutions)
        if challenges != args.challenges.read_bytes() or solutions != args.solutions.read_bytes():
            raise ValueError('scoring data changed during loading')
        rows = [_object(line) for line in results.splitlines() if line.strip()]
        identity = receipt['development_manifest_sha256']
        if (any(row.get('development_manifest_sha256') != identity for row in rows) or
                not set(manifest['task_ids']) <= set(tasks)):
            raise ValueError('development result identity differs')
        report = score_comparison(manifest['task_ids'], rows, verifier)
        report.update(protocol='lt-phase-a-development', claim_eligible=False,
                      development_manifest_sha256=identity,
                      solutions_sha256=hashlib.sha256(solutions).hexdigest(),
                      results_sha256=hashlib.sha256(results).hexdigest(),
                      run_identity_independently_verified=False)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'development scoring failed: {type(exc).__name__}', file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
