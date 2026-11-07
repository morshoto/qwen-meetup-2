"""Run one bounded, registered synthetic Phase B canary training attempt."""
import argparse
import json
from pathlib import Path

from src.arc.lt_phase_b_canary_dispatch import run_canary

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'challenges', 'official-archive', 'python', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-collection-sha256', required=True)
    parser.add_argument('--expected-canary-sha256', required=True)
    args = parser.parse_args()
    result = run_canary(source_root=ROOT, run_dir=args.run_dir,
        expected_collection_sha256=args.expected_collection_sha256, expected_canary_sha256=args.expected_canary_sha256,
        challenge_path=args.challenges, official_archive=args.official_archive,
        python_executable=args.python, model_path=args.model_path)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
