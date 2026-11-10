"""Run a bounded registered Phase B collection batch after local preflight."""
import argparse
import json
from pathlib import Path

from src.arc.lt_phase_b_dispatch import run_collection

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'challenges', 'official-archive', 'python', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-run-sha256', required=True)
    parser.add_argument('--limit', type=int, required=True)
    args = parser.parse_args()
    result = run_collection(source_root=ROOT, run_dir=args.run_dir,
        expected_run_sha256=args.expected_run_sha256, challenge_path=args.challenges,
        official_archive=args.official_archive, python_executable=args.python, model_path=args.model_path,
        limit=args.limit)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
