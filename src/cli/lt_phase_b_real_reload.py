"""Supervise one separate registered real-data adapter reload."""
import argparse
import json
from pathlib import Path

from src.arc.lt_phase_b_real_reload_dispatch import run_real_reload_dispatch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'challenges', 'solutions', 'official-archive', 'python', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('training', 'canary', 'collection'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    args = parser.parse_args()
    result = run_real_reload_dispatch(source_root=ROOT, run_dir=args.run_dir,
        expected_training_sha256=args.expected_training_sha256, expected_canary_sha256=args.expected_canary_sha256,
        expected_collection_sha256=args.expected_collection_sha256, challenge_path=args.challenges,
        solution_path=args.solutions, official_archive=args.official_archive, python_executable=args.python,
        model_path=args.model_path)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
