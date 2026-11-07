"""Prepare private Phase B SFT data using the registered tokenizer runtime."""
import argparse
import json
from pathlib import Path

from src.arc.lt_phase_b_prepare_dispatch import run_prepare_sft

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'challenges', 'solutions', 'official-archive', 'python', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('data', 'collection'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    args = parser.parse_args()
    result = run_prepare_sft(source_root=ROOT, run_dir=args.run_dir, expected_data_sha256=args.expected_data_sha256,
        expected_collection_sha256=args.expected_collection_sha256, challenge_path=args.challenges,
        solution_path=args.solutions, official_archive=args.official_archive, python_executable=args.python,
        model_path=args.model_path)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
