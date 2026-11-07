"""Save a private draft collection proposal; never freeze or launch a run."""
import argparse
import json
from pathlib import Path

from src.arc.lt_phase_b_proposal_dispatch import save_collection_proposal

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'pool', 'challenges', 'official-archive', 'python', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = save_collection_proposal(source_root=ROOT, run_dir=args.run_dir, pool_path=args.pool,
        challenge_path=args.challenges, official_archive=args.official_archive,
        python_executable=args.python, model_path=args.model_path)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
