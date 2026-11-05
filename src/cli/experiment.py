"""Run or validate the CPU experiment."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from src.core.config import load_experiment_config
from src.arc.data import load_arc, split_tasks
from src.pipelines.experiment import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.check:
        tasks, _ = load_arc(config.data.challenges, config.data.solutions)
        split_tasks(list(tasks), config.data.split_seed, {x: getattr(config.data, x) for x in ('exploration', 'promotion_gate', 'monitoring')})
        print(json.dumps({'valid': True, 'tasks': len(tasks), 'config': asdict(config)}, indent=2))
    else:
        path = run(config, resume=args.resume, run_dir=args.run_dir)
        print(path)
        state_path = path / 'run_state.json'
        if state_path.exists():
            status = json.loads(state_path.read_text()).get('status')
            if status != 'complete':
                print(f'Experiment did not complete: {status}', file=sys.stderr)
                raise SystemExit(2)

if __name__ == '__main__': main()
