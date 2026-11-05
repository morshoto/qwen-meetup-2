"""Run the finite-budget E5 self-improvement experiment."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.arc.data import load_arc, split_tasks
from src.core.config import load_experiment_config
from src.pipelines.e5 import run_e5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--live', action='store_true', help='required to run guarded MLX inference')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.check:
        challenges = Path(config.data.challenges).expanduser()
        solutions = Path(config.data.solutions).expanduser()
        if not challenges.is_absolute():
            challenges = (Path.cwd() / challenges).resolve()
        if not solutions.is_absolute():
            solutions = (Path.cwd() / solutions).resolve()
        tasks, _ = load_arc(challenges, solutions)
        splits = split_tasks(list(tasks), config.data.split_seed,
                             {'exploration': config.data.exploration,
                              'promotion_gate': config.data.promotion_gate,
                              'monitoring': config.data.monitoring})
        if config.e5.profile != 'cpu_fake' and not args.live:
            live_required = True
        else:
            live_required = False
        print(json.dumps({
            'valid': True,
            'live_required_to_execute': config.e5.profile != 'cpu_fake',
            'live_flag_present': args.live,
            'config_live_check_only': bool(args.live),
            'live_missing': live_required,
            'conditions': config.e5.conditions,
            'generations': list(range(config.e5.generations + 1)),
            'exploration_tasks_per_generation': config.e5.real_arc_tasks_per_generation,
            'synthetic_tasks_per_generation': config.e5.synthetic_tasks_per_generation,
            'split_sizes': {key: len(value) for key, value in splits.items()},
            'model_id': config.model.id,
            'model_revision': config.model.revision,
            'model_backend': config.model.backend,
            'sandbox': config.agent.sandbox,
            'budgets': {
                'model_calls': config.e5.max_model_calls,
                'reserved_tokens': config.e5.max_reserved_tokens,
                'sandbox_runs': config.e5.max_sandbox_runs,
                'training_steps': config.e5.max_training_steps,
                'wall_seconds': config.e5.max_wall_seconds,
                'training_seconds': config.e5.max_training_seconds,
            },
            'config': asdict(config),
        }, indent=2, ensure_ascii=False))
        return
    if config.e5.profile != 'cpu_fake' and not args.live:
        parser.error('Live E5 requires explicit --live; --check only validates settings')
    result = run_e5(config, resume=args.resume, run_dir=args.run_dir, live=args.live)
    print(result)


if __name__ == '__main__':
    main()
