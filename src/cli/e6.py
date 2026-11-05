"""Run the bounded E6 generator-distillation experiment."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.core.config import load_experiment_config
from src.pipelines.e6 import run_e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--live', action='store_true', help='required for guarded Qwen inference')
    parser.add_argument('--allow-training', action='store_true',
                        help='required only for the one bounded E6 QLoRA bootstrap')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.check:
        if args.allow_training and config.e6.profile != 'local_bootstrap':
            parser.error('--allow-training is only valid for local_bootstrap')
        print(json.dumps({
            'valid': True,
            'profile': config.e6.profile,
            'live_required_to_execute': config.e6.profile != 'cpu_fake',
            'live_flag_present': args.live,
            'training_authorization_required': config.e6.profile == 'local_bootstrap',
            'training_authorization_present': args.allow_training,
            'generation_range': list(range(config.e6.max_generations + 1)),
            'min_unique_verified_tasks': config.e6.min_unique_verified_tasks,
            'max_collection_attempts': config.e6.max_collection_attempts,
            'source_mixing_allowed': config.e6.allow_source_mixing,
            'budgets': {
                'model_calls': config.e6.max_model_calls,
                'reserved_tokens': config.e6.max_reserved_tokens,
                'sandbox_runs': config.e6.max_sandbox_runs,
                'training_steps': config.e6.max_training_steps,
                'generations': config.e6.max_generations,
                'wall_seconds': config.e6.max_wall_seconds,
                'training_seconds': config.e6.max_training_seconds,
            },
            'model_id': config.model.id,
            'model_revision': config.model.revision,
            'model_backend': config.model.backend,
            'sandbox': config.agent.sandbox,
            'config': asdict(config),
        }, indent=2, ensure_ascii=False))
        return
    if config.e6.profile != 'cpu_fake' and not args.live:
        parser.error('Live E6 requires explicit --live; --check only validates settings')
    if config.e6.profile == 'local_bootstrap' and not args.allow_training:
        parser.error('E6 bootstrap requires explicit --allow-training')
    if config.e6.profile != 'local_bootstrap' and args.allow_training:
        parser.error('--allow-training is only valid for local_bootstrap')
    result = run_e6(config, resume=args.resume, run_dir=args.run_dir,
                    live=args.live, allow_training=args.allow_training)
    print(result)


if __name__ == '__main__':
    main()
