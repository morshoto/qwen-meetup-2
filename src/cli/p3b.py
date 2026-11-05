"""Explicit P3-B QLoRA pilot and verifier-backed Exploration trainer."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from src.core.config import load_experiment_config
from src.pipelines.p3b import collect_verified_exploration_records, synthetic_pilot_records, train_p3b
from src.pipelines.p3b_compare import compare_adapter


def _output(config, name: str, supplied: Path | None) -> Path:
    return supplied or (Path(config.tracking.output_dir) / f'{name}-{uuid.uuid4().hex[:12]}')


def main() -> None:
    parser = argparse.ArgumentParser(description='P3-B QLoRA training')
    subparsers = parser.add_subparsers(dest='command', required=True)
    pilot = subparsers.add_parser('synthetic-pilot', help='Train and reload an adapter using artificial canary data only')
    pilot.add_argument('--config', required=True)
    pilot.add_argument('--output', type=Path)
    train = subparsers.add_parser('train-from-run', help='Reverify a completed B run and train only verified Exploration solutions')
    train.add_argument('--config', required=True)
    train.add_argument('--source-run', type=Path, required=True)
    train.add_argument('--output', type=Path)
    train.add_argument('--previous-adapter', type=Path)
    compare = subparsers.add_parser('compare', help='Compare B and adapter-backed C on unused Exploration tasks')
    compare.add_argument('--config', required=True)
    compare.add_argument('--training-run', type=Path, required=True)
    compare.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.command == 'synthetic-pilot':
        records = synthetic_pilot_records()
        manifest = {'protocol': 'p3b_synthetic_pilot_v1', 'synthetic_only': True,
                    'arc_data_used': False, 'public_solutions_used': False,
                    'sample_count': len(records)}
        result = train_p3b(config, records, manifest, _output(config, 'p3b1-synthetic', args.output), source='synthetic_pilot')
    elif args.command == 'train-from-run':
        records, manifest = collect_verified_exploration_records(
            args.source_run, required_tasks=config.p3b.training.collection_tasks)
        result = train_p3b(config, records, manifest, _output(config, 'p3b3-qlora', args.output),
                            source='exploration', previous_adapter=args.previous_adapter)
    else:
        result = compare_adapter(config, args.training_run,
                                 _output(config, 'p3b3-compare', args.output))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get('status') == 'failed':
        print('P3-B training did not produce a verified adapter', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
