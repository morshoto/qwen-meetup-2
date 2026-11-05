"""E3 verifier-backed solution collection from unused Exploration tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config import load_experiment_config
from src.pipelines.exploration import check_collect, collect_solutions


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect verifier-approved ARC solutions without training')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--selection-dir', type=Path, required=True,
                        help='completed E2 directory containing its frozen strategy')
    parser.add_argument('--check', action='store_true', help='validate frozen strategy and unused task plan')
    parser.add_argument('--live', action='store_true', help='use the configured guarded local model and Docker')
    parser.add_argument('--dry-run', action='store_true', help='run the deterministic CPU Fake profile')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.live and args.dry_run:
        parser.error('--live and --dry-run are mutually exclusive')
    if args.check and (args.resume or args.run_dir or args.output_dir):
        parser.error('--check cannot be combined with execution output or resume options')
    if args.resume and args.output_dir:
        parser.error('--resume uses --run-dir and cannot use --output-dir')
    if not args.resume and args.run_dir:
        parser.error('--run-dir requires --resume')
    config = load_experiment_config(args.config)
    extras = args.exclude_manifest or None
    if args.check:
        result = check_collect(config, args.selection_dir, live=args.live,
                               extra_manifests=extras)
    else:
        result = collect_solutions(config, args.selection_dir,
                                   output_dir=args.output_dir, live=args.live,
                                   resume=args.resume, run_dir=args.run_dir,
                                   extra_manifests=extras)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
