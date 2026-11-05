"""E2 paired S0-S3 comparison; live inference requires an explicit flag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config import load_experiment_config
from src.pipelines.exploration import (check_compare, compare_strategies,
                                       reselect_completed_comparison)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare pre-registered ARC solution search strategies')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--check', action='store_true', help='validate config and task exclusions without inference')
    parser.add_argument('--live', action='store_true', help='use the configured guarded local model and Docker')
    parser.add_argument('--dry-run', action='store_true', help='run the deterministic CPU Fake profile')
    parser.add_argument('--resume', action='store_true', help='resume an existing interrupted E2 run')
    parser.add_argument('--run-dir', type=Path, help='existing E2 run directory used with --resume')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reselect-from', type=Path,
                        help='derive a corrected selection from a sealed completed E2 Full run without inference')
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.reselect_from:
        if (not args.output_dir or args.live or args.dry_run or args.check or args.resume or
                args.run_dir or args.exclude_manifest):
            parser.error('--reselect-from requires --output-dir and cannot be combined with execution options')
        result = reselect_completed_comparison(args.reselect_from, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    if not args.config:
        parser.error('--config is required unless --reselect-from is used')
    if args.live and args.dry_run:
        parser.error('--live and --dry-run are mutually exclusive')
    if args.check and (args.resume or args.run_dir):
        parser.error('--check cannot be combined with --resume or --run-dir')
    if args.resume and args.output_dir:
        parser.error('--resume uses --run-dir and cannot use --output-dir')
    if not args.resume and args.run_dir:
        parser.error('--run-dir requires --resume')
    config = load_experiment_config(args.config)
    extras = args.exclude_manifest or None
    if args.check:
        result = check_compare(config, live=args.live, extra_manifests=extras)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    result = compare_strategies(config, output_dir=args.output_dir, live=args.live,
                                resume=args.resume, run_dir=args.run_dir,
                                extra_manifests=extras)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
