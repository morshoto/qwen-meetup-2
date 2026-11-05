"""Run E6-F oracle repair and collection without training."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.arc.e6f_config import load_e6f_config
from src.pipelines.e6f import run_e6f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--allow-collection', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    settings, base = load_e6f_config(args.config)
    if args.check:
        print(json.dumps({'valid': True, 'settings': asdict(settings),
                          'training_enabled': False}, indent=2))
        return
    print(run_e6f(settings, base, live=args.live, allow_collection=args.allow_collection,
                  resume=args.resume, run_dir=args.run_dir))


if __name__ == '__main__':
    main()
