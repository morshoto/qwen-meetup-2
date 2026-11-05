"""Run a preregistered, bounded E8 model stage."""
from __future__ import annotations
import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from src.arc.e8_config import MODELS, ORDER, load_e8_config
from src.pipelines.e8 import run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--stage', choices=ORDER)
    parser.add_argument('--allow-32b', action='store_true')
    parser.add_argument('--allow-full', action='store_true')
    parser.add_argument('--full-budget', help='explicit calls,tokens,sandbox,seconds')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--campaign-dir', type=Path)
    parser.add_argument('--smoke-campaign', type=Path, default=Path('artifacts/e8/live-smoke-32b-retry-3600-20261101'))
    args = parser.parse_args()
    s = load_e8_config(args.config)
    if args.check:
        if s.profile == 'full':
            from src.cli.e8_full_prepare import prepare
            print(json.dumps(prepare(Path(args.config), args.smoke_campaign), indent=2)); return
        print(json.dumps({'valid': True, 'settings': asdict(s), 'models': MODELS,
                          'training_enabled': False}, indent=2)); return
    try:
        full_budget = tuple(int(x) for x in args.full_budget.split(',')) if args.full_budget else None
    except ValueError:
        parser.error('invalid --full-budget')
    if s.profile in ('fake', 'fake_full') and (args.live or args.allow_32b or args.allow_full or full_budget is not None):
        parser.error('fake profile cannot use live approval flags')
    if s.profile in ('fake', 'fake_full') and args.stage is None:
        campaign = args.campaign_dir or Path('artifacts/e8') / ('fake-' + uuid.uuid4().hex[:12])
        for stage in ORDER:
            run_stage(s, stage, campaign, resume=args.resume)
        print(campaign); return
    if not args.stage:
        parser.error('live execution requires an explicit stage')
    if not args.campaign_dir:
        parser.error('live execution requires an explicit campaign directory')
    print(run_stage(s, args.stage, args.campaign_dir, live=args.live, allow_32b=args.allow_32b,
                    allow_full=args.allow_full, full_budget=full_budget, resume=args.resume))


if __name__ == '__main__':
    main()
