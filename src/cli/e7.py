"""E7 configuration checks and read-only historical diagnosis."""
from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from src.arc.e7_config import load_e7_config
from src.arc.e7_protocol import schedule
from src.cli.e7_diagnose import diagnose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-full', action='store_true')
    parser.add_argument('--full-budget', help='explicit totals: calls,tokens,sandbox,seconds')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    settings, base = load_e7_config(args.config)
    if args.check:
        print(json.dumps({'valid': True, 'settings': asdict(settings),
                          'schedule': schedule(settings), 'training_enabled': False}, indent=2))
        return
    if args.audit_only:
        if any((args.live, args.smoke, args.allow_full, args.full_budget, args.resume, args.run_dir)):
            parser.error('audit-only cannot be combined with execution flags')
        output = Path('artifacts/e7') / ('diagnosis-' + uuid.uuid4().hex[:12]) / 'diagnostics'
        diagnose(Path('artifacts/e6g/live-smoke'), output)
        print(output)
        return
    if settings.profile == 'live_smoke' and not (args.live and args.smoke):
        parser.error('live smoke requires both --live and --smoke')
    if settings.profile == 'live_full' and not (args.live and args.allow_full):
        parser.error('live full requires --live, --allow-full, and separately authorized budget')
    if settings.profile == 'cpu_fake' and (args.live or args.smoke or args.allow_full):
        parser.error('fake profile does not accept live flags')
    try:
        full_budget = tuple(int(x) for x in args.full_budget.split(',')) if args.full_budget else None
    except ValueError:
        parser.error('--full-budget must be calls,tokens,sandbox,seconds')
    from src.pipelines.e7 import run_e7
    print(run_e7(settings, base, run_dir=args.run_dir, resume=args.resume,
                 live=args.live, smoke=args.smoke, allow_full=args.allow_full, full_budget=full_budget))


if __name__ == '__main__':
    main()
