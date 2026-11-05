"""Check or run one independent E9 Qwen model collection."""
from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.arc.e9_config import MODELS, load_e9_config
from src.pipelines.e9 import PROJECT, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--allow-extra-slots', action='store_true')
    parser.add_argument('--approve-slot-range', help='approved slot range as start:end, zero based')
    parser.add_argument('--allow-training', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--parent-run', type=Path)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    settings = load_e9_config(args.config)
    extra = settings.slot_offset > 0 or settings.slots > 10
    if args.check:
        print(json.dumps({'valid': True, 'settings': asdict(settings),
                          'models': {k: MODELS[k] for k in settings.models},
                          'extra_slots_require_approval': extra,
                          'training_enabled': False}, indent=2))
        return
    if settings.profile == 'fake':
        if any((args.live, args.collect_only, args.allow_extra_slots, args.allow_training,
                args.approve_slot_range is not None, args.parent_run is not None)):
            parser.error('fake E9 cannot use live approval flags')
        root = args.run_dir or PROJECT / 'artifacts/e9' / ('fake-' + uuid.uuid4().hex[:12])
        print('\n'.join(str(path) for path in run(settings, root, resume=args.resume)))
        return
    if not args.live or not args.collect_only:
        parser.error('live generation requires both --live and --collect-only')
    if args.allow_training:
        parser.error('training is a separate gated operation; collection CLI will not start training')
    if extra:
        if not args.allow_extra_slots or not args.approve_slot_range:
            parser.error('additional slots require --allow-extra-slots and --approve-slot-range')
        try:
            start, end = (int(value) for value in args.approve_slot_range.split(':'))
        except (ValueError, TypeError):
            parser.error('--approve-slot-range must be start:end')
        if (start, end) != (settings.slot_offset, settings.slot_offset + settings.slots):
            parser.error('approved range must exactly match the configured slot range')
    elif args.allow_extra_slots or args.approve_slot_range or args.parent_run:
        parser.error('extra-slot approval applies only to additional batches')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = args.run_dir or PROJECT / 'artifacts/e9' / settings.models[0] / f'live-{stamp}'
    paths = run(settings, run_dir, live=True, collect_only=True, resume=args.resume,
                approve_extra_slots=args.allow_extra_slots, parent_run=args.parent_run)
    print('\n'.join(str(path) for path in paths))


if __name__ == '__main__':
    main()
