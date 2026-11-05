"""Offline E1 failure diagnosis for an existing P3-A/B/C run artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.exploration_diagnose import diagnose_run


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze saved ARC run failures without model inference')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = diagnose_run(args.run_dir, args.output_dir)
    print(json.dumps({'status': 'complete', 'output_dir': str(args.output_dir.resolve()),
                      'task_count': summary['task_count'],
                      'candidate_count': summary['candidate_count'],
                      'failure_breakdown': summary['failure_breakdown']},
                     ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
