"""Read-only reconstruction of historical E6-G sandbox accounting."""
from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

from src.pipelines.e6g import _seal_g, _verify_g

UNKNOWN = 'unknown_insufficient_evidence'


def diagnose(source: Path, output: Path) -> None:
    _verify_g(source)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(output, 0o700)
    rows = []
    for arm in ('G0_baseline', 'G1_novelty_feedback'):
        condition = source / 'conditions' / arm
        _verify_g(condition)
        events = [json.loads(line) for line in (condition / 'lineage.jsonl').read_text().splitlines()]
        for event in events:
            if event.get('event') != 'attempt_finished':
                continue
            before, after = event['cost_before'], event['cost_after']
            rows.append({'arm': arm, 'candidate_id': f"{arm}-{event['candidate_index']}",
                'trace_id': event.get('trial_id', UNKNOWN), 'attempt': event['attempt'],
                'phase': event.get('validation_stage', UNKNOWN),
                'trigger': 'generation' if event['attempt'] == 0 else 'repair',
                'code_sha256': event.get('code_sha256', UNKNOWN),
                'input_digest': UNKNOWN, 'timestamp': UNKNOWN,
                'reason': event['category'],
                'sandbox_starts': after['sandbox_runs'] - before['sandbox_runs'],
                'evidence': 'recorded_attempt_counter_delta'})
    protocol = json.loads((source / 'experiment_protocol.json').read_text())
    totals = defaultdict(int)
    for row in rows:
        totals[row['candidate_id']] += row['sandbox_starts']
    data = {'source': str(source), 'attempts': rows, 'candidate_totals': dict(totals),
            'individual_sandbox_events': UNKNOWN,
            'g1_twenty_explanation': 'Recorded 20 starts on one passed attempt. Historical verifier code runs five public and five private inputs twice each; phase allocation is inferred from code and successful completion, not separately logged.',
            'cache_decision': 'No reuse: repeat executions are required for determinism.',
            'legacy_reported_oracles': 2, 'currently_reverified_oracles': 2,
            'strict_training_eligible_legacy_oracles': 0,
            'legacy_limitation': 'Original E6/E6-F identity omitted src/arc/training.py.',
            'protocol': protocol}
    g1 = next(row for row in rows if row['candidate_id'] == 'G1_novelty_feedback-0')
    timeline = []
    for number in range(20):
        timeline.append({'ordinal_inferred': number + 1, 'candidate_id': g1['candidate_id'],
            'trace_id': g1['trace_id'], 'attempt': 0, 'code_sha256': g1['code_sha256'],
            'phase': 'public_validation' if number < 10 else 'private_acceptance',
            'trigger': 'first_execution' if number % 2 == 0 else 'determinism_rerun',
            'input_digest': UNKNOWN, 'timestamp': UNKNOWN,
            'image_digest': protocol.get('sandbox_image_id', UNKNOWN),
            'reason': 'inferred_from_pinned_verifier_and_passed_attempt',
            'evidence': 'inference_not_recorded_individual_event'})
    (output / 'g1_inferred_timeline.json').write_text(json.dumps(timeline, indent=2) + '\n')
    with (output / 'g1_inferred_timeline.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(timeline[0]))
        writer.writeheader(); writer.writerows(timeline)
    with (output / 'attempts.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / 'diagnostics.json').write_text(json.dumps(data, indent=2) + '\n')
    (output / 'report.md').write_text('''# E6-G Sandbox accounting\n\nThe sealed lineage records candidate G1-0 attempt 0 increasing the Sandbox counter from 0 to 20; it passed private acceptance. The historical verifier performs five public and five private inputs, two independent executions each. Thus 10 public and 10 private starts are inferred from the pinned implementation and successful completion. There are no individual Docker trace IDs, input digests, or timestamps in the lineage.\n\nG0-0 used 5 + 5 + 0 = 10 starts; the third response repeated the second code and was not executed. G0-1 used 1 + 1 + 1 = 3. The static checks themselves start no Docker containers. The second G1 candidate was never started because its arm had exhausted the cap of 20. Repeated calls per input are determinism checks; treating them as redundant cached calls would weaken verification. The exact seed-source duplicate was nevertheless fully validated before novelty rejection.\n\nSeparate past work: two offline diagnosis passes used 60 starts and two seed re-audits used 70; these are not E6-G candidate costs. The legacy two records were reverified under current dependencies, but the original static-filter dependency was not pinned. They do not meet the strict historical provenance criterion.\n\nAll per-start fields absent from saved events remain unknown; phase attribution for individual starts is inferred, not an observed timeline. No source artifact was modified.\n''')
    for path in output.iterdir():
        os.chmod(path, 0o600)
    _seal_g(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/e6g/live-smoke'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path('artifacts/e7') / ('diagnosis-' + uuid.uuid4().hex[:12]) / 'diagnostics'
    diagnose(args.source, output)
    print(output)


if __name__ == '__main__':
    main()
