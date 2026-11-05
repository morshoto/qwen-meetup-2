"""Read-only verification of a sealed E7 comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.e6g import _verify_g
from src.pipelines.e6f import _sha


def audit(root: Path) -> dict:
    _verify_g(root)
    protocol = json.loads((root / 'protocol.lock.yml').read_text())
    if (root / 'protocol.sha256').read_text().strip() != _sha(root / 'protocol.lock.yml'):
        raise ValueError('protocol hash mismatch')
    comparison = json.loads((root / 'comparison.json').read_text())
    rows = json.loads((root / 'slots.json').read_text())
    schedule = protocol['slot_manifest']
    if [(r['arm'], r['slot']) for r in rows] != [(s['arm'], s['slot']) for s in schedule]:
        raise ValueError('slot schedule mismatch')
    settings = protocol['settings']
    for row in rows:
        slot = root / 'conditions' / row['arm'] / f"slot-{row['slot']}"
        _verify_g(slot)
        budget = row.get('budget', {})
        if (budget.get('sandbox_runs', 0) > settings['candidate_sandbox'] or
                budget.get('model_calls', 0) > settings['candidate_calls'] or
                budget.get('reserved_tokens', 0) > settings['candidate_tokens']):
            raise ValueError('candidate budget exceeded')
        events_path = slot / 'events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        dispatches = [x for x in events if x.get('event') == 'sandbox_dispatch']
        if len(dispatches) > budget.get('sandbox_runs', 0):
            raise ValueError('unreserved sandbox dispatch')
        if row.get('completion') == 'completed' and len(dispatches) != budget.get('sandbox_runs', 0):
            raise ValueError('completed candidate sandbox trace missing')
        for event in events:
            if event.get('event') == 'generation_requested':
                expected = settings['seed'] + row['slot'] * 3 + event['attempt']
                if event['seed_requested'] != expected:
                    raise ValueError('slot seed mismatch')
                if row['arm'] == 'G0_baseline' and event['feedback_applied']:
                    raise ValueError('baseline received novelty feedback')
    for arm, budget in comparison['arm_budgets'].items():
        if (budget['sandbox_runs'] > settings['arm_sandbox'] or budget['model_calls'] > settings['arm_calls'] or
                budget['reserved_tokens'] > settings['arm_tokens']):
            raise ValueError('arm budget exceeded')
        expected = [r.get('budget', {}) for r in rows if r['arm'] == arm]
        for key in ('sandbox_runs', 'model_calls', 'reserved_tokens'):
            if sum(x.get(key, 0) for x in expected) > budget[key]:
                raise ValueError('arm ledger undercounts candidate reservations')
    total = comparison['total_budget']
    if (total['sandbox_runs'] > settings['total_sandbox'] or total['model_calls'] > settings['total_calls'] or
            total['reserved_tokens'] > settings['total_tokens'] or total['training_steps'] != 0):
        raise ValueError('total budget or training violation')
    for key in ('sandbox_runs', 'model_calls', 'reserved_tokens'):
        if sum(b[key] for b in comparison['arm_budgets'].values()) > total[key]:
            raise ValueError('total ledger undercounts arm reservations')
    return {'seal_valid': True, 'protocol_sha256': _sha(root / 'protocol.lock.yml'),
            'decision': comparison['decision'], 'feasibility_gate': comparison['feasibility_gate'],
            'slots_checked': len(rows), 'note': 'ledger, schedule and trace audit; not semantic proof'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2))


if __name__ == '__main__':
    main()
