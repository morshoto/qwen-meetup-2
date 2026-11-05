"""Supplement the E8 audit with slot separation and paired execution checks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from src.pipelines.e5 import _verify_mutable_file
from src.pipelines.e6g import _seal_g, _verify_g
from src.pipelines.e6f import _sha


def audit(campaign: Path, smoke: Path, output: Path) -> dict:
    for root in (campaign, smoke):
        _verify_mutable_file(root / 'protocol.lock.yml')
        if (root / 'protocol.sha256').read_text().strip() != _sha(root / 'protocol.lock.yml'):
            raise ValueError('protocol hash mismatch')
    full = json.loads((campaign / 'protocol.lock.yml').read_text())
    old = json.loads((smoke / 'protocol.lock.yml').read_text())
    ids = [x['slot'] for x in full['slots']]
    seeds = [y for x in full['slots'] for y in x['attempt_seeds']]
    if len(ids) != 10 or len(set(ids)) != 10 or set(ids) & {x['slot'] for x in old['slots']}:
        raise ValueError('invalid or reused slots')
    if set(seeds) & {v for x in old['slots'] for v in x['attempt_seeds']}:
        raise ValueError('reused seeds')
    summary = {}
    for stage in full['model_order']:
        root = campaign / stage
        _verify_g(root)
        if {int(x.name) for x in (root / 'slots').iterdir()} != set(ids):
            raise ValueError('stage slot set differs')
        verified = 0
        for slot in ids:
            path = root / 'slots' / str(slot)
            _verify_g(path)
            row = json.loads((path / 'result.json').read_text())
            if not row.get('oracle_valid'): continue
            events = [json.loads(x) for x in (path / 'events.jsonl').read_text().splitlines()]
            attempt = max(e['attempt'] for e in events if e['event'] == 'attempt_finished')
            dispatch = [e for e in events if e['event'] == 'sandbox_dispatch' and e['attempt'] == attempt]
            finished = {e['trace_id']: e['status'] for e in events if e['event'] == 'sandbox_finished'}
            groups = defaultdict(Counter)
            for event in dispatch:
                if finished.get(event['trace_id']) != 'ok': raise ValueError('verified sandbox execution failed')
                groups[event['phase']][event['input_digest']] += 1
            if set(groups) != {'public_validation', 'private_acceptance'}:
                raise ValueError('verified phase missing')
            if any(len(group) != 5 or set(group.values()) != {2} for group in groups.values()):
                raise ValueError('verified candidate lacks five input pairs per phase')
            if set(groups['public_validation']) & set(groups['private_acceptance']):
                raise ValueError('public and private inputs overlap')
            verified += 1
        summary[stage] = {'slot_ids': ids, 'verified_with_20_paired_executions': verified,
                          'seal_sha256': _sha(root / 'seal.json')}
    result = {'protocol_sha256': _sha(campaign / 'protocol.lock.yml'), 'smoke_excluded': True,
              'slots': summary, 'scope': 'Checks paired input dispatch and successful sandbox traces; the main audit checks request provenance and ledgers.'}
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    os.chmod(output / 'summary.json', 0o600)
    _seal_g(output)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path)
    p.add_argument('--smoke', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.campaign, a.smoke, a.output), indent=2))


if __name__ == '__main__': main()
