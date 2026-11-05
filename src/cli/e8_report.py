"""Produce a sealed read-only E8 comparison report."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from src.arc.e8_config import ORDER
from src.arc.e6g_novelty import source_hash
from src.arc.ast_fingerprint import stable_ast_sha256
from src.pipelines.e5 import _verify_mutable_file
from src.pipelines.e6g import _seal_g, _verify_g
from src.pipelines.e6f import _sha


def wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z / total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def report(campaign: Path, output: Path) -> dict:
    lock = campaign / 'protocol.lock.yml'
    _verify_mutable_file(lock)
    if (campaign / 'protocol.sha256').read_text().strip() != _sha(lock):
        raise ValueError('campaign protocol hash mismatch')
    _verify_mutable_file(campaign / 'initial_pool.json')
    protocol = json.loads(lock.read_text())
    rows = []
    accepted = []
    comparable = True
    for stage in ORDER:
        root = campaign / stage
        if not root.exists():
            rows.append({'stage': stage, 'status': 'not_executed', 'planned_slots': protocol['settings']['slots']})
            continue
        if not (root / 'seal.json').exists():
            rows.append({'stage': stage, 'status': 'unsealed_in_progress_or_interrupted', 'planned_slots': protocol['settings']['slots']})
            continue
        _verify_g(root)
        result = json.loads((root / 'result.json').read_text())
        if result['campaign_protocol_sha256'] != _sha(lock):
            raise ValueError('stage protocol mismatch')
        if (result['model_id'], result['revision']) != tuple(protocol['model_identities'][stage][:2]):
            comparable = False
        for path in (root / 'slots').glob('*/result.json'):
            item = json.loads(path.read_text()).get('accepted_item')
            if item:
                code = item['record']['messages'][-1]['content']
                accepted.append({'stage': stage, 'source': source_hash(code),
                                 'ast': stable_ast_sha256(code), 'behavior': item['behavior_sha256']})
        row = {k: result.get(k) for k in ('stage', 'status', 'stop_reason', 'model_id', 'revision',
            'planned_slots', 'completed_slots', 'valid_oracles', 'strict_new_oracles', 'source_duplicates',
            'ast_duplicates', 'behavior_matches', 'behavior_evaluated', 'repair_successes',
            'runtime_error_attempts', 'format_error_attempts', 'model_calls', 'reserved_tokens',
            'observed_tokens', 'sandbox_starts', 'elapsed_seconds', 'model_inference_seconds',
            'peak_sampled_server_rss_kib', 'training_gate', 'oom_observed_in_logs',
            'novel_per_planned_slot', 'novel_per_completed_slot', 'cost_per_novel_oracle')}
        row['wilson_95_completed_slots'] = wilson(row['strict_new_oracles'] or 0, row['completed_slots'] or 0)
        row['template_sha256'] = result.get('model_manifest', {}).get('chat_template_sha256')
        row['quantization'] = result.get('model_manifest', {}).get('quantization')
        row['model_load_seconds'] = result.get('canary', {}).get('model_load_seconds')
        requests = [json.loads(line) for log in (root / 'slots').glob('*/logs/e6/llm_requests.jsonl')
                    for line in log.read_text().splitlines() if line.strip()]
        measured = [r for r in requests if isinstance(r.get('usage'), dict) and isinstance(r['usage'].get('completion_tokens'), int)]
        elapsed = sum(r.get('elapsed_seconds') or 0 for r in measured)
        row['generation_completion_tokens'] = sum(r['usage']['completion_tokens'] for r in measured) if measured else None
        row['generation_tokens_per_second'] = row['generation_completion_tokens'] / elapsed if measured and elapsed else None
        row['swap_before_mib'] = result.get('before', {}).get('swap_used_mib')
        row['swap_after_mib'] = result.get('after', {}).get('swap_used_mib')
        row['sampled_min_free_percent'] = min((x['memory_free_percent'] for x in result.get('resource_samples', [])), default=None)
        slots_path = root / 'slots.json'
        slots = json.loads(slots_path.read_text()) if slots_path.exists() else []
        row['slot_status_counts'] = dict(Counter(x.get('status', 'unknown') for x in slots))
        row['seconds_per_novel_oracle'] = (row['elapsed_seconds'] / row['strict_new_oracles']
            if row.get('strict_new_oracles') else None)
        rows.append(row)
    cross = []
    for i, a in enumerate(accepted):
        for b in accepted[i+1:]:
            if a['stage'] != b['stage']:
                cross.append({'stages': [a['stage'], b['stage']], 'same_source': a['source'] == b['source'],
                              'same_ast': a['ast'] == b['ast'], 'sampled_behavior_match': a['behavior'] == b['behavior']})
    summary = {'protocol_sha256': _sha(lock), 'profile': protocol['settings']['profile'],
               'models': rows, 'cross_model_comparisons': cross,
               'comparison_status': ('not_comparable' if not comparable else
                   'complete' if all(r.get('status') == 'completed' for r in rows) else 'inconclusive_resource_limited'),
               'caveat': 'small sample; finite behavior matches do not establish semantic equivalence'}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = ['stage', 'status', 'planned_slots', 'completed_slots', 'valid_oracles', 'strict_new_oracles',
              'ast_duplicates', 'behavior_matches', 'model_calls', 'reserved_tokens', 'observed_tokens',
              'sandbox_starts', 'elapsed_seconds', 'stop_reason']
    with (output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    lines = ['# E8 model scaling comparison', '', '| Model | Status | Completed / planned | Valid | Strict novel | AST duplicates | Calls | Reserved / actual | Docker |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in rows:
        measured = row.get('completed_slots', 0) > 0
        valid = row.get('valid_oracles', '—') if measured else '—'
        novel = row.get('strict_new_oracles', '—') if measured else '—'
        duplicates = row.get('ast_duplicates', '—') if measured else '—'
        lines.append(f"| {row['stage']} | {row['status']} | {row.get('completed_slots', '—')} / {row['planned_slots']} | "
                     f"{valid} | {novel} | {duplicates} | "
                     f"{row.get('model_calls', '—')} | {row.get('reserved_tokens', '—')} / {row.get('observed_tokens', '—')} | {row.get('sandbox_starts', '—')} |")
    lines += ['', 'Comparison status: ' + summary['comparison_status'], '',
              'Slot status counts: ' + '; '.join(f"{r['stage']}: {r.get('slot_status_counts', {})}" for r in rows),
              '', summary['caveat'], '']
    (output / 'report.md').write_text('\n'.join(lines))
    for p in output.iterdir(): os.chmod(p, 0o600)
    _seal_g(output)
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path); p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); print(json.dumps(report(a.campaign,a.output), indent=2))


if __name__ == '__main__': main()
