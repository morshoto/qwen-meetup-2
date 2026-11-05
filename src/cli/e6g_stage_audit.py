"""Join historical Qwen response hashes to recorded validation stages."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from src.cli.e6g_diagnose import PROJECT, RUNS, _verify_sources
from src.pipelines.e6f import _verify_final

OUTPUT = PROJECT / 'artifacts/e6g/stage-audit'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    source_seals = _verify_sources()
    diagnosis = PROJECT / 'artifacts/e6g/diagnosis-v2'
    seed = PROJECT / 'artifacts/e6g/seed-pool-v2'
    _verify_final(diagnosis)
    _verify_final(seed)
    prior = _rows(seed / 'private/pool.jsonl')
    prior_requests = {x['record']['qwen_provenance']['request_id'] for x in prior}
    with (diagnosis / 'duplicates.csv').open(newline='') as stream:
        observed = list(csv.DictReader(stream))
    run_paths = {name: PROJECT / relative for name, relative, _ in RUNS}
    trial_events = {}
    final_events = {}
    for name in ('e6f_smoke', 'e6f_collection'):
        for row in _rows(run_paths[name] / 'oracle_lineage.jsonl'):
            if row.get('event') == 'attempt_finished' and 'attempt' in row:
                trial_events[(name, row['candidate_index'], row['attempt'])] = row
            elif row.get('event') == 'candidate_finished':
                final_events[(name, row['candidate_index'])] = row
    results = []
    for row in observed:
        name = row['run']
        item = {'run': name, 'request_id': row['request_id'], 'candidate_id': row['candidate_id'],
                'generation_attempt': row['generation_attempt'], 'generation_stage': row['generation_stage'],
                'raw_model_text_sha256': row['raw_generated_text_sha256'], 'raw_source_hash': row['raw_source_hash'],
                'canonical_ast_hash': row['canonical_ast_hash'], 'behavior_fingerprint': row['behavior_fingerprint'],
                'behavior_probe_statuses': row['behavior_statuses'], 'extraction_status': row['extraction_status'],
                'static_stage': 'unknown_insufficient_evidence',
                'sandbox_stage': 'unknown_insufficient_evidence',
                'registration_stage': 'unknown_insufficient_evidence',
                'legacy_identity_status': row['legacy_identity_status'], 'evidence': row['source_log']}
        if name.startswith('e6f'):
            candidate, attempt = int(row['candidate_id']), int(row['generation_attempt']) % 3
            trial = trial_events.get((name, candidate, attempt))
            final = final_events.get((name, candidate), {})
            if trial:
                category = trial.get('category', 'unknown_insufficient_evidence')
                stage = trial.get('validation_stage')
                item['recorded_attempt_category'] = category
                item['recorded_validator_stage'] = stage or 'not_recorded'
                if stage in {'public_validation', 'private_acceptance'}:
                    item['static_stage'] = 'passed_to_sandbox'
                    item['sandbox_stage'] = category
                elif category == 'repeated_code':
                    item['static_stage'] = 'skipped_repeated_code'
                    item['sandbox_stage'] = 'not_executed_for_repeated_code'
                elif category in {'syntax_error', 'parse_or_contract_error', 'prohibited_operation',
                                  'static_safety_rejected'}:
                    item['static_stage'] = category
                    item['sandbox_stage'] = 'not_executed_after_static_rejection'
                if category == 'passed':
                    item['registration_stage'] = final.get('status', 'unknown_insufficient_evidence')
                elif category not in {'budget_exhausted', 'model_request_failed'}:
                    item['registration_stage'] = 'not_registered_this_attempt'
                item['evidence'] += f"; {name}/oracle_lineage.jsonl candidate={candidate} attempt={attempt}"
        elif row['request_id'] in prior_requests:
            item['registration_stage'] = 'historical_seed_reaudited'
        elif row['historical_final_status'] == 'sandbox_execution_failed':
            item['static_stage'] = 'passed_to_sandbox'
            item['sandbox_stage'] = 'historical_execution_failure_truncated'
            item['registration_stage'] = 'not_registered_this_attempt'
        elif row['historical_final_status'] == 'duplicate_oracle_fingerprint':
            item['static_stage'] = 'passed_to_sandbox'
            item['sandbox_stage'] = 'historically_verified_before_deduplication'
            item['registration_stage'] = 'duplicate_oracle_fingerprint'
        results.append(item)
    if _verify_sources() != source_seals:
        raise ValueError('historical source changed during stage audit')
    OUTPUT.mkdir(parents=True, mode=0o700, exist_ok=False)
    keys = sorted({key for row in results for key in row})
    with (OUTPUT / 'stages.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    summary = {'protocol': 'e6g_historical_stage_audit_v1', 'model_calls': 0, 'sandbox_starts': 0,
        'source_seals': source_seals, 'diagnosis_seal_sha256': _sha(diagnosis / 'seal.json'),
        'seed_pool_seal_sha256': _sha(seed / 'seal.json'), 'observations': len(results),
        'static_stage_counts': dict(Counter(r['static_stage'] for r in results)),
        'sandbox_stage_counts': dict(Counter(r['sandbox_stage'] for r in results)),
        'registration_stage_counts': dict(Counter(r['registration_stage'] for r in results)),
        'implementation_sha256': _sha(PROJECT / 'src/cli/e6g_stage_audit.py'),
        'limits': ['Unknown stages are not inferred from final candidate status.',
                   'Historical E6-F static-filter identity was incomplete.']}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (OUTPUT / 'report.md').write_text('# E6-G historical stage audit\n\n' +
        f"Joined {len(results)} response hashes to their recorded stages; no model calls or Sandbox starts. "
        'Unrecorded intermediate stages remain unknown. See the owner-only CSV for identifiers and hashes.\n')
    seal = {p.name: _sha(p) for p in OUTPUT.iterdir() if p.is_file()}
    (OUTPUT / 'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    for path in OUTPUT.iterdir(): os.chmod(path, 0o600)
    print(json.dumps({'output': str(OUTPUT), 'observations': len(results),
                      'sandbox_stages': summary['sandbox_stage_counts']}))


if __name__ == '__main__':
    main()
