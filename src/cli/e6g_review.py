"""Trace historical duplicates back to their re-audited seed evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from src.pipelines.e6f import PROJECT, _verify_final

OUTPUT = PROJECT / 'artifacts/e6g/diagnosis-review'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    diagnosis = PROJECT / 'artifacts/e6g/diagnosis'
    expected = json.loads((diagnosis / 'seal.json').read_text())
    actual = {str(p.relative_to(diagnosis)): _sha(p) for p in sorted(diagnosis.rglob('*'))
              if p.is_file() and p.name != 'seal.json'}
    if expected != actual:
        raise ValueError('G1 diagnosis seal mismatch')
    seed = PROJECT / 'artifacts/e6g/seed-pool'
    _verify_final(seed)
    pool = [json.loads(x) for x in (seed / 'private/pool.jsonl').read_text().splitlines() if x]
    seed_text_hashes = {}
    for item in pool:
        row = item['record']
        provenance = row['qwen_provenance']
        for line in Path(provenance['response_log_path']).read_text().splitlines():
            response = json.loads(line)
            if response.get('request_id') == provenance['request_id']:
                seed_text_hashes[row['oracle_ast_sha256']] = hashlib.sha256(
                    response['generated_text'].encode()).hexdigest()
                break
    with (diagnosis / 'duplicates.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['run'] == 'e6f_collection' and
                r['generation_stage'] == 'initial' and r['historical_final_status'] == 'duplicate_oracle']
    results = []
    for row in rows:
        match = next((item['record'] for item in pool
                      if item['record']['oracle_ast_sha256'] == row['canonical_ast_hash']), None)
        if not match:
            results.append({'candidate_id': row['candidate_id'], 'classification': 'unknown_insufficient_evidence'})
            continue
        same_source = row['raw_source_hash'] == match['oracle_sha256']
        same_text = row['raw_generated_text_sha256'] == seed_text_hashes.get(row['canonical_ast_hash'])
        category = ('same_model_text' if same_text else 'extraction_collapsed_difference' if same_source
                    else 'ast_normalization_collapsed_difference')
        results.append({'candidate_id': row['candidate_id'], 'request_id': row['request_id'],
                        'classification': category, 'matches_seed_ast': True,
                        'matches_seed_source': same_source, 'matches_seed_generated_text': same_text})
    summary = {'protocol': 'e6g_duplicate_stage_review_v1', 'source_diagnosis_seal_sha256': _sha(diagnosis / 'seal.json'),
               'source_pool_seal_sha256': _sha(seed / 'seal.json'), 'model_calls': 0, 'sandbox_starts': 0,
               'duplicate_candidates': len(results), 'classification_counts': dict(Counter(x['classification'] for x in results)),
               'legacy_identity_status': 'legacy_identity_incomplete', 'candidates': results}
    OUTPUT.mkdir(parents=True, mode=0o700, exist_ok=False)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (OUTPUT / 'report.md').write_text('# Duplicate stage review\n\n' +
        f"Four E6-F candidates matched a re-audited seed AST: {summary['classification_counts']}. "
        'Only hashes and request identifiers are recorded. The old static-filter hash remains unproven.\n')
    seal = {p.name: _sha(p) for p in OUTPUT.iterdir() if p.is_file()}
    (OUTPUT / 'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    for p in OUTPUT.iterdir(): os.chmod(p, 0o600)
    print(json.dumps({'output': str(OUTPUT), 'classification_counts': summary['classification_counts']}))


if __name__ == '__main__':
    main()
