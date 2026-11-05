"""Verify that saved Qwen text comes from the hashed HTTP response body."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from src.arc.e6g_provenance import verify_response_body
from src.arc.http_backend import BackendError, final_answer
from src.cli.e6g_diagnose import PROJECT, RUNS, _verify_sources
from src.pipelines.e6f import _verify_final

OUTPUT = PROJECT / 'artifacts/e6g/provenance-audit'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    seals = _verify_sources()
    diagnosis = PROJECT / 'artifacts/e6g/diagnosis-v2'
    _verify_final(diagnosis)
    with (diagnosis / 'duplicates.csv').open(newline='') as stream:
        diagnostic = {(row['run'], row['request_id']): row for row in csv.DictReader(stream)}
    outcomes = []
    for name, relative, log_relative in RUNS:
        log = PROJECT / relative / log_relative
        requests = [r for r in _rows(log / 'llm_requests.jsonl') if r.get('kind') == 'e5_generator']
        responses = {r.get('request_id'): r for r in _rows(log / 'llm_responses.jsonl')}
        for request in requests:
            response = responses.get(request.get('request_id'), {})
            row = diagnostic.get((name, request.get('request_id')))
            status = 'verified'
            try:
                body = response['raw_http_body_text']
                content = json.loads(body)['choices'][0]['message']['content']
                if (_sha(body.encode()) != request['response_sha256'] or
                        content != response['generated_text'] or not row or
                        _sha(final_answer(content).encode()) != row['raw_source_hash']):
                    status = 'mismatch'
            except (BackendError, KeyError, IndexError, TypeError, ValueError):
                status = 'unknown_insufficient_evidence'
            outcomes.append({'run': name, 'request_id': request.get('request_id'), 'status': status})
    pool_root = PROJECT / 'artifacts/e6g/seed-pool'
    _verify_final(pool_root)
    pool = _rows(pool_root / 'private/pool.jsonl')
    seed_status = []
    for item in pool:
        row = item['record']
        try:
            verify_response_body(row['qwen_provenance'], row['messages'][-1]['content'])
            status = 'verified'
        except (ValueError, OSError):
            status = 'mismatch'
        seed_status.append({'task_id': row['task_id'], 'status': status})
    if _verify_sources() != seals:
        raise ValueError('historical source changed during provenance audit')
    summary = {'protocol': 'e6g_http_body_provenance_v1', 'source_seals': seals,
        'diagnosis_seal_sha256': _sha((diagnosis/'seal.json').read_bytes()),
        'seed_pool_seal_sha256': _sha((pool_root/'seal.json').read_bytes()),
        'implementation_sha256': {name: _sha((PROJECT/name).read_bytes()) for name in (
            'src/arc/e6g_provenance.py', 'src/cli/e6g_provenance_audit.py',
            'src/arc/mlx_lora.py', 'src/arc/http_backend.py')},
        'model_calls': 0, 'sandbox_starts': 0,
        'response_counts': dict(Counter(item['status'] for item in outcomes)),
        'seed_counts': dict(Counter(item['status'] for item in seed_status)),
        'responses': outcomes, 'seeds': seed_status}
    OUTPUT.mkdir(parents=True, mode=0o700, exist_ok=False)
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (OUTPUT/'report.md').write_text('# Qwen HTTP response provenance audit\n\n' +
        f"Historical generator responses: {summary['response_counts']}. "
        f"Seed Oracles: {summary['seed_counts']}. No model calls or sandbox starts.\n")
    seal = {p.name: _sha(p.read_bytes()) for p in OUTPUT.iterdir() if p.is_file()}
    (OUTPUT/'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    for path in OUTPUT.iterdir(): os.chmod(path, 0o600)
    print(json.dumps({'output': str(OUTPUT), 'response_counts': summary['response_counts'],
                      'seed_counts': summary['seed_counts']}))


if __name__ == '__main__':
    main()
