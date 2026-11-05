"""Offline, sealed diagnosis of historical Qwen Oracle duplication."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from src.arc.ast_fingerprint import stable_ast_sha256, stable_shape_sha256
from src.arc.e5_selfplay import _safe_transform_code
from src.arc.e6_curriculum import _quality_grids, diagnose_e5_artifact
from src.arc.http_backend import BackendError, final_answer
from src.arc.sandbox import DockerSandbox
from src.pipelines.e6 import _verify_seal
from src.pipelines.e6f import PROJECT, _verify_final

OUTPUT = PROJECT / 'artifacts/e6g/diagnosis'
RUNS = (
    ('e5', 'artifacts/e5-live-smoke-20261101-compact-feedback-final', 'logs/e5'),
    ('e6_smoke', 'artifacts/e6/e6-local-smoke-fa35c09b83', 'logs/e6'),
    ('e6_bootstrap', 'artifacts/e6/e6-local-bootstrap-e526bae72f', 'logs/e6'),
    ('e6f_smoke', 'artifacts/e6f/live-smoke', 'logs/e6'),
    ('e6f_collection', 'artifacts/e6f/live-collection', 'logs/e6'),
)


def _hash(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _verify_sources() -> dict[str, str]:
    out = {}
    for name, relative, _ in RUNS:
        root = PROJECT / relative
        if name == 'e5':
            diagnose_e5_artifact(root)
            seal = root / 'root_artifact_hashes.json'
        elif name.startswith('e6f'):
            _verify_final(root)
            seal = root / 'seal.json'
        else:
            _verify_seal(root)
            seal = root / 'artifact_hashes.json'
        out[name] = _hash(seal.read_bytes())
    return out


def main() -> None:
    source_seals = _verify_sources()
    e6_root = PROJECT / 'artifacts/e6/e6-local-bootstrap-e526bae72f'
    old_identity = json.loads((e6_root / 'identity.json').read_text())
    agent = old_identity['config']['agent']
    sandbox = DockerSandbox(image=old_identity['sandbox_image_id'], timeout=agent['timeout_seconds'],
        memory_mb=agent['memory_mb'], cpus=agent['cpus'], pids=agent['pids'],
        max_output_bytes=agent['max_output_bytes'], max_code_bytes=agent['max_code_bytes'])
    probes = _quality_grids(196602, 5)
    probe_sha = _hash(json.dumps(probes, sort_keys=True, separators=(',', ':')))
    seed_records = _rows(e6_root / 'gen-0/sft_generator_distillation.jsonl')
    seed_ast = {row['oracle_ast_sha256'] for row in seed_records}
    observations: list[dict] = []
    behavior_cache: dict[str, dict] = {}
    sandbox_starts = 0
    seen_source: set[str] = set()
    seen_ast: set[str] = set()
    seen_text: set[str] = set()
    for run_name, relative, log_relative in RUNS:
        root = PROJECT / relative
        identity = json.loads((root / 'identity.json').read_text())
        cfg = identity.get('base_config', identity.get('config', {}))
        model = cfg.get('model', {})
        identity_incomplete = 'src/arc/training.py' not in identity.get('implementation_sha256', {})
        logs = root / log_relative
        requests = _rows(logs / 'llm_requests.jsonl')
        responses = {row.get('request_id'): row for row in _rows(logs / 'llm_responses.jsonl')}
        finished = {}
        if (root / 'oracle_lineage.jsonl').is_file():
            finished = {row['candidate_index']: row for row in _rows(root / 'oracle_lineage.jsonl')
                        if row.get('event') == 'candidate_finished'}
        rejections = {}
        if (root / 'gen-0/rejections.jsonl').is_file():
            rejections = {row.get('attempt'): row.get('reason', '')
                          for row in _rows(root / 'gen-0/rejections.jsonl')}
        for request in requests:
            if request.get('kind') != 'e5_generator':
                continue
            request_id = request.get('request_id')
            candidate = request.get('candidate')
            response = responses.get(request_id, {})
            raw_body = response.get('raw_http_body_text')
            text = response.get('generated_text')
            response_verified = (isinstance(raw_body, str) and
                                 _hash(raw_body) == request.get('response_sha256'))
            stage = ('repair' if run_name.startswith('e6f') and type(candidate) is int and candidate % 3 else 'initial')
            lineage = candidate // 3 if run_name.startswith('e6f') and type(candidate) is int else candidate
            result = finished.get(lineage, {}).get('status') if finished else None
            if result is None and not run_name.startswith('e6f'):
                reason = rejections.get(candidate, '')
                result = ('sandbox_execution_failed' if reason.startswith('sandbox execution failed:') else
                          reason if reason in {'duplicate_oracle_fingerprint'} else 'unknown_insufficient_evidence')
            row = {'run': run_name, 'request_id': request_id, 'candidate_id': lineage,
                   'generation_attempt': candidate, 'generation_stage': stage,
                   'response_hash_verified': response_verified,
                   'raw_generated_text_sha256': _hash(text) if isinstance(text, str) else None,
                   'raw_source_hash': None, 'canonical_ast_hash': None, 'shape_hash': None,
                   'behavior_fingerprint': None, 'behavior_statuses': None,
                   'historical_final_status': result or 'unknown_insufficient_evidence',
                   'model_revision': request.get('revision'), 'quantization': model.get('quantization'),
                   'thinking': request.get('thinking_mode_effective'), 'temperature': request.get('temperature'),
                   'requested_seed': request.get('seed_requested'), 'seed_support': request.get('seed_support'),
                   'prompt_hash': request.get('prompt_sha256'),
                   'legacy_identity_status': 'legacy_identity_incomplete' if identity_incomplete else 'complete',
                   'source_log': f'{relative}/{log_relative}/llm_requests.jsonl'}
            if not response_verified or not isinstance(text, str):
                row['extraction_status'] = 'unknown_insufficient_evidence'
                observations.append(row)
                continue
            try:
                code = final_answer(text)
            except BackendError:
                row['extraction_status'] = 'parse_or_contract_error'
                observations.append(row)
                continue
            row['raw_source_hash'] = _hash(code)
            try:
                row['canonical_ast_hash'] = stable_ast_sha256(code)
                row['shape_hash'] = stable_shape_sha256(code)
                row['extraction_status'] = 'parsed'
            except (SyntaxError, ValueError, TypeError, RecursionError):
                row['extraction_status'] = 'ast_parse_error'
                observations.append(row)
                continue
            row['raw_text_seen_before'] = row['raw_generated_text_sha256'] in seen_text
            row['source_seen_before'] = row['raw_source_hash'] in seen_source
            row['ast_seen_before'] = row['canonical_ast_hash'] in seen_ast
            row['matches_initial_pool_ast'] = row['canonical_ast_hash'] in seed_ast
            seen_text.add(row['raw_generated_text_sha256'])
            seen_source.add(row['raw_source_hash'])
            seen_ast.add(row['canonical_ast_hash'])
            key = row['canonical_ast_hash']
            if key not in behavior_cache:
                statuses = []
                if not _safe_transform_code(code):
                    statuses = [{'status': 'static_safety_rejected'}]
                else:
                    for grid in probes:
                        try:
                            sandbox_starts += 1
                            output = sandbox.run(code, grid)
                            statuses.append({'status': 'ok', 'output_sha256': _hash(
                                json.dumps(output, sort_keys=True, separators=(',', ':')))})
                        except TimeoutError:
                            statuses.append({'status': 'timeout_or_resource_limit'})
                        except BackendError:
                            statuses.append({'status': 'sandbox_infrastructure_error'})
                            break
                        except (ValueError, RuntimeError) as exc:
                            statuses.append({'status': 'execution_or_grid_failure', 'error_type': type(exc).__name__})
                behavior_cache[key] = {'statuses': statuses,
                    'fingerprint': _hash(json.dumps(statuses, sort_keys=True, separators=(',', ':')))}
            behavior = behavior_cache[key]
            row['behavior_fingerprint'] = behavior['fingerprint']
            row['behavior_statuses'] = [entry['status'] for entry in behavior['statuses']]
            observations.append(row)
    if _verify_sources() != source_seals:
        raise ValueError('a source artifact changed during offline diagnosis')
    collection = [r for r in observations if r['run'] == 'e6f_collection']
    duplicates = [r for r in collection if r['historical_final_status'] == 'duplicate_oracle' and
                  r['generation_stage'] == 'initial']
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in observations:
        groups[json.dumps({k: row.get(k) for k in ('run', 'model_revision', 'quantization',
                    'thinking', 'temperature', 'prompt_hash')}, sort_keys=True)].append(row)
    summaries = []
    for key, values in groups.items():
        summaries.append({**json.loads(key), 'requests': len(values),
            'distinct_requested_seeds': len({v.get('requested_seed') for v in values}),
            'distinct_source_hashes': len({v['raw_source_hash'] for v in values if v.get('raw_source_hash')}),
            'distinct_ast_hashes': len({v['canonical_ast_hash'] for v in values if v.get('canonical_ast_hash')})})
    implementation = ('src/cli/e6g_diagnose.py', 'src/arc/e5_selfplay.py',
                      'src/arc/e6_curriculum.py', 'src/arc/ast_fingerprint.py',
                      'src/arc/training.py', 'src/arc/sandbox.py', 'src/arc/data.py',
                      'src/arc/http_backend.py', 'src/pipelines/e5.py',
                      'src/pipelines/e6.py', 'src/pipelines/e6f.py')
    summary = {'protocol': 'e6g_historical_diagnostic_v1', 'source_seals': source_seals,
        'implementation_sha256': {name: _hash((PROJECT / name).read_bytes()) for name in implementation},
        'model_calls': 0, 'sandbox_starts': sandbox_starts, 'probe_seed': 196602,
        'probe_input_sha256': probe_sha, 'observations': len(observations),
        'parsed_candidates': sum(r['extraction_status'] == 'parsed' for r in observations),
        'duplicate_collection_candidates': len(duplicates),
        'duplicate_collection_candidates_matching_seed_ast': sum(r.get('matches_initial_pool_ast', False) for r in duplicates),
        'global_repeated_raw_text': sum(r.get('raw_text_seen_before', False) for r in observations),
        'global_repeated_source': sum(r.get('source_seen_before', False) for r in observations),
        'global_repeated_ast': sum(r.get('ast_seen_before', False) for r in observations),
        'behavior_probe_unique_ast_count': len(behavior_cache),
        'behavior_status_counts': dict(Counter(status for item in behavior_cache.values()
                                               for status in [r['status'] for r in item['statuses']])),
        'legacy_identity_incomplete_runs': sorted({r['run'] for r in observations
                                                   if r['legacy_identity_status'] == 'legacy_identity_incomplete'}),
        'settings_groups': summaries,
        'limits': ['A sampled behavior fingerprint is not proof of semantic equivalence.',
                   'Legacy identity omissions and unrecorded validator stages remain unknown.',
                   'Correlation of settings and duplicates does not establish causation.']}
    OUTPUT.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(OUTPUT, 0o700)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    keys = ['run', 'request_id', 'candidate_id', 'generation_attempt', 'generation_stage',
            'response_hash_verified', 'raw_generated_text_sha256', 'raw_source_hash', 'canonical_ast_hash',
            'shape_hash', 'behavior_fingerprint', 'behavior_statuses', 'extraction_status',
            'raw_text_seen_before', 'source_seen_before', 'ast_seen_before', 'matches_initial_pool_ast',
            'historical_final_status', 'model_revision', 'quantization', 'thinking', 'temperature',
            'requested_seed', 'seed_support', 'prompt_hash', 'legacy_identity_status', 'source_log']
    with (OUTPUT / 'duplicates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(observations)
    (OUTPUT / 'report.md').write_text(
        '# E6-G offline Oracle diagnosis\n\n'
        f"Observed {len(observations)} generator responses; model calls: 0; Docker starts: {sandbox_starts}.\n\n"
        f"E6-F Collection had {len(duplicates)} validated duplicate candidates, "
        f"{sum(r.get('matches_initial_pool_ast', False) for r in duplicates)} matching a seed AST.\n\n"
        f"Globally repeated raw model text: {summary['global_repeated_raw_text']}; "
        f"extracted source: {summary['global_repeated_source']}; AST: {summary['global_repeated_ast']}.\n\n"
        'The CSV records stage, hashes, sampled behavior outcomes and model settings; it contains no source or prompts. '
        'Identical sampled behavior is a suspicion, not proof of semantic identity. '
        'Legacy identity omissions and absent validation details remain explicitly unknown.\n')
    seal = {str(p.relative_to(OUTPUT)): _hash(p.read_bytes()) for p in sorted(OUTPUT.rglob('*'))
            if p.is_file() and p.name != 'seal.json'}
    (OUTPUT / 'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    for path in OUTPUT.iterdir():
        os.chmod(path, 0o600)
    print(json.dumps({'output': str(OUTPUT), 'observations': len(observations),
                      'collection_duplicates': len(duplicates), 'sandbox_starts': sandbox_starts}))


if __name__ == '__main__':
    main()
