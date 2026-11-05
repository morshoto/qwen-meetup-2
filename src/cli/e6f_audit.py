"""Audit completed E6-F artifacts without inference or generated-code execution."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.e6_curriculum import diagnose_e5_artifact, grid_fingerprint
from src.arc.http_backend import final_answer
from src.arc.mlx_lora import _training_safe_record
from src.pipelines.e6 import _verify_seal
from src.pipelines.e6f import PROJECT, _verify_final

OUTPUT = PROJECT / 'artifacts/e6f/audit-v2'
SNAPSHOT = PROJECT / 'artifacts/e6f/implementation-snapshot-before-hardening'


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluated_rates(events: list[dict]) -> dict:
    """Separate budget-interrupted initial validation from evaluated failures."""
    terminal = {x['candidate_index']: x for x in events if x.get('event') == 'candidate_finished'}
    first = {x['candidate_index']: x for x in events
             if x.get('event') == 'attempt_finished' and x.get('attempt') == 0}
    excluded = {'budget_exhausted', 'timeout_or_resource_limit', 'model_request_failed',
                'sandbox_infrastructure_error', 'invalid_model_response'}
    comparable = [index for index, trial in first.items()
                  if index in terminal and trial.get('category') not in excluded and
                  terminal[index].get('status') not in {'budget_exhausted', 'interrupted',
                                                       'timeout_or_resource_limit', 'model_request_failed',
                                                       'sandbox_infrastructure_error'}]
    before = sum(first[i].get('category') == 'passed' for i in comparable)
    after = sum(bool(terminal[i].get('validation_pass')) for i in comparable)
    return {'comparable_candidates': len(comparable), 'baseline_passes': before,
            'after_repair_passes': after,
            'baseline_rate': before / len(comparable) if comparable else None,
            'after_repair_rate': after / len(comparable) if comparable else None,
            'excluded_incomplete_candidates': len(first) - len(comparable)}


def _audit_run(root: Path) -> dict:
    _verify_final(root)
    identity = json.loads((root / 'identity.json').read_text())
    archived = []
    for name, expected in identity['implementation_sha256'].items():
        if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != expected:
            snapshot = SNAPSHOT / 'e6f.py'
            if name != 'src/pipelines/e6f.py' or not snapshot.is_file() or \
                    hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
                raise ValueError(f'implementation hash differs and no verified snapshot exists: {name}')
            archived.append(name)
    state = json.loads((root / 'run_state.json').read_text())
    metrics = json.loads((root / 'metrics.json').read_text())
    budget = json.loads((root / 'budget_ledger.json').read_text())
    if (not state['complete'] or state['training_invoked'] or state['adapter_updated'] or
            budget['training_steps'] != 0 or state['public_evaluation_accessed']):
        raise ValueError('unexpected training, adapter update, or public evaluation')
    if any(budget[current] > budget[limit] for current, limit in (
            ('model_calls', 'max_model_calls'), ('reserved_tokens', 'max_reserved_tokens'),
            ('sandbox_runs', 'max_sandbox_runs'))):
        raise ValueError('client budget exceeded')
    events = _rows(root / 'oracle_lineage.jsonl')
    sft = _rows(root / 'sft_generator_distillation.jsonl')
    heldout = {r['task_id']: r for r in _rows(root / 'private/heldout_verifier.jsonl')}
    for row in sft:
        _training_safe_record(row, 'e6_generator_distillation')
        hidden = heldout[row['task_id']]
        if (grid_fingerprint(hidden['inputs']) != row['heldout_input_fingerprint'] or
                grid_fingerprint(hidden['outputs']) != row['heldout_output_fingerprint']):
            raise ValueError('held-out fingerprints differ')
        visible = row['visible_task']
        visible_inputs = [x['input'] for x in visible['train']] + visible['test_inputs']
        canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'))
        if {canonical(x) for x in visible_inputs} & {canonical(x) for x in hidden['inputs']}:
            raise ValueError('held-out and solver inputs overlap')
    if (len({r['task_id'] for r in sft}) != len(sft) or
            len({r['oracle_ast_sha256'] for r in sft}) != len(sft) or
            len({r['oracle_shape_sha256'] for r in sft}) != len(sft) or
            len(sft) != state['unique_verified_oracles'] or
            (root / 'sft_self_solved.jsonl').read_text()):
        raise ValueError('invalid SFT uniqueness or self-solve separation')
    split = json.loads((root / 'split_task_manifest.json').read_text())
    groups = [set(split[k]) for k in ('exploration', 'promotion_gate', 'monitoring')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('ARC Training splits overlap')
    if any(r['task_id'] in set.union(*groups) for r in sft):
        raise ValueError('real ARC task entered synthetic collection')
    sensitive = ['oracle_lineage.jsonl', 'sft_generator_distillation.jsonl',
                 'private/heldout_verifier.jsonl', 'private/accepted_records.jsonl']
    if root.name.startswith('live-'):
        sensitive += ['logs/e6/llm_requests.jsonl', 'logs/e6/llm_responses.jsonl']
    if root.stat().st_mode & 0o077 or any((root / name).stat().st_mode & 0o077 for name in sensitive):
        raise ValueError('private E6-F files or directory are accessible to other users')
    server_ledgers = []
    for path in (root / 'checkpoints/e6').glob('*ledger.sqlite'):
        with sqlite3.connect(f'file:{path.resolve()}?mode=ro&immutable=1', uri=True) as connection:
            token_limit, reserved = connection.execute('select token_limit,reserved from budget').fetchone()
            count, total = connection.execute('select count(*),coalesce(sum(reserved_tokens),0) from requests').fetchone()
        if reserved != total or reserved > token_limit or token_limit > budget['max_reserved_tokens']:
            raise ValueError('server token ledger does not satisfy its cap')
        server_ledgers.append({'token_limit': token_limit, 'reserved_tokens': reserved, 'requests': count})
    if server_ledgers and sum(x['requests'] for x in server_ledgers) != budget['model_calls']:
        raise ValueError('client and server model-call counts differ')
    duplicates = Counter()
    if root.name == 'live-collection':
        response_by_id = {r['request_id']: r for r in _rows(root / 'logs/e6/llm_responses.jsonl')}
        seed = [item['record'] for item in _rows(root / 'private/seed_records.jsonl')]
        for item in events:
            if item.get('event') != 'candidate_finished' or item.get('status') != 'duplicate_oracle':
                continue
            trial = next(x for x in events if x.get('event') == 'attempt_finished' and
                         x.get('candidate_index') == item['candidate_index'] and x.get('category') == 'passed')
            code = final_answer(response_by_id[trial['request_id']]['generated_text']).strip()
            if any(stable_ast_sha256(code) == r['oracle_ast_sha256'] for r in seed):
                duplicates['exact_ast'] += 1
            else:
                duplicates['normalized_shape'] += 1
    return {'run': root.name, 'seal_valid': True, 'state': state, 'budget': budget,
            'server_ledgers': server_ledgers, 'split_sizes': [len(x) for x in groups],
            'archived_implementation_files': archived,
            'unbound_runtime_dependencies': [] if 'src/arc/training.py' in identity['implementation_sha256']
                                            else ['src/arc/training.py'],
            'sft_records_validated': len(sft), 'self_solved_records': 0,
            'private_permissions_valid': True, 'rates_excluding_incomplete': evaluated_rates(events),
            'attempt_categories': dict(Counter(x['category'] for x in events if x.get('event') == 'attempt_finished')),
            'candidate_rejections': metrics['candidate_rejection_counts'], 'duplicate_breakdown': dict(duplicates)}


def main() -> None:
    source = PROJECT / 'artifacts/e6/e6-local-bootstrap-e526bae72f'
    _verify_seal(source)
    e5 = diagnose_e5_artifact(PROJECT / 'artifacts/e5-live-smoke-20261101-compact-feedback-final')
    diagnosis = PROJECT / 'artifacts/e6f/diagnosis'
    expected = json.loads((diagnosis / 'seal.json').read_text())
    actual = {str(p.relative_to(diagnosis)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(diagnosis.rglob('*')) if p.is_file() and p.name != 'seal.json'}
    if expected != actual:
        raise ValueError('offline diagnosis seal mismatch')
    snapshot_expected = json.loads((SNAPSHOT / 'seal.json').read_text())
    snapshot_actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in SNAPSHOT.iterdir() if p.is_file() and p.name != 'seal.json'}
    if snapshot_expected != snapshot_actual:
        raise ValueError('prior implementation snapshot seal mismatch')
    runs = [_audit_run(PROJECT / 'artifacts/e6f' / name)
            for name in ('cpu-fake-e2e-v2', 'live-smoke', 'live-collection')]
    summary = {'protocol': 'e6f_independent_artifact_audit_v1', 'runs': runs,
               'e5_source_root_seal_sha256': e5['source_root_seal_sha256'],
               'e6_source_root_seal_sha256': hashlib.sha256((source / 'artifact_hashes.json').read_bytes()).hexdigest(),
               'diagnosis_seal_valid': True, 'implementation_snapshot_seal_valid': True,
               'model_calls': 0, 'sandbox_starts': 0}
    OUTPUT.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(OUTPUT, 0o700)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (OUTPUT / 'report.md').write_text('# E6-F artifact audit\n\nAll three E6-F run seals, the source E5/E6 seals, '
        'the diagnosis seal, SFT provenance, held-out separation, split disjointness, private permissions, '
        'client/server budgets, and zero training were verified. The collection contains two valid SFT records. '
        'The prior run pinned its pipeline source but did not separately pin src/arc/training.py; '
        'that historical identity limitation is not retroactively repaired.\n')
    seal = {str(p.relative_to(OUTPUT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUTPUT.rglob('*')) if p.is_file() and p.name != 'seal.json'}
    (OUTPUT / 'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    for path in OUTPUT.iterdir():
        if path.is_file(): os.chmod(path, 0o600)
    print(json.dumps({'output': str(OUTPUT), 'runs_verified': len(runs),
                      'collection_rates_excluding_incomplete': runs[-1]['rates_excluding_incomplete']}))


if __name__ == '__main__':
    main()
