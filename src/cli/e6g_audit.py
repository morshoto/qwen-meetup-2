"""Independently audit the sealed E6-G smoke and its comparison evidence."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.e5_selfplay import _generator_prompt
from src.arc.e6_curriculum import grid_fingerprint
from src.arc.e6f_repair import repair_prompt
from src.arc.e6g_config import load_e6g_config
from src.arc.e6g_novelty import NoveltyFeedback, classify_novelty, source_hash
from src.arc.e6g_provenance import verify_response_body
from src.arc.http_backend import final_answer
from src.arc.mlx_lora import _training_safe_record
from src.cli.e6g_diagnose import _verify_sources
from src.pipelines.e6g import (PROJECT, _compat, _condition_metrics, _docker_image_id, _live_provenance,
                              _load_pool, _protocol, _read_jsonl, _sha, _split_evidence, _verify_g)
from src.pipelines.e5 import _verify_mutable_file
from src.pipelines.e6f import _verify_final

OUTPUT = PROJECT / 'artifacts/e6g/audit-v5'
SNAPSHOT = PROJECT / 'artifacts/e6g/implementation-snapshot-before-metric-fix'


def _condition(root: Path, name: str, protocol: dict, initial_pool: list[dict]) -> dict:
    condition = root / 'conditions' / name
    _verify_g(condition)
    result = json.loads((condition / 'result.json').read_text())
    journal = _read_jsonl(condition / 'lineage.jsonl')
    requests = _read_jsonl(condition / 'logs/e6/llm_requests.jsonl')
    responses = {r['request_id']: r for r in _read_jsonl(condition / 'logs/e6/llm_responses.jsonl')}
    request_map = {r['request_id']: r for r in requests}
    if len(requests) != result['model_calls']:
        raise ValueError('condition client call count differs from request log')
    pool = json.loads(json.dumps(initial_pool))
    feedback = NoveltyFeedback(enabled=name == 'G1_novelty_feedback', pool=pool)
    prompt_by_trial: dict[str, str] = {}
    code_by_trial: dict[str, str] = {}
    failure_by_trial: dict[str, tuple[str, str | None]] = {}
    passed_code: dict[int, str] = {}
    seen_code, seen_ast = set(), set()
    self_raw_repeats = self_ast_repeats = 0
    verified_calls = 0
    protocol_sha = _sha(root / 'experiment_protocol.json')
    for event in journal:
        kind = event.get('event')
        if kind == 'generation_requested':
            trial = event['generation_attempt_id']
            candidate = event['candidate_index']
            attempt = int(trial.rsplit('-attempt-', 1)[1])
            if (event['protocol_sha256'] != protocol_sha or
                    event['pool_snapshot_sha256'] != protocol['initial_pool']['pool_snapshot_file_sha256'] or
                    event['model_revision'] != protocol['model']['revision'] or
                    event['seed_requested'] != protocol['settings']['seed'] + candidate * 3 + attempt):
                raise ValueError('generation identity or seed differs from the registered protocol')
            if attempt == 0:
                expected_prompt = feedback.prompt(_generator_prompt())
            else:
                previous_trial = f'{name}-{candidate}-attempt-{attempt - 1}'
                prior_category, prior_type = failure_by_trial[previous_trial]
                expected_prompt = repair_prompt(code_by_trial[previous_trial], prior_category, prior_type)
            if (hashlib.sha256(expected_prompt.encode()).hexdigest() != event['prompt_sha256'] or
                    event['feedback_applied'] != (attempt == 0 and name == 'G1_novelty_feedback' and
                                                  feedback.previous_code is not None)):
                raise ValueError('recorded prompt differs from the registered feedback boundary')
            if event['feedback_applied']:
                previous = feedback.previous_code
                if previous is None or event['feedback_code_sha256'] != source_hash(previous):
                    raise ValueError('feedback did not use the expected Qwen-authored source')
            prompt_by_trial[trial] = expected_prompt
        elif kind == 'novelty_observed':
            trial = event['generation_attempt_id']
            request_id = event['request_id']
            request = request_map[request_id]
            response = responses[request_id]
            code = final_answer(response['generated_text']).strip()
            candidate = event['candidate_index']
            attempt = int(trial.rsplit('-attempt-', 1)[1])
            evidence = _live_provenance(condition, prompt_by_trial[trial], candidate * 3 + attempt, code)
            if evidence['request_id'] != request_id or request.get('seed_requested') != protocol['settings']['seed'] + candidate * 3 + attempt:
                raise ValueError('request provenance or requested seed differs')
            if (event['raw_source_hash'] != source_hash(code) or
                    event['canonical_ast_hash'] != stable_ast_sha256(code) or
                    event['classification'] != feedback.observe(code, pool)):
                raise ValueError('stored novelty classification does not match its code')
            raw, tree = source_hash(code), stable_ast_sha256(code)
            self_raw_repeats += raw in seen_code
            self_ast_repeats += tree in seen_ast
            seen_code.add(raw); seen_ast.add(tree)
            code_by_trial[trial] = code
            verified_calls += 1
        elif kind == 'attempt_finished':
            trial = f"{name}-{event['candidate_index']}-attempt-{event['attempt']}"
            failure_by_trial[trial] = (event['category'], event.get('exception_type'))
            if event['category'] == 'passed':
                passed_code[event['candidate_index']] = code_by_trial[trial]
        elif kind == 'candidate_finished':
            if event.get('oracle_valid') and event.get('novelty') != 'reserved_task_rejected':
                code = passed_code[event['candidate_index']]
                fingerprint = event['behavior_fingerprint']
                if (event['novelty'] != classify_novelty(code, fingerprint, pool) or
                        event['behavior_match'] != any(x['behavior_sha256'] == fingerprint for x in pool)):
                    raise ValueError('final novelty or sampled behavior classification differs')
                exact = [x for x in pool if source_hash(x['record']['messages'][-1]['content']) == source_hash(code)]
                if exact and any(x['behavior_sha256'] != fingerprint for x in exact):
                    raise ValueError('identical source did not match the re-audited seed behavior')
            if event.get('accepted_path'):
                item = json.loads((condition / event['accepted_path']).read_text())
                pool.append(item)
    finished = [x for x in journal if x.get('event') == 'candidate_finished']
    completed = [x for x in finished if x['completion'] == 'completed']
    if (len(completed) != result['completed_candidates'] or len(finished) != result['started_candidates'] or
            sum(bool(x.get('verified_novel')) for x in completed) != result['verified_novel_oracles'] or
            sum(bool(x.get('oracle_valid')) for x in completed) != result['oracle_valid'] or
            verified_calls != len(requests)):
        raise ValueError('condition results do not match the append-only trial journal')
    condition_budget = json.loads((condition / 'budget.json').read_text())
    recomputed = _condition_metrics(name, journal, condition_budget, condition,
        protocol['initial_pool']['pool_snapshot_file_sha256'], protocol['settings']['candidates_per_condition'])
    corrected_keys = {'behavior_evaluated_candidates', 'sampled_behavior_suspect_rate'}
    if any(result.get(key) != value for key, value in recomputed.items() if key not in corrected_keys):
        raise ValueError('condition metrics do not match the journal and budget')
    if (result['initial_pool_snapshot_sha256'] != protocol['initial_pool']['pool_snapshot_file_sha256'] or
            result['model_calls'] > result['budget_limits']['model_calls'] or
            result['reserved_tokens'] > result['budget_limits']['reserved_tokens'] or
            result['sandbox_runs'] > result['budget_limits']['sandbox_runs'] or result['training_steps']):
        raise ValueError('condition pool, budget, or no-training invariant failed')
    server = []
    for path in sorted((condition / 'checkpoints/e6').glob('*ledger.sqlite')):
        with sqlite3.connect(f'file:{path.resolve()}?mode=ro&immutable=1', uri=True) as connection:
            limit, reserved = connection.execute('select token_limit,reserved from budget').fetchone()
            count, total = connection.execute('select count(*),coalesce(sum(reserved_tokens),0) from requests').fetchone()
        if limit > result['budget_limits']['reserved_tokens'] or reserved != total or reserved > limit:
            raise ValueError('guarded server ledger exceeds its condition limit')
        server.append({'token_limit': limit, 'reserved_tokens': reserved, 'requests': count})
    if sum(x['requests'] for x in server) != len(requests):
        raise ValueError('guarded server and client request counts differ')
    sensitive = [condition/'lineage.jsonl', condition/'logs/e6/llm_requests.jsonl',
                 condition/'logs/e6/llm_responses.jsonl', condition/'checkpoints/e6/request_cache.json']
    if condition.stat().st_mode & 0o077 or any(p.stat().st_mode & 0o077 for p in sensitive):
        raise ValueError('sensitive condition artifacts are not owner-only')
    return {'seal_valid': True, 'requests_verified_against_http_bodies': verified_calls,
            'prompt_boundary_verified': True, 'self_repeated_source_emissions': self_raw_repeats,
            'self_repeated_ast_emissions': self_ast_repeats, 'model_emissions': len(requests),
            'request_seed_support': dict(Counter(str(r.get('seed_support')) for r in requests)),
            'server_ledgers': server,
            'corrected_behavior_evaluated_candidates': recomputed['behavior_evaluated_candidates'],
            'corrected_sampled_behavior_suspect_rate': recomputed['sampled_behavior_suspect_rate'],
            'historical_sampled_behavior_suspect_rate': result['sampled_behavior_suspect_rate'],
            'unstarted_candidates': result['candidate_limit'] - result['started_candidates'],
            'candidate_statuses': dict(Counter(x['status'] for x in finished))}


def main() -> None:
    root = PROJECT / 'artifacts/e6g/live-smoke'
    _verify_g(root)
    _verify_mutable_file(root/'experiment_protocol.json')
    protocol = json.loads((root / 'experiment_protocol.json').read_text())
    comparison = json.loads((root / 'comparison.json').read_text())
    state = json.loads((root / 'run_state.json').read_text())
    settings, base = load_e6g_config(PROJECT / 'configs/e6g.live.smoke.yml')
    cfg = _compat(base, calls=settings.total_calls, tokens=settings.total_reserved_tokens,
                  sandboxes=settings.total_sandbox, seconds=settings.total_wall_seconds, seed=settings.seed)
    image = _docker_image_id(cfg)
    splits, forbidden = _split_evidence(base)
    pool, probes, info = _load_pool(settings, base, image, splits)
    diagnostic = json.loads((PROJECT/'artifacts/e6g/diagnosis-v2/summary.json').read_text())
    if _verify_sources() != diagnostic['source_seals']:
        raise ValueError('a historical E5, E6, or E6-F artifact changed')
    info['pool_snapshot_file_sha256'] = _sha(root/'private/initial_pool.jsonl')
    expected_snapshot = json.loads((SNAPSHOT/'seal.json').read_text())
    actual_snapshot = {p.name: _sha(p) for p in SNAPSHOT.iterdir() if p.is_file() and p.name != 'seal.json'}
    if expected_snapshot != actual_snapshot:
        raise ValueError('prior E6-G implementation snapshot seal mismatch')
    current_protocol = _protocol(settings, base, image, splits, info, probes, protocol['git_evidence'])
    historical_pipeline_hash = protocol['implementation_sha256']['src/pipelines/e6g.py']
    if _sha(SNAPSHOT/'e6g.py') != historical_pipeline_hash:
        raise ValueError('prior E6-G pipeline snapshot does not match the recorded protocol')
    pipeline_changed = current_protocol['implementation_sha256']['src/pipelines/e6g.py'] != historical_pipeline_hash
    current_protocol['implementation_sha256']['src/pipelines/e6g.py'] = historical_pipeline_hash
    if current_protocol != protocol:
        raise ValueError('current dependency, model, data, pool, or configuration differs from protocol')
    if any(set(splits[a]) & set(splits[b]) for a, b in (
            ('exploration', 'promotion_gate'), ('exploration', 'monitoring'), ('promotion_gate', 'monitoring'))):
        raise ValueError('ARC Training splits overlap')
    conditions = {name: _condition(root, name, protocol, pool)
                  for name in ('G0_baseline', 'G1_novelty_feedback')}
    budget = json.loads((root/'budget_ledger.json').read_text())
    if (budget['training_steps'] or state['training_invoked'] or state['adapter_updated'] or
            comparison['promoted'] or comparison['public_evaluation_accessed'] or
            any(budget[x] > budget[y] for x, y in (('model_calls', 'max_model_calls'),
                ('reserved_tokens', 'max_reserved_tokens'), ('sandbox_runs', 'max_sandbox_runs')))):
        raise ValueError('global budget or no-training invariant failed')
    if any(sum(comparison['conditions'][name][key] for name in conditions) != budget[budget_key]
           for key, budget_key in (('model_calls', 'model_calls'), ('reserved_tokens', 'reserved_tokens'),
                                   ('sandbox_runs', 'sandbox_runs'))):
        raise ValueError('condition costs do not add up to the global ledger')
    sft = _read_jsonl(root/'sft_generator_distillation.jsonl')
    labels = {row['task_id']: row for row in _read_jsonl(root/'private/heldout_verifier.jsonl')}
    for row in sft:
        _training_safe_record(row, 'e6_generator_distillation')
        verify_response_body(row['qwen_provenance'], row['messages'][-1]['content'])
        hidden = labels[row['task_id']]
        if (grid_fingerprint(hidden['inputs']) != row['heldout_input_fingerprint'] or
                grid_fingerprint(hidden['outputs']) != row['heldout_output_fingerprint'] or
                row['task_id'] in forbidden):
            raise ValueError('SFT provenance, held-out labels, or task partition is invalid')
    if len(sft) != state['eligible_unique_oracles'] or len({r['oracle_ast_sha256'] for r in sft}) != len(sft):
        raise ValueError('eligible Oracle count or AST uniqueness mismatch')
    private = [root/'private/initial_pool.jsonl', root/'private/combined_records.jsonl',
               root/'private/heldout_verifier.jsonl', root/'sft_generator_distillation.jsonl']
    if root.stat().st_mode & 0o077 or any(path.stat().st_mode & 0o077 for path in private):
        raise ValueError('root SFT or pool files are not owner-only')
    fake = PROJECT/'artifacts/e6g/cpu-fake-e2e'
    _verify_g(fake)
    fake_result = json.loads((fake/'comparison.json').read_text())
    if fake_result['execution_mode'] != 'fake_fixture' or fake_result['training_invoked']:
        raise ValueError('Fake comparison was mislabelled or trained')
    fixed_fake = PROJECT/'artifacts/e6g/cpu-fake-e2e-v2'
    _verify_g(fixed_fake)
    summary = {'protocol': 'e6g_smoke_independent_audit_v1',
        'experiment_protocol_sha256': _sha(root/'experiment_protocol.json'),
        'source_run_seal_sha256': _sha(root/'seal.json'), 'seed_pool': info,
        'implementation_sha256': {name: _sha(PROJECT/name) for name in (
            'src/cli/e6g_audit.py', 'src/arc/e6g_provenance.py', 'src/arc/e6g_novelty.py',
            'src/arc/e6f_repair.py', 'src/pipelines/e6g.py')},
        'split_sizes': {name: len(ids) for name, ids in splits.items()},
        'conditions': conditions, 'global_budget': budget, 'eligible_sft_records': len(sft),
        'fake_seal_valid': True, 'fixed_fake_seal_valid': True, 'historical_source_seals_valid': True,
        'historical_pipeline_snapshot_sha256': historical_pipeline_hash,
        'current_pipeline_differs_from_historical': pipeline_changed,
        'decision': comparison['decision'], 'training_invoked': False, 'adapter_updated': False,
        'model_calls': 0, 'sandbox_starts': 0,
        'limitations': ['Legacy E6-F identities did not separately pin src/arc/training.py.',
                        'Sampled output equality is a suspicion, not proof of semantic equivalence.',
                        'The guarded server contract is self-reported and not remote attestation.']}
    OUTPUT.mkdir(parents=True, mode=0o700, exist_ok=False)
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (OUTPUT/'report.md').write_text('# E6-G independent smoke audit\n\n'
        'Run and condition seals, source and pool identities, prompt boundaries, Qwen HTTP bodies, '
        'seed requests, split disjointness, SFT provenance, budgets, and zero training were checked. '
        'The legacy static-filter identity limitation remains. The archived pipeline matches the historical '
        'protocol; behavior rates are corrected to exclude candidates without fingerprints.\n')
    seal = {p.name: _sha(p) for p in OUTPUT.iterdir() if p.is_file()}
    (OUTPUT/'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    for path in OUTPUT.iterdir(): os.chmod(path, 0o600)
    print(json.dumps({'output': str(OUTPUT), 'decision': comparison['decision'],
                      'eligible_sft_records': len(sft), 'conditions': conditions}))


if __name__ == '__main__':
    main()
