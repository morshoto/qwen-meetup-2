"""E6-F collection with guarded Qwen inference, repair, and no training path."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from src.arc.data import load_arc, split_tasks, validate_grid
from src.arc.e6_curriculum import (_quality_grids, collection_rows, diagnose_e5_artifact,
                                   grid_fingerprint, verify_oracle, write_private_jsonl)
from src.arc.e6f_config import E6FSettings
from src.arc.e6f_repair import run_candidate
from src.arc.http_backend import BackendError
from src.arc.mlx_lora import _training_safe_record
from src.arc.sandbox import DockerSandbox
from src.core.config import ExperimentConfig
from src.pipelines.e5 import (E5Budget, E5CachedBackend, MeteredSandbox, _backend,
                              _docker_image_id, _seal_mutable_file, _verify_mutable_file)
from src.pipelines.e6 import (_compat_e5, _json, _observed_token_usage, _request_provenance,
                              _run_lock, _secure_private_dirs, _verify_seal)

PROJECT = Path(__file__).resolve().parents[2]
EXCLUSIONS = PROJECT / 'configs/p3c.exploration.exclusions.v1.json'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seal(root: Path) -> None:
    files = {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name not in {'seal.json', '.run.lock'}}
    _json(root / 'seal.json', files, private=True)


def _verify_final(root: Path) -> None:
    actual = {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob('*'))
              if p.is_file() and p.name not in {'seal.json', '.run.lock'}}
    if not (root / 'seal.json').is_file() or json.loads((root / 'seal.json').read_text()) != actual:
        raise ValueError('E6-F artifact seal mismatch')


def _append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    _seal_mutable_file(path)


def _prefixes(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for rel in ('logs/e6/llm_requests.jsonl', 'logs/e6/llm_responses.jsonl'):
        path = root / rel
        if path.is_file():
            result[rel] = {'size': path.stat().st_size, 'sha256': _sha(path)}
    return result


def _verify_prefixes(root: Path, prefixes: dict[str, Any]) -> None:
    for rel, expected in prefixes.items():
        path = root / rel
        size = expected['size']
        if not path.is_file() or path.stat().st_size < size:
            raise ValueError('E6-F request evidence was truncated')
        with path.open('rb') as stream:
            digest = hashlib.sha256(stream.read(size)).hexdigest()
        if digest != expected['sha256']:
            raise ValueError('E6-F request evidence prefix changed')


def _compat(base: ExperimentConfig, s: E6FSettings) -> ExperimentConfig:
    config = _compat_e5(base)
    return replace(config,
        e5=replace(config.e5, max_model_calls=s.max_model_calls,
                   max_reserved_tokens=s.max_reserved_tokens, max_sandbox_runs=s.max_sandbox_runs,
                   max_wall_seconds=s.max_wall_seconds, max_training_steps=0),
        budget=replace(config.budget, max_llm_calls=s.max_model_calls, max_tokens=s.max_reserved_tokens,
                       max_container_starts=s.max_sandbox_runs, max_seconds=s.max_wall_seconds,
                       max_candidates=s.max_model_calls))


def _split_evidence(base: ExperimentConfig) -> tuple[dict[str, list[str]], set[str]]:
    challenge = (PROJECT / base.data.challenges).resolve()
    solution = (PROJECT / base.data.solutions).resolve()
    if base.model.backend != 'fake' and (any('training' not in p.name.lower() for p in (challenge, solution)) or
            any(word in p.name.lower() for p in (challenge, solution) for word in ('test', 'evaluation'))):
        raise ValueError('E6-F Live is restricted to ARC Training data')
    tasks, _ = load_arc(challenge, solution)
    split = split_tasks(list(tasks), base.data.split_seed,
        {'exploration': base.data.exploration, 'promotion_gate': base.data.promotion_gate,
         'monitoring': base.data.monitoring})
    exclusions = json.loads(EXCLUSIONS.read_text())
    return split, set(tasks) | set(exclusions['task_ids'])


def _identity(s: E6FSettings, base: ExperimentConfig, image: str,
              splits: dict[str, list[str]]) -> dict[str, Any]:
    implementation = ('src/arc/e6f_config.py', 'src/arc/e6f_repair.py', 'src/arc/e6f_fake.py',
                      'src/pipelines/e6f.py', 'src/cli/e6f.py', 'src/arc/e6_curriculum.py',
                      'src/arc/e5_selfplay.py', 'src/arc/e6_fake.py', 'src/arc/ast_fingerprint.py',
                      'src/arc/data.py', 'src/arc/agent.py', 'src/arc/training.py',
                      'src/arc/sandbox.py', 'sandbox/runner.py',
                      'src/arc/http_backend.py', 'src/arc/prompting.py', 'src/arc/mlx_lora.py',
                      'src/arc/mlx_chat_template.py', 'src/arc/response_diagnostics.py',
                      'src/arc/guarded_contract.py', 'src/arc/guarded_generation.py',
                      'src/arc/guarded_http.py', 'src/arc/token_ledger.py',
                      'src/arc/deadline_watchdog.py', 'src/arc/mlx_token_driver.py',
                      'src/cli/guarded_mlx_server.py', 'src/pipelines/e5.py',
                      'src/pipelines/e6.py', 'src/core/config.py')
    sources = {}
    if s.reuse_existing:
        source = (PROJECT / s.source_artifact).resolve()
        _verify_seal(source)
        sources['e6_root_seal_sha256'] = _sha(source / 'artifact_hashes.json')
        source_identity = json.loads((source / 'identity.json').read_text())
        e5 = (PROJECT / base.e6.e5_artifact).resolve()
        diagnose_e5_artifact(e5)
        if _sha(e5 / 'root_artifact_hashes.json') != source_identity['e5_source_root_seal_sha256']:
            raise ValueError('E5 source seal differs from the E6 record')
        sources['e5_root_seal_sha256'] = _sha(e5 / 'root_artifact_hashes.json')
        if image != source_identity['sandbox_image_id']:
            raise ValueError('current Docker image differs from the sealed E6 image')
    if s.profile == 'live_collection':
        smoke = (PROJECT / s.smoke_artifact).resolve()
        _verify_final(smoke)
        state = json.loads((smoke / 'run_state.json').read_text())
        if not state.get('safe_for_collection'):
            raise ValueError('Live collection requires a completed, safe E6-F smoke')
        sources['smoke_seal_sha256'] = _sha(smoke / 'seal.json')
    data_paths = (PROJECT / base.data.challenges, PROJECT / base.data.solutions, EXCLUSIONS)
    return {'protocol': 'e6f_oracle_reliability_v1', 'settings': asdict(s), 'base_config': asdict(base),
            'sandbox_image_id': image, 'source_seals': sources,
            'data_sha256': {str(p.relative_to(PROJECT)): _sha(p) for p in data_paths},
            'split_manifest_sha256': _hash(splits),
            'implementation_sha256': {name: _sha(PROJECT / name) for name in implementation},
            'training_enabled': False, 'operator_search_used': False, 'public_evaluation_accessed': False}


def _audit_record(row: dict[str, Any], labels: dict[str, Any], *, sandbox: Any,
                  probes: list, revision: str, model_id: str, forbidden_ids: set[str]) -> str:
    _training_safe_record(row, 'e6_generator_distillation')
    if (row['qwen_provenance']['execution_mode'] != 'live_model' or row['base_revision'] != revision or
            row['qwen_provenance'].get('model_id') != model_id or
            row['qwen_provenance'].get('generation') != 0 or row.get('adapter_revision') != 'none'):
        raise ValueError('source_model_or_execution_mismatch')
    if row['task_id'] in forbidden_ids:
        raise ValueError('reserved_or_diagnostic_task_id')
    if labels.get('task_id') != row['task_id']:
        raise ValueError('verifier_task_mismatch')
    hidden_inputs = [validate_grid(x) for x in labels['inputs']]
    hidden_outputs = [validate_grid(x) for x in labels['outputs']]
    if (not hidden_inputs or len(hidden_inputs) != len(hidden_outputs) or
            grid_fingerprint(hidden_inputs) != row['heldout_input_fingerprint'] or
            grid_fingerprint(hidden_outputs) != row['heldout_output_fingerprint']):
        raise ValueError('verifier_fingerprint_mismatch')
    visible = row['visible_task']
    train = visible['train']
    inputs = [item['input'] for item in train] + visible['test_inputs'] + hidden_inputs
    code = row['messages'][-1]['content']
    outputs = verify_oracle(code, inputs, sandbox=sandbox)
    if (outputs[:len(train)] != [item['output'] for item in train] or
            outputs[-len(hidden_outputs):] != hidden_outputs):
        raise ValueError('stored_labels_mismatch')
    if outputs == inputs:
        raise ValueError('identity_only_rejected')
    known = {_hash(grid): output for grid, output in zip(inputs, outputs)}
    behavior = [known[_hash(grid)] if _hash(grid) in known else
                validate_grid(sandbox.run(code, grid)) for grid in probes]
    if behavior == probes:
        raise ValueError('identity_only_rejected')
    return _hash(behavior)


def _prior_records(s: E6FSettings) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    if not s.reuse_existing:
        return []
    source = (PROJECT / s.source_artifact).resolve()
    rows = _read_jsonl(source / 'gen-0/sft_generator_distillation.jsonl')
    labels = {r['task_id']: r for r in _read_jsonl(source / 'gen-0/private/heldout_verifier.jsonl')}
    result = [(row, labels[row['task_id']], 'e6') for row in rows]
    known = {row['task_id'] for row in rows}
    if s.profile == 'live_collection':
        smoke = (PROJECT / s.smoke_artifact).resolve()
        for item in _read_jsonl(smoke / 'private/accepted_records.jsonl'):
            row = dict(item['record'])
            private = row.pop('_verifier_only')
            if row['task_id'] not in known:
                result.append((row, {'task_id': row['task_id'], **private}, 'e6f_smoke'))
                known.add(row['task_id'])
    return result


def _probe_inputs(s: E6FSettings) -> list:
    """Use five sealed synthetic inputs as private cross-oracle probes when available."""
    prior = _prior_records(s)
    if not prior:
        return _quality_grids(88000, 5)
    row, labels, _ = prior[0]
    visible = row['visible_task']
    pool = ([item['input'] for item in visible['train']] + visible['test_inputs'] + labels['inputs'])
    probes = [validate_grid(grid) for grid in pool[:5]]
    if (len(probes) != 5 or len({_hash(grid) for grid in probes}) != 5 or
            not any(len(a) == len(b) and len(a[0]) == len(b[0])
                    for i, a in enumerate(probes) for b in probes[i + 1:])):
        raise ValueError('sealed source does not provide five distinct comparable probe inputs')
    return probes


def _metrics(events: list[dict[str, Any]], budget: dict[str, Any], root: Path,
             accepted: list[dict[str, Any]], seed_count: int, target: int, status: str) -> dict[str, Any]:
    finished = [e for e in events if e.get('event') == 'candidate_finished' and e.get('status') != 'interrupted']
    attempts = [e for e in events if e.get('event') == 'attempt_finished']
    before = [e for e in attempts if e.get('attempt') == 0]
    baseline = sum(bool(e.get('baseline_pass')) for e in finished)
    after = sum(bool(e.get('validation_pass')) for e in finished)
    def cost(rows, key):
        return sum(max(0, e.get('cost_after', {}).get(key, 0) - e.get('cost_before', {}).get(key, 0)) for e in rows)
    def rate(numerator, denominator):
        return numerator / denominator if denominator else None
    initial_tokens, total_tokens = cost(before, 'reserved_tokens'), cost(attempts, 'reserved_tokens')
    initial_sandbox, total_sandbox = cost(before, 'sandbox_runs'), cost(attempts, 'sandbox_runs')
    requests_path = root / 'logs/e6/llm_requests.jsonl'
    requests = _read_jsonl(requests_path) if requests_path.is_file() else []
    def observed(rows):
        candidates = {e['candidate_index'] * 3 + e['attempt'] for e in rows
                      if e.get('cost_after', {}).get('model_calls', 0) >
                         e.get('cost_before', {}).get('model_calls', 0)}
        matches = [r for r in requests if r.get('kind') == 'e5_generator' and r.get('candidate') in candidates]
        known = [r['usage']['total_tokens'] for r in matches
                 if isinstance(r.get('usage'), dict) and type(r['usage'].get('total_tokens')) is int]
        return {'total_tokens': sum(known) if candidates and len(known) == len(candidates) else None,
                'known_calls': len(known), 'expected_calls': len(candidates)}
    baseline_observed, all_observed = observed(before), observed(attempts)
    return {'status': status, 'ready_for_e6_training': len(accepted) >= target,
            'target_unique_oracles': target, 'unique_verified_oracles': len(accepted),
            'reaudited_source_oracles': seed_count, 'new_unique_oracles': len(accepted) - seed_count,
            'reaudited_e6_oracles': sum(item['origin'] == 'e6' for item in accepted),
            'reaudited_smoke_oracles': sum(item['origin'] == 'e6f_smoke' for item in accepted),
            'initial_candidates': len(finished), 'baseline_validation_successes': baseline,
            'interrupted_candidates': sum(e.get('status') == 'interrupted' for e in events
                                          if e.get('event') == 'candidate_finished'),
            'after_repair_validation_successes': after,
            'baseline_success_rate': rate(baseline, len(finished)),
            'after_repair_success_rate': rate(after, len(finished)),
            'initial_attempt_reserved_tokens': initial_tokens, 'all_attempt_reserved_tokens': total_tokens,
            'initial_attempt_sandbox_runs': initial_sandbox, 'all_attempt_sandbox_runs': total_sandbox,
            'baseline_successes_per_1000_reserved_tokens': rate(1000 * baseline, initial_tokens),
            'after_repair_successes_per_1000_reserved_tokens': rate(1000 * after, total_tokens),
            'baseline_successes_per_sandbox_run': rate(baseline, initial_sandbox),
            'after_repair_successes_per_sandbox_run': rate(after, total_sandbox),
            'initial_attempt_observed_usage': baseline_observed, 'all_attempt_observed_usage': all_observed,
            'baseline_successes_per_1000_observed_tokens': rate(1000 * baseline, baseline_observed['total_tokens']),
            'after_repair_successes_per_1000_observed_tokens': rate(1000 * after, all_observed['total_tokens']),
            'candidate_rejection_counts': dict(Counter(e['status'] for e in finished if e['status'] != 'passed')),
            'self_solved_synthetic_records': 0,
            'budget': {**budget, **_observed_token_usage(root / 'logs/e6/llm_requests.jsonl')},
            'training_steps': 0, 'weights_updated': False, 'adapter_promoted': False}


def run_e6f(s: E6FSettings, base: ExperimentConfig, *, live: bool = False,
            allow_collection: bool = False, resume: bool = False,
            run_dir: str | Path | None = None) -> Path:
    fake = s.profile == 'cpu_fake'
    if fake and (live or allow_collection):
        raise ValueError('Fake E6-F does not accept live flags')
    if not fake and not live:
        raise ValueError('Live E6-F requires --live')
    if s.profile == 'live_collection' and not allow_collection:
        raise ValueError('Live collection requires --allow-collection')
    if s.profile != 'live_collection' and allow_collection:
        raise ValueError('--allow-collection is restricted to the collection profile')
    namespace = PROJECT / 'artifacts/e6f'
    fixed = namespace / ('live-smoke' if s.profile == 'live_smoke' else 'live-collection')
    root = (Path(run_dir).expanduser().resolve() if run_dir else
            ((namespace / f'fake-{uuid.uuid4().hex[:10]}') if fake else fixed)).resolve()
    if not fake and root != fixed.resolve():
        raise ValueError('Live E6-F uses the fixed profile run directory to enforce one attempt')
    if root.exists() and not resume and any(root.iterdir()):
        raise FileExistsError('E6-F run exists; use same-run --resume')
    if resume and not root.is_dir():
        raise ValueError('E6-F resume requires an existing run')
    # Preflight the immutable sources and smoke gate before claiming a live run.
    config = _compat(base, s)
    image = 'fake' if fake else _docker_image_id(config)
    splits, forbidden_ids = _split_evidence(base)
    identity = _identity(s, base, image, splits)
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    with _run_lock(root):
        _secure_private_dirs(root)
        (root / 'private').mkdir(exist_ok=True, mode=0o700)
        os.chmod(root / 'private', 0o700)
        identity_path = root / 'identity.json'
        if resume:
            _verify_mutable_file(identity_path)
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('E6-F resume identity, data, source, or implementation differs')
            if (root / 'seal.json').is_file():
                _verify_final(root)
                return root
        else:
            _json(identity_path, identity, private=True)
            _seal_mutable_file(identity_path)
            _json(root / 'config.snapshot.json', {'settings': asdict(s), 'base_config': asdict(base)}, private=True)
            _json(root / 'split_task_manifest.json', splits, private=True)
        journal = root / 'oracle_lineage.jsonl'
        if resume and journal.exists():
            _verify_mutable_file(journal)
        events = _read_jsonl(journal) if journal.exists() else []
        for event in reversed(events):
            if event.get('log_prefixes'):
                _verify_prefixes(root, event['log_prefixes'])
                break
        def emit(row: dict[str, Any]) -> None:
            _append(journal, row)
            events.append(row)
        budget = E5Budget(root / 'budget.json', config, resume=resume)
        if fake:
            from src.arc.e6f_fake import FakeSandbox
            sandbox = MeteredSandbox(FakeSandbox(), budget)
        else:
            a = config.agent
            sandbox = MeteredSandbox(DockerSandbox(image=image, timeout=a.timeout_seconds,
                memory_mb=a.memory_mb, cpus=a.cpus, pids=a.pids,
                max_output_bytes=a.max_output_bytes, max_code_bytes=a.max_code_bytes,
                remaining=budget.remaining), budget)
        probes = _probe_inputs(s)
        accepted: list[dict[str, Any]] = []
        status = 'insufficient_verified_oracles'
        fatal = False
        source_path = root / 'private/seed_records.jsonl'
        try:
            if source_path.exists() and any(e.get('event') == 'seed_audit_finished' for e in events):
                _verify_mutable_file(source_path)
                _verify_mutable_file(root / 'seed_audit.json')
                accepted = _read_jsonl(source_path)
            elif any(e.get('event') == 'seed_audit_started' for e in events):
                status, fatal = 'interrupted_seed_audit', True
            else:
                emit({'event': 'seed_audit_started'})
                seed_audit = []
                for row, labels, origin in _prior_records(s):
                    try:
                        behavior = _audit_record(row, labels, sandbox=sandbox, probes=probes,
                                                 revision=base.model.revision, model_id=base.model.id,
                                                 forbidden_ids=forbidden_ids)
                        record = dict(row, _verifier_only={'inputs': labels['inputs'], 'outputs': labels['outputs']})
                        if (len(collection_rows([x['record'] for x in accepted] + [record])['accepted']) != len(accepted) + 1 or
                                behavior in {x['behavior_sha256'] for x in accepted}):
                            raise ValueError('duplicate_or_equivalent_oracle')
                        accepted.append({'record': record, 'behavior_sha256': behavior, 'origin': origin})
                        seed_audit.append({'task_id': row['task_id'], 'origin': origin, 'status': 'accepted'})
                    except TimeoutError as exc:
                        status = 'budget_exhausted' if 'budget' in str(exc).lower() else 'timeout_or_resource_limit'
                        fatal = True
                        seed_audit.append({'task_id': row.get('task_id'), 'origin': origin, 'status': status})
                        break
                    except BackendError:
                        status, fatal = 'sandbox_infrastructure_error', True
                        seed_audit.append({'task_id': row.get('task_id'), 'origin': origin, 'status': status})
                        break
                    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                        reason = str(exc)
                        allowed = {'source_model_or_execution_mismatch', 'reserved_or_diagnostic_task_id',
                                   'verifier_task_mismatch', 'verifier_fingerprint_mismatch',
                                   'stored_labels_mismatch', 'identity_only_rejected',
                                   'probe_input_overlap', 'duplicate_or_equivalent_oracle'}
                        seed_audit.append({'task_id': row.get('task_id'), 'origin': origin,
                                           'status': 'rejected', 'error_type': type(exc).__name__,
                                           'reason': reason if reason in allowed else 'source_reaudit_failed'})
                write_private_jsonl(source_path, accepted)
                _seal_mutable_file(source_path)
                _json(root / 'seed_audit.json', seed_audit, private=True)
                _seal_mutable_file(root / 'seed_audit.json')
                emit({'event': 'seed_audit_finished', 'accepted': len(accepted), 'status': status})
            seed_count = len(accepted)
            finished = {e['candidate_index']: e for e in events if e.get('event') == 'candidate_finished'}
            started = {e['candidate_index'] for e in events if e.get('event') == 'candidate_started'}
            for index in sorted(started - set(finished)):
                emit({'event': 'candidate_finished', 'candidate_index': index, 'status': 'interrupted',
                      'baseline_pass': False, 'validation_pass': False})
            finished = {e['candidate_index']: e for e in events if e.get('event') == 'candidate_finished'}
            for index, event in sorted(finished.items()):
                if event.get('accepted_path'):
                    path = root / event['accepted_path']
                    if not path.is_file() or _sha(path) != event['accepted_sha256']:
                        raise ValueError('completed E6-F candidate evidence changed')
                    accepted.append(json.loads(path.read_text()))
            cache = root / 'checkpoints/e6/request_cache.json'
            if cache.exists():
                _verify_mutable_file(cache)
                if json.loads(cache.read_text()).get('pending') is not None:
                    status, fatal = 'interrupted_uncertain_request', True
            remaining = [i for i in range(s.initial_candidates) if i not in finished]
            if len(accepted) >= s.target_unique_oracles:
                status = 'ready_for_e6_training'
            elif not fatal and remaining:
                if fake:
                    from contextlib import nullcontext
                    from src.arc.e6f_fake import FakeBackend
                    context = nullcontext(E5CachedBackend(FakeBackend(), budget, cache))
                else:
                    label = 'e6f-' + uuid.uuid4().hex[:10]
                    context = _backend(config, root, budget, cache, adapter=None, label=label)
                with context as backend:
                    for index in remaining:
                        if len(accepted) >= s.target_unique_oracles:
                            status = 'ready_for_e6_training'
                            break
                        emit({'event': 'candidate_started', 'candidate_index': index})
                        def generate(prompt, candidate_index, attempt):
                            call_index = candidate_index * 3 + attempt
                            backend.scope = f'e6f-{candidate_index}-{attempt}'
                            code = backend.generate(prompt, kind='e5_generator', candidate=call_index)
                            if fake:
                                provenance = {'source_type': 'generator_distillation', 'request_id': f'fake-{call_index}',
                                    'raw_response_sha256': hashlib.sha256(code.encode()).hexdigest(),
                                    'model_revision': base.model.revision, 'model_id': base.model.id,
                                    'execution_mode': 'fake_fixture', 'thinking_mode': 'off', 'generation': 0}
                            else:
                                provenance = _request_provenance(root, prompt, call_index)
                                if not provenance or provenance.pop('response_matches_code', None) != code.strip():
                                    raise ValueError('Qwen provenance mismatch')
                            return code, provenance
                        result = run_candidate(index=index, max_repairs=s.max_repair_attempts,
                            generate=generate, sandbox=sandbox, emit=emit,
                            counters=lambda: dict(budget.state), probe_inputs=probes)
                        outcome = result['status']
                        event = {'event': 'candidate_finished', 'candidate_index': index,
                                 'status': outcome, 'baseline_pass': result.get('baseline_pass', False),
                                 'validation_pass': outcome == 'passed', 'log_prefixes': _prefixes(root)}
                        if outcome == 'passed':
                            record = result['record']
                            clean = {k: v for k, v in record.items() if k != '_verifier_only'}
                            _training_safe_record(clean, 'e6_generator_distillation')
                            previous = [x['record'] for x in accepted]
                            if record['task_id'] in forbidden_ids:
                                outcome = 'reserved_or_diagnostic_task_id'
                            elif len(collection_rows(previous + [record])['accepted']) != len(previous) + 1:
                                outcome = 'duplicate_oracle'
                            elif result['behavior_sha256'] in {x['behavior_sha256'] for x in accepted}:
                                outcome = 'sampled_behavior_duplicate'
                            else:
                                item = {'record': record, 'behavior_sha256': result['behavior_sha256'], 'origin': 'new'}
                                path = root / f'private/candidate-{index}.json'
                                _json(path, item, private=True)
                                event.update(accepted_path=str(path.relative_to(root)), accepted_sha256=_sha(path))
                                accepted.append(item)
                            event['status'] = outcome
                        emit(event)
                        if outcome in {'budget_exhausted', 'timeout_or_resource_limit',
                                       'sandbox_infrastructure_error', 'model_request_failed'}:
                            status, fatal = outcome, True
                            break
        except TimeoutError as exc:
            status = 'budget_exhausted' if 'budget' in str(exc).lower() else 'model_server_timeout'
            fatal = True
        except BackendError:
            status, fatal = 'sandbox_infrastructure_error', True
        except RuntimeError as exc:
            if 'guarded server exited' not in str(exc):
                raise
            status, fatal = 'model_server_unavailable', True
        finally:
            budget.finish()
        if not fatal and len(accepted) >= s.target_unique_oracles:
            status = 'ready_for_e6_training'
        records = [x['record'] for x in accepted]
        write_private_jsonl(root / 'private/accepted_records.jsonl', accepted)
        write_private_jsonl(root / 'sft_generator_distillation.jsonl',
                            [{k: v for k, v in row.items() if k != '_verifier_only'} for row in records])
        write_private_jsonl(root / 'sft_self_solved.jsonl', [])
        write_private_jsonl(root / 'verified_oracles.jsonl', [
            {'task_id': row['task_id'], 'oracle_sha256': row['oracle_sha256'],
             'oracle_ast_sha256': row['oracle_ast_sha256'], 'oracle_shape_sha256': row['oracle_shape_sha256'],
             'oracle_code': row['messages'][-1]['content'], 'qwen_provenance': row['qwen_provenance']}
            for row in records])
        write_private_jsonl(root / 'private/heldout_verifier.jsonl', [
            {'task_id': row['task_id'], **row['_verifier_only']} for row in records])
        write_private_jsonl(root / 'rejections.jsonl', [e for e in events
            if e.get('event') == 'candidate_finished' and e.get('status') != 'passed'])
        metrics = _metrics(events, budget.state, root, accepted, seed_count, s.target_unique_oracles, status)
        _json(root / 'metrics.json', metrics, private=True)
        _json(root / 'budget_ledger.json', {**metrics['budget'],
              'max_model_calls': s.max_model_calls, 'max_reserved_tokens': s.max_reserved_tokens,
              'max_sandbox_runs': s.max_sandbox_runs, 'max_wall_seconds': s.max_wall_seconds}, private=True)
        safe = bool(not fake and s.profile == 'live_smoke' and budget.state['model_calls'] > 0 and
                    status not in {'sandbox_infrastructure_error', 'model_request_failed',
                                   'interrupted_seed_audit', 'interrupted_uncertain_request',
                                   'timeout_or_resource_limit', 'model_server_timeout',
                                   'model_server_unavailable'})
        _json(root / 'run_state.json', {'status': status, 'complete': True, 'safe_for_collection': safe,
              'unique_verified_oracles': len(accepted), 'ready_for_e6_training': metrics['ready_for_e6_training'],
              'training_invoked': False, 'adapter_updated': False, 'public_evaluation_accessed': False}, private=True)
        (root / 'report.md').write_text(
            f"# E6-F result\n\nStatus: {status}. Verified unique oracles: {len(accepted)}. "
            f"Reaudited sources: {seed_count}; new: {len(accepted)-seed_count}.\n\n"
            f"Baseline validation successes: {metrics['baseline_validation_successes']}/"
            f"{metrics['initial_candidates']}; after repair: {metrics['after_repair_validation_successes']}/"
            f"{metrics['initial_candidates']}. Rejections: {metrics['candidate_rejection_counts']}.\n\n"
            f"Model calls: {budget.state['model_calls']}; reserved tokens: {budget.state['reserved_tokens']}; "
            f"observed tokens: {metrics['budget']['observed_total_tokens']}; "
            f"sandbox runs: {budget.state['sandbox_runs']}; elapsed seconds: {budget.state['elapsed_seconds']:.2f}. "
            'Training was not invoked.\n', encoding='utf-8')
        os.chmod(root / 'report.md', 0o600)
        for relative in ('logs/e6', 'checkpoints/e6'):
            for path in (root / relative).rglob('*'):
                if path.is_file():
                    os.chmod(path, 0o600)
                elif path.is_dir():
                    os.chmod(path, 0o700)
        # Revalidate all source seals before sealing the output.
        if _identity(s, base, image, splits) != identity:
            raise ValueError('E6-F identity or source seal changed during the run')
        _seal(root)
        return root
