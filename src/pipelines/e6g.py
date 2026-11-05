"""Pre-registered, symmetric E6-G Oracle novelty comparison."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import uuid
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from src.arc.ast_fingerprint import stable_ast_sha256, stable_shape_sha256
from src.arc.e5_selfplay import _generator_prompt
from src.arc.e6_curriculum import _quality_grids, diagnose_e5_artifact, write_private_jsonl
from src.arc.e6f_config import load_e6f_config
from src.arc.e6f_repair import run_candidate
from src.arc.e6g_config import E6GSettings
from src.arc.e6g_novelty import (FEEDBACK_INSTRUCTION, FEEDBACK_SUFFIX, NoveltyFeedback,
                                 classify_novelty, source_hash)
from src.arc.e6g_provenance import verify_response_body
from src.arc.http_backend import BackendError
from src.arc.mlx_lora import _training_safe_record
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.sandbox import DockerSandbox
from src.core.config import ExperimentConfig
from src.pipelines.e5 import (E5Budget, E5CachedBackend, MeteredSandbox, _backend,
                              _canonical, _docker_image_id, _seal_mutable_file, _verify_mutable_file)
from src.pipelines.e6 import (_compat_e5, _json, _observed_token_usage, _request_provenance,
                              _run_lock, _secure_private_dirs, _verify_seal)
from src.pipelines.e6f import (_append, _hash, _prefixes, _probe_inputs, _read_jsonl,
                               _sha, _split_evidence, _verify_final, _verify_prefixes)

PROJECT = Path(__file__).resolve().parents[2]
CONDITIONS = ('G0_baseline', 'G1_novelty_feedback')


def _tree(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob('*'))
            if path.is_file() and path != root / 'seal.json' and path.name != '.run.lock'}


def _seal_g(root: Path) -> None:
    _json(root / 'seal.json', _tree(root), private=True)


def _verify_g(root: Path) -> None:
    if not (root / 'seal.json').is_file() or json.loads((root / 'seal.json').read_text()) != _tree(root):
        raise ValueError('E6-G artifact seal mismatch')


def _compat(base: ExperimentConfig, *, calls: int, tokens: int, sandboxes: int,
            seconds: int, seed: int) -> ExperimentConfig:
    config = _compat_e5(base)
    return replace(config, experiment=replace(config.experiment, seed=seed),
        e5=replace(config.e5, max_model_calls=calls, max_reserved_tokens=tokens,
                   max_sandbox_runs=sandboxes, max_wall_seconds=seconds, max_training_steps=0),
        budget=replace(config.budget, max_llm_calls=calls, max_tokens=tokens,
                       max_container_starts=sandboxes, max_seconds=seconds, max_candidates=calls))


class PairedBudget:
    """Charge each dispatch to both the global ledger and one equal sub-ledger."""
    def __init__(self, total: E5Budget, condition: E5Budget):
        self.total, self.condition = total, condition

    @property
    def state(self):
        return self.condition.state

    def remaining(self):
        return min(self.total.remaining(), self.condition.remaining())

    def reserve_model(self, prompt: str, kind: str) -> None:
        self.remaining()
        messages = [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}]
        reserved = self.condition.config.model.max_tokens + len(_canonical(messages).encode()) + 1024
        for budget in (self.total, self.condition):
            c, st = budget.config, budget.state
            if (st['model_calls'] + 1 > min(c.e5.max_model_calls, c.budget.max_llm_calls) or
                    st['reserved_tokens'] + reserved > min(c.e5.max_reserved_tokens, c.budget.max_tokens)):
                raise TimeoutError('paired model or token budget exhausted')
        self.total.reserve_model(prompt, kind)
        self.condition.reserve_model(prompt, kind)

    def reserve_sandbox(self) -> None:
        self.remaining()
        for budget in (self.total, self.condition):
            c, st = budget.config, budget.state
            if st['sandbox_runs'] + 1 > min(c.e5.max_sandbox_runs, c.budget.max_container_starts):
                raise TimeoutError('paired sandbox budget exhausted')
        self.total.reserve_sandbox()
        self.condition.reserve_sandbox()


def _fake_pool(base: ExperimentConfig, probes: list) -> list[dict[str, Any]]:
    from src.arc.e6_curriculum import verify_oracle
    from src.arc.e6f_fake import FakeSandbox
    from src.pipelines.e6 import _fake_records
    sandbox = FakeSandbox()
    records = _fake_records(base, 0)
    pool = []
    for index, record in enumerate(records[1:3]):
        outputs = verify_oracle(record['messages'][-1]['content'], probes, sandbox=sandbox)
        pool.append({'record': record, 'behavior_sha256': _hash(outputs), 'origin': 'fake_fixture',
                     'historical_duplicate_observed': index == 0, 'audit_status': 'fake_fixture'})
    return pool


def _load_pool(s: E6GSettings, base: ExperimentConfig, image: str, splits: dict[str, list[str]]):
    if s.profile == 'cpu_fake':
        probes = _quality_grids(88000, 5)
        pool = _fake_pool(base, probes)
        return pool, probes, {'mode': 'fake_fixture', 'eligible_seed_oracles': 0,
                              'pool_sha256': _hash(pool), 'legacy_identity_incomplete': False}
    pool_root = (PROJECT / s.pool_path).resolve()
    _verify_final(pool_root)
    manifest = json.loads((pool_root / 'pool_manifest.json').read_text())
    pool = _read_jsonl(pool_root / 'private/pool.jsonl')
    e6f, _ = load_e6f_config(PROJECT / 'configs/e6f.live.smoke.yml')
    probes = _probe_inputs(e6f)
    if (manifest['pool_sha256'] != _sha(pool_root / 'private/pool.jsonl') or
            manifest['model_revision'] != base.model.revision or manifest['model_id'] != base.model.id or
            manifest['quantization'] != base.model.quantization or manifest['sandbox_image_id'] != image or
            manifest['probe_inputs_sha256'] != _hash(probes) or
            manifest['split_manifest_sha256'] != _hash(splits)):
        raise ValueError('E6-G seed pool identity differs from the comparison')
    for name, expected in manifest['implementation_sha256'].items():
        if _sha(PROJECT / name) != expected:
            raise ValueError(f'E6-G seed re-audit dependency changed: {name}')
    e6_root = PROJECT / e6f.source_artifact
    e5_root = PROJECT / base.e6.e5_artifact
    f_root = PROJECT / 'artifacts/e6f/live-collection'
    _verify_seal(e6_root)
    _verify_final(f_root)
    e5 = diagnose_e5_artifact(e5_root)
    if (manifest['source_e6_seal_sha256'] != _sha(e6_root / 'artifact_hashes.json') or
            manifest['source_e6f_collection_seal_sha256'] != _sha(f_root / 'seal.json') or
            manifest['source_e5_seal_sha256'] != e5['source_root_seal_sha256']):
        raise ValueError('E6-G historical source seal changed')
    diagnosis = PROJECT / 'artifacts/e6g/diagnosis'
    expected = json.loads((diagnosis / 'seal.json').read_text())
    actual = {str(path.relative_to(diagnosis)): _sha(path) for path in sorted(diagnosis.rglob('*'))
              if path.is_file() and path.name != 'seal.json'}
    if expected != actual or manifest['source_diagnosis_seal_sha256'] != _sha(diagnosis / 'seal.json'):
        raise ValueError('E6-G diagnosis seal changed')
    diagnosis_v2 = PROJECT / 'artifacts/e6g/diagnosis-v2'
    _verify_final(diagnosis_v2)
    if _sha(diagnosis_v2 / 'duplicates.csv') != _sha(diagnosis / 'duplicates.csv'):
        raise ValueError('reproduced G1 classification differs from the seed evidence')
    diagnostic_identity = json.loads((diagnosis_v2 / 'summary.json').read_text())
    for name, expected_sha in diagnostic_identity['implementation_sha256'].items():
        if _sha(PROJECT / name) != expected_sha:
            raise ValueError(f'E6-G diagnostic dependency changed: {name}')
    provenance_root = PROJECT / 'artifacts/e6g/provenance-audit-v2'
    _verify_final(provenance_root)
    provenance_audit = json.loads((provenance_root / 'summary.json').read_text())
    audited_pool = PROJECT / 'artifacts/e6g/seed-pool'
    _verify_final(audited_pool)
    if (provenance_audit['diagnosis_seal_sha256'] != _sha(diagnosis_v2 / 'seal.json') or
            provenance_audit['seed_pool_seal_sha256'] != _sha(audited_pool / 'seal.json') or
            _sha(audited_pool / 'private/pool.jsonl') != manifest['pool_sha256'] or
            provenance_audit['seed_counts'] != {'verified': len(pool)} or
            provenance_audit['response_counts'] != {'verified': diagnostic_identity['observations']}):
        raise ValueError('E6-G historical Qwen HTTP response evidence is incomplete')
    for name, expected_sha in provenance_audit['implementation_sha256'].items():
        if _sha(PROJECT / name) != expected_sha:
            raise ValueError(f'E6-G provenance audit dependency changed: {name}')
    if len(pool) != manifest['historical_verified'] or len(pool) != manifest['eligible_seed_oracles']:
        raise ValueError('E6-G seed pool count differs from its audit')
    for item in pool:
        row = {k: v for k, v in item['record'].items() if k != '_verifier_only'}
        _training_safe_record(row, 'e6_generator_distillation')
        if item['audit_status'] != 'historical_verified' or row['qwen_provenance']['execution_mode'] != 'live_model':
            raise ValueError('E6-G seed pool has an unverifiable or non-Qwen entry')
        verify_response_body(row['qwen_provenance'], row['messages'][-1]['content'])
    return pool, probes, {'mode': 'live_model', 'eligible_seed_oracles': len(pool),
        'pool_sha256': manifest['pool_sha256'], 'seed_pool_seal_sha256': _sha(pool_root / 'seal.json'),
        'provenance_checked_pool_seal_sha256': _sha(audited_pool / 'seal.json'),
        'diagnosis_v2_seal_sha256': _sha(diagnosis_v2 / 'seal.json'),
        'provenance_audit_seal_sha256': _sha(provenance_root / 'seal.json'),
        'legacy_identity_incomplete': manifest['legacy_identity_incomplete']}


def _protocol(s: E6GSettings, base: ExperimentConfig, image: str, splits: dict,
              pool_info: dict, probes: list, git_evidence: dict) -> dict:
    names = ('src/arc/e6g_config.py', 'src/arc/e6g_novelty.py', 'src/arc/e6g_fake.py',
             'src/arc/e6g_provenance.py', 'src/cli/e6g_provenance_audit.py',
             'src/pipelines/e6g.py', 'src/cli/e6g.py', 'src/cli/e6g_diagnose.py',
             'src/cli/e6g_seed.py', 'src/cli/e6g_review.py', 'src/arc/e6f_repair.py',
             'src/arc/e6f_config.py', 'src/arc/e6f_fake.py', 'src/pipelines/e6f.py',
             'src/arc/e6_curriculum.py', 'src/arc/e5_selfplay.py', 'src/arc/training.py',
             'src/arc/ast_fingerprint.py', 'src/arc/data.py', 'src/arc/agent.py',
             'src/arc/e6_fake.py', 'src/arc/sandbox.py', 'sandbox/runner.py', 'src/arc/http_backend.py',
             'src/arc/prompting.py', 'src/arc/mlx_lora.py', 'src/arc/mlx_chat_template.py',
             'src/arc/response_diagnostics.py', 'src/arc/guarded_contract.py',
             'src/arc/guarded_generation.py', 'src/arc/guarded_http.py', 'src/arc/token_ledger.py',
             'src/arc/deadline_watchdog.py', 'src/arc/mlx_token_driver.py',
             'src/cli/guarded_mlx_server.py', 'src/pipelines/e5.py', 'src/pipelines/e6.py', 'src/core/config.py')
    order = list(CONDITIONS)
    random.Random(s.seed).shuffle(order)
    model_path = Path(base.p3b.training.model_path).expanduser()
    tokenizer_path = model_path / 'tokenizer_config.json'
    tokenizer = json.loads(tokenizer_path.read_text()) if s.profile != 'cpu_fake' else {}
    template = tokenizer.get('chat_template')
    if s.profile != 'cpu_fake' and (not isinstance(template, str) or model_path.name != base.model.revision):
        raise ValueError('pinned tokenizer or model snapshot is invalid')
    data = (PROJECT / base.data.challenges, PROJECT / base.data.solutions,
            PROJECT / 'configs/p3c.exploration.exclusions.v1.json')
    return {'protocol': 'e6g_paired_comparison_v1', 'settings': asdict(s),
        'base_config': asdict(base), 'base_config_sha256': _sha(PROJECT / s.base_config),
        'condition_order': order, 'initial_pool': pool_info,
        'probe_inputs_sha256': _hash(probes), 'split_manifest_sha256': _hash(splits),
        'data_sha256': {str(p.relative_to(PROJECT)): _sha(p) for p in data},
        'model': {'id': base.model.id, 'revision': base.model.revision,
                  'quantization': base.model.quantization, 'temperature': base.model.temperature,
                  'thinking': base.model.thinking, 'adapter': 'none',
                  'mlx_lm_version': base.p3b.training.mlx_lm_version},
        'sandbox_image_id': image, 'tokenizer_config_sha256': _sha(tokenizer_path) if template else None,
        'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest() if template else None,
        'generator_prompt_sha256': hashlib.sha256(_generator_prompt().encode()).hexdigest(),
        'system_prompt_sha256': hashlib.sha256(FINAL_ANSWER_SYSTEM_PROMPT.encode()).hexdigest(),
        'feedback_template_sha256': hashlib.sha256((FEEDBACK_INSTRUCTION + FEEDBACK_SUFFIX).encode()).hexdigest(),
        'candidate_seed_rule': 'seed + candidate_index * 3 + repair_index',
        'seed_control_note': 'Guarded driver source applies requested seed; request logs label seed support unverified.',
        'conditions_share_initial_pool_only': True, 'human_authored_feedback_intervention': True,
        'behavior_match_policy': 'sampled match is a suspicion; conservatively excluded from novel count',
        'training_enabled': False, 'operator_search_used': False, 'public_evaluation_accessed': False,
        'guarded_server_contract_note': 'same-process ledger and server-reported contract; not remote attestation',
        'implementation_sha256': {name: _sha(PROJECT / name) for name in names}, 'git_evidence': git_evidence}


def _capture_git(root: Path) -> dict[str, str]:
    private = root / 'private'
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = {}
    for name, args in (('status.txt', ['status', '--porcelain=v1']),
                       ('diff-unstaged.patch', ['diff', '--no-ext-diff', '--no-color']),
                       ('diff-staged.patch', ['diff', '--cached', '--no-ext-diff', '--no-color'])):
        run = subprocess.run(['git', *args], cwd=PROJECT, capture_output=True, check=True)
        path = private / name
        path.write_bytes(run.stdout)
        os.chmod(path, 0o600)
        result[name] = _sha(path)
    result['head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT,
                                    capture_output=True, text=True, check=True).stdout.strip()
    return result


def _live_provenance(root: Path, prompt: str, candidate: int, code: str) -> dict[str, Any]:
    evidence = _request_provenance(root, prompt, candidate)
    if not evidence or evidence.pop('response_matches_code', None) != code.strip():
        raise ValueError('Qwen request/response provenance mismatch')
    verify_response_body(evidence, code.strip())
    return evidence


def _condition_metrics(name: str, events: list[dict], budget: dict, root: Path,
                       snapshot_sha: str, candidate_limit: int) -> dict:
    rows = [r for r in events if r.get('event') == 'candidate_finished']
    completed = [r for r in rows if r.get('completion') == 'completed']
    n = len(completed)
    novel = sum(r.get('verified_novel', False) for r in completed)
    valid = sum(r.get('oracle_valid', False) for r in completed)
    raw = sum(r.get('raw_duplicate', False) for r in completed)
    ast = sum(r.get('ast_duplicate', False) for r in completed)
    behavior_evaluated = [r for r in completed if 'behavior_fingerprint' in r]
    behavioral = sum(r.get('behavior_match', False) for r in behavior_evaluated)
    repaired = {event['candidate_index'] for event in events
                if event.get('event') == 'attempt_started' and event.get('attempt', 0) > 0}
    repair_denominator = sum(row['candidate_index'] in repaired for row in completed)
    repair_passes = sum(r.get('repair_pass', False) for r in completed)
    repair_novel = sum(r.get('repair_pass', False) and r.get('verified_novel', False) for r in completed)
    observed = _observed_token_usage(root / 'logs/e6/llm_requests.jsonl')
    usage_complete = bool(budget['model_calls'] and observed['calls_with_usage'] == budget['model_calls'])
    safe_rate = lambda numerator, denominator: numerator / denominator if denominator else None
    return {'condition': name, 'initial_pool_snapshot_sha256': snapshot_sha,
        'candidate_limit': candidate_limit, 'started_candidates': len(rows),
        'completed_candidates': n, 'incomplete_candidates': len(rows) - n,
        'oracle_valid': valid, 'verified_novel_oracles': novel,
        'verified_novel_per_completed_trial': safe_rate(novel, n),
        'raw_duplicate_candidates': raw, 'ast_duplicate_candidates': ast,
        'behavior_evaluated_candidates': len(behavior_evaluated), 'sampled_behavior_suspects': behavioral,
        'raw_duplicate_rate': safe_rate(raw, n), 'ast_duplicate_rate': safe_rate(ast, n),
        'sampled_behavior_suspect_rate': safe_rate(behavioral, len(behavior_evaluated)),
        'repair_attempted_candidates': repair_denominator, 'repair_validation_successes': repair_passes,
        'repair_validation_success_rate': safe_rate(repair_passes, repair_denominator),
        'repair_novel_successes': repair_novel,
        'repair_novel_success_rate': safe_rate(repair_novel, repair_denominator),
        'execution_errors': sum(r.get('had_execution_error', False) for r in completed),
        'candidate_outcomes': dict(Counter(r.get('status') for r in rows)),
        'feedback_applied_candidates': sum(r.get('feedback_applied', False) for r in rows),
        'model_calls': budget['model_calls'], 'reserved_tokens': budget['reserved_tokens'],
        **observed, 'observed_usage_complete': usage_complete,
        'sandbox_runs': budget['sandbox_runs'], 'wall_seconds': budget['elapsed_seconds'],
        'novel_per_100k_reserved_tokens': safe_rate(100000 * novel, budget['reserved_tokens']),
        'novel_per_100k_observed_tokens': safe_rate(100000 * novel, observed['observed_total_tokens'])
                                           if usage_complete else None,
        'training_steps': budget['training_steps']}


def _run_condition(name: str, root: Path, settings: E6GSettings, base: ExperimentConfig,
                   total_budget: E5Budget, image: str, initial_pool: list[dict], probes: list,
                   forbidden_ids: set[str], snapshot_sha: str, protocol_sha: str, resume: bool) -> dict:
    fake = settings.profile == 'cpu_fake'
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    _secure_private_dirs(root)
    (root / 'private').mkdir(exist_ok=True, mode=0o700)
    os.chmod(root / 'private', 0o700)
    result_path = root / 'result.json'
    if result_path.is_file():
        _verify_g(root)
        _verify_mutable_file(result_path)
        return json.loads(result_path.read_text())
    journal = root / 'lineage.jsonl'
    if journal.exists():
        _verify_mutable_file(journal)
    events = _read_jsonl(journal) if journal.exists() else []
    for event in reversed(events):
        if event.get('log_prefixes'):
            _verify_prefixes(root, event['log_prefixes'])
            break
    def emit(item: dict) -> None:
        _append(journal, item)
        events.append(item)
    config = _compat(base, calls=settings.per_condition_calls,
        tokens=settings.per_condition_reserved_tokens, sandboxes=settings.per_condition_sandbox,
        seconds=settings.per_condition_wall_seconds, seed=settings.seed)
    budget_path = root / 'budget.json'
    if resume and journal.exists() and not budget_path.exists():
        raise ValueError('condition budget journal is missing on resume')
    budget = E5Budget(budget_path, config, resume=resume and budget_path.exists())
    paired = PairedBudget(total_budget, budget)
    # E5's helpers are annotated nominally as E5Budget; this wrapper implements
    # the subset they consume while charging both ledgers before dispatch.
    paired_for_e5 = cast(E5Budget, paired)
    if fake:
        from src.arc.e6g_fake import FakeSandbox
        sandbox = MeteredSandbox(FakeSandbox(), paired_for_e5)
    else:
        a = config.agent
        sandbox = MeteredSandbox(DockerSandbox(image=image, timeout=a.timeout_seconds,
            memory_mb=a.memory_mb, cpus=a.cpus, pids=a.pids, max_output_bytes=a.max_output_bytes,
            max_code_bytes=a.max_code_bytes, remaining=paired.remaining), paired_for_e5)
    # A separate deep copy is essential: neither arm ever sees the other's records.
    pool = json.loads(json.dumps(initial_pool))
    feedback = NoveltyFeedback(enabled=name == 'G1_novelty_feedback', pool=pool, allow_fixture=fake)
    finished = {e['candidate_index']: e for e in events if e.get('event') == 'candidate_finished'}
    started = {e['candidate_index'] for e in events if e.get('event') == 'candidate_started'}
    for index in sorted(started - set(finished)):
        emit({'event': 'candidate_finished', 'candidate_index': index,
              'candidate_id': f'{name}-{index}', 'status': 'interrupted', 'completion': 'incomplete',
              'oracle_valid': False, 'verified_novel': False})
    finished = {e['candidate_index']: e for e in events if e.get('event') == 'candidate_finished'}
    for _, event in sorted(finished.items()):
        if event.get('accepted_path'):
            file = root / event['accepted_path']
            if not file.is_file() or _sha(file) != event['accepted_sha256']:
                raise ValueError('completed condition candidate changed')
            pool.append(json.loads(file.read_text()))
        if event.get('feedback_path'):
            file = root / event['feedback_path']
            if not file.is_file() or _sha(file) != event['feedback_sha256']:
                raise ValueError('condition feedback evidence changed')
            feedback.previous_code = json.loads(file.read_text())['code']
    cache = root / 'checkpoints/e6/request_cache.json'
    stop = 'completed'
    if cache.exists():
        _verify_mutable_file(cache)
        if json.loads(cache.read_text()).get('pending') is not None:
            stop = 'interrupted_uncertain_request'
    remaining = [i for i in range(settings.candidates_per_condition) if i not in finished]
    try:
        if stop == 'completed' and remaining:
            context: Any
            if fake:
                from contextlib import nullcontext
                from src.arc.e6g_fake import FakeBackend
                context = nullcontext(E5CachedBackend(FakeBackend(name), paired_for_e5, cache))
            else:
                context = _backend(config, root, paired_for_e5, cache, adapter=None,
                                   label='e6g-' + uuid.uuid4().hex[:10])
            with context as backend:
                for index in remaining:
                    if (budget.state['model_calls'] >= settings.per_condition_calls or
                            budget.state['sandbox_runs'] >= settings.per_condition_sandbox or
                            budget.state['reserved_tokens'] >= settings.per_condition_reserved_tokens):
                        stop = 'budget_exhausted'
                        break
                    candidate_id = f'{name}-{index}'
                    emit({'event': 'candidate_started', 'candidate_index': index, 'candidate_id': candidate_id,
                          'pool_snapshot_sha256': snapshot_sha, 'protocol_sha256': protocol_sha,
                          'model_revision': base.model.revision})
                    found: list[dict[str, Any]] = []
                    applied = False
                    def generate(prompt, candidate_index, attempt):
                        nonlocal applied
                        call_index = candidate_index * 3 + attempt
                        actual_prompt = feedback.prompt(prompt) if attempt == 0 else prompt
                        if attempt == 0:
                            applied = actual_prompt != prompt
                        emit({'event': 'generation_requested', 'candidate_index': index,
                              'generation_attempt_id': f'{candidate_id}-attempt-{attempt}',
                              'protocol_sha256': protocol_sha, 'pool_snapshot_sha256': snapshot_sha,
                              'model_revision': base.model.revision, 'seed_requested': settings.seed + call_index,
                              'feedback_applied': actual_prompt != prompt,
                              'feedback_code_sha256': source_hash(feedback.previous_code)
                                  if actual_prompt != prompt and feedback.previous_code else None,
                              'prompt_sha256': hashlib.sha256(actual_prompt.encode()).hexdigest()})
                        backend.scope = f'{name}-{candidate_index}-{attempt}'
                        code = backend.generate(actual_prompt, kind='e5_generator', candidate=call_index)
                        provenance: dict[str, Any]
                        if fake:
                            provenance = {'source_type': 'generator_distillation', 'request_id': f'fake-{name}-{call_index}',
                                'raw_response_sha256': hashlib.sha256(code.encode()).hexdigest(),
                                'model_revision': base.model.revision, 'model_id': base.model.id,
                                'execution_mode': 'fake_fixture', 'thinking_mode': 'off', 'generation': 0}
                        else:
                            provenance = _live_provenance(root, actual_prompt, call_index, code)
                        classification = feedback.observe(code, pool)
                        found.append({'classification': classification, 'code': code,
                                      'request_id': provenance['request_id']})
                        try:
                            ast_sha, shape_sha = stable_ast_sha256(code), stable_shape_sha256(code)
                        except (SyntaxError, ValueError, TypeError, RecursionError):
                            ast_sha, shape_sha = None, None
                        emit({'event': 'novelty_observed', 'candidate_index': index,
                              'generation_attempt_id': f'{candidate_id}-attempt-{attempt}',
                              'classification': classification, 'request_id': provenance['request_id'],
                              'raw_source_hash': source_hash(code), 'canonical_ast_hash': ast_sha,
                              'shape_hash': shape_sha})
                        return code, provenance
                    result = run_candidate(index=index, max_repairs=settings.max_repairs,
                        generate=generate, sandbox=sandbox, emit=emit,
                        counters=lambda: dict(budget.state), probe_inputs=probes)
                    outcome = result['status']
                    event = {'event': 'candidate_finished', 'candidate_index': index, 'candidate_id': candidate_id,
                        'completion': 'incomplete' if outcome in {'budget_exhausted', 'timeout_or_resource_limit',
                            'sandbox_infrastructure_error', 'model_request_failed'} else 'completed',
                        'status': outcome, 'oracle_valid': outcome == 'passed', 'verified_novel': False,
                        'repair_pass': bool(result.get('repair_pass')), 'feedback_applied': applied,
                        'raw_duplicate': any(x['classification'] == 'duplicate_source' for x in found),
                        'ast_duplicate': any(x['classification'] in {'duplicate_source', 'duplicate_ast'} for x in found),
                        'had_execution_error': any(x.get('event') == 'attempt_finished' and
                            x.get('candidate_index') == index and x.get('category') == 'runtime_exception' for x in events),
                        'novelty': None, 'behavior_match': False, 'log_prefixes': _prefixes(root)}
                    if outcome == 'passed':
                        record = result['record']
                        clean = {k: v for k, v in record.items() if k != '_verifier_only'}
                        _training_safe_record(clean, 'e6_generator_distillation')
                        if record['task_id'] in forbidden_ids:
                            novelty = 'reserved_task_rejected'
                        else:
                            event['behavior_match'] = result['behavior_sha256'] in {
                                item.get('behavior_sha256') for item in pool}
                            event['behavior_fingerprint'] = result['behavior_sha256']
                            novelty = classify_novelty(record['messages'][-1]['content'], result['behavior_sha256'], pool)
                        event['novelty'] = novelty
                        event['status'] = novelty if novelty != 'novel' else 'verified_novel_oracle'
                        if novelty == 'novel':
                            item = {'record': record, 'behavior_sha256': result['behavior_sha256'],
                                    'origin': name, 'historical_duplicate_observed': False}
                            path = root / f'private/candidate-{index}.json'
                            _json(path, item, private=True)
                            event.update(verified_novel=True, accepted_path=str(path.relative_to(root)),
                                         accepted_sha256=_sha(path))
                            pool.append(item)
                    duplicate = next((x for x in reversed(found)
                                      if x['classification'] in {'duplicate_source', 'duplicate_ast'}), None)
                    if name == 'G1_novelty_feedback' and duplicate:
                        path = root / f'private/feedback-{index}.json'
                        _json(path, duplicate, private=True)
                        event.update(feedback_path=str(path.relative_to(root)), feedback_sha256=_sha(path))
                    emit(event)
                    if event['completion'] == 'incomplete':
                        stop = outcome
                        break
    except TimeoutError as exc:
        stop = 'budget_exhausted' if 'budget' in str(exc).lower() else 'model_server_timeout'
    except BackendError:
        stop = 'sandbox_infrastructure_error'
    except RuntimeError as exc:
        if 'guarded server exited' not in str(exc):
            raise
        stop = 'model_server_unavailable'
    finally:
        budget.finish()
    metrics = _condition_metrics(name, events, budget.state, root, snapshot_sha, settings.candidates_per_condition)
    metrics['stop_reason'] = stop
    metrics['budget_limits'] = {'model_calls': settings.per_condition_calls,
        'reserved_tokens': settings.per_condition_reserved_tokens,
        'sandbox_runs': settings.per_condition_sandbox, 'wall_seconds': settings.per_condition_wall_seconds}
    metrics['records'] = pool[len(initial_pool):]
    _json(result_path, metrics, private=True)
    _seal_mutable_file(result_path)
    for relative in ('logs/e6', 'checkpoints/e6'):
        for path in (root / relative).rglob('*'):
            if path.is_file(): os.chmod(path, 0o600)
    _seal_g(root)
    return metrics


def run_e6g(s: E6GSettings, base: ExperimentConfig, *, live: bool = False,
            approve_full: bool = False, resume: bool = False,
            run_dir: str | Path | None = None) -> Path:
    fake = s.profile == 'cpu_fake'
    if fake and (live or approve_full):
        raise ValueError('Fake E6-G cannot use live approval flags')
    if not fake and not live:
        raise ValueError('Live E6-G requires --live')
    if s.profile == 'live_compare' and not approve_full:
        raise ValueError('Full comparison requires --approve-full-comparison')
    if s.profile != 'live_compare' and approve_full:
        raise ValueError('--approve-full-comparison is only valid for the full comparison')
    namespace = PROJECT / 'artifacts/e6g'
    fixed = namespace / ('live-smoke' if s.profile == 'live_smoke' else 'live-compare')
    root = (Path(run_dir).expanduser().resolve() if run_dir else
            ((namespace / f'fake-{uuid.uuid4().hex[:10]}') if fake else fixed)).resolve()
    if not fake and root != fixed.resolve():
        raise ValueError('Live E6-G uses the fixed profile run directory to prevent budget bypass')
    if root.exists() and not resume and any(root.iterdir()):
        raise FileExistsError('E6-G run exists; only same-run --resume is allowed')
    if resume and not root.is_dir():
        raise ValueError('E6-G resume requires an existing run')
    total_config = _compat(base, calls=s.total_calls, tokens=s.total_reserved_tokens,
                           sandboxes=s.total_sandbox, seconds=s.total_wall_seconds, seed=s.seed)
    image = 'fake' if fake else _docker_image_id(total_config)
    splits, forbidden = _split_evidence(base)
    pool, probes, pool_info = _load_pool(s, base, image, splits)
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    with _run_lock(root):
        (root / 'private').mkdir(exist_ok=True, mode=0o700)
        os.chmod(root / 'private', 0o700)
        protocol_path = root / 'experiment_protocol.json'
        pool_path = root / 'private/initial_pool.jsonl'
        if resume:
            _verify_mutable_file(protocol_path)
            protocol = json.loads(protocol_path.read_text())
            for name, expected in protocol['git_evidence'].items():
                if name == 'head':
                    current = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT,
                                             capture_output=True, text=True, check=True).stdout.strip()
                else:
                    current = _sha(root / 'private' / name)
                if current != expected:
                    raise ValueError('E6-G git evidence changed')
            pool_info['pool_snapshot_file_sha256'] = _sha(pool_path)
            if pool_info['pool_snapshot_file_sha256'] != protocol['initial_pool']['pool_snapshot_file_sha256']:
                raise ValueError('E6-G fixed initial pool changed')
            current_protocol = _protocol(s, base, image, splits, pool_info, probes, protocol['git_evidence'])
            if current_protocol != protocol:
                raise ValueError('E6-G resume config, code, source, model, sandbox, or split identity changed')
            if (root / 'seal.json').is_file():
                _verify_g(root)
                return root
        else:
            write_private_jsonl(pool_path, pool)
            pool_info['pool_snapshot_file_sha256'] = _sha(pool_path)
            if not fake and pool_info['pool_snapshot_file_sha256'] != pool_info['pool_sha256']:
                raise ValueError('E6-G copied initial pool differs from the sealed source')
            git_evidence = _capture_git(root)
            protocol = _protocol(s, base, image, splits, pool_info, probes, git_evidence)
            _json(protocol_path, protocol, private=True)
            _seal_mutable_file(protocol_path)
            _json(root / 'split_task_manifest.json', splits, private=True)
        total_path = root / 'budget.json'
        total_budget = E5Budget(total_path, total_config, resume=resume and total_path.exists())
        results = {}
        try:
            for condition in protocol['condition_order']:
                results[condition] = _run_condition(condition, root / 'conditions' / condition,
                    s, base, total_budget, image, pool, probes, forbidden,
                    protocol['initial_pool']['pool_snapshot_file_sha256'], _sha(protocol_path), resume)
        finally:
            total_budget.finish()
        for item in results.values():
            if item['training_steps'] != 0:
                raise ValueError('E6-G observed an unexpected training reservation')
        if (sum(r['model_calls'] for r in results.values()) > total_budget.state['model_calls'] or
                sum(r['sandbox_runs'] for r in results.values()) > total_budget.state['sandbox_runs'] or
                sum(r['reserved_tokens'] for r in results.values()) > total_budget.state['reserved_tokens']):
            raise ValueError('condition ledger exceeds the global ledger')
        # Merge only after both arms finish. This union never enters either prompt.
        union = json.loads(json.dumps(pool))
        cross_arm_rejections = []
        for condition in protocol['condition_order']:
            for item in results[condition]['records']:
                record = item['record']
                classification = classify_novelty(record['messages'][-1]['content'], item['behavior_sha256'], union)
                if classification == 'novel':
                    clean = {key: value for key, value in record.items() if key != '_verifier_only'}
                    _training_safe_record(clean, 'e6_generator_distillation')
                    if not fake:
                        verify_response_body(clean['qwen_provenance'], clean['messages'][-1]['content'])
                    union.append(item)
                else:
                    cross_arm_rejections.append({'condition': condition, 'task_id': record['task_id'],
                                                 'reason': classification})
        write_private_jsonl(root / 'private/combined_records.jsonl', union)
        write_private_jsonl(root / 'sft_generator_distillation.jsonl', [
            {key: value for key, value in item['record'].items() if key != '_verifier_only'} for item in union])
        write_private_jsonl(root / 'private/heldout_verifier.jsonl', [
            {'task_id': item['record']['task_id'], **item['record']['_verifier_only']} for item in union])
        write_private_jsonl(root / 'sft_self_solved.jsonl', [])
        public_results = {name: {key: value for key, value in result.items() if key != 'records'}
                          for name, result in results.items()}
        g0, g1 = (public_results[name] for name in CONDITIONS)
        if g0['verified_novel_oracles'] == g1['verified_novel_oracles'] == 0:
            decision = 'no_demonstrated_gain'
        elif (g0['completed_candidates'] == g1['completed_candidates'] == s.candidates_per_condition and
              g1['verified_novel_oracles'] > g0['verified_novel_oracles']):
            decision = 'novelty_feedback_effect_observed'
        else:
            decision = 'inconclusive'
        eligible_count = 0 if fake else len(union)
        eligible = not fake and eligible_count >= s.min_verified_tasks
        comparison = {'protocol': 'e6g_paired_comparison_result_v1',
            'execution_mode': 'fake_fixture' if fake else 'live_model',
            'condition_order': protocol['condition_order'], 'conditions': public_results,
            'decision': decision, 'interpretation': 'exploratory small sample; no statistical superiority claim',
            'balance': {'declared_condition_caps_equal': True,
                        'completed_candidate_counts_equal': g0['completed_candidates'] == g1['completed_candidates'],
                        'model_call_difference_g1_minus_g0': g1['model_calls'] - g0['model_calls'],
                        'sandbox_difference_g1_minus_g0': g1['sandbox_runs'] - g0['sandbox_runs'],
                        'reserved_token_difference_g1_minus_g0': g1['reserved_tokens'] - g0['reserved_tokens'],
                        'budget_stop_observed': any(r['stop_reason'] == 'budget_exhausted' for r in (g0, g1))},
            'same_initial_pool_snapshot_sha256': protocol['initial_pool']['pool_snapshot_file_sha256'],
            'seed_pool_status': pool_info, 'cross_arm_rejections': cross_arm_rejections,
            'historical_verified_oracles': pool_info['eligible_seed_oracles'],
            'eligible_unique_oracles': eligible_count, 'fixture_union_count': len(union) if fake else None,
            'eligible_for_future_training': eligible,
            'training_gate_status': 'eligible_for_future_training' if eligible else 'insufficient_verified_oracles',
            'training_invoked': False, 'adapter_updated': False, 'promoted': False,
            'global_budget': total_budget.state, 'public_evaluation_accessed': False}
        _json(root / 'comparison.json', comparison, private=True)
        _json(root / 'budget_ledger.json', {**total_budget.state,
            'max_model_calls': s.total_calls, 'max_reserved_tokens': s.total_reserved_tokens,
            'max_sandbox_runs': s.total_sandbox, 'max_wall_seconds': s.total_wall_seconds}, private=True)
        _json(root / 'run_state.json', {'status': comparison['training_gate_status'], 'complete': True,
            'decision': decision, 'eligible_unique_oracles': eligible_count,
            'eligible_for_future_training': eligible, 'training_invoked': False, 'adapter_updated': False}, private=True)
        lines = ['# E6-G paired comparison', '', f'Decision: {decision}.',
                 f"Eligible unique Oracles: {eligible_count}/{s.min_verified_tasks}; training was not invoked.", '',
                 '| Condition | Completed | Valid | Novel | Calls | Reserved | Observed | Sandbox | Stop |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for condition in CONDITIONS:
            r = public_results[condition]
            lines.append(f"| {condition} | {r['completed_candidates']} | {r['oracle_valid']} | "
                         f"{r['verified_novel_oracles']} | {r['model_calls']} | {r['reserved_tokens']} | "
                         f"{r['observed_total_tokens']} | {r['sandbox_runs']} | {r['stop_reason']} |")
        lines += ['', 'Small exploratory comparison; no statistical or ARC improvement claim.', '']
        (root / 'report.md').write_text('\n'.join(lines))
        os.chmod(root / 'report.md', 0o600)
        # Source pool, code, data and split identities must still match the pre-registration.
        end_pool, end_probes, end_info = _load_pool(s, base, image, splits)
        end_info['pool_snapshot_file_sha256'] = _sha(pool_path)
        if _protocol(s, base, image, splits, end_info, end_probes, protocol['git_evidence']) != protocol:
            raise ValueError('E6-G protocol or source identity changed during execution')
        _seal_g(root)
        return root
