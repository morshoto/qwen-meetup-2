"""E7 protocol identity and read-only preflight helpers."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.arc.e6g_config import E6GSettings
from src.arc.e7_config import E7Settings
from src.arc.e7_protocol import schedule
from src.pipelines.e6g import PROJECT, _load_pool, _protocol
from src.pipelines.e6f import _sha, _split_evidence
from src.pipelines.e5 import _docker_image_id


def preflight(settings: E7Settings, base: Any, git_evidence: dict[str, str]) -> tuple[dict, list, list, set]:
    """Validate dependencies and assemble all identities before model dispatch."""
    fake = settings.profile == 'cpu_fake'
    e6g = E6GSettings(profile='cpu_fake' if fake else 'live_smoke',
        base_config=settings.base_config, candidates_per_condition=settings.slots,
        max_repairs=settings.max_repairs, per_condition_calls=settings.arm_calls,
        per_condition_reserved_tokens=settings.arm_tokens, per_condition_sandbox=settings.arm_sandbox,
        per_condition_wall_seconds=settings.arm_seconds, total_calls=settings.total_calls,
        total_reserved_tokens=settings.total_tokens, total_sandbox=settings.total_sandbox,
        total_wall_seconds=settings.total_seconds, seed=settings.seed, pool_path=settings.pool_path)
    image = 'fake' if fake else _docker_image_id(base)
    splits, forbidden = _split_evidence(base)
    pool, probes, info = _load_pool(e6g, base, image, splits)
    old = _protocol(e6g, base, image, splits, info, probes, git_evidence)
    names = ('src/arc/e7_budget.py', 'src/arc/e7_config.py', 'src/arc/e7_protocol.py',
             'src/arc/e7_sandbox.py', 'src/pipelines/e7.py', 'src/cli/e7.py')
    identity = {'protocol': 'e7_paired_generation_slots_v1', 'settings': asdict(settings),
                'e6g_identity': old, 'slot_manifest': schedule(settings),
                'e7_implementation_sha256': {name: _sha(PROJECT / name) for name in names},
                'rules': {'novelty_gain': 'all slots complete and G1 novel count exceeds G0',
                          'no_gain': 'all slots complete and G1 novel count does not exceed G0',
                          'otherwise': 'inconclusive_budget_or_missing_data',
                          'verifier': 'five public and five private inputs, twice each',
                          'cache': 'no sandbox execution reuse',
                          'candidate_exhaustion': 'incomplete; never scored as failed',
                          'independence': 'separate mutable pool per arm',
                          'seed_note': 'matching requested seeds do not guarantee identical output'}}
    return identity, pool, probes, forbidden


def _configuration(base: Any, calls: int, tokens: int, sandbox: int, seconds: int, seed: int) -> Any:
    from src.pipelines.e6g import _compat
    return _compat(base, calls=calls, tokens=tokens, sandboxes=sandbox, seconds=seconds, seed=seed)


def _slot(*, root: Path, arm: str, index: int, settings: E7Settings, base: Any,
          total: Any, arm_budget: Any, pool: list, probes: list, forbidden: set,
          feedback: Any, image: str, fake: bool) -> dict:
    import contextlib
    import json
    import time
    from src.arc.e6_curriculum import _quality_grids
    from src.arc.e6f_repair import run_candidate
    from src.arc.http_backend import BackendError
    from src.arc.mlx_lora import _training_safe_record
    from src.pipelines.e6 import _observed_token_usage
    from src.arc.e6g_novelty import classify_novelty
    from src.arc.e7_budget import LayeredBudget
    from src.arc.e7_sandbox import TracedSandbox, _digest
    from src.arc.e6g_fake import FakeBackend, FakeSandbox
    from src.arc.sandbox import DockerSandbox
    from src.pipelines.e5 import E5Budget, E5CachedBackend, _backend
    from src.pipelines.e6f import _append
    from src.pipelines.e6g import _live_provenance

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    config = _configuration(base, settings.candidate_calls, settings.candidate_tokens,
                            settings.candidate_sandbox, settings.candidate_seconds, settings.seed)
    candidate_budget = E5Budget(root / 'budget.json', config, resume=False)
    layered = LayeredBudget(candidate_budget, arm_budget, total)
    events: list[dict] = []
    def emit(event: dict) -> None:
        events.append(event)
        _append(root / 'events.jsonl', event)
    if fake:
        underlying = FakeSandbox()
    else:
        a = config.agent
        underlying = DockerSandbox(image=image, timeout=a.timeout_seconds, memory_mb=a.memory_mb,
                                   cpus=a.cpus, pids=a.pids, max_output_bytes=a.max_output_bytes,
                                   max_code_bytes=a.max_code_bytes, remaining=layered.remaining)
    sandbox = TracedSandbox(underlying, layered, image, emit)
    sandbox.candidate_id = f'{arm}-{index}'
    cache = root / 'request_cache.json'
    started = time.monotonic()
    result: dict[str, Any] = {'status': 'interrupted', 'baseline_pass': False, 'repair_pass': False}
    found: list[dict[str, Any]] = []
    try:
        context = (contextlib.nullcontext(E5CachedBackend(FakeBackend(arm), layered, cache)) if fake
                   else _backend(config, root, layered, cache, adapter=None, label=f'e7-{arm}-{index}'))
        with context as backend:
            def generate(prompt: str, candidate_index: int, attempt: int):
                sandbox.attempt = attempt
                sandbox.public_inputs = {_digest(grid) for grid in
                                         _quality_grids(71000 + index * 100 + attempt, 5)}
                actual = feedback.prompt(prompt) if attempt == 0 else prompt
                call_index = candidate_index * 3 + attempt
                emit({'event': 'generation_requested', 'candidate_index': index, 'attempt': attempt,
                      'seed_requested': settings.seed + call_index,
                      'base_prompt_bytes': len(prompt.encode()), 'actual_prompt_bytes': len(actual.encode()),
                      'prompt_sha256': hashlib.sha256(actual.encode()).hexdigest(),
                      'feedback_applied': actual != prompt})
                backend.scope = f'{arm}-{index}-{attempt}'
                code = backend.generate(actual, kind='e5_generator', candidate=call_index)
                if fake:
                    provenance = {'source_type': 'generator_distillation',
                        'request_id': f'fake-{arm}-{call_index}',
                        'raw_response_sha256': hashlib.sha256(code.encode()).hexdigest(),
                        'model_revision': base.model.revision, 'model_id': base.model.id,
                        'execution_mode': 'fake_fixture', 'thinking_mode': 'off', 'generation': 0}
                else:
                    provenance = _live_provenance(root, actual, call_index, code)
                classification = feedback.observe(code, pool)
                found.append({'classification': classification, 'code': code})
                return code, provenance
            result = run_candidate(index=index, max_repairs=settings.max_repairs, generate=generate,
                                   sandbox=sandbox, emit=emit, counters=lambda: dict(candidate_budget.state),
                                   probe_inputs=probes)
    except TimeoutError:
        result = {'status': 'budget_exhausted'}
    except (BackendError, RuntimeError):
        result = {'status': 'sandbox_infrastructure_error'}
    finally:
        candidate_budget.finish()
        for path in root.rglob('*'):
            if path.is_file():
                path.chmod(0o600)
    outcome = result['status']
    if outcome == 'budget_exhausted' and layered.last_exhausted:
        outcome = ('candidate_budget_exhausted' if layered.last_exhausted == 'candidate'
                   else 'experiment_inconclusive_budget')
    incomplete = outcome in {'budget_exhausted', 'candidate_budget_exhausted', 'experiment_inconclusive_budget', 'timeout_or_resource_limit',
                             'sandbox_infrastructure_error', 'model_request_failed'}
    requests_path = root / 'logs/e6/llm_requests.jsonl'
    requests = ([json.loads(line) for line in requests_path.read_text().splitlines() if line.strip()]
                if requests_path.is_file() else [])
    row: dict[str, Any] = {'arm': arm, 'slot': index, 'status': outcome,
        'completion': 'incomplete' if incomplete else 'completed',
        'budget_scope': layered.last_exhausted,
        'oracle_valid': outcome == 'passed', 'verified_novel': False,
        'repair_pass': bool(result.get('repair_pass')),
        'raw_duplicate': any(x['classification'] == 'duplicate_source' for x in found),
        'ast_duplicate': any(x['classification'] in {'duplicate_source', 'duplicate_ast'} for x in found),
        'budget': dict(candidate_budget.state), 'elapsed_seconds': time.monotonic() - started,
        'observed_usage': _observed_token_usage(requests_path),
        'model_inference_seconds': sum(float(r.get('elapsed_seconds', 0)) for r in requests),
        'attempts': [e for e in events if e.get('event') == 'attempt_finished'],
        'sandbox_successful_executions': sum(e.get('event') == 'sandbox_finished' and e.get('status') == 'ok' for e in events),
        'sandbox_finished_executions': sum(e.get('event') == 'sandbox_finished' for e in events),
        'phase_starts': {phase: sum(e.get('event') == 'sandbox_dispatch' and e.get('phase') == phase for e in events)
                         for phase in ('public_validation', 'private_acceptance')},
        'prompts': [e for e in events if e.get('event') == 'generation_requested'],
        'feedback_code': feedback.previous_code if arm == 'G1_novelty_feedback' else None}
    if outcome == 'passed':
        record = result['record']
        clean = {k: v for k, v in record.items() if k != '_verifier_only'}
        _training_safe_record(clean, 'e6_generator_distillation')
        behavior = result['behavior_sha256']
        novelty = ('reserved_task_rejected' if record['task_id'] in forbidden else
                   classify_novelty(record['messages'][-1]['content'], behavior, pool))
        row.update(novelty=novelty, behavior_sha256=behavior,
                   behavior_match=any(item.get('behavior_sha256') == behavior for item in pool))
        if novelty == 'novel':
            row['verified_novel'] = True
            row['accepted_item'] = {'record': record, 'behavior_sha256': behavior,
                                    'origin': arm, 'historical_duplicate_observed': False}
    return row


def run_e7(settings: E7Settings, base: Any, *, run_dir: Path | None = None,
           resume: bool = False, live: bool = False, smoke: bool = False,
           allow_full: bool = False, full_budget: tuple[int, int, int, int] | None = None) -> Path:
    """Execute registered slots; uncertain interrupted slots fail closed on resume."""
    import csv
    import json
    import os
    import uuid
    from src.arc.e6g_novelty import NoveltyFeedback, classify_novelty
    from src.arc.e7_protocol import ARMS, arm_metrics, feasibility, selection
    from src.pipelines.e5 import E5Budget, _seal_mutable_file, _verify_mutable_file
    from src.pipelines.e6 import _json, _run_lock
    from src.pipelines.e6g import _capture_git, _seal_g, _verify_g
    from src.pipelines.e6f import _hash, _split_evidence

    fake = settings.profile == 'cpu_fake'
    if fake and (live or smoke or allow_full):
        raise ValueError('fake profile cannot use live flags')
    if settings.profile == 'live_smoke' and not (live and smoke and not allow_full):
        raise ValueError('live smoke requires explicit live and smoke flags')
    if settings.profile == 'live_full' and not (live and allow_full and not smoke):
        raise ValueError('live full requires explicit live and full approval')
    if settings.profile == 'live_full' and full_budget != (settings.total_calls, settings.total_tokens,
                                                              settings.total_sandbox, settings.total_seconds):
        raise ValueError('live full requires explicit matching total budget confirmation')
    if settings.profile != 'live_full' and full_budget is not None:
        raise ValueError('full budget confirmation applies only to full profile')
    root = (run_dir or PROJECT / 'artifacts/e7' / (settings.profile + '-' + uuid.uuid4().hex[:12])).resolve()
    if not root.is_relative_to((PROJECT / 'artifacts/e7').resolve()):
        raise ValueError('E7 run must reside in artifacts/e7')
    if resume and not root.is_dir():
        raise ValueError('resume requires an existing run directory')
    if root.exists() and not resume and any(root.iterdir()):
        raise FileExistsError('run already exists')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with _run_lock(root):
        protocol_path = root / 'protocol.lock.yml'
        if resume:
            _verify_mutable_file(protocol_path)
            if (root / 'protocol.sha256').read_text().strip() != _sha(protocol_path):
                raise ValueError('E7 protocol hash mismatch')
            registered = json.loads(protocol_path.read_text())
            git = registered['e6g_identity']['git_evidence']
            for name, digest in git.items():
                if name != 'head' and _sha(root / 'private' / name) != digest:
                    raise ValueError('registered git evidence changed')
            import subprocess
            if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=PROJECT).strip() != git['head']:
                raise ValueError('registered git head changed')
            identity, initial_pool, probes, forbidden = preflight(settings, base, git)
            if registered != identity:
                raise ValueError('E7 protocol, settings or dependency hash mismatch')
            if (root / 'seal.json').exists():
                _verify_g(root)
                return root
        else:
            git = _capture_git(root)
            identity, initial_pool, probes, forbidden = preflight(settings, base, git)
            _json(protocol_path, identity, private=True)
            _seal_mutable_file(protocol_path)
            (root / 'protocol.sha256').write_text(_sha(protocol_path) + '\n')
            (root / 'protocol.sha256').chmod(0o600)
            _json(root / 'slot_manifest.json', schedule(settings), private=True)
            splits, _ = _split_evidence(base)
            _json(root / 'split_manifest.json', splits, private=True)
        total_config = _configuration(base, settings.total_calls, settings.total_tokens,
                                      settings.total_sandbox, settings.total_seconds, settings.seed)
        if resume and not (root / 'budget.json').is_file():
            raise ValueError('E7 total budget ledger missing')
        total = E5Budget(root / 'budget.json', total_config, resume=resume)
        arms = {}
        for arm in ARMS:
            arm_root = root / 'conditions' / arm
            arm_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            config = _configuration(base, settings.arm_calls, settings.arm_tokens,
                                    settings.arm_sandbox, settings.arm_seconds, settings.seed)
            if resume and not (arm_root / 'budget.json').is_file():
                raise ValueError('E7 arm budget ledger missing')
            arms[arm] = E5Budget(arm_root / 'budget.json', config, resume=resume)
        pools = {arm: json.loads(json.dumps(initial_pool)) for arm in ARMS}
        feedbacks = {arm: NoveltyFeedback(enabled=arm == ARMS[1], pool=pools[arm], allow_fixture=fake)
                     for arm in ARMS}
        rows: list[dict] = []
        blocked_arms: set[str] = set()
        global_stop = False
        try:
            for item in schedule(settings):
                arm, index = item['arm'], item['slot']
                slot_dir = root / 'conditions' / arm / f'slot-{index}'
                result_path = slot_dir / 'result.json'
                marker = slot_dir / 'started.json'
                if result_path.exists():
                    _verify_g(slot_dir)
                    _verify_mutable_file(result_path)
                    row = json.loads(result_path.read_text())
                elif marker.exists():
                    _verify_mutable_file(marker)
                    row = {'arm': arm, 'slot': index, 'status': 'interrupted_uncertain',
                           'completion': 'incomplete', 'verified_novel': False}
                    _json(result_path, row, private=True)
                    _seal_mutable_file(result_path)
                    _seal_g(slot_dir)
                elif global_stop or arm in blocked_arms:
                    slot_dir.mkdir(parents=True, mode=0o700)
                    row = {'arm': arm, 'slot': index, 'status': 'not_started_after_stop',
                           'completion': 'incomplete', 'verified_novel': False}
                    _json(result_path, row, private=True)
                    _seal_mutable_file(result_path)
                    _seal_g(slot_dir)
                else:
                    slot_dir.mkdir(parents=True, mode=0o700)
                    _json(marker, {'slot': index, 'arm': arm, 'protocol_sha256': _sha(protocol_path)}, private=True)
                    _seal_mutable_file(marker)
                    row = _slot(root=slot_dir, arm=arm, index=index, settings=settings, base=base,
                                total=total, arm_budget=arms[arm], pool=pools[arm], probes=probes,
                                forbidden=forbidden, feedback=feedbacks[arm],
                                image=identity['e6g_identity']['sandbox_image_id'], fake=fake)
                    _json(result_path, row, private=True)
                    _seal_mutable_file(result_path)
                    _seal_g(slot_dir)
                if row.get('accepted_item'):
                    pools[arm].append(row['accepted_item'])
                if row.get('feedback_code'):
                    feedbacks[arm].previous_code = row['feedback_code']
                rows.append(row)
                if row.get('budget_scope') == 'arm':
                    blocked_arms.add(arm)
                if row.get('budget_scope') == 'total' or row.get('status') in {
                        'sandbox_infrastructure_error', 'model_request_failed'}:
                    global_stop = True
        finally:
            for ledger in arms.values():
                ledger.finish()
            total.finish()
        comparison = selection(rows, settings)
        combined = json.loads(json.dumps(initial_pool))
        unique_new = []
        for row in rows:
            item = row.get('accepted_item')
            if item and classify_novelty(item['record']['messages'][-1]['content'],
                                        item['behavior_sha256'], combined) == 'novel':
                combined.append(item)
                unique_new.append(item)
        strict_new = 0 if fake else len(unique_new)
        gate = feasibility(strict_count=strict_new, new_strict_count=strict_new,
            selection_decision=comparison['decision'],
            resource_limited=any(row.get('status') in {'budget_exhausted', 'candidate_budget_exhausted', 'experiment_inconclusive_budget', 'timeout_or_resource_limit'} for row in rows))
        if fake:
            gate = 'not_applicable_fake_fixture'
        comparison['arm_metrics'] = {arm: arm_metrics(rows, arm, settings.slots) for arm in ARMS}
        comparison.update({'feasibility_gate': gate, 'legacy_reported_oracles': 0 if fake else 2,
            'currently_reverified_oracles': 0 if fake else 2,
            'strict_training_eligible_oracles': strict_new, 'new_strict_oracles': strict_new,
            'training_invoked': False, 'adapter_updated': False, 'promoted': False,
            'execution_mode': 'fake_fixture' if fake else 'live_model',
            'total_budget': total.state, 'arm_budgets': {arm: ledger.state for arm, ledger in arms.items()}})
        public_rows = [{k: v for k, v in row.items() if k not in {'accepted_item', 'feedback_code'}} for row in rows]
        _json(root / 'comparison.json', comparison, private=True)
        _json(root / 'slots.json', public_rows, private=True)
        _json(root / 'stop_reason.json', {'decision': comparison['decision'],
              'slots': [{'arm': r['arm'], 'slot': r['slot'], 'status': r['status']} for r in rows]}, private=True)
        with (root / 'slots.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['arm', 'slot', 'completion', 'status', 'oracle_valid',
                                                         'verified_novel', 'ast_duplicate', 'repair_pass'],
                                    extrasaction='ignore')
            writer.writeheader(); writer.writerows(public_rows)
        os.chmod(root / 'slots.csv', 0o600)
        lines = ['# E7 selection', '', f"Decision: {comparison['decision']}", '',
                 f'Feasibility: {gate}', '',
                 '| Arm | Completed / planned | Novel | AST duplicates / completed | Calls | Reserved / observed tokens | Sandbox |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            m = comparison['arm_metrics'][arm]
            lines.append(f"| {arm} | {m['completed_slots']} / {m['planned_slots']} | {m['verified_novel_oracles']} | "
                         f"{m['ast_duplicate_candidates']} / {m['completed_slots']} | {m['model_calls']} | "
                         f"{m['reserved_tokens']} / {m['observed_tokens']} | {m['sandbox_starts']} |")
        lines += ['', 'Results are exploratory; a small sample does not establish model superiority.', '']
        (root / 'selection_report.md').write_text('\n'.join(lines))
        os.chmod(root / 'selection_report.md', 0o600)
        for path in root.rglob('*'):
            if path.is_file():
                os.chmod(path, 0o600)
        final_identity, _, _, _ = preflight(settings, base, git)
        if final_identity != identity:
            raise ValueError('E7 protocol identity changed during execution')
        _seal_g(root)
        return root
