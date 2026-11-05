"""Sequential, bounded model scaling using E7's candidate verifier."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from src.arc.e8_config import E8Settings, MODELS, ORDER
from src.arc.e8_resource import ResourceMonitor, check_resources, snapshot
from src.arc.e6g_novelty import NoveltyFeedback
from src.arc.e7_config import E7Settings
from src.core.config import load_experiment_config
from src.pipelines.e5 import E5Budget, _backend, _docker_image_id, _seal_mutable_file, _verify_mutable_file
from src.pipelines.e6 import _json, _observed_token_usage, _run_lock
from src.pipelines.e6f import _sha, _split_evidence
from src.pipelines.e6g import PROJECT, PairedBudget, _load_pool, _seal_g, _verify_g
from src.pipelines.e6g import _protocol as e6g_protocol
from src.arc.e6g_config import E6GSettings
from src.pipelines.e7 import _configuration, _slot

CANARIES = ('Reply with the single word READY.',
            'Return only JSON with a field named grid containing a rectangular 2 by 2 array of integers.')


def training_gate(strict_new_oracles: int, *, fake: bool) -> str:
    if type(strict_new_oracles) is not int or strict_new_oracles < 0:
        raise ValueError('invalid Oracle count')
    if fake:
        return 'not_applicable_fake_fixture'
    return 'eligible_for_future_planning' if strict_new_oracles >= 5 else 'below_five'


def _base(s: E8Settings):
    return load_experiment_config(PROJECT / s.base_config)


def _e7_settings(s: E8Settings) -> E7Settings:
    return E7Settings(profile='cpu_fake' if s.profile in ('fake', 'fake_full') else 'live_smoke',
        base_config=s.base_config, slots=s.slots, candidate_sandbox=s.candidate_sandbox,
        arm_sandbox=s.sandbox, total_sandbox=s.sandbox, candidate_calls=3,
        arm_calls=s.calls, total_calls=s.calls, candidate_tokens=15000,
        arm_tokens=s.reserved_tokens, total_tokens=s.reserved_tokens,
        candidate_seconds=s.candidate_seconds, arm_seconds=s.wall_seconds,
        total_seconds=s.wall_seconds, max_repairs=s.max_repairs, seed=s.seed)


def _source_pool(s: E8Settings, base: Any, image: str):
    e = _e7_settings(s)
    # The legacy pool was audited using 8B; it remains a fixed duplicate-only pool
    # for every model, not a statement about a larger model's provenance.
    old = E6GSettings(profile='cpu_fake' if s.profile in ('fake', 'fake_full') else 'live_smoke',
        base_config=s.base_config, candidates_per_condition=s.slots, max_repairs=s.max_repairs,
        per_condition_calls=s.calls, per_condition_reserved_tokens=s.reserved_tokens,
        per_condition_sandbox=s.sandbox, per_condition_wall_seconds=s.wall_seconds,
        total_calls=s.calls * 2, total_reserved_tokens=s.reserved_tokens * 2,
        total_sandbox=s.sandbox * 2, total_wall_seconds=s.wall_seconds * 2, seed=s.seed)
    splits, forbidden = _split_evidence(base)
    pool, probes, info = _load_pool(old, base, image, splits)
    identity = e6g_protocol(old, base, image, splits, info, probes, {})
    return pool, probes, forbidden, splits, identity


def _protocol(s: E8Settings, base: Any, image: str, source: dict) -> dict:
    names = ['src/arc/__init__.py', 'src/arc/e5_plugins.py',
             'src/arc/e8_config.py', 'src/arc/e8_resource.py', 'src/pipelines/e8.py', 'src/cli/e8.py',
             'src/arc/e7_budget.py', 'src/arc/e7_config.py', 'src/arc/e7_protocol.py',
             'src/arc/e7_sandbox.py', 'src/pipelines/e7.py']
    manifests = ({key: _model_manifest(_model_dir(model, revision), revision)
                  for key, (model, revision, _) in MODELS.items()} if s.profile == 'full' else {})
    return {'protocol': 'e8_model_scaling_v1', 'settings': asdict(s), 'model_identities': {k: list(v) for k, v in MODELS.items()},
        'model_file_manifests': manifests,
        'model_order': list(ORDER), 'arm': 'G0_baseline', 'slots': [
            {'slot': i, 'attempt_seeds': [s.seed + i * 3 + j for j in range(s.max_repairs + 1)]}
            for i in range(s.slot_offset, s.slot_offset + s.slots)],
        'canary_prompts_sha256': [hashlib.sha256(x.encode()).hexdigest() for x in CANARIES],
        'canaries_count_against_budget': True, 'initial_pool': source['initial_pool'],
        'verifier_identity': source, 'sandbox_image_id': image,
        'dependencies': {name: _sha(PROJECT / name) for name in names},
        'rules': {'legacy_strict_count': 0, 'minimum_strict_oracles_per_model': 5,
                  'feedback': False, 'sampled_behavior': 'suspicion, not semantic proof',
                  'training': False, 'adapter': 'none', 'operator_search': False,
                  'seed': 'requested seed is not a deterministic guarantee',
                  'resource_stop': 'incomplete, not zero-success evidence',
                  'aggregate_budget': {'calls': s.calls * 3, 'reserved_tokens': s.reserved_tokens * 3,
                                       'sandbox': s.sandbox * 3, 'wall_seconds': s.wall_seconds * 3},
                  'slot_budget': {'calls': 3, 'reserved_tokens': 15000, 'sandbox': s.candidate_sandbox}}}


def _model_dir(model: str, revision: str) -> Path:
    return Path.home() / '.cache/huggingface/hub' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision


def _download(model: str, revision: str, seconds: int) -> None:
    # The revision is an immutable commit; huggingface_hub performs LFS integrity checks.
    cmd = ['/tmp/qwen-p3a-mlx-venv/bin/python', '-c',
           'from huggingface_hub import snapshot_download; import sys; '
           'print(snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2]))', model, revision]
    subprocess.run(cmd, check=True, timeout=seconds)


def _model_manifest(path: Path, revision: str) -> dict:
    if path.name != revision or not path.is_dir():
        raise ValueError('model revision snapshot missing')
    files = {str(p.relative_to(path)): _sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
    if not any(x.endswith('.safetensors') for x in files) or 'config.json' not in files or 'tokenizer_config.json' not in files:
        raise ValueError('required model files missing')
    config = json.loads((path / 'config.json').read_text())
    quant = config.get('quantization', config.get('quantization_config', {}))
    if config.get('model_type') != 'qwen3' or not isinstance(quant, dict) or quant.get('bits') != 4 or quant.get('group_size') != 64:
        raise ValueError('model is not the expected Qwen3 4-bit group-size 64')
    tokenizer = json.loads((path / 'tokenizer_config.json').read_text())
    template = tokenizer.get('chat_template')
    if not isinstance(template, str) or not template:
        raise ValueError('model chat template is absent')
    return {'revision': revision, 'files_sha256': files, 'quantization': quant,
            'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
            'tokenizer_config_sha256': files['tokenizer_config.json']}


def _check_no_other_model_server() -> None:
    ps = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True, timeout=10)
    if any('src.cli.guarded_mlx_server' in line or 'mlx_lm.server' in line for line in ps.splitlines()):
        raise RuntimeError('another MLX server is already running')
    if any('ollama serve' in line for line in ps.splitlines()):
        import urllib.request
        try:
            with urllib.request.urlopen('http://127.0.0.1:11434/api/ps', timeout=3) as response:
                data = json.load(response)
            if not isinstance(data.get('models'), list) or data['models']:
                raise RuntimeError('Ollama has a loaded model or its state is ambiguous')
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError('cannot verify Ollama model residency') from exc


def _canaries(root: Path, config: Any, budget: Any, fake: bool) -> dict:
    from src.pipelines.e5 import _canonical
    if fake:
        for p in CANARIES:
            budget.reserve_model(p, 'canary')
        return {'mode': 'fixture', 'neutral': 'READY', 'grid_json': True,
                'model_load_seconds': None, 'first_token_seconds': None}
    started = time.monotonic()
    cache = root / 'canary-cache.json'
    with _backend(config, root, budget, cache, adapter=None, label='e8-canary') as backend:
        loaded = time.monotonic() - started
        backend.scope = 'canary-neutral'
        a = backend.generate(CANARIES[0], kind='e8_canary', candidate=10000)
        backend.scope = 'canary-grid'
        b = backend.generate(CANARIES[1], kind='direct', candidate=10001)
    if a.strip().strip('.').upper() != 'READY':
        raise ValueError('neutral canary did not return READY')
    parsed = json.loads(b)
    grid = parsed.get('grid') if isinstance(parsed, dict) else None
    if (not isinstance(grid, list) or len(grid) != 2 or
        any(not isinstance(row, list) or len(row) != 2 or any(type(x) is not int for x in row) for row in grid)):
        raise ValueError('grid JSON canary failed')
    requests = [json.loads(x) for x in (root / 'logs/e6/llm_requests.jsonl').read_text().splitlines()]
    if any(r.get('thinking_mode_effective') != 'off' or r.get('finish_reason') != 'stop' for r in requests):
        raise ValueError('canary thinking or finish reason failed')
    return {'mode': 'live', 'neutral': 'READY', 'grid_json': True,
            'model_load_seconds': loaded, 'first_token_seconds': None,
            'request_metrics': [{k: r.get(k) for k in ('elapsed_seconds', 'usage', 'finish_reason',
                'thinking_mode_effective', 'thinking_control')} for r in requests]}


def run_stage(s: E8Settings, stage: str, campaign: Path, *, live: bool = False,
              allow_32b: bool = False, allow_full: bool = False,
              full_budget: tuple[int, int, int, int] | None = None, resume: bool = False) -> Path:
    if stage not in ORDER:
        raise ValueError('unknown model stage')
    fake = s.profile in ('fake', 'fake_full')
    if fake and (live or allow_32b or allow_full):
        raise ValueError('fake profile cannot use live flags')
    if not fake and not live:
        raise ValueError('live stage requires --live')
    if not fake and stage == '32b' and not allow_32b:
        raise ValueError('32B requires explicit independent permission')
    if s.profile == 'full' and (not allow_full or full_budget !=
                                (s.calls, s.reserved_tokens, s.sandbox, s.wall_seconds)):
        raise ValueError('full requires explicit permission and matching total budget')
    if s.profile != 'full' and (allow_full or full_budget is not None):
        raise ValueError('full approval is valid only for full profile')
    campaign = campaign.resolve()
    if not campaign.is_relative_to((PROJECT / 'artifacts/e8').resolve()):
        raise ValueError('campaign must be in artifacts/e8')
    if stage != '8b' and not campaign.exists():
        raise ValueError('start with 8B')
    campaign.mkdir(parents=True, exist_ok=True, mode=0o700)
    campaign.chmod(0o700)
    with _run_lock(campaign):
        base = _base(s)
        image = 'fake' if fake else _docker_image_id(base)
        pool, probes, forbidden, splits, source = _source_pool(s, base, image)
        protocol = _protocol(s, base, image, source)
        lock = campaign / 'protocol.lock.yml'
        if lock.exists():
            _verify_mutable_file(lock)
            if json.loads(lock.read_text()) != protocol:
                raise ValueError('campaign protocol or implementation identity changed')
        else:
            if stage != '8b':
                raise ValueError('campaign must start with 8B')
            _json(lock, protocol, private=True); _seal_mutable_file(lock)
            (campaign / 'protocol.sha256').write_text(_sha(lock) + '\n')
            (campaign / 'protocol.sha256').chmod(0o600)
            _json(campaign / 'initial_pool.json', pool, private=True)
            _seal_mutable_file(campaign / 'initial_pool.json')
            _json(campaign / 'split_manifest.json', splits, private=True)
            _seal_mutable_file(campaign / 'split_manifest.json')
        _verify_mutable_file(campaign / 'initial_pool.json')
        if json.loads((campaign / 'initial_pool.json').read_text()) != pool:
            raise ValueError('shared seed pool changed')
        if stage != '8b':
            prior = campaign / ORDER[ORDER.index(stage) - 1]
            _verify_g(prior)
            previous = json.loads((prior / 'result.json').read_text())
            if previous['status'] != 'completed':
                raise ValueError('prior model stage did not complete')
        root = campaign / stage
        if root.exists() and (root / 'seal.json').exists():
            if not resume:
                raise FileExistsError('completed model stage exists; use resume')
            _verify_g(root)
            return root
        if root.exists() and not resume:
            raise FileExistsError('model stage exists; use resume')
        if resume and not root.exists():
            raise ValueError('resume requires an existing stage')
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        return _stage_body(s, stage, root, base, pool, probes, forbidden, image,
                           _sha(lock), fake=fake, resume=resume)


def _stage_body(s: E8Settings, stage: str, root: Path, base: Any, initial_pool: list,
                probes: list, forbidden: set, image: str, protocol_sha: str,
                *, fake: bool, resume: bool) -> Path:
    model, revision, size = MODELS[stage]
    start = time.monotonic()
    before = snapshot(root)
    reason = check_resources(before, before, min_free_percent=s.min_memory_free_percent,
        max_swap_increase_mib=s.max_swap_increase_mib,
        min_disk_bytes=s.min_free_disk_gib * 1024**3 + (0 if fake or _model_dir(model, revision).is_dir() else size))
    canary: dict = {}
    rows: list[dict] = []
    model_info: dict = {}
    preparation_seconds = 0.0
    monitor: ResourceMonitor | None = None
    total = arm = None
    status = 'resource_infeasible' if reason else 'completed'
    error = reason
    try:
        if reason:
            raise RuntimeError(reason)
        if not fake:
            _check_no_other_model_server()
            model_path = _model_dir(model, revision)
            if not model_path.is_dir():
                if resume:
                    raise RuntimeError('interrupted_download_not_retried')
                _download(model, revision, min(3600 if s.profile == 'full' else 600, s.wall_seconds))
            model_info = _model_manifest(model_path, revision)
            base = replace(base, model=replace(base.model, id=model, revision=revision),
                p3b=replace(base.p3b, training=replace(base.p3b.training, model_path=str(model_path))))
        else:
            model_info = {'model_id': model, 'revision': revision, 'fixture': True}
        model_info['model_id'] = model
        preparation_seconds = time.monotonic() - start
        identity = {'model': model_info, 'campaign_protocol_sha256': protocol_sha,
                    'preparation_seconds': preparation_seconds, 'preflight': before, 'image_digest': image}
        identity_path = root / 'model.lock.json'
        if resume:
            _verify_mutable_file(identity_path)
            old_identity = json.loads(identity_path.read_text())
            if old_identity['model'] != model_info or old_identity['campaign_protocol_sha256'] != protocol_sha:
                raise ValueError('model identity changed on resume')
            preparation_seconds = old_identity['preparation_seconds']
        else:
            _json(identity_path, identity, private=True); _seal_mutable_file(identity_path)
        remaining_seconds = int(s.wall_seconds - preparation_seconds)
        if remaining_seconds <= 0:
            raise TimeoutError('model preparation exhausted wall budget')
        cfg = _configuration(base, s.calls, s.reserved_tokens, s.sandbox, remaining_seconds, s.seed)
        if resume and any(not (root / f'{x}.json').exists() for x in ('total-budget', 'arm-budget')):
            raise ValueError('resume lacks budget ledger')
        total = E5Budget(root / 'total-budget.json', cfg, resume=resume)
        arm = E5Budget(root / 'arm-budget.json', cfg, resume=resume)
        budget = PairedBudget(total, arm)
        monitor = ResourceMonitor(root, before, min_free_percent=s.min_memory_free_percent,
            max_swap_increase_mib=s.max_swap_increase_mib, min_disk_bytes=s.min_free_disk_gib * 1024**3)
        with monitor:
            canary_path = root / 'canary.json'
            if canary_path.exists():
                _verify_mutable_file(canary_path)
                canary = json.loads(canary_path.read_text())
            elif resume and (root / 'canary-started.json').exists():
                raise RuntimeError('interrupted_canary_not_retried')
            else:
                _json(root / 'canary-started.json', {'protocol_sha256': protocol_sha}, private=True)
                _seal_mutable_file(root / 'canary-started.json')
                canary = _canaries(root, cfg, budget, fake)
                _json(canary_path, canary, private=True); _seal_mutable_file(canary_path)
            if monitor.stop_reason:
                raise RuntimeError(monitor.stop_reason)
            pool = json.loads(json.dumps(initial_pool))
            feedback = NoveltyFeedback(enabled=False, pool=pool, allow_fixture=fake)
            for i in range(s.slot_offset, s.slot_offset + s.slots):
                slotroot = root / 'slots' / str(i)
                result_path = slotroot / 'result.json'
                marker = slotroot / 'started.json'
                if result_path.exists():
                    _verify_g(slotroot)
                    row = json.loads(result_path.read_text())
                elif marker.exists():
                    row = {'arm': 'G0_baseline', 'slot': i, 'completion': 'incomplete',
                           'status': 'interrupted_uncertain', 'verified_novel': False}
                    _json(result_path, row, private=True); _seal_mutable_file(result_path); _seal_g(slotroot)
                else:
                    if monitor.stop_reason:
                        raise RuntimeError(monitor.stop_reason)
                    slotroot.mkdir(parents=True, exist_ok=True, mode=0o700)
                    _json(marker, {'slot': i, 'protocol_sha256': protocol_sha}, private=True)
                    _seal_mutable_file(marker)
                    row = _slot(root=slotroot, arm='G0_baseline', index=i, settings=_e7_settings(s),
                        base=base, total=total, arm_budget=arm, pool=pool, probes=probes,
                        forbidden=forbidden, feedback=feedback, image=image, fake=fake)
                    _json(result_path, row, private=True); _seal_mutable_file(result_path); _seal_g(slotroot)
                rows.append(row)
                if row.get('accepted_item'):
                    pool.append(row['accepted_item'])
                if row.get('status') in {'model_request_failed', 'sandbox_infrastructure_error',
                                         'experiment_inconclusive_budget'}:
                    status = 'incomplete_due_to_resource_cap'
                    error = row['status']
                    break
    except Exception as exc:
        status = 'resource_infeasible' if (reason or (monitor and monitor.stop_reason)) else 'incomplete_due_to_resource_cap'
        error = (monitor.stop_reason if monitor and monitor.stop_reason else type(exc).__name__ + ': ' + str(exc))
    finally:
        if arm: arm.finish()
        if total: total.finish()
    if not fake and model_info and status == 'completed':
        try:
            if _model_manifest(_model_dir(model, revision), revision) != {k: v for k, v in model_info.items() if k != 'model_id'}:
                status, error = 'protocol_invalid', 'model files changed during stage'
        except Exception as exc:
            status, error = 'protocol_invalid', 'model identity recheck failed: ' + type(exc).__name__
    after = snapshot(root)
    oom = any(any(term in path.read_text(errors='replace').lower() for term in ('out of memory', 'memoryerror', 'metal out of memory'))
              for path in root.rglob('*server.log'))
    if oom:
        status, error = 'resource_infeasible', 'oom_reported_in_server_log'
    if monitor and monitor.stop_reason:
        status, error = 'resource_infeasible', monitor.stop_reason
    elif len(rows) != s.slots or any(r['completion'] != 'completed' for r in rows):
        if status == 'completed': status, error = 'incomplete_due_to_resource_cap', 'incomplete_slot'
    try:
        original_base = _base(s)
        _, _, _, _, final_source = _source_pool(s, original_base, image)
        final_protocol = _protocol(s, original_base, image, final_source)
        if final_protocol != json.loads((root.parent / 'protocol.lock.yml').read_text()):
            status, error = 'protocol_invalid', 'dependencies changed during stage'
    except Exception as exc:
        status, error = 'protocol_invalid', 'source recheck failed: ' + type(exc).__name__
    strict = 0 if fake or status == 'protocol_invalid' else sum(bool(r.get('verified_novel')) for r in rows)
    usage = []
    for log in root.rglob('llm_requests.jsonl'):
        usage.append(_observed_token_usage(log))
    calls = total.state['model_calls'] if total else 0
    known = sum(x['calls_with_usage'] for x in usage)
    observed = sum(x['observed_total_tokens'] or 0 for x in usage) if calls and known == calls else None
    samples = monitor.samples if monitor else []
    summary = {'stage': stage, 'model_id': model, 'revision': revision, 'profile': s.profile,
        'status': status, 'stop_reason': error, 'planned_slots': s.slots,
        'completed_slots': sum(r['completion'] == 'completed' for r in rows),
        'valid_oracles': sum(bool(r.get('oracle_valid')) for r in rows),
        'strict_new_oracles': strict, 'fixture_novel_oracles': sum(bool(r.get('verified_novel')) for r in rows) if fake else None,
        'source_duplicates': sum(bool(r.get('raw_duplicate')) for r in rows),
        'ast_duplicates': sum(bool(r.get('ast_duplicate')) for r in rows),
        'behavior_matches': sum(bool(r.get('behavior_match')) for r in rows),
        'behavior_evaluated': sum('behavior_sha256' in r for r in rows),
        'repair_successes': sum(bool(r.get('repair_pass')) for r in rows),
        'repair_attempted_candidates': sum(any(a.get('attempt', 0) > 0 for a in r.get('attempts', [])) for r in rows),
        'runtime_error_attempts': sum(a.get('category') == 'runtime_exception' for r in rows for a in r.get('attempts', [])),
        'format_error_attempts': sum(a.get('category') in {'syntax_error', 'parse_or_contract_error', 'invalid_model_response'} for r in rows for a in r.get('attempts', [])),
        'novel_per_planned_slot': strict / s.slots,
        'novel_per_completed_slot': strict / sum(r['completion'] == 'completed' for r in rows) if any(r['completion'] == 'completed' for r in rows) else None,
        'reserved_tokens': total.state['reserved_tokens'] if total else 0, 'observed_tokens': observed,
        'model_calls': calls, 'sandbox_starts': total.state['sandbox_runs'] if total else 0,
        'elapsed_seconds': time.monotonic() - start,
        'model_inference_seconds': sum(r.get('model_inference_seconds', 0) for r in rows),
        'canary': canary, 'before': before, 'after': after, 'resource_samples': samples,
        'peak_sampled_server_rss_kib': max((rss for item in samples for rss in item['owned_server_rss_kib']), default=None),
        'training_gate': training_gate(strict, fake=fake),
        'oom_observed_in_logs': oom,
        'training_invoked': False, 'adapter_updated': False, 'promoted': False,
        'cost_per_novel_oracle': None if strict == 0 else (total.state['reserved_tokens'] / strict if total else None),
        'model_manifest': model_info, 'campaign_protocol_sha256': protocol_sha}
    public_rows = [{k: v for k, v in r.items() if k not in ('accepted_item', 'feedback_code')} for r in rows]
    _json(root / 'slots.json', public_rows, private=True)
    _json(root / 'result.json', summary, private=True)
    for path in root.rglob('*'):
        if path.is_file(): path.chmod(0o600)
    _seal_g(root)
    return root
