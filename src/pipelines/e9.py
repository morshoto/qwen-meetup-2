"""Independent, provenance-bound Qwen3-14B and Qwen3-32B E9 runs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from src.arc.e7_config import E7Settings
from src.arc.e8_config import E8Settings
from src.arc.e8_resource import ResourceMonitor, check_resources, snapshot
from src.arc.e9_config import E9Settings, MODELS
from src.arc.e6g_novelty import NoveltyFeedback
from src.arc.ast_fingerprint import stable_ast_sha256
from src.core.config import load_experiment_config
from src.pipelines.e5 import (E5Budget, _docker_image_id, _seal_mutable_file,
                              _verify_mutable_file)
from src.pipelines.e6 import (_json, _observed_token_usage, _run_lock,
                              _secure_private_dirs)
from src.pipelines.e6f import _sha, _split_evidence
from src.pipelines.e6g import (_load_pool, _seal_g, _verify_g, _live_provenance,
                               PairedBudget)
from src.pipelines.e7 import _slot
from src.pipelines.e8 import (_base as _e8_base, _canaries, _check_no_other_model_server,
                              _model_dir, _model_manifest, _source_pool)

PROJECT = Path(__file__).resolve().parents[2]
E8_SEED_CAMPAIGN = PROJECT / 'artifacts/e8/full-rerun-preregistered-20261102'
E8_SEED_AUDIT = PROJECT / 'artifacts/e8/audit-full-rerun-20261104'
MIN_STRICT_ORACLES = 5
IMPLEMENTATION_FILES = (
    'src/arc/e9_config.py', 'src/pipelines/e9.py', 'src/cli/e9.py',
    'src/arc/e7_config.py', 'src/arc/e7_budget.py', 'src/arc/e7_protocol.py',
    'src/arc/e7_sandbox.py', 'src/pipelines/e7.py',
    'src/arc/e6f_repair.py', 'src/arc/e6g_novelty.py',
    'src/arc/e6g_provenance.py', 'src/arc/ast_fingerprint.py',
    'src/arc/e8_config.py', 'src/arc/e8_resource.py', 'src/pipelines/e8.py',
    'src/arc/e5_selfplay.py', 'src/arc/e5_plugins.py', 'src/pipelines/e5.py',
    'src/pipelines/e6.py', 'src/pipelines/e6f.py', 'src/pipelines/e6g.py',
    'src/arc/data.py', 'src/arc/sandbox.py', 'sandbox/runner.py',
    'src/arc/prompting.py', 'src/arc/http_backend.py', 'src/arc/guarded_contract.py',
    'src/arc/guarded_generation.py', 'src/arc/guarded_http.py',
    'src/arc/token_ledger.py', 'src/arc/deadline_watchdog.py',
    'src/cli/guarded_mlx_server.py', 'src/core/config.py',
    'src/arc/training.py', 'src/arc/mlx_lora.py', 'src/arc/mlx_chat_template.py',
    'src/arc/operator_search.py', 'uv.lock',
)


def _config_for(base: Any, *, calls: int, tokens: int, sandboxes: int,
                seconds: int, seed: int) -> Any:
    from src.pipelines.e6g import _compat
    return _compat(base, calls=calls, tokens=tokens, sandboxes=sandboxes,
                   seconds=seconds, seed=seed)


def _e8_settings(settings: E9Settings) -> E8Settings:
    return E8Settings(
        profile='fake_full' if settings.profile == 'fake' else 'full',
        base_config=settings.base_config, slots=min(settings.slots, 40),
        candidate_sandbox=settings.candidate_sandbox, sandbox=settings.sandbox,
        calls=settings.calls, reserved_tokens=settings.reserved_tokens,
        wall_seconds=settings.wall_seconds, candidate_seconds=settings.candidate_seconds,
        max_repairs=settings.max_repairs, seed=settings.seed,
        min_free_disk_gib=settings.min_free_disk_gib,
        min_memory_free_percent=settings.min_memory_free_percent,
        max_swap_increase_mib=settings.max_swap_increase_mib,
        slot_offset=settings.slot_offset,
    )


def _e7_settings(settings: E9Settings) -> E7Settings:
    return E7Settings(
        profile='cpu_fake' if settings.profile == 'fake' else 'live_full',
        base_config=settings.base_config, slots=settings.slots,
        candidate_sandbox=settings.candidate_sandbox,
        arm_sandbox=settings.sandbox, total_sandbox=settings.sandbox,
        candidate_calls=3, arm_calls=settings.calls, total_calls=settings.calls,
        candidate_tokens=15000, arm_tokens=settings.reserved_tokens,
        total_tokens=settings.reserved_tokens, candidate_seconds=settings.candidate_seconds,
        arm_seconds=settings.wall_seconds, total_seconds=settings.wall_seconds,
        max_repairs=settings.max_repairs, seed=settings.seed,
    )


def _environment(base: Any, fake: bool) -> dict[str, Any]:
    value: dict[str, Any] = {'python': sys.version, 'platform': platform.platform(),
                             'machine': platform.machine(), 'fake': fake}
    if fake:
        return value
    executable = base.p3b.training.python_executable
    code = ('import importlib.metadata as m, json; names=("mlx", "mlx-lm", "transformers", "tokenizers", "numpy"); '
            'print(json.dumps({n:m.version(n) for n in names}))')
    run = subprocess.run([executable, '-c', code], check=True, capture_output=True,
                         text=True, timeout=20)
    value['mlx_runtime'] = json.loads(run.stdout)
    value['python_executable'] = executable
    return value


def _verify_audit(root: Path) -> dict[str, Any]:
    _verify_g(root)
    return json.loads((root / 'summary.json').read_text(encoding='utf-8'))


def _e8_existing(model_key: str, fake: bool, image: str) -> tuple[list[dict], list[dict], dict]:
    """Re-audit E8 strict records; incomplete E7 records remain duplicate-only."""
    if fake:
        return [], [], {'eligible': 0, 'rejected': 0, 'reasons': {'fake_fixture': 0}}
    _verify_audit(E8_SEED_AUDIT)
    protocol_path = E8_SEED_CAMPAIGN / 'protocol.lock.yml'
    _verify_mutable_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    required_top = {'src/arc/e7_config.py', 'src/arc/e7_protocol.py',
                    'src/arc/e5_plugins.py', 'src/pipelines/e8.py'}
    verifier = protocol.get('verifier_identity', {})
    nested = verifier.get('implementation_sha256', {})
    required_nested = {'src/arc/e6f_repair.py', 'src/arc/e6g_provenance.py',
                       'src/arc/training.py', 'src/arc/ast_fingerprint.py'}
    if not required_top.issubset(protocol.get('dependencies', {})) or not required_nested.issubset(nested):
        raise ValueError('E8 source record lacks required strict provenance dependencies')
    for name, expected in {**protocol['dependencies'], **nested}.items():
        if _sha(PROJECT / name) != expected:
            raise ValueError('E8 source dependency changed: ' + name)
    if protocol.get('sandbox_image_id') != image:
        raise ValueError('E8 seed sandbox digest differs from E9')
    stage = E8_SEED_CAMPAIGN / model_key
    _verify_g(stage)
    result = json.loads((stage / 'result.json').read_text(encoding='utf-8'))
    if (result.get('status') != 'completed' or result.get('model_id') != MODELS[model_key][0] or
            result.get('revision') != MODELS[model_key][1] or
            result.get('campaign_protocol_sha256') != _sha(protocol_path)):
        raise ValueError('E8 strict seed model or protocol identity differs')
    audit = _verify_audit(E8_SEED_AUDIT)
    if not audit.get('seal_valid') or not audit.get('stages', {}).get(model_key, {}).get('summary_recomputed'):
        raise ValueError('E8 strict seed was not independently audited')
    strict_rows = []
    for slot in protocol['slots']:
        slot_root = stage / 'slots' / str(slot['slot'])
        if not slot_root.is_dir():
            continue
        _verify_g(slot_root)
        row = json.loads((slot_root / 'result.json').read_text(encoding='utf-8'))
        if row.get('verified_novel') and row.get('oracle_valid'):
            item = row.get('accepted_item')
            if not isinstance(item, dict) or not isinstance(item.get('record'), dict):
                raise ValueError('E8 strict row lacks its sealed Oracle record')
            strict_rows.append((slot_root, row, item))
    if len(strict_rows) != result.get('strict_new_oracles'):
        raise ValueError('E8 strict count differs from sealed candidate records')
    existing, pool_items = [], []
    for slot_root, row, item in strict_rows:
        record = item['record']
        provenance = record.get('qwen_provenance', {})
        code = record.get('messages', [{}])[-1].get('content', '')
        if (provenance.get('execution_mode') != 'live_model' or
                provenance.get('model_id') != MODELS[model_key][0] or
                provenance.get('model_revision', provenance.get('revision')) != MODELS[model_key][1] or
                not record.get('task_id') or not code or
                record.get('oracle_ast_sha256') != stable_ast_sha256(code)):
            raise ValueError('E8 strict record provenance is incomplete')
        clean = {k: v for k, v in record.items() if k != '_verifier_only'}
        source = {
            'source_model_id': MODELS[model_key][0], 'model_revision': MODELS[model_key][1],
            'generation': 0, 'oracle_source': 'generator_distillation',
            'task_id': record['task_id'], 'canonical_ast_hash': record['oracle_ast_sha256'],
            'behavior_fingerprint': item['behavior_sha256'],
            'generator_prompt_hash': row['prompts'][0]['prompt_sha256'],
            'verifier_revision': nested['src/arc/e6f_repair.py'],
            'static_checker_hash': nested['src/arc/e6f_repair.py'],
            'docker_image_digest': protocol['sandbox_image_id'],
            'seal': _sha(slot_root / 'seal.json'), 'source_run': str(E8_SEED_CAMPAIGN),
            'source_protocol_sha256': _sha(protocol_path),
        }
        existing.append({'record': clean, **source})
        pool_items.append({'record': record, 'behavior_sha256': item['behavior_sha256'],
                           'origin': 'E8_strict_existing', 'historical_duplicate_observed': False})
    return existing, pool_items, {'eligible': len(existing), 'rejected': 2,
                                   'reasons': {'E7_missing_static_checker_hash': 2}}


def _load_parent_pool(parent_run: Path | None, model_key: str, expected: dict) -> tuple[list[dict], list[dict]]:
    if parent_run is None:
        return [], []
    parent = parent_run.resolve()
    if not parent.is_dir() or not (parent / 'seal.json').is_file():
        raise ValueError('additional slots require a sealed parent E9 run')
    _verify_g(parent)
    old_protocol = json.loads((parent / 'protocol.lock.yml').read_text(encoding='utf-8'))
    if (old_protocol.get('model_key') != model_key or
            old_protocol.get('model_identity') != expected['model_identity'] or
            old_protocol.get('sandbox_image_id') != expected['sandbox_image_id'] or
            old_protocol.get('initial_pool_sha256') != expected['initial_pool_sha256'] or
            old_protocol.get('e9_source_hashes') != expected['e9_source_hashes']):
        raise ValueError('parent E9 run model, pool, sandbox or implementation differs')
    records = [json.loads(line) for line in (parent / 'strict_oracles.jsonl').read_text().splitlines()
               if line and not json.loads(line).get('source_run')]
    pool = []
    for row in records:
        if row.get('source_model_id') != expected['model_identity'][0] or row.get('model_revision') != expected['model_identity'][1]:
            raise ValueError('parent E9 run contains cross-model training data')
        record = row.get('record')
        if not isinstance(record, dict) or record.get('task_id') != row.get('task_id'):
            raise ValueError('parent E9 Oracle record is malformed')
        pool.append({'record': record, 'behavior_sha256': row['behavior_fingerprint'],
                     'origin': 'E9_parent', 'historical_duplicate_observed': False})
    return records, pool


def _dependency_hashes() -> dict[str, str]:
    return {name: _sha(PROJECT / name) for name in IMPLEMENTATION_FILES}


def _protocol(settings: E9Settings, model_key: str, model_manifest: dict,
              image: str, pool_identity: dict, existing: list[dict],
              parent_records: list[dict], environment: dict, git_head: str) -> dict:
    from src.arc.e5_selfplay import _generator_prompt
    from src.arc.e6f_repair import repair_prompt
    from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
    slot_rows = [
        {'slot': i, 'attempt_seeds': [settings.seed + i * 3 + j for j in range(settings.max_repairs + 1)]}
        for i in range(settings.slot_offset, settings.slot_offset + settings.slots)
    ]
    e8_protocol = E8_SEED_CAMPAIGN / 'protocol.lock.yml'
    return {
        'protocol': 'e9_dual_model_autonomous_rsi_v1', 'settings': asdict(settings),
        'model_key': model_key, 'model_identity': list(MODELS[model_key]),
        'model_manifest': model_manifest, 'model_order': list(settings.models),
        'slot_manifest': slot_rows, 'sandbox_image_id': image,
        'initial_pool': pool_identity, 'initial_pool_sha256': pool_identity['pool_sha256'],
        'e9_source_hashes': _dependency_hashes(), 'git_head': git_head,
        'dependency_lock_sha256': _sha(PROJECT / 'uv.lock'), 'environment': environment,
        'prompt_hashes': {
            'system': hashlib.sha256(FINAL_ANSWER_SYSTEM_PROMPT.encode()).hexdigest(),
            'generator': hashlib.sha256(_generator_prompt().encode()).hexdigest(),
            'repair_template': hashlib.sha256(repair_prompt('x = 1', 'syntax_error', 'SyntaxError').encode()).hexdigest(),
        },
        'existing_oracle_policy': {
            'e8_campaign_protocol_sha256': _sha(e8_protocol) if e8_protocol.is_file() else None,
            'e8_strict_eligible_count': len(existing),
            'e7_strict_eligible_count': 0,
            'e7_rejection': 'historical static-checker dependency hash missing',
            'e8_artifacts_are_not_e9_new': True,
        },
        'parent_e9_strict_count': len(parent_records),
        'parent_e9_oracles_sha256': hashlib.sha256(json.dumps(
            parent_records, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
        'budgets': {
            'max_slots': settings.slots, 'slot_calls': 3, 'total_calls': settings.calls,
            'reserved_tokens': settings.reserved_tokens, 'sandbox_starts': settings.sandbox,
            'wall_seconds': settings.wall_seconds, 'candidate_seconds': settings.candidate_seconds,
            'calls_per_slot': 3, 'candidate_tokens': 15000,
            'max_repairs': settings.max_repairs,
        },
        'rules': {
            'initial_models_independent': True, 'pool_updates_from_same_model_only': True,
            'oracle_source': 'generator_distillation', 'strict_gate': MIN_STRICT_ORACLES,
            'static_checker_relaxed': False, 'hidden_labels_in_feedback': False,
            'operator_search': False, 'training': False, 'adapter': 'none',
            'monitoring_run': False, 'public_evaluation_run': False,
            'extra_slots_require_approval': True,
            'novelty': 'source_and_ast_duplicates_rejected; shape_or_behavior_suspects_rejected',
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rows),
                    encoding='utf-8')
    path.chmod(0o600)


def _oracle_manifest(item: dict, row: dict, model_key: str, image: str,
                     verifier_hash: str, seal_hash: str) -> dict:
    record = item['record']
    code = record['messages'][-1]['content']
    clean_record = {key: value for key, value in record.items() if key != '_verifier_only'}
    provenance = record.get('qwen_provenance', {})
    model_id, revision, _size = MODELS[model_key]
    if (provenance.get('execution_mode') not in {'live_model', 'fake_fixture'} or
            provenance.get('model_id') not in (None, model_id) or
            record.get('base_revision') not in (None, revision) or
            record.get('oracle_ast_sha256') != stable_ast_sha256(code)):
        raise ValueError('E9 Oracle provenance or AST hash mismatch')
    prompt_events = row.get('prompts', [])
    success = [event for event in row.get('attempts', []) if event.get('category') == 'passed']
    successful_attempt = success[-1].get('attempt') if success else None
    selected = [event for event in prompt_events if event.get('attempt') == successful_attempt]
    if not selected and len(prompt_events) == 1:
        selected = prompt_events
    if not selected:
        raise ValueError('E9 Oracle is missing its generator prompt hash')
    return {
        'record': clean_record, 'source_model_id': model_id, 'model_revision': revision,
        'generation': 0, 'oracle_source': 'generator_distillation', 'task_id': record['task_id'],
        'canonical_ast_hash': stable_ast_sha256(code),
        'behavior_fingerprint': item['behavior_sha256'],
        'generator_prompt_hash': selected[-1]['prompt_sha256'],
        'verifier_revision': verifier_hash, 'static_checker_hash': verifier_hash,
        'docker_image_digest': image, 'seal': seal_hash,
    }


def _write_snapshots(root: Path, settings: E9Settings, base: Any, environment: dict,
                     *, resume: bool) -> dict:
    status = subprocess.run(['git', 'status', '--short'], cwd=PROJECT, check=True,
                            capture_output=True, text=True).stdout
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, check=True,
                          capture_output=True, text=True).stdout.strip()
    config_text = yaml.safe_dump({'e9': asdict(settings), 'base_experiment': asdict(base)}, sort_keys=False)
    if resume:
        for name in ('git-status.txt', 'config.snapshot.yml', 'environment.json', 'dependency-lock.json'):
            _verify_mutable_file(root / name)
        if ((root / 'git-status.txt').read_text(encoding='utf-8') != status or
                (root / 'config.snapshot.yml').read_text(encoding='utf-8') != config_text or
                json.loads((root / 'environment.json').read_text(encoding='utf-8')) != environment):
            raise ValueError('E9 source status, config or environment changed on resume')
    else:
        (root / 'git-status.txt').write_text(status, encoding='utf-8')
        (root / 'git-status.txt').chmod(0o600)
        _seal_mutable_file(root / 'git-status.txt')
        (root / 'config.snapshot.yml').write_text(config_text, encoding='utf-8')
        (root / 'config.snapshot.yml').chmod(0o600)
        _seal_mutable_file(root / 'config.snapshot.yml')
        _json(root / 'environment.json', environment, private=True)
        _seal_mutable_file(root / 'environment.json')
        _json(root / 'dependency-lock.json', {'path': 'uv.lock', 'sha256': _sha(PROJECT / 'uv.lock')}, private=True)
        _seal_mutable_file(root / 'dependency-lock.json')
    if json.loads((root / 'dependency-lock.json').read_text()) != {
            'path': 'uv.lock', 'sha256': _sha(PROJECT / 'uv.lock')}:
        raise ValueError('E9 dependency lock changed on resume')
    return {'head': head, 'status_sha256': _sha(root / 'git-status.txt')}


def _slot_outcome(row: dict) -> str:
    if row.get('completion') != 'completed':
        if 'budget' in str(row.get('status', '')).lower():
            return 'budget_stopped'
        if 'resource' in str(row.get('status', '')).lower():
            return 'resource_stopped'
        if 'interrupted' in str(row.get('status', '')).lower():
            return 'error'
        return 'error'
    if row.get('verified_novel'):
        return 'completed'
    status = str(row.get('status', ''))
    if 'duplicate' in status or row.get('novelty', '').startswith(('duplicate', 'normalized_shape', 'sampled_behavior')):
        return 'duplicate'
    if 'private_acceptance' in status or status == 'hidden_failed':
        return 'hidden_failed'
    if status in {'passed', 'novel'}:
        return 'completed'
    return 'invalid'


def _audit_sft(rows: list[dict], model_key: str, image: str,
               model_manifest: dict) -> tuple[list[dict], dict]:
    model_id, revision, _ = MODELS[model_key]
    template_hash = model_manifest.get('chat_template_sha256')
    checker_hash = _sha(PROJECT / 'src/arc/e6f_repair.py')
    seen_tasks: set[str] = set()
    seen_asts: set[str] = set()
    seen_behaviors: set[str] = set()
    result, reasons = [], {}
    for row in rows:
        reason = None
        record = row.get('record')
        if (row.get('source_model_id') != model_id or row.get('model_revision') != revision or
                row.get('generation') != 0):
            reason = 'model_or_generation_mismatch'
        elif row.get('oracle_source') not in {'generator_distillation', 'self_solved_synthetic', 'self_solved_real_arc'}:
            reason = 'unknown_oracle_source'
        elif row.get('docker_image_digest') != image or row.get('seal') is None:
            reason = 'provenance_seal_or_sandbox_mismatch'
        elif row.get('verifier_revision') != checker_hash or row.get('static_checker_hash') != checker_hash:
            reason = 'unknown_verifier_or_static_checker_hash'
        elif not row.get('canonical_ast_hash') or not row.get('behavior_fingerprint') or not row.get('generator_prompt_hash'):
            reason = 'required_provenance_field_missing'
        elif not isinstance(record, dict) or '_verifier_only' in record or 'hidden_labels' in record:
            reason = 'hidden_verifier_data_in_training_record'
        elif record.get('task_id') != row.get('task_id') or not row.get('task_id'):
            reason = 'task_binding_mismatch'
        elif (record.get('adapter_revision', 'none') != 'none' or
                record.get('base_revision') not in (None, revision)):
            reason = 'adapter_or_base_revision_mismatch'
        elif (record.get('qwen_provenance', {}).get('model_id') not in (None, model_id) or
                record.get('qwen_provenance', {}).get('model_revision') not in (None, revision)):
            reason = 'cross_model_provenance'
        elif row.get('canonical_ast_hash') in seen_asts:
            reason = 'duplicate_ast'
        elif row.get('behavior_fingerprint') in seen_behaviors:
            reason = 'duplicate_behavior'
        elif row.get('task_id') in seen_tasks:
            reason = 'duplicate_task_id'
        messages = record.get('messages', []) if isinstance(record, dict) else []
        if reason is None and (not isinstance(messages, list) or len(messages) < 3 or
                [m.get('role') for m in messages] != ['system', 'user', 'assistant'] or
                any(not isinstance(m.get('content'), str) or not m.get('content') for m in messages)):
            reason = 'invalid_chat_record'
        if reason is None and (not template_hash or row.get('oracle_source') == 'self_solved_real_arc'):
            # Real ARC self-solves are out of scope for this synthetic-only E9 run.
            reason = 'chat_template_missing_or_real_arc_source_not_preregistered'
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        seen_tasks.add(row['task_id'])
        seen_asts.add(row['canonical_ast_hash'])
        seen_behaviors.add(row['behavior_fingerprint'])
        result.append({
            'id': f"{model_key}-g0-{row['task_id']}", 'task_id': row['task_id'],
            'source_model_id': model_id, 'model_revision': revision, 'generation': 0,
            'oracle_source': row['oracle_source'], 'canonical_ast_hash': row['canonical_ast_hash'],
            'behavior_fingerprint': row['behavior_fingerprint'],
            'generator_prompt_hash': row['generator_prompt_hash'],
            'verifier_revision': row['verifier_revision'], 'static_checker_hash': row['static_checker_hash'],
            'docker_image_digest': image, 'seal': row['seal'], 'messages': messages,
            'thinking_mode': 'off', 'chat_template_sha256': template_hash,
            'mask_prompt': True, 'loss_target': 'assistant_completion_only',
        })
    return result, {'status': 'passed' if not reasons else 'rejected_rows_present',
                    'eligible_records': len(result), 'rejected_records': sum(reasons.values()),
                    'rejection_reasons': reasons, 'thinking_mode': 'off',
                    'chat_template_sha256': template_hash,
                    'loss_target': 'assistant_completion_only', 'mask_prompt': True,
                    'verified_model_identity': [model_id, revision]}


def run_model(settings: E9Settings, model_key: str, run_dir: Path, *, live: bool = False,
              collect_only: bool = True, resume: bool = False, approve_extra_slots: bool = False,
              parent_run: Path | None = None, allow_training: bool = False) -> Path:
    """Execute one model run; additional batches require an explicit CLI approval."""
    if model_key not in settings.models:
        raise ValueError('model is not registered in this config')
    fake = settings.profile == 'fake'
    if fake and (live or approve_extra_slots or parent_run is not None or allow_training):
        raise ValueError('fake E9 cannot use live approval flags')
    if not fake and (not live or not collect_only):
        raise ValueError('live E9 collection requires --live --collect-only')
    additional = settings.slot_offset > 0 or settings.slots > 10
    if additional and (fake or not approve_extra_slots):
        raise ValueError('additional slots require explicit approval')
    if allow_training:
        raise ValueError('E9 training requires the separate training preflight CLI')

    root = run_dir.expanduser().resolve()
    expected_root = (PROJECT / 'artifacts/e9').resolve()
    model_root = (expected_root / model_key).resolve()
    if not root.is_relative_to(expected_root) or (not fake and not root.is_relative_to(model_root)):
        raise ValueError('E9 run directory must be under artifacts/e9/<model>')
    if resume and not root.is_dir():
        raise ValueError('resume requires an existing E9 run')
    if root.exists() and not resume and any(root.iterdir()):
        raise FileExistsError('E9 run directory already contains files')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    with _run_lock(root):
        start = time.monotonic()
        base_seed = load_experiment_config(PROJECT / settings.base_config)
        fake_image = 'fake'
        image = fake_image if fake else _docker_image_id(base_seed)
        e8_settings = _e8_settings(settings)
        pool, probes, forbidden, splits, source_identity = _source_pool(e8_settings, base_seed, image)
        base_pool_sha = hashlib.sha256(json.dumps(
            pool, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        e8_14, pool_14, audit_14 = _e8_existing('14b', fake, image)
        e8_32, pool_32, audit_32 = _e8_existing('32b', fake, image)
        existing, existing_audit = (e8_14, audit_14) if model_key == '14b' else (e8_32, audit_32)
        fixed_pool = list(pool) + pool_14 + pool_32
        pool_identity = {
            'e6g_identity': source_identity,
            'base_e6g_pool_sha256': base_pool_sha,
            'pool_sha256': hashlib.sha256(json.dumps(
                fixed_pool, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
            'e8_duplicate_reference_count': len(pool_14) + len(pool_32),
        }
        parent_records, parent_items = _load_parent_pool(parent_run, model_key, {
            'model_identity': list(MODELS[model_key]), 'sandbox_image_id': image,
            'initial_pool_sha256': pool_identity['pool_sha256'],
            'e9_source_hashes': _dependency_hashes(),
        }) if parent_run is not None else ([], [])
        pool = list(fixed_pool) + parent_items
        pool.extend(parent_items)
        model_id, revision, size = MODELS[model_key]
        before = snapshot(root) if not fake else {'disk_free_bytes': None, 'swap_used_mib': 0,
                                                  'memory_free_percent': 100, 'fixture': True}
        resource_reason = None if fake else check_resources(
            before, before, min_free_percent=settings.min_memory_free_percent,
            max_swap_increase_mib=settings.max_swap_increase_mib,
            min_disk_bytes=settings.min_free_disk_gib * 1024**3 +
                (0 if _model_dir(model_id, revision).is_dir() else size))
        environment = _environment(base_seed, fake)
        model_manifest = ({'model_id': model_id, 'revision': revision, 'fixture': True} if fake else
                          {'model_id': model_id, 'revision': revision,
                           'resource_preflight_stopped': resource_reason} if resource_reason else None)
        if not fake and not resource_reason:
            _check_no_other_model_server()
            model_path = _model_dir(model_id, revision)
            if not model_path.is_dir():
                if resume:
                    raise RuntimeError('interrupted_model_download_not_retried')
                from src.pipelines.e8 import _download
                _download(model_id, revision, min(3600, settings.wall_seconds))
            model_manifest = _model_manifest(model_path, revision)
        assert model_manifest is not None
        base = replace(base_seed, model=replace(base_seed.model, id=model_id, revision=revision),
                       p3b=replace(base_seed.p3b, training=replace(
                           base_seed.p3b.training, model_path=str(_model_dir(model_id, revision))))) if not fake else base_seed
        git = _write_snapshots(root, settings, base, environment, resume=resume)
        protocol = _protocol(settings, model_key, model_manifest, image, pool_identity,
                             existing, parent_records, environment, git['head'])
        protocol['git_status_sha256'] = git['status_sha256']
        protocol['split_manifest_sha256'] = hashlib.sha256(
            json.dumps(splits, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        protocol['starting_novelty_pool_sha256'] = hashlib.sha256(
            json.dumps(pool, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        protocol_path = root / 'protocol.lock.yml'
        if resume:
            _verify_mutable_file(protocol_path)
            if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
                raise ValueError('E9 protocol, model, data, config or implementation changed on resume')
            if (root / 'seal.json').is_file():
                _verify_g(root)
                return root
            _verify_mutable_file(root / 'initial_pool.json')
            _verify_mutable_file(root / 'novelty_pool.json')
            if (json.loads((root / 'initial_pool.json').read_text()) != fixed_pool or
                    json.loads((root / 'novelty_pool.json').read_text()) != pool):
                raise ValueError('E9 same-model novelty pool changed on resume')
        else:
            _json(protocol_path, protocol, private=True)
            _seal_mutable_file(protocol_path)
            (root / 'protocol.sha256').write_text(_sha(protocol_path) + '\n', encoding='utf-8')
            (root / 'protocol.sha256').chmod(0o600)
            _json(root / 'initial_pool.json', fixed_pool, private=True)
            _seal_mutable_file(root / 'initial_pool.json')
            _json(root / 'novelty_pool.json', pool, private=True)
            _seal_mutable_file(root / 'novelty_pool.json')
            _json(root / 'split_manifest.json', splits, private=True)
            _seal_mutable_file(root / 'split_manifest.json')
            _json(root / 'existing_audit.json', existing_audit, private=True)
            _seal_mutable_file(root / 'existing_audit.json')
            _write_jsonl(root / 'existing_oracles.jsonl', existing)

        remaining_seconds = int(settings.wall_seconds - (time.monotonic() - start))
        if remaining_seconds <= 0:
            resource_reason = resource_reason or 'preparation_exhausted_wall_budget'
            remaining_seconds = 1
        # E8 model snapshots are also copied into a private model-lock record;
        # model weights themselves remain in the Hugging Face cache.
        model_lock = {'model': model_manifest, 'protocol_sha256': _sha(protocol_path),
                      'sandbox_image_id': image}
        if resume:
            _verify_mutable_file(root / 'model.lock.json')
            old_model_lock = json.loads((root / 'model.lock.json').read_text())
            if any(old_model_lock.get(k) != v for k, v in model_lock.items()):
                raise ValueError('E9 model identity changed on resume')
        else:
            model_lock['resource_preflight'] = before
            model_lock['resource_preflight_stop'] = resource_reason
            _json(root / 'model.lock.json', model_lock, private=True)
            _seal_mutable_file(root / 'model.lock.json')

        total_config = _config_for(base, calls=settings.calls, tokens=settings.reserved_tokens,
                                   sandboxes=settings.sandbox, seconds=remaining_seconds, seed=settings.seed)
        total_path, model_budget_path = root / 'total-budget.json', root / 'model-budget.json'
        if resume and (not total_path.is_file() or not model_budget_path.is_file()):
            raise ValueError('E9 resume lacks one or more budget ledgers')
        total = E5Budget(total_path, total_config, resume=resume)
        model_budget = E5Budget(model_budget_path, total_config, resume=resume)
        paired = PairedBudget(total, model_budget)
        monitor: ResourceMonitor | None = None
        rows: list[dict] = []
        accepted_new: list[dict] = []
        status, stop_reason = ('resource_infeasible', resource_reason) if resource_reason else ('completed', None)
        canary: dict = {}
        try:
            monitor_context = (ResourceMonitor(root, before,
                min_free_percent=settings.min_memory_free_percent,
                max_swap_increase_mib=settings.max_swap_increase_mib,
                min_disk_bytes=settings.min_free_disk_gib * 1024**3) if not fake and not resource_reason
                else None)
            if monitor_context:
                monitor = monitor_context
                monitor.__enter__()
            if resource_reason:
                raise RuntimeError(resource_reason)
            canary_path = root / 'canary.json'
            if canary_path.exists():
                _verify_mutable_file(canary_path)
                canary = json.loads(canary_path.read_text())
            elif resume and (root / 'canary-started.json').exists():
                raise RuntimeError('interrupted_canary_not_retried')
            else:
                _json(root / 'canary-started.json', {'protocol_sha256': _sha(protocol_path)}, private=True)
                _seal_mutable_file(root / 'canary-started.json')
                canary = _canaries(root, total_config, paired, fake)
                _json(canary_path, canary, private=True)
                _seal_mutable_file(canary_path)
            feedback = NoveltyFeedback(enabled=False, pool=pool, allow_fixture=fake)
            for index in range(settings.slot_offset, settings.slot_offset + settings.slots):
                if monitor and monitor.stop_reason:
                    status, stop_reason = 'resource_stopped', monitor.stop_reason
                    break
                slot_root = root / 'slots' / str(index)
                result_path, marker = slot_root / 'result.json', slot_root / 'started.json'
                if result_path.is_file():
                    _verify_g(slot_root)
                    row = json.loads(result_path.read_text(encoding='utf-8'))
                elif marker.is_file():
                    _verify_mutable_file(marker)
                    row = {'arm': 'G0_baseline', 'slot': index, 'completion': 'incomplete',
                           'status': 'interrupted_uncertain', 'verified_novel': False}
                    _json(result_path, row, private=True)
                    _seal_mutable_file(result_path)
                    _seal_g(slot_root)
                else:
                    slot_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                    _json(marker, {'slot': index, 'protocol_sha256': _sha(protocol_path)}, private=True)
                    _seal_mutable_file(marker)
                    row = _slot(root=slot_root, arm='G0_baseline', index=index,
                                settings=_e7_settings(settings), base=base, total=total,
                                arm_budget=model_budget, pool=pool, probes=probes,
                                forbidden=forbidden, feedback=feedback, image=image, fake=fake)
                    _json(result_path, row, private=True)
                    _seal_mutable_file(result_path)
                    _seal_g(slot_root)
                rows.append(row)
                if row.get('accepted_item'):
                    item = row['accepted_item']
                    oracle = _oracle_manifest(item, row, model_key, image,
                                              _sha(PROJECT / 'src/arc/e6f_repair.py'),
                                              _sha(slot_root / 'seal.json'))
                    accepted_new.append(oracle)
                    pool.append({'record': item['record'], 'behavior_sha256': item['behavior_sha256'],
                                 'origin': 'E9_same_model', 'historical_duplicate_observed': False})
                if row.get('status') in {'model_request_failed', 'sandbox_infrastructure_error',
                                         'experiment_inconclusive_budget'}:
                    status, stop_reason = 'budget_stopped', row['status']
                    break
        except Exception as exc:
            if resource_reason or (monitor and monitor.stop_reason):
                status = 'resource_stopped'
                stop_reason = monitor.stop_reason if monitor and monitor.stop_reason else resource_reason
            else:
                status = 'error'
                stop_reason = type(exc).__name__ + ': ' + str(exc)
        finally:
            if monitor:
                monitor.__exit__(None, None, None)
            model_budget.finish()
            total.finish()

        after = snapshot(root) if not fake else {'fixture': True}
        if monitor and monitor.stop_reason:
            status, stop_reason = 'resource_stopped', monitor.stop_reason
        if len(rows) != settings.slots and status == 'completed':
            status, stop_reason = 'budget_stopped', 'not_all_preregistered_slots_completed'
        strict_new = sum(bool(row.get('verified_novel')) for row in rows) if not fake else 0
        if len(accepted_new) != strict_new and not fake:
            status, stop_reason = 'error', 'strict_rows_missing_training_safe_oracle_record'
            strict_new = len(accepted_new)
        strict_rows = existing + parent_records + accepted_new
        sft_rows, sft_audit = _audit_sft(strict_rows, model_key, image, model_manifest)
        existing_ids = {row['task_id'] for row in existing}
        parent_ids = {row['task_id'] for row in parent_records}
        new_ids = {row['task_id'] for row in accepted_new}
        eligible_ids = {row['task_id'] for row in sft_rows}
        strict_existing = len(existing_ids & eligible_ids)
        strict_parent = len(parent_ids & eligible_ids)
        strict_new = len(new_ids & eligible_ids) if not fake else 0
        strict_total = len(sft_rows)
        rejection_reasons = dict(existing_audit.get('reasons', {}))
        for key, value in sft_audit['rejection_reasons'].items():
            rejection_reasons[key] = rejection_reasons.get(key, 0) + value
        training = {
            'strict_eligible_existing': strict_existing, 'strict_parent_e9': strict_parent,
            'strict_new_e9': strict_new, 'strict_total': strict_total,
            'rejected': existing_audit.get('rejected', 0) + sft_audit['rejected_records'],
            'rejection_reasons': rejection_reasons,
            'trained': False, 'weights_updated': False, 'reloaded': False,
            'promotion': 'not_applicable',
            'reason': 'insufficient_strict_oracles' if strict_total < MIN_STRICT_ORACLES
                      else 'explicit_training_authorization_required',
            'minimum_strict_oracles': MIN_STRICT_ORACLES,
            'adapter_updated': False, 'preflight_run': False,
        }
        _write_jsonl(root / 'strict_oracles.jsonl', strict_rows)
        _write_jsonl(root / 'sft.jsonl', sft_rows)
        _json(root / 'sft-audit.json', sft_audit, private=True)
        _json(root / 'comparison.json', {
            'status': 'training_not_run' if strict_total < MIN_STRICT_ORACLES else 'authorization_required',
            'baseline_adapter': 'none', 'candidate_adapter': None,
            'B_C_comparison_run': False, 'promotion': 'not_applicable',
            'monitoring_run': False, 'public_evaluation_run': False,
        }, private=True)
        _json(root / 'training-preflight.json', {
            'status': 'not_run_insufficient_strict_oracles' if strict_total < MIN_STRICT_ORACLES
                      else 'not_run_requires_separate_training_authorization',
            'attempted': False, 'canary_training': False,
            'model_id': model_id, 'revision': revision,
            'mlx_lm_version': environment.get('mlx_runtime', {}).get('mlx-lm'),
            'lora_rank': base.p3b.training.lora_rank,
            'max_seq_length': base.p3b.training.max_seq_length,
            'max_training_steps': 5, 'network_disabled_when_started': True,
            'reason': training['reason'],
        }, private=True)
        public_rows = [{k: v for k, v in row.items() if k not in ('accepted_item', 'feedback_code')}
                       for row in rows]
        _json(root / 'slots.json', public_rows, private=True)
        _json(root / 'training-gate.json', training, private=True)
        calls = total.state['model_calls']
        usage_rows = [_observed_token_usage(path) for path in root.rglob('llm_requests.jsonl')]
        observed_calls = sum(item['calls_with_usage'] for item in usage_rows)
        observed_sum = sum(item['observed_total_tokens'] or 0 for item in usage_rows)
        observed = observed_sum if observed_calls == calls else None
        result = {
            'protocol': 'e9_dual_model_autonomous_rsi_v1', 'model_key': model_key,
            'model_id': model_id, 'revision': revision, 'model_manifest': model_manifest,
            'status': status, 'stop_reason': stop_reason,
            'planned_slots': settings.slots,
            'slot_range': [settings.slot_offset, settings.slot_offset + settings.slots - 1],
            'completed_slots': sum(row.get('completion') == 'completed' for row in rows),
            'outcomes': {name: sum(_slot_outcome(row) == name for row in rows)
                         for name in ('completed', 'invalid', 'duplicate', 'hidden_failed',
                                      'budget_stopped', 'resource_stopped', 'error')},
            'valid_oracles': sum(bool(row.get('oracle_valid')) for row in rows),
            'verified_new_oracles': strict_new, 'training_gate': training,
            'model_calls': calls, 'reserved_tokens': total.state['reserved_tokens'],
            'observed_tokens': observed, 'sandbox_starts': total.state['sandbox_runs'],
            'elapsed_seconds': time.monotonic() - start,
            'budget_elapsed_seconds': total.state['elapsed_seconds'],
            'model_inference_seconds': sum(float(row.get('model_inference_seconds', 0)) for row in rows),
            'canary': canary, 'before': before, 'after': after,
            'resource_samples': monitor.samples if monitor else [],
            'resource_stop_reason': monitor.stop_reason if monitor else None,
            'strict_existing_oracles': strict_existing, 'strict_parent_e9_oracles': strict_parent,
            'training_invoked': False, 'weights_updated': False, 'reloaded': False,
            'promoted': False, 'monitoring_run': False, 'public_evaluation_run': False,
            'operator_search_used': False, 'campaign_protocol_sha256': _sha(protocol_path),
        }
        _json(root / 'result.json', result, private=True)
        for path in root.rglob('*'):
            if path.is_file():
                path.chmod(0o600)
        _seal_g(root)
        return root


def run(settings: E9Settings, run_root: Path, *, live: bool = False,
        collect_only: bool = True, resume: bool = False,
        approve_extra_slots: bool = False, parent_run: Path | None = None,
        allow_training: bool = False) -> list[Path]:
    roots = []
    for model_key in settings.models:
        model_root = run_root / model_key if len(settings.models) > 1 else run_root
        roots.append(run_model(settings, model_key, model_root, live=live,
            collect_only=collect_only, resume=resume, approve_extra_slots=approve_extra_slots,
            parent_run=parent_run, allow_training=allow_training))
    return roots
