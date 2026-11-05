"""Paired S0-S3 search comparison and verifier-backed Exploration collection."""
from __future__ import annotations

import csv
import ast
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.arc.agent import FakeLLM
from src.arc.data import Task, Verifier, load_arc, split_tasks
from src.arc.exploration_registry import PROTOCOL as EXCLUSION_PROTOCOL
from src.arc.exploration_registry import build_registry, validate_registry
from src.arc.http_backend import (BackendError, OpenAICompatibleBackend,
                                  verify_guarded_contract, verify_ollama_revision)
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.sandbox import DockerSandbox, FakeSandbox
from src.arc.search import (BudgetedBackend, BudgetedSandbox, SearchBudget,
                            SearchBudgetError, StrategyRun, run_strategy,
                            static_code_check)
from src.core.config import ExperimentConfig
from src.pipelines.p3a import run_lock


ROOT = Path(__file__).resolve().parents[2]
SELECTION_PROTOCOL = 'p3c_exploration_strategy_selection_v1'
FROZEN_PROTOCOL = 'p3c_frozen_search_config_v1'
COLLECTION_PROTOCOL = 'p3c_verified_solution_collection_v1'
RECOVERABLE_RESPONSE_ERRORS = {'malformed_thinking', 'invalid_response',
                               'missing_response', 'invalid_json',
                               'code_extraction_error'}
SFT_PROTOCOL = 'arc_exploration_sft_v1'
IMPLEMENTATION_FILES = (
    'src/core/config.py', 'src/arc/agent.py', 'src/arc/data.py', 'src/arc/search.py',
    'src/arc/exploration_registry.py', 'src/arc/sandbox.py', 'src/arc/scoring.py',
    'src/arc/prompting.py', 'src/arc/response_diagnostics.py', 'src/arc/http_backend.py',
    'src/arc/guarded_http.py', 'src/arc/guarded_generation.py',
    'src/arc/guarded_contract.py', 'src/arc/mlx_token_driver.py',
    'src/arc/deadline_watchdog.py', 'src/arc/token_ledger.py',
    'src/pipelines/exploration.py', 'src/cli/exploration_compare.py',
    'src/cli/exploration_collect.py', 'src/cli/guarded_mlx_server.py',
    'sandbox/Dockerfile', 'sandbox/runner.py', 'uv.lock',
)


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if os.name != 'nt':
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_yaml(value: Any) -> bytes:
    return yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode('utf-8')


def _data(config: ExperimentConfig) -> tuple[dict[str, Task], Verifier, dict[str, list[str]]]:
    tasks, verifier = load_arc(config.data.challenges, config.data.solutions)
    splits = split_tasks(list(tasks), config.data.split_seed,
                         {name: getattr(config.data, name)
                          for name in ('exploration', 'promotion_gate', 'monitoring')})
    return tasks, verifier, splits


def _load_registry(config: ExperimentConfig, *, extra_manifests: list[str | Path] | None = None,
                   additional_ids: list[str] | None = None) -> dict[str, Any]:
    manifest = Path(config.search.exclusion_manifest).expanduser()
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    registry = build_registry(ROOT, additional_manifests=[manifest, *(extra_manifests or [])])
    for task_id in additional_ids or []:
        registry['exclusions'].setdefault(task_id, []).append('current_e2_comparison')
    registry['exclusions'] = {key: sorted(set(value)) for key, value in sorted(registry['exclusions'].items())}
    registry['task_ids'] = sorted(registry['exclusions'])
    registry['sources'] = sorted({source for values in registry['exclusions'].values() for source in values})
    return registry


def resolve_comparison_tasks(config: ExperimentConfig, *,
                             extra_manifests: list[str | Path] | None = None) -> tuple[list[str], dict[str, Any], dict[str, list[str]]]:
    if not config.search.enabled:
        raise ValueError('E2/E3 requires an explicitly enabled search config group')
    tasks, _, splits = _data(config)
    if config.experiment.evaluation_split != 'exploration':
        raise ValueError('E2 comparison is restricted to the Exploration split')
    registry = _load_registry(config, extra_manifests=extra_manifests)
    excluded = set(registry['task_ids'])
    if config.search.comparison_task_ids:
        selected = list(config.search.comparison_task_ids)
        if len(selected) != config.search.comparison_task_count:
            raise ValueError('explicit comparison task IDs must equal comparison_task_count')
        if not set(selected) <= set(splits['exploration']):
            raise ValueError('comparison task IDs must all belong to Exploration')
        if set(selected) & excluded:
            raise ValueError('comparison task IDs include a previously used or diagnostic task')
    else:
        selected = [task_id for task_id in splits['exploration'] if task_id not in excluded]
        selected = selected[:config.search.comparison_task_count]
    if len(selected) != config.search.comparison_task_count:
        raise ValueError(f'insufficient unused Exploration tasks: need {config.search.comparison_task_count}, found {len(selected)}')
    if any(task_id not in tasks for task_id in selected):
        raise ValueError('comparison task ID is absent from the configured dataset')
    return selected, registry, splits


METHOD_TIE_BREAK_ORDER = ['S0', 'S1', 'S2', 'S3']


def _selection_policy(config: ExperimentConfig) -> dict[str, Any]:
    return {'protocol': 'p3c_e2_preregistered_selection_policy_v1',
            'version': 1,
            'run_profile': config.tracking.profile,
            'required_paired_tasks': config.search.comparison_task_count,
            'paired_task_rule': 'include only tasks with a recorded result for all four strategies',
            'ranking': [
                {'metric': 'complete_train_match_tasks', 'direction': 'descending',
                 'definition': 'count task outcomes whose selected code matched every public train output'},
                {'metric': 'macro_average_final_train_match_rate', 'direction': 'descending',
                 'definition': 'mean of each paired task final-code exact-match fraction; tasks have equal weight'},
                {'metric': 'parser_and_sandbox_error_events', 'direction': 'ascending',
                 'definition': 'candidate parse failures plus train sandbox execution failures'},
                {'metric': 'reserved_tokens', 'direction': 'ascending',
                 'definition': 'client reservation totals under equal task-strategy ceilings'},
                {'metric': 'fixed_strategy_order', 'direction': 'ascending',
                 'order': METHOD_TIE_BREAK_ORDER}],
            'held_out_verifier': {'allowed_after_candidate_selection': True,
                                  'reported_as_diagnostic_only': True,
                                  'used_for_strategy_selection': False,
                                  'used_for_tie_break': False},
            'inconclusive_status': 'inconclusive_fallback when all strategies have zero full-train-match tasks',
            'smoke_status': 'not_established; E2 Smoke cannot select or freeze a strategy',
            'selection_policy_hash_algorithm': 'sha256'}


def _strategy_order(config: ExperimentConfig, task_id: str) -> list[str]:
    digest = hashlib.sha256(f'{config.experiment.seed}:{task_id}:strategy-order-v1'.encode()).digest()
    offset = digest[0] % len(METHOD_TIE_BREAK_ORDER)
    return METHOD_TIE_BREAK_ORDER[offset:] + METHOD_TIE_BREAK_ORDER[:offset]


def _split_manifest(config: ExperimentConfig, splits: dict[str, list[str]]) -> dict[str, Any]:
    split_bytes = _json_bytes(splits)
    return {'protocol': 'arc_training_split_manifest_v1', 'seed': config.data.split_seed,
            'counts': {name: len(ids) for name, ids in splits.items()},
            'split_sha256': hashlib.sha256(split_bytes).hexdigest(),
            'split_ids': splits,
            'challenge_sha256': _sha(config.data.challenges),
            'solution_sha256': _sha(config.data.solutions),
            'public_evaluation_accessed': False}


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, timeout=5).stdout.strip()
        changed = subprocess.run(['git', 'diff', 'HEAD', '--name-only', '-z'], cwd=ROOT,
                                 check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 timeout=10).stdout.decode().split('\0')
        untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '-z'],
                                   cwd=ROOT, check=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, timeout=10).stdout.decode().split('\0')
        paths = sorted(set(path for path in [*changed, *untracked] if path and Path(path).is_file()))
        file_hashes = {path: _sha(ROOT / path) for path in paths}
        diff_digest = hashlib.sha256(_json_bytes(file_hashes)).hexdigest()
        return {'commit': commit, 'working_tree_file_hashes': file_hashes,
                'working_tree_diff_sha256': diff_digest}
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return {'commit': None, 'working_tree_file_hashes': {},
                'working_tree_diff_sha256': None, 'status': 'unavailable'}


def _model_snapshot_metadata(config: ExperimentConfig) -> dict[str, Any]:
    model_dir = Path(config.model.local_model_path).expanduser() if config.model.local_model_path else None
    if model_dir is None:
        return {'path': None, 'snapshot_directory_matches_revision': None,
                'tokenizer_files': {}, 'tokenizer_template_sha256': None}
    model_dir = model_dir.resolve()
    tokenizer_files = {}
    for name in ('tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json',
                 'tokenizer.model', 'config.json'):
        path = model_dir / name
        if path.is_file():
            tokenizer_files[name] = _sha(path)
    template_sha = None
    tokenizer_config = model_dir / 'tokenizer_config.json'
    if tokenizer_config.is_file():
        try:
            value = json.loads(tokenizer_config.read_text(encoding='utf-8'))
            template = value.get('chat_template')
            if isinstance(template, str):
                template_sha = hashlib.sha256(template.encode()).hexdigest()
            elif isinstance(template, list):
                template_sha = hashlib.sha256(_json_bytes(template)).hexdigest()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    return {'path': str(model_dir), 'exists': model_dir.is_dir(),
            'snapshot_directory_matches_revision': model_dir.name == config.model.revision,
            'tokenizer_files': tokenizer_files, 'tokenizer_template_sha256': template_sha}


def _artifact_seal(root: Path, names: list[str], *, protocol: str) -> dict[str, Any]:
    hashes = {name: _sha(root / name) for name in names if (root / name).is_file()}
    body = {'protocol': protocol, 'algorithm': 'sha256', 'artifacts': hashes}
    body['seal_sha256'] = hashlib.sha256(_json_bytes(body)).hexdigest()
    return body


def _verify_artifact_seal(root: Path, *, name: str = 'artifact_seal.json') -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        raise ValueError(f'artifact seal is missing: {name}')
    seal = json.loads(path.read_text(encoding='utf-8'))
    body = dict(seal)
    expected = body.pop('seal_sha256', None)
    if expected != hashlib.sha256(_json_bytes(body)).hexdigest():
        raise ValueError(f'artifact seal is invalid: {name}')
    for artifact, digest in body.get('artifacts', {}).items():
        if not (root / artifact).is_file() or _sha(root / artifact) != digest:
            raise ValueError(f'artifact hash mismatch: {artifact}')
    return seal


def check_compare(config: ExperimentConfig, *, live: bool = False,
                  extra_manifests: list[str | Path] | None = None) -> dict[str, Any]:
    profile = config.tracking.profile
    if profile not in {'cpu_fake', 'e2_smoke', 'e2_full'}:
        raise ValueError('E2 requires the cpu_fake, e2_smoke, or e2_full profile')
    if live and profile not in {'e2_smoke', 'e2_full'}:
        raise ValueError('live E2 requires an explicit e2_smoke or e2_full profile')
    if live:
        _validate_live_budget(config)
    task_ids, registry, splits = resolve_comparison_tasks(config, extra_manifests=extra_manifests)
    image_id = None
    contract = None
    if live:
        if config.model.backend != 'guarded_mlx' or config.agent.sandbox != 'docker':
            raise ValueError('live E2 requires the guarded MLX backend and Docker sandbox')
        if config.model.adapter_sha256 != 'none':
            raise ValueError('E2 comparison must use the unadapted base model')
        model_dir = Path(config.model.local_model_path).expanduser()
        if not model_dir.is_dir() or model_dir.resolve().name != config.model.revision:
            raise ValueError('live E2 requires the configured local model snapshot named by its revision')
        image_id = _docker_image_id(config)
        contract = _guarded_preflight(config, required_modes=('off', 'on'))
        if contract.get('reserved_tokens') != 0:
            raise ValueError('a new live E2 run requires a fresh guarded token ledger')
    policy = _selection_policy(config)
    policy_bytes = _canonical_yaml(policy)
    splits_manifest = _split_manifest(config, splits)
    orders = {task_id: _strategy_order(config, task_id) for task_id in task_ids}
    theoretical_calls = len(task_ids) * len(config.search.strategies) * config.search.max_calls_per_task
    return {'status': 'ready', 'execution_mode': 'live' if live else 'cpu_fake',
            'run_profile': profile,
            'comparison_task_ids': task_ids, 'split': 'exploration',
            'strategy_order_by_task': orders,
            'strategy_tie_break_order': METHOD_TIE_BREAK_ORDER,
            'selection_policy': policy,
            'selection_policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
            'split_manifest_sha256': splits_manifest['split_sha256'],
            'exploration_size': len(splits['exploration']),
            'excluded_task_count': len(registry['task_ids']),
            'exclusion_registry_sha256': hashlib.sha256(_json_bytes(registry)).hexdigest(),
            'strategies': config.search.strategies,
            'expected_maximum_llm_calls': theoretical_calls,
            'run_limits': {'seconds': config.budget.max_seconds,
                           'calls': config.budget.max_llm_calls,
                           'reserved_tokens': config.budget.max_tokens,
                           'max_candidates': config.budget.max_candidates,
                           'sandbox_starts': config.budget.max_container_starts},
            'per_task_limits': {'calls': config.search.max_calls_per_task,
                                'reserved_tokens': config.search.max_reserved_tokens_per_task,
                                'sandbox_starts': config.search.max_sandbox_starts_per_task,
                                'seconds': config.search.max_seconds_per_task,
                                'requested_completion_tokens': config.model.max_tokens},
            'thinking_policy': config.search.thinking_policy,
            'thinking_canary': contract.get('thinking_canary') if contract else None,
            'guarded_contract': contract,
            'docker_image_id': image_id}


def _validate_live_budget(config: ExperimentConfig) -> None:
    limits = {
        'e2_smoke': (3600, 32, 250000, 120),
        'e2_full': (5400, 80, 500000, 240),
        'e3_smoke': (3600, 8, 60000, 24),
        'e3_collection': (10800, 80, 500000, 240),
    }
    if config.tracking.profile not in limits:
        raise ValueError('live execution requires a bounded E2/E3 profile')
    seconds, calls, tokens, containers = limits[config.tracking.profile]
    if (config.budget.max_seconds > seconds or config.budget.max_llm_calls > calls or
            config.budget.max_tokens > tokens or config.budget.max_container_starts > containers):
        raise ValueError(f'{config.tracking.profile} exceeds its registered run budget')
    if config.tracking.profile == 'e2_smoke' and config.search.comparison_task_count != 2:
        raise ValueError('E2 Smoke is fixed at two paired tasks')
    if config.tracking.profile == 'e2_full' and config.search.comparison_task_count != 5:
        raise ValueError('E2 Full is fixed at five paired tasks')
    if config.tracking.profile == 'e3_smoke' and (
            config.search.collection_max_tasks != 2 or config.search.target_unique_verified_tasks != 2):
        raise ValueError('E3 Smoke is fixed at two tasks and two verified successes')
    if config.tracking.profile == 'e3_collection' and (
            config.search.collection_max_tasks > 20 or config.search.target_unique_verified_tasks != 5):
        raise ValueError('E3 Collection is capped at twenty tasks and targets five verified successes')


class SearchFakeBackend(FakeLLM):
    """CPU-only deterministic model stand-in with distinct candidate source texts."""
    _codes = (
        'def transform(grid):\n    return [row[:] for row in grid]',
        'def transform(grid):\n    return [list(row) for row in grid]',
        'def transform(grid):\n    result = []\n    for row in grid:\n        result.append(row[:])\n    return result',
    )

    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        if kind == 'analysis':
            return 'The demonstrations preserve each row and cell; test this rule against every training pair.'
        if kind == 'direct':
            return super().generate(prompt, kind=kind, candidate=candidate)
        return self._codes[candidate % len(self._codes)]


def _new_budget(config: ExperimentConfig, *, scope_state: dict[str, Any] | None = None,
                persist: Any = None) -> SearchBudget:
    return SearchBudget(max_seconds=config.budget.max_seconds,
                        max_calls=config.budget.max_llm_calls,
                        max_tokens=config.budget.max_tokens,
                        max_candidates=config.budget.max_candidates,
                        max_sandbox_starts=config.budget.max_container_starts,
                        max_calls_per_task=config.search.max_calls_per_task,
                        max_tokens_per_task=config.search.max_reserved_tokens_per_task,
                        max_sandbox_per_task=config.search.max_sandbox_starts_per_task,
                        max_seconds_per_task=config.search.max_seconds_per_task,
                        state=scope_state, on_change=persist)


def _backend(config: ExperimentConfig, *, live: bool, thinking: str, log_path: Path,
             budget: SearchBudget, scope: str, safe_events: list[dict[str, Any]],
             prompt_events: list[dict[str, Any]] | None = None,
             before_dispatch: Any = None, on_response: Any = None) -> Any:
    if not live:
        return BudgetedBackend(SearchFakeBackend(), budget, scope,
                               max_tokens=config.model.max_tokens, prompt_events=prompt_events,
                               before_dispatch=before_dispatch, on_response=on_response)
    model = config.model
    backend = OpenAICompatibleBackend(model=model.id, revision=model.revision,
        base_url=model.base_url, temperature=model.temperature, max_tokens=model.max_tokens,
        seed=config.experiment.seed, timeout=model.timeout_seconds, thinking=thinking,
        allow_remote=model.allow_remote, log_path=log_path,
        api_style='openai_compatible', guarded=True,
        event_callback=lambda event: _record_backend_event(event, safe_events, scope, budget))
    return BudgetedBackend(backend, budget, scope, max_tokens=model.max_tokens,
                           prompt_events=prompt_events, before_dispatch=before_dispatch,
                           on_response=on_response)


def _record_backend_event(event: dict[str, Any], events: list[dict[str, Any]],
                          scope: str, budget: SearchBudget) -> None:
    budget.record_actual(scope, event.get('usage'))
    events.append({'scope': scope, 'kind': event.get('kind'),
                   'candidate': event.get('candidate'),
                   'response_sha256': event.get('response_sha256'),
                   'finish_reason': event.get('finish_reason'),
                   'diagnostic_category': event.get('diagnostic_category'),
                   'usage': event.get('usage'), 'error': event.get('error'),
                   'thinking_control': event.get('thinking_control'),
                   'thinking_mode_requested': event.get('thinking_mode_requested'),
                   'thinking_mode_effective': event.get('thinking_mode_effective'),
                   'elapsed_seconds': event.get('elapsed_seconds')})


def _sandbox(config: ExperimentConfig, *, live: bool, budget: SearchBudget, scope: str,
             docker_image_id: str | None = None) -> Any:
    if not live:
        base = FakeSandbox()
    else:
        agent = config.agent
        base = DockerSandbox(agent.sandbox_image, timeout=agent.timeout_seconds,
                             memory_mb=agent.memory_mb, cpus=agent.cpus, pids=agent.pids,
                             max_output_bytes=agent.max_output_bytes,
                             max_code_bytes=agent.max_code_bytes)
        if docker_image_id:
            base.image = docker_image_id
    return BudgetedSandbox(base, budget, scope)


def _docker_image_id(config: ExperimentConfig) -> str:
    docker = shutil.which('docker')
    if not docker:
        raise BackendError('sandbox_unavailable', 'Docker unavailable; refusing live code execution')
    try:
        result = subprocess.run([docker, 'image', 'inspect', config.agent.sandbox_image,
                                 '--format', '{{.Id}}'], check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=min(10, config.model.timeout_seconds),
                                text=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BackendError('sandbox_unavailable', 'configured Docker runner image is unavailable') from exc
    image_id = result.stdout.strip()
    if not image_id:
        raise BackendError('sandbox_unavailable', 'Docker runner image identity is unavailable')
    return image_id


def _task_seed(config: ExperimentConfig, task_id: str) -> int:
    task_hash = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)
    return config.experiment.seed * 100000 + task_hash % 100000


def _extract_attempts(run: StrategyRun, train_count: int) -> list[dict[str, Any]]:
    if run.candidates:
        return [{key: value for key, value in row.items() if key != 'examples'} | {
            'train_feedback': row.get('examples', [])} for row in run.candidates]
    output = []
    for index, attempt in enumerate(run.result.history):
        code = attempt.get('code') if isinstance(attempt, dict) else None
        if not isinstance(code, str) or not code:
            output.append({'candidate_index': index, 'code': None, 'code_sha256': None,
                           'parse_status': 'not_recorded', 'train_match_count': None,
                           'execution_errors': None, 'static_filter_pass': None,
                           'complete_train_match': None})
            continue
        try:
            ast.parse(code)
            parse_status = 'parsed'
        except (SyntaxError, ValueError, RecursionError):
            parse_status = 'error'
        feedback = attempt.get('feedback')
        known = isinstance(feedback, list)
        matched = max(0, train_count - len(feedback)) if known else None
        error_detail = (feedback.splitlines()[-1][:512]
                        if isinstance(feedback, str) and feedback else None)
        error_kind = 'execution_error' if error_detail else None
        passed, reasons, complexity = static_code_check(code)
        output.append({'candidate_index': index, 'code': code,
                       'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                       'parse_status': parse_status, 'train_match_count': matched,
                       'train_example_count': train_count,
                       'execution_errors': 0 if known else int(error_kind is not None),
                       'error_kind': error_kind,
                       'execution_error_detail': error_detail,
                       'static_filter_pass': passed, 'static_filter_reasons': reasons,
                       'code_complexity': complexity,
                       'complete_train_match': bool(known and matched == train_count and passed) if known else None})
    return output


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    radius = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return max(0.0, center - radius), min(1.0, center + radius)


def _strategy_metrics(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    subset = [row for row in rows if row['strategy'] == strategy]
    n = len(subset)
    successes = sum(row['pass_at_1'] is True for row in subset)
    parsed = [attempt for row in subset for attempt in row['attempts'] if attempt.get('parse_status') in {'parsed', 'error'}]
    valid = sum(item.get('parse_status') == 'parsed' for item in parsed)
    fm = [item for item in parsed if type(item.get('train_match_count')) is int]
    examples = sum(item.get('train_match_count', 0) for item in fm)
    example_total = sum(item.get('train_example_count', 0) for item in fm)
    repairs = []
    if strategy in {'S0', 'S2'}:
        for row in subset:
            feedback = [item.get('train_match_count') for item in row['attempts']]
            feedback = [item for item in feedback if type(item) is int]
            if len(feedback) >= 2:
                repairs.append(feedback[-1] > feedback[0])
    calls = sum(row['budget'].get('calls', 0) for row in subset)
    reserved = sum(row['budget'].get('reserved_tokens', 0) for row in subset)
    actual_calls = sum(row['budget'].get('actual_token_calls', 0) for row in subset)
    actual_values = [row['budget'].get('actual_tokens') for row in subset]
    actual_tokens = sum(value for value in actual_values if type(value) is int)
    sandbox_starts = sum(row['budget'].get('sandbox_starts', 0) for row in subset)
    errors = sum(row['error'] is not None for row in subset)
    response_events = [event for row in subset for event in row.get('response_events', [])]
    parse_failures = {'invalid_json', 'thinking_unclosed', 'missing_final_answer',
                      'generation_truncated', 'code_extraction_error', 'transport_error',
                      'budget_rejected', 'invalid_response', 'missing_response'}
    parse_observations = [event.get('diagnostic_category') for event in response_events
                          if isinstance(event.get('diagnostic_category'), str)]
    format_error_count = sum(category in parse_failures for category in parse_observations)
    low, high = _wilson(successes, n)
    return {'strategy': strategy, 'tasks': n, 'task_successes': successes,
            'pass_at_1': successes / n if n else None,
            'pass_at_1_wilson_95': [low, high],
            'valid_code_rate': valid / len(parsed) if parsed else None,
            'valid_code_numerator': valid, 'valid_code_denominator': len(parsed),
            'train_example_match_rate': examples / example_total if example_total else None,
            'train_examples_matched': examples, 'train_examples_total': example_total,
            'repair_improvement_rate': sum(repairs) / len(repairs) if repairs else None,
            'repair_improvement_available_count': len(repairs),
            'response_parse_rate': ((len(parse_observations) - format_error_count) / len(parse_observations)
                                    if parse_observations else None),
            'response_parse_denominator': len(parse_observations),
            'response_parse_not_available_count': max(0, sum(row['budget'].get('calls', 0) for row in subset) - len(parse_observations)),
            'format_error_rate': format_error_count / len(parse_observations) if parse_observations else None,
            'format_error_not_available_count': max(0, sum(row['budget'].get('calls', 0) for row in subset) - len(parse_observations)),
            'calls': calls, 'reserved_tokens': reserved,
            'actual_tokens': actual_tokens if actual_calls else None,
            'actual_token_calls': actual_calls,
            'sandbox_starts': sandbox_starts,
            'elapsed_seconds': sum(row['elapsed_seconds'] for row in subset),
            'budget_exhausted_tasks': sum(row['budget_exhausted'] for row in subset),
            'error_tasks': errors}


def _method_selection_metrics(rows: list[dict[str, Any]], strategies: list[str],
                              comparison_task_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    by_task_strategy = {(row.get('task_id'), row.get('strategy')): row for row in rows}
    paired = [task_id for task_id in comparison_task_ids
              if all((task_id, strategy) in by_task_strategy and
                     by_task_strategy[(task_id, strategy)].get('cell_status') == 'complete'
                     for strategy in strategies)]
    metrics: list[dict[str, Any]] = []
    parser_failure_categories = {
        'invalid_json', 'thinking_unclosed', 'missing_final_answer',
        'generation_truncated', 'code_extraction_error', 'transport_error',
        'budget_rejected', 'invalid_response', 'missing_response',
        'malformed_thinking',
    }
    for strategy in strategies:
        cells = [by_task_strategy[(task_id, strategy)] for task_id in paired]
        rates = [float(cell.get('final_train_match_rate') or 0.0) for cell in cells]
        full_count = sum(cell.get('final_train_full_match') is True for cell in cells)
        parse_errors = 0
        sandbox_errors = 0
        for cell in cells:
            candidate_parse_errors = sum(
                candidate.get('parse_status') == 'error'
                for candidate in cell.get('attempts', []))
            response_parse_errors = sum(
                event.get('diagnostic_category') in parser_failure_categories or
                (isinstance(event.get('error'), dict) and
                 event['error'].get('kind') in RECOVERABLE_RESPONSE_ERRORS)
                for event in cell.get('response_events', []))
            # A malformed response may also be represented by a failed candidate.
            # Count at least one event per failure source without double-counting
            # the same response when both views are present.
            parse_errors += max(candidate_parse_errors, response_parse_errors)
            for candidate in cell.get('attempts', []):
                value = candidate.get('execution_errors')
                if type(value) is int:
                    sandbox_errors += value
            if cell.get('error') in {'sandbox_unavailable', 'timeout', 'sandbox_error'}:
                sandbox_errors += 1
        metrics.append({'strategy': strategy, 'paired_tasks': len(paired),
                        'complete_train_match_tasks': full_count,
                        'macro_average_final_train_match_rate': (sum(rates) / len(rates) if rates else None),
                        'parser_error_events': parse_errors,
                        'sandbox_error_events': sandbox_errors,
                        'parser_and_sandbox_error_events': parse_errors + sandbox_errors,
                        'reserved_tokens': sum(int(cell.get('budget', {}).get('reserved_tokens', 0))
                                               for cell in cells),
                        'calls': sum(int(cell.get('budget', {}).get('calls', 0)) for cell in cells),
                        'heldout_verified_successes_diagnostic_only': sum(cell.get('pass_at_1') is True
                                                                           for cell in cells)})
    return metrics, paired


def reselect_completed_comparison(source_dir: str | Path,
                                  output_dir: str | Path) -> dict[str, Any]:
    """Derive a corrected E2 selection from a sealed completed run, offline.

    The source run is verified and left untouched. Inference logs and candidate
    results are copied byte-for-byte; only the deterministic selection outputs,
    their provenance note, and the new artifact seal are written to the copy.
    """
    source = Path(source_dir).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('reselection source and output must be separate, non-nested directories')
    source_seal = _verify_artifact_seal(source)
    source_seal_sha = _sha(source / 'artifact_seal.json')
    comparison_path = source / 'comparison.json'
    manifest_path = source / 'selection_manifest.json'
    policy_path = source / 'selection_policy.yml'
    frozen_path = source / 'frozen_search_config.yml'
    required = (comparison_path, manifest_path, policy_path, frozen_path,
                source / 'results.jsonl', source / 'config.snapshot.yml')
    if any(not path.is_file() for path in required):
        raise ValueError('completed E2 source is missing a required selection artifact')
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (comparison.get('protocol') != 'p3c_e2_comparison_v2' or
            comparison.get('run_profile') != 'e2_full' or
            comparison.get('inference_mode') != 'live' or
            comparison.get('status') != 'complete' or
            manifest.get('run_profile') != 'e2_full' or
            manifest.get('inference_mode') != 'live' or
            manifest.get('status') != 'complete'):
        raise ValueError('offline reselection requires a completed live E2 Full run')
    if len(comparison.get('comparison_task_ids', [])) != 5:
        raise ValueError('offline reselection requires the registered five-task E2 Full')
    if not isinstance(source_seal, dict) or not source_seal.get('artifacts'):
        raise ValueError('E2 source artifact seal is empty')

    rows = _read_jsonl(source / 'results.jsonl')
    source_frozen = yaml.safe_load(frozen_path.read_bytes())
    if not isinstance(source_frozen, dict) or not isinstance(source_frozen.get('model'), dict):
        raise ValueError('E2 frozen config is invalid')
    task_ids = comparison['comparison_task_ids']
    metrics, paired = _method_selection_metrics(rows, METHOD_TIE_BREAK_ORDER, task_ids)
    if paired != task_ids:
        raise ValueError('offline reselection requires all five paired strategy results')
    ranking = _rank_methods(metrics)
    if not ranking:
        raise ValueError('offline reselection produced no strategy ranking')
    selected = ranking[0]['strategy']
    status = ('selected_for_collection'
              if ranking[0]['complete_train_match_tasks'] > 0
              else 'inconclusive_fallback')

    if target.exists():
        raise FileExistsError(f'E2 reselection output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('.run.lock'))
    os.chmod(target, 0o700)
    frozen = source_frozen
    selected_thinking_mode = ('on' if selected == 'S2'
                              else frozen['model'].get('base_thinking'))

    comparison = dict(comparison)
    comparison.update({'selection_status': status,
                       'method_selection_metrics': metrics,
                       'method_selection_ranking': ranking,
                       'selected_strategy': selected,
                       'selection_rederived_from': str(source),
                       'selection_rederived_without_inference': True})
    manifest = dict(manifest)
    manifest.update({'selection_status': status,
                     'selected_strategy': selected,
                     'selected_thinking_mode': selected_thinking_mode,
                     'selection_reason': ranking[0],
                     'method_selection_metrics': metrics,
                     'selection_rederived_from': str(source),
                     'selection_rederived_without_inference': True})
    frozen['strategy'] = selected
    frozen['thinking_mode'] = selected_thinking_mode
    frozen['selection_status'] = status
    frozen['selection_reason'] = ranking[0]
    frozen['implementation_sha256'] = {
        name: _sha(ROOT / name) for name in IMPLEMENTATION_FILES}
    frozen['prompt_sha256'] = hashlib.sha256(_json_bytes({
        'system_prompt': FINAL_ANSWER_SYSTEM_PROMPT,
        'agent_prompt_source_sha256': _sha(ROOT / 'src/arc/agent.py'),
        'search_prompt_source_sha256': _sha(ROOT / 'src/arc/search.py')})).hexdigest()
    frozen_bytes = _canonical_yaml(frozen)
    _atomic_write(target / 'frozen_search_config.yml', frozen_bytes)
    frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    comparison.update({'frozen_search_config': 'frozen_search_config.yml',
                       'frozen_search_config_sha256': frozen_sha})
    manifest.update({'frozen_search_config': 'frozen_search_config.yml',
                     'frozen_search_config_sha256': frozen_sha})
    _atomic_write(target / 'comparison.json', _json_bytes(comparison, pretty=True))
    _atomic_write(target / 'selection_manifest.json', _json_bytes(manifest, pretty=True))

    reselection = {
        'protocol': 'p3c_e2_offline_reselection_v1',
        'source_run': str(source),
        'source_artifact_seal_sha256': source_seal_sha,
        'source_selection_policy_sha256': _sha(source / 'selection_policy.yml'),
        'selection_metric': 'parser_and_sandbox_error_events',
        'corrected_response_error_categories': sorted({
            event.get('diagnostic_category')
            for row in rows for event in row.get('response_events', [])
            if event.get('diagnostic_category') in {
                'invalid_json', 'thinking_unclosed', 'missing_final_answer',
                'generation_truncated', 'code_extraction_error', 'transport_error',
                'budget_rejected', 'invalid_response', 'missing_response',
                'malformed_thinking'}}),
        'method_selection_metrics': metrics,
        'method_selection_ranking': ranking,
        'selected_strategy': selected,
        'selection_status': status,
        'model_requests_made': 0,
        'raw_inference_artifacts_copied_unchanged': True,
        'source_artifact_seal_verified': True,
        'reason': 'generation_truncated response errors were omitted from parser error ranking',
    }
    _atomic_write(target / 'reselection.json', _json_bytes(reselection, pretty=True))
    report_path = target / 'report.md'
    report_lines = ['# S0–S3 paired Exploration comparison', '',
                    '- Status: `complete`', '- Profile: `e2_full`',
                    f'- Selection: `{selected}` ({status})',
                    f'- Paired tasks: {len(paired)}/{len(task_ids)}',
                    '- Execution mode: `live`',
                    '- Method selection uses complete public-train outcomes, parser/sandbox errors, reserved tokens, and the fixed S0>S1>S2>S3 tie order.',
                    '- Held-out verifier results are diagnostic only and never select or break method ties.',
                    '', '| Strategy | Pass@1 diagnostic | Tasks | Wilson 95% | Valid code | Train match | Full train tasks | Parse errors | Sandbox errors | Calls | Reserved | Actual | Sandbox | Seconds |',
                    '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    selection_by_strategy = {row['strategy']: row for row in metrics}
    for row in comparison.get('strategy_metrics', []):
        select_row = selection_by_strategy.get(row.get('strategy'), {})
        interval = row.get('pass_at_1_wilson_95') or [None, None]
        low, high = interval
        interval_text = 'NA' if low is None else f'{low:.3f}–{high:.3f}'
        report_lines.append(
            f"| {row['strategy']} | {row.get('pass_at_1')} | {row.get('tasks')} | {interval_text} | "
            f"{row.get('valid_code_rate')} | {row.get('train_example_match_rate')} | "
            f"{select_row.get('complete_train_match_tasks', 0)} | "
            f"{select_row.get('parser_error_events', 0)} | "
            f"{select_row.get('sandbox_error_events', 0)} | {row.get('calls')} | "
            f"{row.get('reserved_tokens')} | {row.get('actual_tokens')} | "
            f"{row.get('sandbox_starts')} | {row.get('elapsed_seconds', 0):.2f} |")
    if status == 'inconclusive_fallback':
        report_lines.extend(('', 'No strategy achieved a full-train-match task; the selected method is an inconclusive deterministic fallback.'))
    report_lines.extend(('', '## Offline reselection', '',
                         f'This derived artifact corrects response-error accounting from `{source}`. '
                         'It reuses the original sealed results and inference logs without new model requests. '
                         'The source run remains unchanged.', '',
                         f'Selection policy SHA-256: `{comparison["selection_policy_sha256"]}`',
                         f'Frozen config SHA-256: `{frozen_sha}`',
                         f'Used task IDs: `{", ".join(task_ids)}`'))
    report = '\n'.join(report_lines) + '\n'
    _atomic_write(report_path, report.encode('utf-8'))

    artifact_names = sorted(set(source_seal['artifacts']) | {'report.md', 'reselection.json'})
    seal = _artifact_seal(target, artifact_names, protocol='p3c_e2_artifact_seal_v1')
    _atomic_write(target / 'artifact_seal.json', _json_bytes(seal, pretty=True))
    checkpoint_path = target / 'checkpoint.json'
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        checkpoint['artifact_sha256'] = {'artifact_seal.json': _sha(target / 'artifact_seal.json')}
        _atomic_write(checkpoint_path, _json_bytes(checkpoint, pretty=True))
    _verify_artifact_seal(target)
    return {'status': 'complete', 'output_dir': str(target),
            'source_output_dir': str(source), 'selected_strategy': selected,
            'selection_status': status, 'paired_task_count': len(paired),
            'method_selection_metrics': metrics, 'method_selection_ranking': ranking,
            'frozen_search_config_sha256': frozen_sha, 'model_requests_made': 0,
            'source_artifact_seal_sha256': source_seal_sha,
            'artifact_seal_sha256': _sha(target / 'artifact_seal.json')}


def _rank_methods(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tie_order = {strategy: index for index, strategy in enumerate(METHOD_TIE_BREAK_ORDER)}
    return sorted(metrics, key=lambda row: (
        -int(row['complete_train_match_tasks']),
        -(float(row['macro_average_final_train_match_rate'])
          if row['macro_average_final_train_match_rate'] is not None else -1.0),
        int(row['parser_and_sandbox_error_events']),
        int(row['reserved_tokens']),
        tie_order.get(row['strategy'], len(tie_order))))


def _guarded_preflight(config: ExperimentConfig, *, required_modes: tuple[str, ...]) -> dict[str, Any]:
    contract = verify_guarded_contract(config.model.base_url, config.model.id,
        config.model.revision, config.budget.max_tokens, config.model.max_tokens,
        config.model.timeout_seconds, config.model.thinking,
        expected_max_prompt_tokens=config.model.max_prompt_tokens,
        required_thinking_modes=required_modes)
    if config.agent.sandbox != 'docker':
        raise ValueError('live strategy comparison requires DockerSandbox')
    if config.model.backend == 'ollama_native':
        verify_ollama_revision(config.model.base_url, config.model.id,
                               config.model.revision, config.model.timeout_seconds)
    return contract


def _write_e2_cells(target: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(target / 'results.jsonl', b''.join(_json_bytes(row) for row in rows))
    traces = []
    for row in rows:
        for attempt in row.get('attempts', []):
            traces.append({'task_id': row['task_id'], 'strategy': row['strategy'],
                           'candidate_index': attempt.get('candidate_index'),
                           'code': attempt.get('code'), 'code_sha256': attempt.get('code_sha256'),
                           'parse_status': attempt.get('parse_status'),
                           'error_kind': attempt.get('error_kind'),
                           'train_match_count': attempt.get('train_match_count'),
                           'train_example_count': attempt.get('train_example_count'),
                           'execution_errors': attempt.get('execution_errors'),
                           'execution_error_detail': attempt.get('execution_error_detail'),
                           'static_filter_pass': attempt.get('static_filter_pass'),
                           'static_filter_reasons': attempt.get('static_filter_reasons'),
                           'complete_train_match': attempt.get('complete_train_match'),
                           'duplicate_of': attempt.get('duplicate_of'),
                           'train_feedback': attempt.get('train_feedback', []),
                           'thinking_mode': row.get('thinking_mode'),
                           'response_events': row.get('response_events', []),
                           'prompt_hash_events': row.get('prompt_events', [])})
    _atomic_write(target / 'candidate_traces.jsonl', b''.join(_json_bytes(row) for row in traces))


def _write_e2_run_checkpoint(target: Path, checkpoint: dict[str, Any]) -> None:
    _atomic_write(target / 'budget_ledger.json', _json_bytes(checkpoint['budget'], pretty=True))
    checkpoint['artifact_sha256'] = {
        name: _sha(target / name) for name in (
            'config.snapshot.yml', 'preflight.json', 'task_manifest.json',
            'selection_policy.yml', 'exclusion_manifest.json',
            'comparison_identity.json', 'results.jsonl', 'candidate_traces.jsonl',
            'candidate_attempts.jsonl', 'budget_ledger.json')
        if (target / name).is_file()}
    _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
    state_path = target / 'run_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
    if state.get('status') != 'complete':
        state.update({'status': checkpoint.get('status', 'running'),
                      'trained': False, 'weights_updated': False})
        _atomic_write(state_path, _json_bytes(state, pretty=True))


def _contract_identity(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {key: item for key, item in value.items() if key != 'reserved_tokens'}


def _verify_e2_resume(target: Path, config: ExperimentConfig, *, live: bool,
                      extra_manifests: list[str | Path] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = ('comparison_identity.json', 'preflight.json', 'task_manifest.json',
                'selection_policy.yml', 'exclusion_manifest.json', 'checkpoint.json',
                'run_state.json')
    if any(not (target / name).is_file() for name in required):
        raise ValueError('E2 resume artifact is incomplete')
    identity = json.loads((target / 'comparison_identity.json').read_text(encoding='utf-8'))
    preflight = json.loads((target / 'preflight.json').read_text(encoding='utf-8'))
    task_manifest = json.loads((target / 'task_manifest.json').read_text(encoding='utf-8'))
    checkpoint = json.loads((target / 'checkpoint.json').read_text(encoding='utf-8'))
    config_sha = hashlib.sha256(_canonical_yaml(asdict(config))).hexdigest()
    if (identity.get('config_sha256') != config_sha or identity.get('run_profile') != config.tracking.profile or
            identity.get('inference_mode') != ('live' if live else 'fake_cpu')):
        raise ValueError('E2 resume config, profile, or execution mode differs from its original run')
    if identity.get('implementation_sha256') != {name: _sha(ROOT / name) for name in IMPLEMENTATION_FILES}:
        raise ValueError('E2 implementation changed since the interrupted run')
    for name, expected in checkpoint.get('artifact_sha256', {}).items():
        if not (target / name).is_file() or _sha(target / name) != expected:
            raise ValueError(f'E2 resume artifact integrity check failed: {name}')
    split = _split_manifest(config, _data(config)[2])
    if task_manifest.get('split_manifest_sha256') != split['split_sha256']:
        raise ValueError('E2 resume split manifest changed')
    if task_manifest.get('challenge_sha256') != split['challenge_sha256'] or task_manifest.get('solution_sha256') != split['solution_sha256']:
        raise ValueError('E2 resume ARC Training data changed')
    policy_bytes = (target / 'selection_policy.yml').read_bytes()
    if hashlib.sha256(policy_bytes).hexdigest() != identity.get('selection_policy_sha256'):
        raise ValueError('E2 resume selection policy was edited')
    saved_registry = json.loads((target / 'exclusion_manifest.json').read_text(encoding='utf-8'))
    validate_registry(saved_registry)
    current = _load_registry(config, extra_manifests=extra_manifests)
    try:
        own_prefix = target.relative_to(ROOT).as_posix().rstrip('/') + '/'
    except ValueError:
        own_prefix = ''
    original_exclusions = saved_registry['exclusions']
    for task_id in task_manifest.get('comparison_task_ids', []):
        external_sources = [source for source in current['exclusions'].get(task_id, [])
                            if not own_prefix or not source.startswith(own_prefix)]
        if external_sources and not set(external_sources) <= set(original_exclusions.get(task_id, [])):
            raise ValueError('a comparison task was used by another run after this E2 run started')
    current_contract = None
    current_image_id = None
    if live:
        current_contract = _guarded_preflight(config, required_modes=('off', 'on'))
        current_image_id = _docker_image_id(config)
    if current_image_id != identity.get('docker_image_id') or _contract_identity(current_contract) != _contract_identity(preflight.get('guarded_contract')):
        raise ValueError('E2 resume guarded server ledger/source or Docker image identity changed')
    return identity, preflight, checkpoint


def _compare_strategies_body(config: ExperimentConfig, target: Path, *, live: bool,
                             resume: bool, extra_manifests: list[str | Path] | None,
                             preflight: dict[str, Any] | None) -> dict[str, Any]:
    tasks, verifier, splits = _data(config)
    start = time.monotonic()
    if resume:
        identity, preflight_data, checkpoint = _verify_e2_resume(
            target, config, live=live, extra_manifests=extra_manifests)
        task_manifest = json.loads((target / 'task_manifest.json').read_text(encoding='utf-8'))
        task_ids = task_manifest['comparison_task_ids']
        registry = json.loads((target / 'exclusion_manifest.json').read_text(encoding='utf-8'))
        policy = yaml.safe_load((target / 'selection_policy.yml').read_text(encoding='utf-8'))
        contract = preflight_data.get('guarded_contract')
        docker_image_id = identity.get('docker_image_id')
        budget_state = checkpoint['budget']
        rows = _read_jsonl(target / 'results.jsonl')
        candidate_attempts = _read_jsonl(target / 'candidate_attempts.jsonl')
        completed = set(checkpoint.get('completed_cells', []))
        if checkpoint.get('pending_cell'):
            pending = checkpoint['pending_cell']
            scope = pending['scope']
            scope_usage = budget_state.get('scopes', {}).get(scope, {})
            if scope_usage.get('calls', 0) > 0:
                rows.append({'task_id': pending['task_id'], 'strategy': pending['strategy'],
                             'cell_status': 'request_uncertain', 'pass_at_1': None,
                             'final_train_match_count': None, 'final_train_example_count': None,
                             'final_train_match_rate': None, 'final_train_full_match': False,
                             'error': 'request_uncertain', 'error_detail': 'in-flight E2 cell skipped on resume',
                             'selected_candidate_index': None, 'attempts': [],
                             'budget': scope_usage, 'elapsed_seconds': 0.0,
                             'budget_exhausted': False, 'thinking_mode': pending['thinking_mode'],
                             'response_events': [], 'prompt_events': []})
                completed.add(pending['cell_key'])
                checkpoint.setdefault('uncertain_cells', []).append(pending['cell_key'])
            checkpoint['pending_cell'] = None
            checkpoint['pending_request'] = None
            checkpoint['completed_cells'] = sorted(completed)
            _write_e2_cells(target, rows)
            _write_e2_run_checkpoint(target, checkpoint)
    else:
        assert preflight is not None
        task_ids, registry, splits = resolve_comparison_tasks(config, extra_manifests=extra_manifests)
        if task_ids != preflight['comparison_task_ids']:
            raise ValueError('E2 task selection changed after preflight')
        contract = preflight.get('guarded_contract')
        docker_image_id = preflight.get('docker_image_id')
        policy = preflight['selection_policy']
        target.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(target, 0o700)
        config_bytes = _canonical_yaml(asdict(config))
        _atomic_write(target / 'config.snapshot.yml', config_bytes)
        split_manifest = _split_manifest(config, splits)
        registry_bytes = _json_bytes(registry, pretty=True)
        policy_bytes = _canonical_yaml(policy)
        _atomic_write(target / 'exclusion_manifest.json', registry_bytes)
        _atomic_write(target / 'selection_policy.yml', policy_bytes)
        for name in ('llm_requests.jsonl', 'candidate_attempts.jsonl', 'results.jsonl',
                     'candidate_traces.jsonl'):
            _atomic_write(target / name, b'')
        os.chmod(target / 'llm_requests.jsonl', 0o600)
        _atomic_write(target / 'requests.jsonl', b'')
        os.chmod(target / 'requests.jsonl', 0o600)
        source_hashes = {name: _sha(ROOT / name) for name in IMPLEMENTATION_FILES}
        identity = {'protocol': 'p3c_exploration_comparison_run_v2',
                    'status': 'reserved_before_dispatch',
                    'run_profile': config.tracking.profile,
                    'inference_mode': 'live' if live else 'fake_cpu',
                    'comparison_task_ids': task_ids, 'split': 'exploration',
                    'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                    'selection_policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
                    'split_manifest_sha256': split_manifest['split_sha256'],
                    'exclusion_manifest_sha256': hashlib.sha256(registry_bytes).hexdigest(),
                    'implementation_sha256': source_hashes,
                    'model_id': config.model.id, 'model_revision': config.model.revision,
                    'quantization': config.model.quantization,
                    'adapter_sha256': config.model.adapter_sha256,
                    'docker_image_id': docker_image_id,
                    'guarded_ledger_id': contract.get('ledger_id') if contract else None}
        _atomic_write(target / 'comparison_identity.json', _json_bytes(identity, pretty=True))
        _atomic_write(target / 'run_state.json', _json_bytes({
            'status': 'running', 'run_profile': config.tracking.profile,
            'trained': False, 'weights_updated': False,
            'inference_mode': 'live' if live else 'fake_cpu'}, pretty=True))
        task_manifest = {'protocol': 'p3c_e2_task_manifest_v1',
                         'run_profile': config.tracking.profile,
                         'comparison_task_ids': task_ids,
                         'strategy_order_by_task': preflight['strategy_order_by_task'],
                         'strategies': config.search.strategies,
                         'split': 'exploration',
                         'split_manifest_sha256': split_manifest['split_sha256'],
                         'split_counts': split_manifest['counts'],
                         'challenge_sha256': split_manifest['challenge_sha256'],
                         'solution_sha256': split_manifest['solution_sha256'],
                         'exclusion_manifest_sha256': identity['exclusion_manifest_sha256'],
                         'task_selection_seed': config.data.split_seed,
                         'method_order_seed': config.experiment.seed,
                         'promotion_gate_accessed': False, 'monitoring_accessed': False,
                         'public_evaluation_accessed': False}
        _atomic_write(target / 'task_manifest.json', _json_bytes(task_manifest, pretty=True))
        preflight_data = dict(preflight)
        preflight_data.update({'checked_at_utc': subprocess.run(
            ['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, check=False).stdout.strip() or None,
            'git': _git_provenance(), 'model_snapshot': _model_snapshot_metadata(config),
            'implementation_sha256': source_hashes,
            'system_prompt_sha256': hashlib.sha256(FINAL_ANSWER_SYSTEM_PROMPT.encode()).hexdigest(),
            'prompt_source_sha256': {name: _sha(ROOT / name) for name in
                                     ('src/arc/agent.py', 'src/arc/search.py', 'src/arc/prompting.py')},
            'server_source_sha256': contract.get('source_sha256') if contract else None})
        _atomic_write(target / 'preflight.json', _json_bytes(preflight_data, pretty=True))
        budget_state = {'calls': 0, 'candidate_calls': 0, 'reserved_tokens': 0,
                        'actual_tokens': 0, 'actual_token_calls': 0,
                        'sandbox_starts': 0, 'elapsed_seconds': 0.0, 'scopes': {}}
        checkpoint = {'protocol': 'p3c_e2_resume_checkpoint_v1',
                      'status': 'running', 'config_sha256': identity['config_sha256'],
                      'selection_policy_sha256': identity['selection_policy_sha256'],
                      'task_manifest_sha256': _sha(target / 'task_manifest.json'),
                      'completed_cells': [], 'uncertain_cells': [],
                      'pending_cell': None, 'pending_request': None,
                      'candidate_dispatches': [], 'budget': budget_state}
        _write_e2_cells(target, [])
        _atomic_write(target / 'candidate_attempts.jsonl', b'')
        _write_e2_run_checkpoint(target, checkpoint)

    results_path = target / 'results.jsonl'
    rows = _read_jsonl(results_path)
    completed = set(checkpoint.get('completed_cells', []))
    candidate_attempts = _read_jsonl(target / 'candidate_attempts.jsonl')
    if resume and checkpoint.get('status') == 'complete':
        return {'status': 'complete', 'output_dir': str(target),
                'selected_strategy': json.loads((target / 'selection_manifest.json').read_text()).get('selected_strategy'),
                'selection_status': json.loads((target / 'selection_manifest.json').read_text()).get('selection_status'),
                'resumed_completed_run': True, 'budget': checkpoint['budget']}

    task_ids = task_manifest['comparison_task_ids']
    order_map = task_manifest['strategy_order_by_task']
    budget_holder: dict[str, SearchBudget] = {}

    def persist_budget(state: dict[str, Any]) -> None:
        checkpoint['budget'] = state
        _write_e2_run_checkpoint(target, checkpoint)

    budget = _new_budget(config, scope_state=checkpoint['budget'], persist=persist_budget)
    budget_holder['budget'] = budget
    status, stop_reason = 'complete', None
    try:
        for task_id in task_ids:
            for strategy in order_map[task_id]:
                cell_key = f'{task_id}:{strategy}'
                if cell_key in completed:
                    continue
                scope = cell_key
                mode = 'on' if strategy == 'S2' else config.model.thinking
                if config.model.backend == 'fake':
                    mode = 'server_default'
                checkpoint['pending_cell'] = {'cell_key': cell_key, 'task_id': task_id,
                                              'strategy': strategy, 'scope': scope,
                                              'thinking_mode': mode}
                checkpoint['pending_request'] = None
                _write_e2_run_checkpoint(target, checkpoint)
                task = tasks[task_id]
                task_attempt_index = sum(1 for item in candidate_attempts if item.get('scope') == scope)
                response_events: list[dict[str, Any]] = []
                prompt_events: list[dict[str, Any]] = []

                def before_dispatch(event: dict[str, Any]) -> None:
                    nonlocal task_attempt_index
                    attempt_id = hashlib.sha256(
                        f'{scope}:{task_attempt_index}:{event["prompt_sha256"]}'.encode()).hexdigest()[:24]
                    row = {'attempt_id': attempt_id, 'task_id': task_id, 'strategy': strategy,
                           'scope': scope, 'candidate_call_index': task_attempt_index,
                           'candidate_index': (event['candidate'] - _task_seed(config, task_id)
                                               if event['kind'] == 'code' and strategy in {'S1', 'S3'}
                                               else task_attempt_index if event['kind'] == 'code' else None),
                           'kind': event['kind'], 'prompt_sha256': event['prompt_sha256'],
                           'reserved_tokens': event['reserved_tokens'],
                           'thinking_mode_requested': mode, 'status': 'reserved_before_dispatch'}
                    task_attempt_index += 1
                    candidate_attempts.append(row)
                    checkpoint['candidate_dispatches'] = candidate_attempts
                    checkpoint['pending_request'] = attempt_id
                    _atomic_write(target / 'candidate_attempts.jsonl',
                                  b''.join(_json_bytes(item) for item in candidate_attempts))
                    _write_e2_run_checkpoint(target, checkpoint)

                def on_response(event: dict[str, Any]) -> None:
                    attempt = next((item for item in reversed(candidate_attempts)
                                   if item.get('scope') == scope and item.get('status') == 'reserved_before_dispatch'), None)
                    if attempt:
                        attempt.update({'status': event['status'],
                                        'response_sha256': event.get('response_sha256'),
                                        'error_kind': event.get('error_kind')})
                    checkpoint['candidate_dispatches'] = candidate_attempts
                    _atomic_write(target / 'candidate_attempts.jsonl',
                                  b''.join(_json_bytes(item) for item in candidate_attempts))
                    _write_e2_run_checkpoint(target, checkpoint)

                def on_backend_event(event: dict[str, Any]) -> None:
                    _record_backend_event(event, response_events, scope, budget)
                    attempt = next((item for item in reversed(candidate_attempts)
                                   if item.get('scope') == scope and item.get('status') in
                                   {'reserved_before_dispatch', 'request_ended_with_error'}), None)
                    if attempt:
                        attempt.update({'request_id': event.get('request_id'),
                                        'response_sha256': event.get('response_sha256'),
                                        'http_diagnostic_category': event.get('diagnostic_category'),
                                        'finish_reason': event.get('finish_reason'),
                                        'actual_usage': event.get('usage'),
                                        'thinking_mode_effective': event.get('thinking_mode_effective'),
                                        'thinking_control': event.get('thinking_control')})
                        if event.get('response_sha256'):
                            attempt['status'] = 'http_response_received'
                    checkpoint['candidate_dispatches'] = candidate_attempts
                    _atomic_write(target / 'candidate_attempts.jsonl',
                                  b''.join(_json_bytes(item) for item in candidate_attempts))
                    _write_e2_run_checkpoint(target, checkpoint)

                backend = _backend(config, live=live, thinking=mode,
                                   log_path=target / 'llm_requests.jsonl', budget=budget,
                                   scope=scope, safe_events=response_events,
                                   prompt_events=prompt_events,
                                   before_dispatch=before_dispatch, on_response=on_response)
                if live:
                    backend.backend.event_callback = on_backend_event
                sandbox = _sandbox(config, live=live, budget=budget, scope=scope,
                                   docker_image_id=json.loads(
                                       (target / 'preflight.json').read_text(encoding='utf-8')
                                   ).get('docker_image_id'))
                task_started = time.monotonic()
                try:
                    run = run_strategy(task, strategy, backend, sandbox,
                                       max_repairs=config.agent.max_repairs,
                                       independent_candidates=config.search.independent_candidates,
                                       base_candidate=_task_seed(config, task_id))
                    attempts = _extract_attempts(run, len(task.train))
                    selected_index = run.selected_index
                    if strategy in {'S1', 'S3'} and selected_index is not None:
                        final_attempt = next((item for item in attempts
                                              if item.get('candidate_index') == selected_index), None)
                    else:
                        final_attempt = attempts[-1] if attempts else None
                    final_train_count = (final_attempt.get('train_match_count')
                                         if final_attempt else None)
                    final_train_count = final_train_count if type(final_train_count) is int else 0
                    final_rate = final_train_count / len(task.train) if task.train else 0.0
                    final_full = bool(final_attempt and final_attempt.get('complete_train_match') is True)
                    selected_code = run.result.code
                    verifier_ran = bool(selected_code and run.result.outputs)
                    score = verifier.matches(task_id, run.result.outputs) if verifier_ran else False
                    row = {'task_id': task_id, 'strategy': strategy,
                           'execution_order_index': order_map[task_id].index(strategy),
                           'cell_status': 'complete', 'pass_at_1': score,
                           'heldout_verifier_ran': verifier_ran,
                           'final_train_match_count': final_train_count,
                           'final_train_example_count': len(task.train),
                           'final_train_match_rate': final_rate,
                           'final_train_full_match': final_full,
                           'error': run.result.error_kind or ('invalid' if run.result.error else None),
                           'error_detail': run.result.error,
                           'selected_candidate_index': selected_index,
                           'final_code_sha256': hashlib.sha256(selected_code.encode()).hexdigest()
                                               if isinstance(selected_code, str) else None,
                           'attempts': attempts, 'budget': budget.usage(scope),
                           'elapsed_seconds': time.monotonic() - task_started,
                           'budget_exhausted': False, 'thinking_mode': mode,
                           'response_events': response_events,
                           'prompt_events': prompt_events,
                           'selection_used_test_labels': False}
                    rows.append(row)
                    completed.add(cell_key)
                    checkpoint['completed_cells'] = sorted(completed)
                    checkpoint['pending_cell'] = None
                    checkpoint['pending_request'] = None
                    _write_e2_cells(target, rows)
                    _write_e2_run_checkpoint(target, checkpoint)
                except SearchBudgetError as exc:
                    row = {'task_id': task_id, 'strategy': strategy,
                           'execution_order_index': order_map[task_id].index(strategy),
                           'cell_status': 'budget_exhausted', 'pass_at_1': None,
                           'heldout_verifier_ran': False,
                           'final_train_match_count': None, 'final_train_example_count': len(task.train),
                           'final_train_match_rate': None, 'final_train_full_match': False,
                           'error': 'budget_exhausted', 'error_detail': str(exc),
                           'selected_candidate_index': None, 'attempts': [],
                           'budget': budget.state.get('scopes', {}).get(scope, {}),
                           'elapsed_seconds': time.monotonic() - task_started,
                           'budget_exhausted': True, 'thinking_mode': mode,
                           'response_events': response_events, 'prompt_events': prompt_events,
                           'selection_used_test_labels': False}
                    rows.append(row); completed.add(cell_key)
                    checkpoint['completed_cells'] = sorted(completed)
                    checkpoint['pending_cell'] = None; checkpoint['pending_request'] = None
                    status, stop_reason = 'budget_exhausted', str(exc)
                    _write_e2_cells(target, rows); _write_e2_run_checkpoint(target, checkpoint)
                    raise
                except BackendError as exc:
                    known_response = exc.kind in RECOVERABLE_RESPONSE_ERRORS
                    cell_status = 'complete' if known_response else (
                        'sandbox_unavailable' if exc.kind == 'sandbox_unavailable' else 'request_uncertain')
                    row = {'task_id': task_id, 'strategy': strategy,
                           'execution_order_index': order_map[task_id].index(strategy),
                           'cell_status': cell_status, 'pass_at_1': False if known_response else None,
                           'heldout_verifier_ran': False,
                           'final_train_match_count': 0 if known_response else None,
                           'final_train_example_count': len(task.train),
                           'final_train_match_rate': 0.0 if known_response else None,
                           'final_train_full_match': False,
                           'error': exc.kind, 'error_detail': str(exc),
                           'selected_candidate_index': None, 'attempts': [],
                           'budget': budget.state.get('scopes', {}).get(scope, {}),
                           'elapsed_seconds': time.monotonic() - task_started,
                           'budget_exhausted': False, 'thinking_mode': mode,
                           'response_events': response_events, 'prompt_events': prompt_events,
                           'selection_used_test_labels': False}
                    rows.append(row)
                    if known_response:
                        completed.add(cell_key)
                        checkpoint['completed_cells'] = sorted(completed)
                        checkpoint['pending_cell'] = None; checkpoint['pending_request'] = None
                    else:
                        status, stop_reason = cell_status, str(exc)
                    _write_e2_cells(target, rows); _write_e2_run_checkpoint(target, checkpoint)
                    if not known_response:
                        raise
            if status != 'complete':
                break
    except SearchBudgetError:
        pass
    except BackendError:
        pass
    except KeyboardInterrupt:
        status, stop_reason = 'interrupted', 'interrupted; the in-flight cell will not be replayed on resume'
    finally:
        budget_state = budget.finish()
        checkpoint['budget'] = budget_state
        checkpoint['status'] = status
        checkpoint['stop_reason'] = stop_reason
        _write_e2_run_checkpoint(target, checkpoint)

    rows = _read_jsonl(target / 'results.jsonl')
    metric_rows = [_strategy_metrics(rows, strategy) for strategy in METHOD_TIE_BREAK_ORDER]
    selection_metrics, paired_ids = _method_selection_metrics(rows, METHOD_TIE_BREAK_ORDER, task_ids)
    selection_ranking = _rank_methods(selection_metrics)
    expected_full = (len(paired_ids) == len(task_ids) and status == 'complete')
    profile = config.tracking.profile
    selected = None
    selection_status = 'not_established'
    freeze_allowed = False
    if profile == 'cpu_fake' and expected_full:
        selected = selection_ranking[0]['strategy'] if selection_ranking else None
        selection_status = 'cpu_fake_selection_only' if selected else 'not_established'
        freeze_allowed = bool(selected)
    elif profile == 'e2_full' and live and expected_full:
        selected = selection_ranking[0]['strategy'] if selection_ranking else None
        if selected and selection_ranking[0]['complete_train_match_tasks'] > 0:
            selection_status = 'selected_for_collection'
        elif selected:
            selection_status = 'inconclusive_fallback'
        freeze_allowed = bool(selected)
    elif profile == 'e2_smoke':
        selection_status = 'not_established_smoke_only'
    elif profile == 'e2_full' and not live:
        selection_status = 'not_established_fake_run'
    if status != 'complete' or len(paired_ids) != len(task_ids):
        selected = None
        selection_status = 'not_established'
        freeze_allowed = False

    comparison = {'protocol': 'p3c_e2_comparison_v2', 'run_profile': profile,
                  'status': status, 'selection_status': selection_status,
                  'inference_mode': 'live' if live else 'fake_cpu',
                  'comparison_task_ids': task_ids, 'requested_paired_task_count': len(task_ids),
                  'complete_paired_task_ids': paired_ids,
                  'complete_paired_task_count': len(paired_ids),
                  'strategy_metrics': metric_rows,
                  'method_selection_metrics': selection_metrics,
                  'method_selection_ranking': selection_ranking,
                  'selected_strategy': selected,
                  'selection_policy_sha256': _sha(target / 'selection_policy.yml'),
                  'heldout_verifier_used_for_selection': False,
                  'stop_reason': stop_reason,
                  'budget': budget_state,
                  'exploration_only': True,
                  'promotion_gate_accessed': False, 'monitoring_accessed': False,
                  'public_evaluation_accessed': False}

    frozen_sha = None
    if freeze_allowed:
        contract = json.loads((target / 'preflight.json').read_text(encoding='utf-8')).get('guarded_contract')
        model = {'id': config.model.id, 'revision': config.model.revision,
                 'backend': config.model.backend, 'quantization': config.model.quantization,
                 'adapter_sha256': config.model.adapter_sha256,
                 'temperature': config.model.temperature, 'max_tokens': config.model.max_tokens,
                 'base_thinking': config.model.thinking,
                 'local_model_path': config.model.local_model_path}
        frozen = {'protocol': FROZEN_PROTOCOL, 'run_profile': profile,
                  'inference_mode': 'live' if live else 'fake_cpu',
                  'model': model, 'agent': asdict(config.agent), 'data': asdict(config.data),
                  'seed': config.experiment.seed,
                  'search_method': {'strategies': config.search.strategies,
                                    'max_calls_per_task': config.search.max_calls_per_task,
                                    'max_reserved_tokens_per_task': config.search.max_reserved_tokens_per_task,
                                    'max_sandbox_starts_per_task': config.search.max_sandbox_starts_per_task,
                                    'max_seconds_per_task': config.search.max_seconds_per_task,
                                    'independent_candidates': config.search.independent_candidates,
                                    'thinking_policy': config.search.thinking_policy,
                                    'selection_rules': config.search.selection_rules},
                  'comparison_task_ids': task_ids, 'selection_policy_sha256': comparison['selection_policy_sha256'],
                  'strategy': selected, 'thinking_mode': 'on' if selected == 'S2' else config.model.thinking,
                  'selection_status': selection_status,
                  'selection_reason': next((row for row in selection_metrics if row['strategy'] == selected), None),
                  'system_prompt_sha256': hashlib.sha256(FINAL_ANSWER_SYSTEM_PROMPT.encode()).hexdigest(),
                  'chat_template_sha256': _model_snapshot_metadata(config).get('tokenizer_template_sha256'),
                  'prompt_sha256': hashlib.sha256(_json_bytes({
                      'system_prompt': FINAL_ANSWER_SYSTEM_PROMPT,
                      'agent_prompt_source_sha256': _sha(ROOT / 'src/arc/agent.py'),
                      'search_prompt_source_sha256': _sha(ROOT / 'src/arc/search.py')})).hexdigest(),
                  'implementation_sha256': {name: _sha(ROOT / name) for name in IMPLEMENTATION_FILES},
                  'server_source_sha256': contract.get('source_sha256') if contract else None,
                  'data_sha256': {'challenges': _sha(config.data.challenges),
                                  'solutions': _sha(config.data.solutions)},
                  'split_manifest_sha256': json.loads((target / 'task_manifest.json').read_text())['split_manifest_sha256'],
                  'exclusion_manifest_sha256': _sha(target / 'exclusion_manifest.json'),
                  'sandbox_mode': 'docker' if live else 'fake',
                  'sandbox_image_id': json.loads((target / 'preflight.json').read_text()).get('docker_image_id'),
                  'guarded_ledger_id': contract.get('ledger_id') if contract else None}
        frozen_bytes = _canonical_yaml(frozen)
        _atomic_write(target / 'frozen_search_config.yml', frozen_bytes)
        frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
        comparison['frozen_search_config'] = 'frozen_search_config.yml'
        comparison['frozen_search_config_sha256'] = frozen_sha
    else:
        (target / 'frozen_search_config.yml').unlink(missing_ok=True)
        comparison['frozen_search_config'] = None
        comparison['frozen_search_config_sha256'] = None

    _atomic_write(target / 'comparison.json', _json_bytes(comparison, pretty=True))
    _write_jsonl(target / 'candidate_attempts.jsonl', candidate_attempts)
    _atomic_write(target / 'requests.jsonl', (target / 'llm_requests.jsonl').read_bytes())
    os.chmod(target / 'requests.jsonl', 0o600)
    per_task_rows = []
    for row in rows:
        per_task_rows.append({key: row.get(key) for key in (
            'task_id', 'strategy', 'execution_order_index', 'cell_status',
            'final_train_match_count', 'final_train_example_count', 'final_train_match_rate',
            'final_train_full_match', 'heldout_verifier_ran', 'pass_at_1', 'error',
            'selected_candidate_index', 'budget', 'elapsed_seconds', 'thinking_mode')})
    if per_task_rows:
        with (target / 'strategy_results.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = list(per_task_rows[0])
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(per_task_rows)
    else:
        _atomic_write(target / 'strategy_results.csv', b'')
    if metric_rows:
        with (target / 'strategy_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = list(metric_rows[0])
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(metric_rows)

    frozen_manifest = {'protocol': SELECTION_PROTOCOL, 'version': 2,
                       'status': status, 'run_profile': profile,
                       'selection_status': selection_status,
                       'selected_strategy': selected,
                       'selected_thinking_mode': ('on' if selected == 'S2' else config.model.thinking) if selected else None,
                       'selection_reason': next((row for row in selection_metrics if row['strategy'] == selected), None),
                       'selection_policy_sha256': comparison['selection_policy_sha256'],
                       'comparison_task_ids': task_ids, 'split': 'exploration',
                       'paired_task_count': len(paired_ids), 'requested_task_count': len(task_ids),
                       'method_selection_metrics': selection_metrics,
                       'strategy_metrics': metric_rows,
                       'strategies': metric_rows,
                       'heldout_verifier_used_for_selection': False,
                       'exclusion_manifest_sha256': _sha(target / 'exclusion_manifest.json'),
                       'frozen_search_config_sha256': frozen_sha,
                       'frozen_search_config': 'frozen_search_config.yml' if frozen_sha else None,
                       'model': {'id': config.model.id, 'revision': config.model.revision,
                                 'quantization': config.model.quantization,
                                 'adapter_sha256': config.model.adapter_sha256},
                       'budget': budget_state, 'stop_reason': stop_reason,
                       'inference_mode': 'live' if live else 'fake_cpu',
                       'guarded_contract': json.loads((target / 'preflight.json').read_text()).get('guarded_contract'),
                       'check': json.loads((target / 'preflight.json').read_text())}
    _atomic_write(target / 'selection_manifest.json', _json_bytes(frozen_manifest, pretty=True))
    report = ['# S0–S3 paired Exploration comparison', '',
              f'- Status: `{status}`', f'- Profile: `{profile}`',
              f'- Selection: `{selected}` ({selection_status})',
              f'- Paired tasks: {len(paired_ids)}/{len(task_ids)}',
              f'- Execution mode: `{comparison["inference_mode"]}`',
              '- Method selection uses only complete public-train outcomes, parser/sandbox errors, reserved tokens, and fixed S0>S1>S2>S3 tie order.',
              '- Independent held-out verifier results are reported after candidate selection and never select or break method ties.',
              '- S1 selects the first statically safe candidate that parses and executes on every train input; S3 ranks candidates by train exact matches.',
              '- Wilson 95% intervals describe the small sample and do not establish general model quality.', '',
              '| Strategy | Pass@1 diagnostic | Tasks | Wilson 95% | Valid code | Train match | Full train tasks | Parse errors | Sandbox errors | Calls | Reserved | Actual | Sandbox | Seconds |',
              '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    selection_by_strategy = {row['strategy']: row for row in selection_metrics}
    for row in metric_rows:
        select_row = selection_by_strategy.get(row['strategy'], {})
        low, high = row['pass_at_1_wilson_95']
        interval = 'NA' if low is None else f'{low:.3f}–{high:.3f}'
        report.append(f"| {row['strategy']} | {row['pass_at_1']} | {row['tasks']} | {interval} | {row['valid_code_rate']} | {row['train_example_match_rate']} | {select_row.get('complete_train_match_tasks', 0)} | {select_row.get('parser_error_events', 0)} | {select_row.get('sandbox_error_events', 0)} | {row['calls']} | {row['reserved_tokens']} | {row['actual_tokens']} | {row['sandbox_starts']} | {row['elapsed_seconds']:.2f} |")
    if selection_status == 'inconclusive_fallback':
        report.extend(('', 'No strategy achieved a full-train-match task; the selected method is an inconclusive deterministic fallback.'))
    if profile in {'e2_smoke'}:
        report.extend(('', 'Smoke results do not establish a winner and do not produce a frozen collection config.'))
    if stop_reason:
        report.extend(('', f'Stopped: `{stop_reason}`.'))
    report.extend(('', f'Selection policy SHA-256: `{comparison["selection_policy_sha256"]}`',
                   f'Frozen config SHA-256: `{frozen_sha}`' if frozen_sha else 'Frozen config: not produced.',
                   f'Used task IDs: `{", ".join(task_ids)}`'))
    _atomic_write(target / 'report.md', ('\n'.join(report) + '\n').encode())
    run_state = {'status': status, 'run_profile': profile, 'trained': False,
                 'weights_updated': False, 'inference_mode': comparison['inference_mode'],
                 'selection_status': selection_status, 'stop_reason': stop_reason,
                 'elapsed_seconds': time.monotonic() - start}
    _atomic_write(target / 'run_state.json', _json_bytes(run_state, pretty=True))
    seal = _artifact_seal(target, [
        'config.snapshot.yml', 'preflight.json', 'task_manifest.json', 'selection_policy.yml',
        'requests.jsonl', 'llm_requests.jsonl', 'llm_responses.jsonl', 'candidate_attempts.jsonl',
        'candidate_traces.jsonl', 'results.jsonl', 'strategy_results.csv',
        'strategy_comparison.csv', 'comparison.json', 'selection_manifest.json',
        'frozen_search_config.yml', 'budget_ledger.json', 'exclusion_manifest.json', 'run_state.json'],
        protocol='p3c_e2_artifact_seal_v1')
    _atomic_write(target / 'artifact_seal.json', _json_bytes(seal, pretty=True))
    checkpoint['status'] = status
    checkpoint['artifact_sha256'] = {'artifact_seal.json': _sha(target / 'artifact_seal.json')}
    _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
    return {'status': status, 'output_dir': str(target), 'selected_strategy': selected,
            'selection_status': selection_status, 'comparison_task_ids': task_ids,
            'paired_task_count': len(paired_ids), 'method_selection_metrics': selection_metrics,
            'strategy_metrics': metric_rows, 'frozen_search_config_sha256': frozen_sha,
            'budget': budget_state, 'stop_reason': stop_reason}


def compare_strategies(config: ExperimentConfig, *, output_dir: str | Path | None = None,
                       live: bool = False, resume: bool = False,
                       run_dir: str | Path | None = None,
                       extra_manifests: list[str | Path] | None = None) -> dict[str, Any]:
    if resume:
        if run_dir is None or output_dir is not None:
            raise ValueError('E2 resume requires --run-dir and forbids --output-dir')
        target = Path(run_dir).expanduser().resolve()
        if not target.is_dir():
            raise ValueError('E2 resume directory does not exist')
        state_path = target / 'run_state.json'
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state.get('status') == 'complete':
                seal_path = target / 'artifact_seal.json'
                if not seal_path.is_file():
                    raise ValueError('completed E2 run lacks an artifact seal')
                seal = json.loads(seal_path.read_text(encoding='utf-8'))
                expected = seal.pop('seal_sha256', None)
                if expected != hashlib.sha256(_json_bytes(seal)).hexdigest():
                    raise ValueError('completed E2 artifact seal is invalid')
                for name, digest in seal.get('artifacts', {}).items():
                    if not (target / name).is_file() or _sha(target / name) != digest:
                        raise ValueError(f'completed E2 artifact hash mismatch: {name}')
                return {'status': 'complete', 'output_dir': str(target),
                        'resumed_completed_run': True,
                        'selected_strategy': json.loads((target / 'selection_manifest.json').read_text()).get('selected_strategy')}
        preflight = None
    else:
        if run_dir is not None:
            raise ValueError('--run-dir is only valid with --resume')
        preflight = check_compare(config, live=live, extra_manifests=extra_manifests)
        target = Path(output_dir).expanduser().resolve() if output_dir else (
            Path(config.tracking.output_dir).expanduser().resolve() / f'e2-compare-{uuid.uuid4().hex[:12]}')
        if target.exists():
            raise FileExistsError(f'E2 output already exists: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(mode=0o700)
    with run_lock(target):
        return _compare_strategies_body(config, target, live=live, resume=resume,
                                        extra_manifests=extra_manifests,
                                        preflight=preflight)


def _selection(selection_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    root = Path(selection_dir).expanduser().resolve()
    manifest_path = root / 'selection_manifest.json'
    frozen_path = root / 'frozen_search_config.yml'
    exclusion_path = root / 'exclusion_manifest.json'
    if any(not path.is_file() for path in (manifest_path, frozen_path, exclusion_path)):
        raise ValueError('E2 selection artifact is incomplete')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    frozen_bytes = frozen_path.read_bytes()
    frozen = yaml.safe_load(frozen_bytes)
    if (not isinstance(manifest, dict) or manifest.get('protocol') != SELECTION_PROTOCOL or
            manifest.get('status') != 'complete' or not manifest.get('selected_strategy') or
            manifest.get('frozen_search_config_sha256') != hashlib.sha256(frozen_bytes).hexdigest() or
            not isinstance(frozen, dict) or frozen.get('protocol') != FROZEN_PROTOCOL or
            frozen.get('strategy') != manifest.get('selected_strategy') or
            frozen.get('run_profile') != manifest.get('run_profile') or
            manifest.get('exclusion_manifest_sha256') != _sha(exclusion_path)):
        raise ValueError('E2 selection or frozen config integrity check failed')
    return manifest, frozen, exclusion_path.read_bytes()


def _validate_frozen_config(config: ExperimentConfig, frozen: dict[str, Any]) -> None:
    if frozen.get('model') != {'id': config.model.id, 'revision': config.model.revision,
                               'backend': config.model.backend, 'quantization': config.model.quantization,
                               'adapter_sha256': config.model.adapter_sha256,
                               'temperature': config.model.temperature,
                               'max_tokens': config.model.max_tokens,
                               'base_thinking': config.model.thinking,
                               'local_model_path': config.model.local_model_path}:
        raise ValueError('E3 model identity/settings differ from the frozen E2 config')
    if frozen.get('agent') != asdict(config.agent) or frozen.get('data') != asdict(config.data):
        raise ValueError('E3 agent or data split differs from the frozen E2 config')
    expected_search_method = {
        'strategies': config.search.strategies,
        'max_calls_per_task': config.search.max_calls_per_task,
        'max_reserved_tokens_per_task': config.search.max_reserved_tokens_per_task,
        'max_sandbox_starts_per_task': config.search.max_sandbox_starts_per_task,
        'max_seconds_per_task': config.search.max_seconds_per_task,
        'independent_candidates': config.search.independent_candidates,
        'thinking_policy': config.search.thinking_policy,
        'selection_rules': config.search.selection_rules,
    }
    if frozen.get('search_method') != expected_search_method:
        raise ValueError('E3 search method differs from the frozen E2 config')
    if frozen.get('seed') != config.experiment.seed:
        raise ValueError('E3 seed differs from the frozen E2 config')
    current_sources = {name: _sha(ROOT / name) for name in IMPLEMENTATION_FILES}
    if frozen.get('implementation_sha256') != current_sources:
        raise ValueError('E3 implementation or prompt source differs from the frozen E2 run')
    current_data = {'challenges': _sha(config.data.challenges), 'solutions': _sha(config.data.solutions)}
    if frozen.get('data_sha256') != current_data:
        raise ValueError('E3 ARC Training files differ from the frozen E2 data')


def _collection_plan(config: ExperimentConfig, selection_dir: str | Path, *,
                     live: bool = False,
                     extra_manifests: list[str | Path] | None = None) -> dict[str, Any]:
    if not config.search.enabled:
        raise ValueError('E2/E3 requires an explicitly enabled search config group')
    selection, frozen, exclusion_bytes = _selection(selection_dir)
    _validate_frozen_config(config, frozen)
    if live:
        if config.tracking.profile not in {'e3_smoke', 'e3_collection'}:
            raise ValueError('live E3 requires an explicit e3_smoke or e3_collection profile')
        if (selection.get('run_profile') != 'e2_full' or
                selection.get('inference_mode') != 'live' or
                selection.get('status') != 'complete' or
                selection.get('selection_status') not in {'selected_for_collection', 'inconclusive_fallback'} or
                selection.get('paired_task_count') != selection.get('requested_task_count') or
                not selection.get('frozen_search_config_sha256')):
            raise ValueError('live E3 requires a completed, paired E2 Full live selection')
    else:
        if config.tracking.profile != 'cpu_fake':
            raise ValueError('CPU Fake E3 requires the explicit cpu_fake profile')
        if (selection.get('run_profile') != 'cpu_fake' or
                selection.get('inference_mode') != 'fake_cpu' or
                selection.get('selection_status') != 'cpu_fake_selection_only'):
            raise ValueError('CPU Fake E3 requires a completed CPU Fake selection')
    if config.rsi.trainer != 'dry_run' or config.model.adapter_sha256 != 'none':
        raise ValueError('E3 collection cannot train or use an adapter')
    tasks, _, splits = _data(config)
    registry = json.loads(exclusion_bytes)
    validate_registry(registry)
    current = _load_registry(config, extra_manifests=extra_manifests,
                             additional_ids=selection['comparison_task_ids'])
    combined = dict(current['exclusions'])
    for task_id, sources in registry['exclusions'].items():
        combined.setdefault(task_id, []).extend(sources)
    exclusions = {task_id: sorted(set(sources)) for task_id, sources in sorted(combined.items())}
    comparison_ids = set(selection['comparison_task_ids'])
    if not comparison_ids <= set(splits['exploration']):
        raise ValueError('E2 comparison manifest contains a task outside Exploration')
    eligible = [task_id for task_id in splits['exploration']
                if task_id not in exclusions and task_id not in comparison_ids]
    if any(task_id not in tasks for task_id in eligible):
        raise ValueError('eligible Exploration task is absent from the configured challenge file')
    return {'selection': selection, 'frozen': frozen,
            'exclusion_registry': {'protocol': EXCLUSION_PROTOCOL, 'version': 1,
                                   'exclusions': exclusions, 'task_ids': sorted(exclusions),
                                   'sources': sorted({source for values in exclusions.values() for source in values})},
            'splits': splits, 'eligible_task_ids': eligible,
            'target_unique_verified_tasks': config.search.target_unique_verified_tasks,
            'collection_max_tasks': config.search.collection_max_tasks}


def check_collect(config: ExperimentConfig, selection_dir: str | Path, *, live: bool = False,
                  extra_manifests: list[str | Path] | None = None) -> dict[str, Any]:
    plan = _collection_plan(config, selection_dir, live=live, extra_manifests=extra_manifests)
    if live:
        _validate_live_budget(config)
        if config.model.backend != 'guarded_mlx' or config.agent.sandbox != 'docker':
            raise ValueError('live E3 requires the guarded MLX backend and Docker sandbox')
        contract = _guarded_preflight(config, required_modes=(plan['selection']['selected_thinking_mode'],)
                                      if plan['selection']['selected_thinking_mode'] in {'on', 'off'} else ())
        if contract.get('reserved_tokens') != 0:
            raise ValueError('a new live E3 run requires a fresh guarded token ledger')
        image_id = _docker_image_id(config)
        if image_id != plan['frozen'].get('sandbox_image_id'):
            raise ValueError('E3 Docker runner image differs from the frozen E2 image')
    scheduled = plan['eligible_task_ids'][:config.search.collection_max_tasks]
    return {'status': 'ready', 'execution_mode': 'live' if live else 'cpu_fake',
            'selected_strategy': plan['selection']['selected_strategy'],
            'selected_thinking_mode': plan['selection']['selected_thinking_mode'],
            'eligible_exploration_task_count': len(plan['eligible_task_ids']),
            'scheduled_task_count': len(scheduled), 'scheduled_task_ids': scheduled,
            'target_unique_verified_tasks': config.search.target_unique_verified_tasks,
            'exclusion_count': len(plan['exclusion_registry']['task_ids']),
            'comparison_task_ids': plan['selection']['comparison_task_ids'],
            'split': 'exploration',
            'budget_limits': {'seconds': config.budget.max_seconds,
                              'calls': config.budget.max_llm_calls,
                              'reserved_tokens': config.budget.max_tokens,
                              'sandbox_starts': config.budget.max_container_starts}}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f'invalid JSONL row in {path}')
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(path, b''.join(_json_bytes(row) for row in rows))


def _checkpoint_artifact_hashes(run_dir: Path) -> dict[str, str]:
    names = ('config.snapshot.yml', 'source_selection_manifest.json',
             'selection_manifest.snapshot.json', 'verified_solutions.jsonl',
             'candidate_traces.jsonl', 'llm_requests.jsonl', 'llm_responses.jsonl')
    return {name: _sha(run_dir / name) for name in names if (run_dir / name).is_file()}


def _collection_artifacts(run_dir: Path, checkpoint: dict[str, Any], *,
                          selection: dict[str, Any], registry: dict[str, Any],
                          stop_reason: str, status: str) -> dict[str, Any]:
    solutions = [row for row in _read_jsonl(run_dir / 'verified_solutions.jsonl')]
    sft = [row['sft_record'] for row in solutions]
    traces = _read_jsonl(run_dir / 'candidate_traces.jsonl')
    attempted_ids = sorted(set(checkpoint.get('attempted_task_ids', [])))
    accepted_ids = {row.get('task_id') for row in solutions}
    trace_by_task = {row.get('task_id'): row for row in traces}
    attempted_rows = []
    rejected_rows = []
    for task_id in attempted_ids:
        trace = trace_by_task.get(task_id, {})
        accepted = task_id in accepted_ids
        reason = None if accepted else (
            trace.get('error_kind') or trace.get('error') or
            ('independent_verifier_failed' if trace.get('independent_verifier_pass') is False
             else trace.get('status') or 'no_verified_solution'))
        attempted_rows.append({'task_id': task_id,
                               'status': 'accepted' if accepted else 'rejected_or_incomplete',
                               'selected_candidate_index': trace.get('selected_candidate_index'),
                               'independent_verifier_pass': trace.get('independent_verifier_pass'),
                               'rejection_reason': reason})
        candidates = trace.get('candidate_traces', [])
        if isinstance(candidates, list) and candidates:
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                is_selected = candidate.get('candidate_index') == trace.get('selected_candidate_index')
                rejected_rows.append({'task_id': task_id,
                                      'candidate_index': candidate.get('candidate_index'),
                                      'code_sha256': candidate.get('code_sha256'),
                                      'parse_status': candidate.get('parse_status'),
                                      'train_match_count': candidate.get('train_match_count'),
                                      'train_example_count': candidate.get('train_example_count'),
                                      'static_filter_pass': candidate.get('static_filter_pass'),
                                      'execution_errors': candidate.get('execution_errors'),
                                      'rejection_reason': (None if accepted and is_selected else
                                          'not_selected_by_train_policy' if not is_selected else reason)})
        elif not accepted:
            rejected_rows.append({'task_id': task_id, 'candidate_index': None,
                                  'code_sha256': None, 'rejection_reason': reason})
    _write_jsonl(run_dir / 'attempted_tasks.jsonl', attempted_rows)
    _write_jsonl(run_dir / 'accepted_solutions.jsonl', solutions)
    _write_jsonl(run_dir / 'rejected_candidates.jsonl', rejected_rows)
    _write_jsonl(run_dir / 'sft_train.jsonl', sft)
    registry_bytes = _json_bytes(registry, pretty=True)
    _atomic_write(run_dir / 'exclusion_manifest.json', registry_bytes)
    _atomic_write(run_dir / 'budget_ledger.json', _json_bytes(checkpoint.get('budget', {}), pretty=True))
    checkpoint['artifact_sha256'] = _checkpoint_artifact_hashes(run_dir)
    _atomic_write(run_dir / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
    files = ['attempted_tasks.jsonl', 'accepted_solutions.jsonl', 'rejected_candidates.jsonl',
             'verified_solutions.jsonl', 'sft_train.jsonl', 'budget_ledger.json',
             'exclusion_manifest.json', 'source_selection_manifest.json', 'checkpoint.json']
    hashes = {name: _sha(run_dir / name) for name in files if (run_dir / name).is_file()}
    unique_ids = sorted({row.get('task_id') for row in solutions if isinstance(row.get('task_id'), str)})
    rejection_counts: dict[str, int] = {}
    for row in rejected_rows:
        reason = str(row.get('rejection_reason') or 'not_rejected')
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    manifest = {'protocol': COLLECTION_PROTOCOL, 'version': 1,
                'status': status, 'stop_reason': stop_reason,
                'selection_manifest_sha256': _sha(run_dir / 'selection_manifest.snapshot.json'),
                'selected_strategy': selection['selected_strategy'],
                'selected_thinking_mode': selection['selected_thinking_mode'],
                'task_ids': attempted_ids,
                'attempted_task_count': len(attempted_ids),
                'verified_unique_task_ids': unique_ids,
                'verified_unique_task_count': len(unique_ids),
                'sft_record_count': len(sft), 'test_outputs_in_sft_messages': False,
                'test_outputs_in_messages': False,
                'verified_record_ids': sorted(record['id'] for record in sft),
                'independent_verifier_protocol': 'all_test_inputs_exact_match_v1',
                'code_verification_mode': ('docker_reexecution' if checkpoint.get('live')
                                           else 'fake_cpu_not_training_eligible'),
                'verification': 'all test inputs compared by independent Verifier after code selection',
                'static_filter_is_heuristic': True,
                'rejection_reason_counts': rejection_counts,
                'data_provenance': 'self-generated code filtered with public ARC Training labels; not unlabeled training data',
                'split_access': {'exploration_inputs_and_train_labels': True,
                                 'exploration_test_inputs': True,
                                 'exploration_test_labels_used_only_by_independent_verifier': True,
                                 'promotion_gate': False, 'monitoring': False,
                                 'validation': False, 'final': False,
                                 'public_evaluation': False},
                'inference_mode': 'live' if checkpoint.get('live') else 'fake_cpu',
                'exclusion_manifest_sha256': hashlib.sha256(registry_bytes).hexdigest(),
                'artifact_sha256': hashes,
                'trained': False, 'weights_updated': False}
    _atomic_write(run_dir / 'collection_manifest.json', _json_bytes(manifest, pretty=True))
    summary = {'status': status, 'stop_reason': stop_reason,
               'inference_mode': 'live' if checkpoint.get('live') else 'fake_cpu',
               'explored_unique_tasks': len(attempted_ids),
               'verified_unique_tasks': len(unique_ids), 'sft_record_count': len(sft),
               'target_unique_verified_tasks': checkpoint['target_unique_verified_tasks'],
               'attempted_task_ids': attempted_ids,
               'verified_task_ids': unique_ids,
               'trained': False, 'weights_updated': False,
               'budget': checkpoint.get('budget', {})}
    _atomic_write(run_dir / 'collection_summary.json', _json_bytes(summary, pretty=True))
    seal = _artifact_seal(run_dir, [
        'config.snapshot.yml', 'source_selection_manifest.json', 'attempted_tasks.jsonl',
        'accepted_solutions.jsonl', 'rejected_candidates.jsonl', 'sft_train.jsonl',
        'collection_manifest.json', 'budget_ledger.json', 'exclusion_manifest.json',
        'collection_summary.json'], protocol='p3c_e3_artifact_seal_v1')
    _atomic_write(run_dir / 'artifact_seal.json', _json_bytes(seal, pretty=True))
    return summary


def _collect_solutions_body(config: ExperimentConfig, selection_dir: str | Path, *,
                            output_dir: str | Path | None = None, live: bool = False,
                            resume: bool = False, run_dir: str | Path | None = None,
                            extra_manifests: list[str | Path] | None = None,
                            precreated: bool = False) -> dict[str, Any]:
    plan = _collection_plan(config, selection_dir, live=live, extra_manifests=extra_manifests)
    if live:
        if config.model.backend != 'guarded_mlx' or config.agent.sandbox != 'docker':
            raise ValueError('live E3 requires the guarded MLX backend and Docker sandbox')
        contract = _guarded_preflight(config, required_modes=(plan['selection']['selected_thinking_mode'],)
                                      if plan['selection']['selected_thinking_mode'] in {'on', 'off'} else ())
        _validate_live_budget(config)
        if not resume and contract.get('reserved_tokens') != 0:
            raise ValueError('a new live E3 run requires a fresh guarded token ledger')
        docker_image_id = _docker_image_id(config)
        if docker_image_id != plan['frozen'].get('sandbox_image_id'):
            raise ValueError('E3 Docker runner image differs from the frozen E2 image')
    else:
        contract = None
        docker_image_id = None
    if resume:
        if run_dir is None:
            raise ValueError('--resume requires --run-dir')
        target = Path(run_dir).expanduser().resolve()
        checkpoint_path = target / 'checkpoint.json'
        if not target.is_dir() or not checkpoint_path.is_file():
            raise ValueError('resume directory has no checkpoint')
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if live and (checkpoint.get('guarded_ledger_id') != contract.get('ledger_id') or
                     checkpoint.get('docker_image_id') != docker_image_id or
                     checkpoint.get('guarded_contract') != _contract_identity(contract)):
            raise ValueError('E3 resume guarded ledger, server contract, or Docker image differs from its interrupted run')
        config_identity = {'selection_sha256': _sha(Path(selection_dir).expanduser().resolve() / 'selection_manifest.json'),
                           'frozen_sha256': _sha(Path(selection_dir).expanduser().resolve() / 'frozen_search_config.yml'),
                           'challenge_sha256': _sha(config.data.challenges),
                           'solution_sha256': _sha(config.data.solutions),
                           'exclusion_manifest_sha256': _sha(config.search.exclusion_manifest if Path(config.search.exclusion_manifest).is_absolute() else ROOT / config.search.exclusion_manifest),
                           'extra_manifest_sha256': [_sha(Path(path).expanduser().resolve()) for path in extra_manifests or []],
                           'config': asdict(config), 'live': live}
        config_sha = hashlib.sha256(_json_bytes(config_identity)).hexdigest()
        if checkpoint.get('config_sha256') != config_sha:
            raise ValueError('resume config, selection, model, or data hash differs from the original run')
        for name, expected in checkpoint.get('artifact_sha256', {}).items():
            if Path(name).name != name or not (target / name).is_file() or _sha(target / name) != expected:
                raise ValueError(f'resume artifact integrity check failed: {name}')
        if checkpoint.get('status') == 'target_reached':
            _verify_artifact_seal(target)
            return _collection_artifacts(target, checkpoint, selection=plan['selection'],
                                          registry=plan['exclusion_registry'],
                                          stop_reason='target_reached', status='target_reached') | {'run_dir': str(target)}
        if checkpoint.get('pending_task_id'):
            uncertain = checkpoint['pending_task_id']
            trace_rows = _read_jsonl(target / 'candidate_traces.jsonl')
            has_trace = uncertain in {row.get('task_id') for row in trace_rows}
            has_solution = uncertain in {row.get('task_id') for row in _read_jsonl(target / 'verified_solutions.jsonl')}
            if not has_trace and not has_solution:
                checkpoint.setdefault('uncertain_task_ids', []).append(uncertain)
                trace_rows.append({'task_id': uncertain,
                                   'strategy': plan['selection']['selected_strategy'],
                                   'status': 'uncertain_skipped_on_resume',
                                   'code': None,
                                   'error_kind': 'uncertain_prior_request',
                                   'selection_used_test_labels': False})
                _write_jsonl(target / 'candidate_traces.jsonl', trace_rows)
            checkpoint['pending_task_id'] = None
            checkpoint['artifact_sha256'] = _checkpoint_artifact_hashes(target)
            _atomic_write(checkpoint_path, _json_bytes(checkpoint, pretty=True))
    else:
        if run_dir is not None:
            raise ValueError('--run-dir is only valid with --resume')
        target = Path(output_dir).expanduser().resolve() if output_dir else (
            Path(config.tracking.output_dir).expanduser().resolve() / f'e3-collect-{uuid.uuid4().hex[:12]}')
        if precreated:
            if not target.is_dir():
                raise ValueError('pre-created E3 run directory is missing')
        else:
            if target.exists():
                raise FileExistsError(f'E3 collection output already exists: {target}')
            target.mkdir(parents=True, mode=0o700)
        selection_root = Path(selection_dir).expanduser().resolve()
        selection_snapshot = (selection_root / 'selection_manifest.json').read_bytes()
        _atomic_write(target / 'selection_manifest.snapshot.json', selection_snapshot)
        _atomic_write(target / 'source_selection_manifest.json', selection_snapshot)
        _atomic_write(target / 'config.snapshot.yml', _canonical_yaml(asdict(config)))
        for name in ('verified_solutions.jsonl', 'candidate_traces.jsonl', 'llm_requests.jsonl'):
            _atomic_write(target / name, b'')
            if name == 'llm_requests.jsonl': os.chmod(target / name, 0o600)
        _atomic_write(target / 'llm_responses.jsonl', b'')
        os.chmod(target / 'llm_responses.jsonl', 0o600)
        for name in ('attempted_tasks.jsonl', 'accepted_solutions.jsonl',
                     'rejected_candidates.jsonl', 'sft_train.jsonl'):
            _atomic_write(target / name, b'')
        config_identity = {'selection_sha256': hashlib.sha256(selection_snapshot).hexdigest(),
                           'frozen_sha256': _sha(selection_root / 'frozen_search_config.yml'),
                           'challenge_sha256': _sha(config.data.challenges),
                           'solution_sha256': _sha(config.data.solutions),
                           'exclusion_manifest_sha256': _sha(config.search.exclusion_manifest if Path(config.search.exclusion_manifest).is_absolute() else ROOT / config.search.exclusion_manifest),
                           'extra_manifest_sha256': [_sha(Path(path).expanduser().resolve()) for path in extra_manifests or []],
                           'config': asdict(config), 'live': live}
        config_sha = hashlib.sha256(_json_bytes(config_identity)).hexdigest()
        checkpoint = {'protocol': 'p3c_exploration_collection_checkpoint_v1',
                      'config_sha256': config_sha, 'status': 'running', 'cursor': 0,
                      'live': live,
                      'guarded_ledger_id': contract.get('ledger_id') if contract else None,
                      'docker_image_id': docker_image_id,
                      'guarded_contract': _contract_identity(contract),
                      'eligible_task_ids': plan['eligible_task_ids'],
                      'attempted_task_ids': [], 'uncertain_task_ids': [],
                      'pending_task_id': None, 'target_unique_verified_tasks': config.search.target_unique_verified_tasks,
                      'budget': {'calls': 0, 'reserved_tokens': 0, 'actual_tokens': 0,
                                 'actual_token_calls': 0, 'sandbox_starts': 0,
                                 'elapsed_seconds': 0.0, 'scopes': {}}}
        _atomic_write(target / 'budget_ledger.json', _json_bytes(checkpoint['budget'], pretty=True))
        checkpoint['artifact_sha256'] = _checkpoint_artifact_hashes(target)
        _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
    assert target is not None
    tasks, verifier, splits = _data(config)
    eligible = checkpoint.get('eligible_task_ids', [])
    if not isinstance(eligible, list) or not set(eligible) <= set(splits['exploration']):
        raise ValueError('checkpoint task schedule is invalid or leaves Exploration')
    # On resume the initial eligible schedule is authoritative and immutable.
    selection = plan['selection']
    registry = plan['exclusion_registry']

    def persist_budget(_: dict[str, Any]) -> None:
        checkpoint['budget'] = budget.state
        _atomic_write(target / 'budget_ledger.json', _json_bytes(checkpoint['budget'], pretty=True))
        _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))

    budget = _new_budget(config, scope_state=checkpoint['budget'], persist=persist_budget)
    status, stop_reason = 'running', None
    stop_now = False
    try:
        while checkpoint['cursor'] < len(eligible):
            successes = {row.get('task_id') for row in _read_jsonl(target / 'verified_solutions.jsonl')}
            if len(successes) >= checkpoint['target_unique_verified_tasks']:
                status, stop_reason = 'target_reached', 'target_reached'
                break
            if len(checkpoint['attempted_task_ids']) >= config.search.collection_max_tasks:
                status, stop_reason = 'budget_exhausted', 'collection task limit reached'
                break
            task_id = eligible[checkpoint['cursor']]
            checkpoint['cursor'] += 1
            checkpoint['attempted_task_ids'].append(task_id)
            checkpoint['pending_task_id'] = task_id
            _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
            task = tasks[task_id]
            strategy = selection['selected_strategy']
            scope = f'{task_id}:{strategy}'
            mode = selection['selected_thinking_mode']
            task_prompt_events: list[dict[str, Any]] = []
            task_response_events: list[dict[str, Any]] = []
            backend = _backend(config, live=live, thinking=mode,
                               log_path=target / 'llm_requests.jsonl', budget=budget,
                               scope=scope, safe_events=task_response_events,
                               prompt_events=task_prompt_events)
            sandbox = _sandbox(config, live=live, budget=budget, scope=scope,
                               docker_image_id=docker_image_id)
            start = time.monotonic()
            try:
                run = run_strategy(task, strategy, backend, sandbox,
                                   max_repairs=config.agent.max_repairs,
                                   independent_candidates=config.search.independent_candidates,
                                   base_candidate=_task_seed(config, task_id))
                trace = {'task_id': task_id, 'strategy': strategy, 'thinking_mode': mode,
                         'code': run.result.code, 'error': run.result.error,
                         'error_kind': run.result.error_kind,
                         'prompt_hash_events': task_prompt_events,
                         'response_events': task_response_events,
                         'candidate_traces': run.candidates,
                         'selected_candidate_index': run.selected_index,
                         'selection_used_test_labels': False}
                code = run.result.code
                verified = False
                outputs: list[Any] = []
                filters, filter_reasons, _ = static_code_check(code) if isinstance(code, str) else (False, ['missing_code'], None)
                if code and filters and run.result.error is None:
                    # Independent re-execution covers every demonstration and every
                    # test input, while expected test outputs remain inside Verifier.
                    train_matches = True
                    try:
                        for train_input, train_output in task.train:
                            train_matches = validate_task_output(sandbox.run(code, train_input), train_output) and train_matches
                        outputs = [sandbox.run(code, test_input) for test_input in task.test_inputs]
                        verified = train_matches and verifier.matches(task_id, outputs)
                    except BackendError:
                        raise
                    except (ValueError, RuntimeError, TimeoutError):
                        verified = False
                trace.update({'static_filter_pass': filters,
                              'static_filter_reasons': filter_reasons,
                              'independent_verifier_pass': verified,
                              'test_output_count': len(outputs),
                              'elapsed_seconds': time.monotonic() - start,
                              'budget': budget.usage(scope)})
                trace_rows = _read_jsonl(target / 'candidate_traces.jsonl')
                trace_rows.append(trace)
                _write_jsonl(target / 'candidate_traces.jsonl', trace_rows)
                if verified and code is not None:
                    user_content = json.dumps(task.prompt_data(), sort_keys=True, ensure_ascii=False, separators=(',', ':'))
                    record_id = hashlib.sha256((user_content + '\0' + code).encode()).hexdigest()
                    sft_record = {'id': record_id, 'task_id': task_id, 'split': 'exploration',
                                  'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                                               {'role': 'user', 'content': user_content},
                                               {'role': 'assistant', 'content': code}]}
                    solution = {'task_id': task_id, 'strategy': strategy,
                                'model_revision': config.model.revision,
                                'adapter_sha256': config.model.adapter_sha256,
                                'code': code, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                                'prompt_sha256': hashlib.sha256(user_content.encode()).hexdigest(),
                                'template_sha256': hashlib.sha256(FINAL_ANSWER_SYSTEM_PROMPT.encode()).hexdigest(),
                                'search_prompt_sha256': plan['frozen'].get('prompt_sha256'),
                                'generation_prompt_hashes': [event['prompt_sha256'] for event in task_prompt_events],
                                'generation_response_events': task_response_events,
                                'verifier': {'protocol': 'independent_all_test_inputs_exact_match_v1',
                                             'all_test_inputs_passed': True,
                                             'test_output_count': len(outputs)},
                                'provenance': {'selection_manifest_sha256': _sha(target / 'selection_manifest.snapshot.json'),
                                               'selection_task_ids': selection['comparison_task_ids'],
                                               'run_mode': 'live' if live else 'fake_cpu'},
                                'predicted_outputs': outputs, 'sft_record': sft_record}
                    solution_rows = _read_jsonl(target / 'verified_solutions.jsonl')
                    if task_id not in {row.get('task_id') for row in solution_rows}:
                        solution_rows.append(solution)
                        _write_jsonl(target / 'verified_solutions.jsonl', solution_rows)
                checkpoint['artifact_sha256'] = _checkpoint_artifact_hashes(target)
                checkpoint['pending_task_id'] = None
                checkpoint['status'] = 'running'
                _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
            except SearchBudgetError as exc:
                trace_rows = _read_jsonl(target / 'candidate_traces.jsonl')
                trace_rows.append({'task_id': task_id, 'strategy': strategy,
                                   'thinking_mode': mode, 'status': 'budget_exhausted',
                                   'code': None, 'error': 'budget_exhausted',
                                   'selection_used_test_labels': False,
                                   'budget': checkpoint.get('budget', {}).get('scopes', {}).get(scope, {})})
                _write_jsonl(target / 'candidate_traces.jsonl', trace_rows)
                checkpoint['artifact_sha256'] = _checkpoint_artifact_hashes(target)
                checkpoint['pending_task_id'] = None
                checkpoint['status'] = 'budget_exhausted'
                _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
                status, stop_reason, stop_now = 'budget_exhausted', str(exc), True
                break
            except BackendError as exc:
                if exc.kind in RECOVERABLE_RESPONSE_ERRORS:
                    trace_rows = _read_jsonl(target / 'candidate_traces.jsonl')
                    trace_rows.append({'task_id': task_id, 'strategy': strategy,
                                       'thinking_mode': mode, 'code': None,
                                       'error': exc.kind, 'error_kind': exc.kind,
                                       'prompt_hash_events': task_prompt_events,
                                       'response_events': task_response_events,
                                       'selection_used_test_labels': False,
                                       'budget': checkpoint.get('budget', {}).get('scopes', {}).get(scope, {})})
                    _write_jsonl(target / 'candidate_traces.jsonl', trace_rows)
                    checkpoint['artifact_sha256'] = _checkpoint_artifact_hashes(target)
                    checkpoint['pending_task_id'] = None
                    _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
                    continue
                # A dispatched request can be uncertain. Keep it marked pending so
                # resume consumes the ID without replaying the request.
                checkpoint['status'] = 'request_uncertain' if exc.kind != 'sandbox_unavailable' else 'sandbox_unavailable'
                _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
                status, stop_reason, stop_now = checkpoint['status'], str(exc), True
                break
        if status == 'running':
            successes = {row.get('task_id') for row in _read_jsonl(target / 'verified_solutions.jsonl')}
            if len(successes) >= checkpoint['target_unique_verified_tasks']:
                status, stop_reason = 'target_reached', 'target_reached'
            elif len(checkpoint['attempted_task_ids']) >= config.search.collection_max_tasks and checkpoint['cursor'] < len(eligible):
                status, stop_reason = 'budget_exhausted', 'collection task limit reached'
            elif checkpoint['cursor'] >= len(eligible):
                status, stop_reason = 'insufficient_verified_tasks', 'exploration_exhausted'
    except KeyboardInterrupt:
        status, stop_reason, stop_now = 'interrupted', 'interrupted; pending task will be skipped on resume', True
    finally:
        checkpoint['budget'] = budget.finish()
        checkpoint['status'] = status
        checkpoint['stop_reason'] = stop_reason
        _atomic_write(target / 'checkpoint.json', _json_bytes(checkpoint, pretty=True))
    summary = _collection_artifacts(target, checkpoint, selection=selection,
                                    registry=registry, stop_reason=stop_reason or status,
                                    status=status)
    return summary | {'run_dir': str(target), 'stop_reason': stop_reason,
                      'execution_mode': 'live' if live else 'fake_cpu',
                      'guarded_contract': contract}


def collect_solutions(config: ExperimentConfig, selection_dir: str | Path, *,
                      output_dir: str | Path | None = None, live: bool = False,
                      resume: bool = False, run_dir: str | Path | None = None,
                      extra_manifests: list[str | Path] | None = None) -> dict[str, Any]:
    """Serialize each E3 run/resume so two processes cannot replay one task."""
    if resume:
        if run_dir is None:
            raise ValueError('--resume requires --run-dir')
        target = Path(run_dir).expanduser().resolve()
        if not target.is_dir():
            raise ValueError('resume directory is missing')
    else:
        if run_dir is not None:
            raise ValueError('--run-dir is only valid with --resume')
        target = Path(output_dir).expanduser().resolve() if output_dir else (
            Path(config.tracking.output_dir).expanduser().resolve() / f'e3-collect-{uuid.uuid4().hex[:12]}')
        if target.exists():
            raise FileExistsError(f'E3 collection output already exists: {target}')
        target.mkdir(parents=True, mode=0o700)
    with run_lock(target):
        return _collect_solutions_body(config, selection_dir, output_dir=target,
                                       live=live, resume=resume,
                                       run_dir=target if resume else None,
                                       extra_manifests=extra_manifests,
                                       precreated=not resume)


def validate_task_output(actual: Any, expected: list[list[int]]) -> bool:
    from src.arc.data import validate_grid
    try:
        return validate_grid(actual) == expected
    except (TypeError, ValueError):
        return False

