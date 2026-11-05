"""Evaluate the existing code agent on a bounded synthetic ARC suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from src.arc.agent import code_agent
from src.arc.data import Task, validate_grid
from src.arc.http_backend import (BackendError, OpenAICompatibleBackend,
                                  verify_guarded_contract)
from src.arc.sandbox import DockerSandbox
from src.arc.search import BudgetedBackend, BudgetedSandbox, SearchBudget
from src.core.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[2]
CHALLENGES = ROOT / 'tests/fixtures/e4_synthetic_challenges.v1.json'
SOLUTIONS = ROOT / 'tests/fixtures/e4_synthetic_solutions.v1.json'
PROTOCOL = 'e4_synthetic_agent_eval_v2'
MODEL_MAX_TOKENS = 768
MODEL_MAX_PROMPT_TOKENS = 32768
RUN_LIMITS = {'seconds': 3600, 'calls': 30, 'reserved_tokens': 200000,
              'sandbox_starts': 160}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                indent=2) + '\n', encoding='utf-8')


def _write_private_jsonl(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n')
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _docker_image_id(image: str) -> str:
    docker = shutil.which('docker')
    if not docker:
        raise BackendError('sandbox_unavailable', 'Docker unavailable; refusing code execution')
    result = subprocess.run([docker, 'image', 'inspect', image, '--format', '{{.Id}}'],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=10, text=True)
    if not result.stdout.strip():
        raise BackendError('sandbox_unavailable', 'sandbox image identity is unavailable')
    return result.stdout.strip()


def _candidate_from_result(result: Any) -> tuple[str | None, int | None, str | None]:
    if result.code:
        return result.code, len(result.history), None
    if result.history:
        item = result.history[-1]
        code = item.get('code') if isinstance(item, dict) else None
        feedback = item.get('feedback') if isinstance(item, dict) else None
        return code if isinstance(code, str) and code else None, len(result.history), (
            'execution_error' if not isinstance(feedback, list) else 'train_mismatch')
    return None, 0, result.error_kind or 'no_candidate'


def _load_challenges(path: Path) -> dict[str, Task]:
    data = json.loads(path.read_text(encoding='utf-8'))
    tasks = {}
    for task_id, item in data.items():
        train = tuple((validate_grid(example['input']), validate_grid(example['output']))
                      for example in item['train'])
        test_inputs = tuple(validate_grid(example['input']) for example in item['test'])
        tasks[task_id] = Task(task_id, train, test_inputs)
    return tasks


def run(output_dir: Path, *, base_url: str, seed: int,
        prompt_variant: str = 'structured') -> dict[str, Any]:
    config = load_experiment_config(ROOT / 'configs/e2.live.full.yml')
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite existing E4 output: {output_dir}')
    output_dir.mkdir(parents=True)
    tasks = _load_challenges(CHALLENGES)
    contract = verify_guarded_contract(
        base_url, config.model.id, config.model.revision,
        RUN_LIMITS['reserved_tokens'], MODEL_MAX_TOKENS,
        config.model.timeout_seconds, 'off',
        expected_max_prompt_tokens=MODEL_MAX_PROMPT_TOKENS,
        required_thinking_modes=('off',))
    if contract.get('reserved_tokens') != 0:
        raise BackendError('budget_unverified', 'E4 requires a fresh guarded token ledger')
    image_id = _docker_image_id(config.agent.sandbox_image)
    artifact = {'protocol': PROTOCOL, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'model': config.model.id, 'revision': config.model.revision,
                'adapter_sha256': 'none', 'thinking': 'off', 'temperature': 0.7,
                'seed': seed, 'max_new_tokens': MODEL_MAX_TOKENS,
                'max_prompt_tokens': MODEL_MAX_PROMPT_TOKENS,
                'challenges_sha256': _sha(CHALLENGES),
                'solutions_sha256': _sha(SOLUTIONS),
                'challenge_count': len(tasks), 'training_pair_count': sum(len(t.train) for t in tasks.values()),
                'heldout_input_count': sum(len(t.test_inputs) for t in tasks.values()),
                'heldout_labels_passed_to_agent': False,
                'agent_prompt_variant': prompt_variant,
                'docker_image': config.agent.sandbox_image, 'docker_image_id': image_id,
                'guarded_contract': {key: contract.get(key) for key in (
                    'contract', 'model', 'revision', 'adapter_sha256', 'thinking_mode',
                    'thinking_modes', 'thinking_canary', 'token_limit', 'reserved_tokens',
                    'max_prompt_tokens', 'max_new_tokens', 'ledger_id', 'driver_type',
                    'driver_runtime_version', 'driver_source_sha256')},
                'limits': RUN_LIMITS}
    _write_json(output_dir / 'manifest.json', artifact)

    budget_state: dict[str, Any] = {'calls': 0, 'candidate_calls': 0,
                                    'reserved_tokens': 0, 'actual_tokens': 0,
                                    'actual_token_calls': 0, 'sandbox_starts': 0,
                                    'elapsed_seconds': 0.0, 'scopes': {}}
    budget = SearchBudget(max_seconds=RUN_LIMITS['seconds'], max_calls=RUN_LIMITS['calls'],
                          max_tokens=RUN_LIMITS['reserved_tokens'],
                          max_sandbox_starts=RUN_LIMITS['sandbox_starts'],
                          max_calls_per_task=3, max_tokens_per_task=18000,
                          max_sandbox_per_task=20, max_seconds_per_task=300,
                          state=budget_state)
    response_events: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    result_path = output_dir / 'baseline_results.jsonl'

    for index, (task_id, task) in enumerate(sorted(tasks.items())):
        scope = task_id
        prompt_events: list[dict[str, Any]] = []
        backend = OpenAICompatibleBackend(
            model=config.model.id, revision=config.model.revision, base_url=base_url,
            temperature=config.model.temperature, max_tokens=MODEL_MAX_TOKENS,
            seed=seed, timeout=config.model.timeout_seconds, thinking='off',
            allow_remote=False, guarded=True,
            event_callback=lambda event, task_scope=scope: (
                budget.record_actual(task_scope, event.get('usage')),
                response_events.append({'scope': task_scope, 'kind': event.get('kind'),
                                        'elapsed_seconds': event.get('elapsed_seconds'),
                                        'finish_reason': event.get('finish_reason'),
                                        'diagnostic_category': event.get('diagnostic_category'),
                                        'error_kind': (event.get('error') or {}).get('kind'),
                                        'usage': event.get('usage')})))
        wrapped_backend = BudgetedBackend(backend, budget, scope,
                                          max_tokens=MODEL_MAX_TOKENS,
                                          prompt_events=prompt_events,
                                          analysis_prompt_mode=prompt_variant == 'structured')
        backend.analysis_prompt_mode = prompt_variant == 'structured'
        sandbox = BudgetedSandbox(DockerSandbox(
            config.agent.sandbox_image, timeout=config.agent.timeout_seconds,
            memory_mb=config.agent.memory_mb, cpus=config.agent.cpus,
            pids=config.agent.pids, max_output_bytes=config.agent.max_output_bytes,
            max_code_bytes=config.agent.max_code_bytes), budget, scope)
        started = time.monotonic()
        error_kind = None
        try:
            result = code_agent(task, wrapped_backend, sandbox, max_repairs=1,
                                candidate=700000 + index,
                                prompt_variant=prompt_variant)
            code, attempts, failure = _candidate_from_result(result)
            error_kind = result.error_kind
            train_hits = 0
            train_errors = 0
            outputs: list[Any] = []
            if code is not None:
                for input_grid, expected in task.train:
                    try:
                        actual = sandbox.run(code, input_grid)
                        train_hits += actual == expected
                    except Exception:
                        train_errors += 1
                if train_errors == 0:
                    try:
                        outputs = [sandbox.run(code, grid) for grid in task.test_inputs]
                    except Exception:
                        outputs = []
            else:
                outputs = []
            row = {'task_id': task_id, 'status': 'complete',
                   'candidate_attempts': attempts, 'failure_category': failure,
                   'agent_error_kind': error_kind, 'code': code,
                   'code_sha256': hashlib.sha256(code.encode()).hexdigest() if code else None,
                   'analysis_sha256': hashlib.sha256(result.analysis.encode()).hexdigest()
                                     if result.analysis else None,
                   'train_match_count': train_hits,
                   'train_example_count': len(task.train),
                   'train_match_rate': train_hits / len(task.train),
                   'train_execution_errors': train_errors,
                   'complete_train_match': train_hits == len(task.train) and train_errors == 0,
                   'test_outputs': outputs,
                   'attempt_trace': result.history,
                   'prompt_hash_events': prompt_events,
                   'response_events': [event for event in response_events if event['scope'] == scope],
                   'budget': budget.usage(scope),
                   'elapsed_seconds': time.monotonic() - started}
        except BackendError as exc:
            error_kind = exc.kind
            row = {'task_id': task_id, 'status': 'budget_or_backend_error',
                   'error_kind': exc.kind, 'error': str(exc),
                   'prompt_hash_events': prompt_events,
                   'response_events': [event for event in response_events if event['scope'] == scope],
                   'budget': budget_state.get('scopes', {}).get(scope, {}),
                   'elapsed_seconds': time.monotonic() - started}
            results.append(row)
            _write_private_jsonl(result_path, row)
            break
        results.append(row)
        _write_private_jsonl(result_path, row)

    # Keep heldout labels unopened until every model call and candidate choice
    # is complete. They are used only to score saved predictions below.
    heldout_labels = json.loads(SOLUTIONS.read_text(encoding='utf-8'))
    for row in results:
        if row.get('status') != 'complete':
            continue
        expected = heldout_labels[row['task_id']]
        actual = row.get('test_outputs', [])
        row['heldout_match_count'] = sum(a == b for a, b in zip(actual, expected))
        row['heldout_input_count'] = len(expected)
        row['complete_heldout_match'] = (len(actual) == len(expected) and
                                         row['heldout_match_count'] == len(expected))
    result_path.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n'
                                     for row in results), encoding='utf-8')
    final_budget = budget.finish()
    _write_json(output_dir / 'budget.json', final_budget)
    complete = [row for row in results if row.get('status') == 'complete']
    summary = {'protocol': PROTOCOL, 'status': 'complete' if len(complete) == len(tasks) else 'incomplete',
               'task_count': len(tasks), 'completed_tasks': len(complete),
               'full_train_match_tasks': sum(row['complete_train_match'] for row in complete),
               'train_matches': sum(row['train_match_count'] for row in complete),
               'train_examples': sum(row['train_example_count'] for row in complete),
               'train_match_rate': (sum(row['train_match_count'] for row in complete) /
                                    sum(row['train_example_count'] for row in complete)) if complete else None,
               'heldout_matches': sum(row['heldout_match_count'] for row in complete),
               'heldout_inputs': sum(row['heldout_input_count'] for row in complete),
               'heldout_match_rate': (sum(row['heldout_match_count'] for row in complete) /
                                      sum(row['heldout_input_count'] for row in complete)) if complete else None,
               'budget': final_budget,
               'failure_categories': {key: sum(row.get('failure_category') == key for row in complete)
                                      for key in ('train_mismatch', 'execution_error', 'no_candidate')},
               'diagnostic_response_counts': {},
               'results_sha256': _sha(result_path)}
    for event in response_events:
        key = event.get('diagnostic_category') or event.get('error_kind') or 'unknown'
        summary['diagnostic_response_counts'][key] = summary['diagnostic_response_counts'].get(key, 0) + 1
    _write_json(output_dir / 'baseline_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8081/v1')
    parser.add_argument('--seed', type=int, default=4100)
    parser.add_argument('--prompt-variant', choices=('legacy', 'structured'),
                        default='structured')
    args = parser.parse_args()
    result = run(args.output_dir, base_url=args.base_url, seed=args.seed,
                 prompt_variant=args.prompt_variant)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
