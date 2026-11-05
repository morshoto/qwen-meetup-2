"""Budgeted E5 autonomous self-improvement pipeline.

Only inference, data isolation, verification, training, and experiment control
are implemented here. The model supplies every synthetic transformation and
every self-solved program.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

import yaml

from src.arc.agent import direct
from src.arc.data import Task, Verifier, load_arc, split_tasks
from src.arc.e5_selfplay import (SyntheticTask, generate_synthetic_task,
                                 make_sft_record, solve_task,
                                 verify_real_solution)
from src.arc.e5_plugins import FakePluginEvaluator, evaluate_and_store
from src.arc.guarded_contract import AUDITED_GENERATE_STEP
from src.arc.http_backend import (BackendError, OpenAICompatibleBackend,
                                  verify_guarded_contract)
from src.arc.mlx_token_driver import adapter_sha256
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.sandbox import DockerSandbox, FakeSandbox, Sandbox
from src.core.config import ExperimentConfig
from src.arc.mlx_lora import train_adapter


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _seal_mutable_file(path: Path) -> None:
    _json(path.with_name(path.name + '.sha256.json'), {'sha256': _sha(path)})


def _verify_mutable_file(path: Path) -> None:
    seal = path.with_name(path.name + '.sha256.json')
    if not path.is_file() or not seal.is_file():
        raise ValueError(f'E5 checkpoint or journal is missing its integrity seal: {path.name}')
    expected = json.loads(seal.read_text(encoding='utf-8')).get('sha256')
    if expected != _sha(path):
        raise ValueError(f'E5 checkpoint or journal hash mismatch: {path.name}')


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha(path)
            for path in sorted(root.rglob('*')) if path.is_file() and
            path.name not in {'artifact_hashes.json', '.run.lock'}}


def _seal_generation(root: Path) -> None:
    _json(root / 'artifact_hashes.json', _tree_hashes(root))


def _verify_generation(root: Path) -> None:
    manifest_path = root / 'artifact_hashes.json'
    if not manifest_path.is_file():
        raise ValueError(f'completed E5 generation lacks an artifact hash manifest: {root.name}')
    expected = json.loads(manifest_path.read_text(encoding='utf-8'))
    actual = _tree_hashes(root)
    if expected != actual:
        raise ValueError(f'E5 generation artifact hash mismatch: {root.name}')


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _bounded_window(task_ids: list[str], offset: int, count: int) -> list[str]:
    if type(offset) is not int or offset < 0 or type(count) is not int or count < 0:
        raise ValueError('invalid E5 task window')
    if offset + count > len(task_ids):
        raise ValueError('E5 task window exceeds its split')
    return task_ids[offset:offset + count]


def _run_path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve()
    if not target.is_relative_to(resolved_root):
        raise ValueError('E5 checkpoint or log path escaped its run directory')
    return target


class E5Budget:
    """Persistent pre-dispatch reservations for calls, tokens, containers, and training."""
    def __init__(self, path: Path, config: ExperimentConfig, *, resume: bool):
        self.path, self.config = path, config
        self.started = time.monotonic()
        if path.exists():
            _verify_mutable_file(path)
            self.state = json.loads(path.read_text(encoding='utf-8'))
        else:
            self.state = {'model_calls': 0, 'reserved_tokens': 0, 'sandbox_runs': 0,
                          'training_steps': 0, 'elapsed_seconds': 0.0, 'active_started_unix': None}
        if path.exists() and not resume:
            raise FileExistsError('E5 budget journal exists; use --resume to continue')
        if self.state.get('active_started_unix') is not None:
            downtime = time.time() - self.state['active_started_unix']
            if downtime < 0:
                raise RuntimeError('clock moved backwards; cannot safely resume E5 wall budget')
            self.state['elapsed_seconds'] += downtime
        self.elapsed_before = float(self.state.get('elapsed_seconds', 0.0))
        self.state['active_started_unix'] = time.time()
        self.save()

    def save(self) -> None:
        _json(self.path, self.state)
        _seal_mutable_file(self.path)

    def elapsed(self) -> float:
        return self.elapsed_before + time.monotonic() - self.started

    def remaining(self) -> float:
        remaining = self.config.e5.max_wall_seconds - self.elapsed()
        if remaining <= 0:
            raise TimeoutError('E5 wall-clock budget exhausted')
        return remaining

    def reserve_model(self, prompt: str, kind: str) -> None:
        self.remaining()
        messages = [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}]
        reserved = self.config.model.max_tokens + len(_canonical(messages).encode('utf-8')) + 1024
        if (self.state['model_calls'] + 1 > self.config.e5.max_model_calls or
                self.state['model_calls'] + 1 > self.config.budget.max_llm_calls or
                self.state['reserved_tokens'] + reserved > self.config.e5.max_reserved_tokens or
                self.state['reserved_tokens'] + reserved > self.config.budget.max_tokens):
            raise TimeoutError(f'E5 model-call or token budget exhausted before {kind}')
        self.state['model_calls'] += 1
        self.state['reserved_tokens'] += reserved
        self.save()

    def reserve_sandbox(self) -> None:
        self.remaining()
        if (self.state['sandbox_runs'] + 1 > self.config.e5.max_sandbox_runs or
                self.state['sandbox_runs'] + 1 > self.config.budget.max_container_starts):
            raise TimeoutError('E5 sandbox-run budget exhausted')
        self.state['sandbox_runs'] += 1
        self.save()

    def reserve_training(self, steps: int) -> None:
        self.remaining()
        if (steps <= 0 or self.state['training_steps'] + steps > self.config.e5.max_training_steps):
            raise TimeoutError('E5 training-step budget exhausted')
        self.state['training_steps'] += steps
        self.save()

    def finish(self) -> None:
        self.state['elapsed_seconds'] = self.elapsed()
        self.state['active_started_unix'] = None
        self.save()


class E5CachedBackend:
    """Cache complete responses and fail closed when a prior request is uncertain."""
    def __init__(self, backend: Any, budget: E5Budget, cache_path: Path):
        self.backend, self.budget, self.path = backend, budget, cache_path
        if cache_path.exists():
            _verify_mutable_file(cache_path)
        self.state = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {
            'responses': {}, 'response_kinds': {}, 'call_kinds': {}, 'pending': None}
        self.state.setdefault('response_kinds', {})
        self.state.setdefault('call_kinds', {})
        if self.state.get('pending') is not None:
            raise RuntimeError('uncertain prior E5 inference request; inspect the journal before retrying')
        self.scope = ''

    def save(self) -> None:
        _json(self.path, self.state)
        _seal_mutable_file(self.path)

    def response_count(self, kind: str) -> int:
        return sum(value == kind for value in self.state['response_kinds'].values())

    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        key = hashlib.sha256((self.scope + '\0' + kind + '\0' + str(candidate) + '\0' + prompt).encode()).hexdigest()
        if key in self.state['responses']:
            return self.state['responses'][key]
        self.budget.reserve_model(prompt, kind)
        self.state['call_kinds'][kind] = self.state['call_kinds'].get(kind, 0) + 1
        self.state['pending'] = {'key': key, 'kind': kind,
                                 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        self.save()
        response = self.backend.generate(prompt, kind=kind, candidate=candidate)
        self.state['responses'][key] = response
        self.state['response_kinds'][key] = kind
        self.state['pending'] = None
        self.save()
        return response


class MeteredSandbox:
    def __init__(self, sandbox: Sandbox, budget: E5Budget):
        self.sandbox, self.budget = sandbox, budget

    def run(self, code: str, grid: list[list[int]]) -> list[list[int]]:
        self.budget.reserve_sandbox()
        return self.sandbox.run(code, grid)


class FakeE5Backend:
    """Scripted protocol fixture. It is never used for Live evidence."""
    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        if kind == 'e5_generator':
            return 'def transform(grid):\n    return [row[:] for row in grid]'
        if kind == 'e5_self_solve' or kind == 'code':
            return 'def transform(grid):\n    return [list(row) for row in grid]'
        if kind == 'direct':
            try:
                payload = json.loads(prompt)
                return json.dumps([item['input'] for item in payload['task']['test']])
            except (KeyError, TypeError, ValueError):
                return '[]'
        if kind == 'e5_plugin':
            return 'def search_strategy(task_context):\n    return ""'
        return ''


class E5GuardedServer:
    def __init__(self, config: ExperimentConfig, root: Path, budget: E5Budget,
                 *, adapter: Path | None, label: str):
        self.config, self.root, self.budget = config, root, budget
        self.adapter, self.label = adapter, label
        self.port = self._free_port()
        self.process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.contract: dict[str, Any] | None = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            return int(sock.getsockname()[1])

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}/v1'

    def start(self) -> dict[str, Any]:
        training = self.config.p3b.training
        if not training.python_executable or not training.model_path:
            raise ValueError('E5 guarded MLX requires configured local Python and model paths')
        model = Path(training.model_path).expanduser().resolve()
        if model.name != self.config.model.revision or not model.is_dir():
            raise ValueError('E5 guarded MLX model path must be a local pinned revision directory')
        expected_adapter_hash = adapter_sha256(self.adapter) if self.adapter else 'none'
        token_limit = max(1, min(self.config.e5.max_reserved_tokens - self.budget.state['reserved_tokens'],
                                 self.config.budget.max_tokens - self.budget.state['reserved_tokens']))
        if token_limit <= 0:
            raise TimeoutError('E5 token budget exhausted before guarded server startup')
        checkpoint_root = _run_path(self.root, self.config.e5.checkpoint_dir)
        log_root = _run_path(self.root, self.config.e5.log_dir)
        ledger = checkpoint_root / f'{self.label}-ledger.sqlite'
        log = log_root / f'{self.label}-server.log'
        ledger.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [training.python_executable, '-m', 'src.cli.guarded_mlx_server',
                   '--model-dir', str(model), '--model-id', self.config.model.id,
                   '--revision', self.config.model.revision,
                   '--expected-mlx-lm-version', training.mlx_lm_version,
                   '--expected-generate-step-sha256', AUDITED_GENERATE_STEP[training.mlx_lm_version],
                   '--ledger', str(ledger), '--total-tokens', str(token_limit),
                   '--max-prompt-tokens', str(self.config.model.max_prompt_tokens),
                   '--max-new-tokens', str(self.config.model.max_tokens),
                   '--temperature', str(self.config.model.temperature),
                   '--thinking', self.config.model.thinking,
                   '--max-request-seconds', str(self.config.model.timeout_seconds),
                   '--port', str(self.port)]
        if self.adapter:
            command.extend(['--adapter-path', str(self.adapter),
                            '--expected-adapter-sha256', expected_adapter_hash])
        self.log_handle = log.open('w', encoding='utf-8')
        self.process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                        stdout=self.log_handle, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + min(self.budget.remaining(), self.config.model.timeout_seconds + 60, 180)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'E5 guarded server exited with status {self.process.returncode}; see {log}')
            try:
                self.contract = verify_guarded_contract(
                    self.base_url, self.config.model.id, self.config.model.revision,
                    self.config.e5.max_reserved_tokens, self.config.model.max_tokens,
                    min(self.config.model.timeout_seconds, self.budget.remaining()),
                    self.config.model.thinking, expected_adapter_sha256=expected_adapter_hash,
                    expected_max_prompt_tokens=self.config.model.max_prompt_tokens)
                if self.contract['reserved_tokens'] != 0:
                    raise BackendError('budget_unverified', 'fresh E5 server ledger is not empty')
                return self.contract
            except BackendError:
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f'E5 guarded server startup timed out; see {log}')

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_handle is not None:
            self.log_handle.close()


@contextmanager
def _run_lock(root: Path) -> Iterator[None]:
    with (root / '.run.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another E5 process is already using this run directory') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _backend(config: ExperimentConfig, root: Path, budget: E5Budget, cache_path: Path,
             *, adapter: Path | None, label: str) -> Iterator[E5CachedBackend]:
    if config.model.backend == 'fake':
        base = FakeE5Backend()
        yield E5CachedBackend(base, budget, cache_path)
        return
    server = E5GuardedServer(config, root, budget, adapter=adapter, label=label)
    try:
        contract = server.start()
        live_config = replace(config, model=replace(config.model, base_url=server.base_url,
                                                    adapter_sha256=contract['adapter_sha256']))
        backend = OpenAICompatibleBackend(
            model=live_config.model.id, revision=live_config.model.revision,
            base_url=live_config.model.base_url, temperature=live_config.model.temperature,
            max_tokens=live_config.model.max_tokens, seed=live_config.experiment.seed,
            timeout=live_config.model.timeout_seconds, thinking=live_config.model.thinking,
            allow_remote=False, log_path=_run_path(root, live_config.e5.log_dir) / 'llm_requests.jsonl',
            api_style='openai_compatible', guarded=True, remaining=budget.remaining,
            event_callback=lambda item: _append_event(_run_path(root, live_config.e5.log_dir) / 'llm_events.jsonl', item))
        yield E5CachedBackend(backend, budget, cache_path)
    finally:
        server.stop()


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n')


def _resolve(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _load_training_data(config: ExperimentConfig, project: Path) -> tuple[dict[str, Task], Verifier, dict[str, list[str]]]:
    challenge_path = _resolve(project, config.data.challenges)
    solution_path = _resolve(project, config.data.solutions)
    if config.e5.profile != 'cpu_fake':
        names = (challenge_path.name.lower(), solution_path.name.lower())
        if not all('training' in name for name in names) or any('evaluation' in name or 'test' in name for name in names):
            raise ValueError('Live E5 may read only ARC Training challenges and solutions')
    tasks, verifier = load_arc(challenge_path, solution_path)
    splits = split_tasks(list(tasks), config.data.split_seed,
                         {'exploration': config.data.exploration,
                          'promotion_gate': config.data.promotion_gate,
                          'monitoring': config.data.monitoring})
    return tasks, verifier, splits


def _docker_image_id(config: ExperimentConfig) -> str:
    if config.agent.sandbox == 'fake':
        return 'fake'
    docker = shutil.which('docker')
    if not docker:
        raise BackendError('sandbox_unavailable', 'Docker is unavailable; E5 refuses to run unisolated code')
    try:
        image_id = subprocess.check_output(
            [docker, 'image', 'inspect', config.agent.sandbox_image, '--format', '{{.Id}}'],
            text=True, stderr=subprocess.DEVNULL, timeout=20).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BackendError('sandbox_unavailable', 'configured E5 Docker image is unavailable locally') from exc
    if not image_id.startswith('sha256:'):
        raise ValueError('Docker did not return an immutable image ID')
    return image_id


def _identity(config: ExperimentConfig, project: Path, sandbox_image_id: str) -> dict[str, Any]:
    challenge_path = _resolve(project, config.data.challenges)
    solution_path = _resolve(project, config.data.solutions)
    implementation = ('src/core/config.py', 'src/arc/e5_selfplay.py', 'src/arc/sandbox.py',
                      'src/arc/agent.py', 'src/arc/mlx_lora.py', 'src/arc/mlx_token_driver.py',
                      'src/arc/guarded_contract.py', 'src/pipelines/e5.py')
    return {
        'protocol': 'e5_autonomous_selfplay_v1',
        'execution_mode': 'fake_fixture' if config.model.backend == 'fake' else 'live_model',
        'e5_profile': config.e5.profile,
        'config': asdict(config),
        'config_sha256': hashlib.sha256(_canonical(asdict(config)).encode()).hexdigest(),
        'data_sha256': {'challenges': _sha(challenge_path), 'solutions': _sha(solution_path)},
        'split': {'seed': config.data.split_seed, 'exploration': config.data.exploration,
                  'promotion_gate': config.data.promotion_gate, 'monitoring': config.data.monitoring},
        'sandbox_image_id': sandbox_image_id,
        'implementation_sha256': {name: _sha(project / name) for name in implementation},
        'e4_operator_search_used': False,
        'public_evaluation_accessed': False,
    }


def _grid_metrics(task_ids: list[str], scored: list[bool], errors: int,
                  task_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {'task_ids': task_ids, 'task_count': len(task_ids), 'scored_tasks': len(scored),
            'errors': errors, 'successes': sum(scored),
            'pass_at_1': (sum(scored) / len(scored)) if scored else None,
            'task_results': task_results}


def _eval_code(backend: E5CachedBackend, sandbox: Sandbox, tasks: dict[str, Task],
               verifier: Verifier, task_ids: list[str], *, attempts: int,
               generation: int, data_source: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    successes: list[bool] = []
    errors = 0
    records: list[dict[str, Any]] = []
    task_results = []
    for task_id in task_ids:
        task = tasks[task_id]
        result = solve_task(backend, sandbox, task, attempts=attempts)
        if result.error:
            errors += 1
            successes.append(False)
            task_results.append({'task_id': task_id, 'success': False,
                                 'error_kind': result.error_kind,
                                 'code_sha256': hashlib.sha256(result.code.encode()).hexdigest() if result.code else None,
                                 'outputs': result.outputs})
            continue
        success = verify_real_solution(task, result, verifier)
        successes.append(success)
        task_results.append({'task_id': task_id, 'success': success,
                             'code_sha256': hashlib.sha256(result.code.encode()).hexdigest() if result.code else None,
                             'outputs': result.outputs})
        if success and data_source:
            record = make_sft_record(task, result, data_source=data_source, generation=generation)
            if record:
                records.append(record)
    return _grid_metrics(task_ids, successes, errors, task_results), records


def _eval_direct(backend: E5CachedBackend, tasks: dict[str, Task], verifier: Verifier,
                 task_ids: list[str]) -> dict[str, Any]:
    successes: list[bool] = []
    errors = 0
    task_results = []
    for task_id in task_ids:
        result = direct(tasks[task_id], backend, candidate=0)
        if result.error:
            errors += 1
            successes.append(False)
            task_results.append({'task_id': task_id, 'success': False,
                                 'error_kind': result.error_kind, 'outputs': result.outputs})
        else:
            success = verifier.matches(task_id, result.outputs)
            successes.append(success)
            task_results.append({'task_id': task_id, 'success': success, 'outputs': result.outputs})
    return _grid_metrics(task_ids, successes, errors, task_results)


def _synthetic_task_artifact(root: Path, task: SyntheticTask) -> None:
    _json(root / 'task.json', {'task_id': task.task.id, 'task': task.task.prompt_data(),
                               'generator_sha256': task.generator_sha256,
                               'generator_ast_sha256': task.generator_ast_sha256})
    _json(root / 'private_verifier.json', {
        'hidden_test_inputs': task.task.test_inputs,
        'hidden_test_outputs': task.hidden_outputs,
        'validation_inputs': task.validation_inputs,
        'validation_outputs': task.validation_outputs,
        'generator_sha256': task.generator_sha256,
    })
    (root / 'generator.py').write_text(task.generator_code, encoding='utf-8')


def _replay_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    _verify_mutable_file(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(item, sort_keys=True, ensure_ascii=False) + '\n' for item in records),
                    encoding='utf-8')
    _seal_mutable_file(path)


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for record in records:
        if record.get('id') not in seen:
            seen.add(record['id'])
            output.append(record)
    return output


def _checked_adapter(root: Path, path_value: Any, expected_hash: str | None) -> Path | None:
    if path_value is None:
        if expected_hash not in (None, 'none'):
            raise ValueError('E5 adapter hash exists without an adapter checkpoint')
        return None
    adapter = Path(str(path_value)).expanduser().resolve()
    if not adapter.is_dir() or not adapter.is_relative_to(root.resolve()):
        raise ValueError('E5 adapter checkpoint is missing or outside the run directory')
    actual = adapter_sha256(adapter)
    if actual != expected_hash:
        raise ValueError('E5 adapter checkpoint hash mismatch')
    return adapter


def _training_batch(new_records: list[dict[str, Any]], replay: list[dict[str, Any]],
                    *, ratio: float, maximum: int, seed: int) -> list[dict[str, Any]]:
    new_records = _dedupe_records(new_records)
    current_ids = {item['id'] for item in new_records}
    eligible_replay = [item for item in _dedupe_records(replay) if item['id'] not in current_ids]
    if not new_records or ratio <= 0 or not eligible_replay:
        return new_records[:maximum]
    requested = len(eligible_replay) if ratio >= 1 else math.ceil(len(new_records) * ratio / (1 - ratio))
    rng = __import__('random').Random(seed)
    rng.shuffle(eligible_replay)
    return (new_records + eligible_replay[:requested])[:maximum]


def _promotion_decision(parent: dict[str, Any], candidate: dict[str, Any], *,
                        expected_tasks: int, minimum_delta: float,
                        rollback_on_regression: bool) -> dict[str, Any]:
    complete = all(item.get('scored_tasks') == expected_tasks and item.get('errors') == 0 and
                   isinstance(item.get('pass_at_1'), (int, float)) for item in (parent, candidate))
    old_score = parent.get('pass_at_1') if complete else None
    new_score = candidate.get('pass_at_1') if complete else None
    regressed = (not complete) or new_score < old_score
    promote = bool(complete and new_score >= old_score + minimum_delta and
                   (not rollback_on_regression or not regressed))
    return {'parent_pass_at_1': old_score, 'candidate_pass_at_1': new_score,
            'complete': complete, 'regressed_or_unverifiable': regressed,
            'decision': 'promote' if promote else 'rollback'}


def _seal_root(root: Path, identity: dict[str, Any]) -> None:
    files = {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob('*'))
             if path.is_file() and path.name not in {'root_artifact_hashes.json', '.run.lock'}}
    _json(root / 'root_artifact_hashes.json', {'identity_sha256': hashlib.sha256(_canonical(identity).encode()).hexdigest(),
                                                'files': files})


def _observed_token_usage(path: Path) -> dict[str, Any]:
    observed = 0
    known = 0
    if not path.is_file():
        return {'observed_tokens': None, 'calls_with_usage': 0}
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            usage = row.get('usage')
            total = usage.get('total_tokens') if isinstance(usage, dict) else None
            if type(total) is int and total >= 0:
                observed += total
                known += 1
        except (ValueError, TypeError):
            continue
    return {'observed_tokens': observed if known else None, 'calls_with_usage': known}


def _verify_root(root: Path, identity: dict[str, Any]) -> None:
    path = root / 'root_artifact_hashes.json'
    if not path.exists():
        return
    expected = json.loads(path.read_text(encoding='utf-8'))
    if expected.get('identity_sha256') != hashlib.sha256(_canonical(identity).encode()).hexdigest():
        raise ValueError('E5 root identity hash mismatch')
    actual = {str(item.relative_to(root)): _sha(item) for item in sorted(root.rglob('*'))
              if item.is_file() and item.name not in {'root_artifact_hashes.json', '.run.lock'}}
    if actual != expected.get('files'):
        raise ValueError('E5 root artifact hash mismatch')


def _train_generation(config: ExperimentConfig, budget: E5Budget, generation_root: Path,
                      records: list[dict[str, Any]], replay: list[dict[str, Any]],
                      active_adapter: Path | None, *, generation: int) -> tuple[dict[str, Any], Path | None]:
    unique_tasks = len({record['task_id'] for record in records})
    result: dict[str, Any] = {
        'status': 'not_scheduled', 'trained': False, 'weights_updated': False,
        'promoted': False, 'rolled_back': False, 'new_sft_records': len(records),
        'new_unique_verified_tasks': unique_tasks, 'adapter_sha256': None,
        'active_adapter_sha256': adapter_sha256(active_adapter) if active_adapter else 'none',
    }
    output = generation_root / 'training-run'
    recovery_path = output / 'training.json'
    if recovery_path.is_file():
        recovered = json.loads(recovery_path.read_text(encoding='utf-8'))
        if recovered.get('source') != 'e5_selfplay':
            raise ValueError('E5 training recovery artifact has unexpected source provenance')
        if recovered.get('status') == 'complete' and recovered.get('trained') is True and recovered.get('weights_updated') is True:
            adapter_path = _checked_adapter(generation_root, recovered.get('adapter_path'),
                                            recovered.get('adapter_sha256'))
            if adapter_path is None:
                raise ValueError('recovered E5 training artifact has no adapter checkpoint')
            actual_hash = adapter_sha256(adapter_path)
            if actual_hash != recovered.get('adapter_sha256'):
                raise ValueError('recovered E5 adapter hash does not match its training manifest')
            return recovered, adapter_path
        # A terminal failed/partial artifact is not silently retried on resume.
        recovered.update(promoted=False, rolled_back=False,
                         previous_adapter_preserved=True, recovered_from_checkpoint=True)
        return recovered, active_adapter
    if unique_tasks < max(config.e5.minimum_successful_tasks,
                          config.p3b.insufficient_data.min_verified_samples,
                          config.p3b.insufficient_data.min_verified_unique_tasks):
        result.update(status='skipped_insufficient_verified_tasks',
                      minimum_successful_tasks=config.e5.minimum_successful_tasks,
                      previous_adapter_preserved=True)
        return result, active_adapter
    if config.rsi.trainer != 'mlx_lora' or config.p3b.training.backend != 'mlx_lora':
        result.update(status='skipped_training_disabled', previous_adapter_preserved=True)
        return result, active_adapter
    training_settings = config.p3b.training
    remaining_steps = config.e5.max_training_steps - budget.state['training_steps']
    steps = min(training_settings.max_steps, remaining_steps)
    if steps <= 0:
        result.update(status='skipped_training_step_budget', previous_adapter_preserved=True)
        return result, active_adapter
    replay_ratio = config.e5.replay_ratio if training_settings.replay_buffer_enabled else 0.0
    batch = _training_batch(records, replay, ratio=replay_ratio,
                            maximum=config.e5.replay_max_records,
                            seed=config.experiment.seed + generation)
    if not batch:
        result.update(status='skipped_empty_training_batch', previous_adapter_preserved=True)
        return result, active_adapter
    budget.reserve_training(steps)
    timeout = int(min(config.e5.max_training_seconds, training_settings.timeout_seconds,
                      max(1, budget.remaining())))
    parameters = {
        'batch_size': training_settings.batch_size,
        'learning_rate': training_settings.learning_rate,
        'max_steps': steps,
        'lora_rank': training_settings.lora_rank,
        'lora_alpha': training_settings.lora_alpha,
        'max_seq_length': training_settings.max_seq_length,
        'num_layers': training_settings.num_layers,
    }
    previous = active_adapter if config.e5.continue_previous_adapter else None
    try:
        trained = train_adapter(batch, model_path=training_settings.model_path,
                                python_executable=training_settings.python_executable,
                                output=output, parameters=parameters,
                                seed=config.experiment.seed + generation,
                                timeout_seconds=timeout, source='e5_selfplay',
                                expected_mlx_lm_version=training_settings.mlx_lm_version,
                                thinking_mode=config.model.thinking, previous_adapter=previous)
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      training_steps_reserved=steps, previous_adapter_preserved=True)
        return result, active_adapter
    result.update(trained)
    result['training_steps_reserved'] = steps
    result['trained'] = trained.get('trained') is True
    result['weights_updated'] = trained.get('weights_updated') is True
    result['promoted'] = False
    result['rolled_back'] = False
    result['new_sft_records'] = len(records)
    result['training_record_count'] = len(batch)
    result['new_unique_verified_tasks'] = unique_tasks
    if result['trained'] and result['weights_updated']:
        adapter_path = Path(str(trained.get('adapter_path', ''))).expanduser().resolve()
        actual_hash = adapter_sha256(adapter_path)
        if actual_hash != trained.get('adapter_sha256'):
            raise ValueError('E5 trained adapter hash does not match its manifest')
        result['adapter_path'] = str(adapter_path)
        result['adapter_sha256'] = actual_hash
        return result, adapter_path
    result['previous_adapter_preserved'] = True
    return result, active_adapter


def _promotion_eval(config: ExperimentConfig, root: Path, budget: E5Budget,
                    requests_path: Path, tasks: dict[str, Task], verifier: Verifier,
                    task_ids: list[str], adapter: Path | None, *, label: str,
                    sandbox_image_id: str = '') -> dict[str, Any]:
    if config.agent.sandbox == 'fake':
        base_sandbox: Sandbox = FakeSandbox()
    else:
        base_sandbox = DockerSandbox(
            image=sandbox_image_id or config.agent.sandbox_image, timeout=config.agent.timeout_seconds,
            memory_mb=config.agent.memory_mb, cpus=config.agent.cpus, pids=config.agent.pids,
            max_output_bytes=config.agent.max_output_bytes,
            max_code_bytes=config.agent.max_code_bytes, remaining=budget.remaining)
    sandbox = MeteredSandbox(base_sandbox, budget)
    with _backend(config, root, budget, requests_path, adapter=adapter, label=label) as backend:
        backend.scope = label
        metric, _ = _eval_code(backend, sandbox, tasks, verifier, task_ids,
                               attempts=config.e5.self_solve_attempts, generation=-1)
    return metric


def run_e5(config: ExperimentConfig, *, resume: bool = False,
           run_dir: str | Path | None = None, live: bool = False) -> Path:
    """Run Gen 0 through the configured final generation with finite budgets."""
    if not config.e5.enabled:
        raise ValueError('E5 is disabled in this configuration')
    if config.e5.profile == 'cpu_fake' and (live or config.model.backend != 'fake'):
        raise ValueError('CPU Fake E5 requires the fake backend and does not accept --live')
    if config.e5.profile != 'cpu_fake' and not live:
        raise ValueError('Live E5 execution requires explicit --live')
    if config.e5.profile != 'cpu_fake' and config.model.adapter_sha256 != 'none':
        raise ValueError('E5 starts from the pinned base model; initial adapters are not accepted')
    project = Path(__file__).resolve().parents[2]
    root = (Path(run_dir).expanduser().resolve() if run_dir else
            _resolve(project, config.tracking.output_dir) / 'e5' /
            f'{config.experiment.name}-{uuid.uuid4().hex[:10]}')
    exists = root.exists()
    if exists and not resume and any(root.iterdir()):
        raise FileExistsError(f'E5 run directory already exists: {root}')
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        sandbox_image_id = _docker_image_id(config)
        identity = _identity(config, project, sandbox_image_id)
        identity_path = root / 'identity.json'
        if identity_path.exists():
            existing = json.loads(identity_path.read_text(encoding='utf-8'))
            if existing != identity:
                raise ValueError('E5 resume config, data, split, or implementation fingerprint mismatch')
        elif resume:
            raise ValueError('cannot resume E5: identity.json is missing')
        else:
            _json(identity_path, identity)
            (root / 'config.snapshot.yml').write_text(
                yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True), encoding='utf-8')
        _verify_root(root, identity)
        state_path = root / 'run_state.json'
        if state_path.is_file():
            completed_state = json.loads(state_path.read_text(encoding='utf-8'))
            if completed_state.get('status') == 'complete':
                if not (root / 'root_artifact_hashes.json').is_file():
                    raise ValueError('completed E5 run is missing its root artifact hash manifest')
                return root

        tasks, verifier, splits = _load_training_data(config, project)
        total_real_tasks = config.e5.real_arc_tasks_per_generation * (config.e5.generations + 1)
        if config.e5.real_arc_tasks_per_generation and (
                config.rsi.exploration_offset + total_real_tasks > len(splits['exploration'])):
            raise ValueError('E5 disjoint per-generation Exploration windows exceed the Exploration split')
        if config.e5.promotion_gate_tasks * config.e5.generations > len(splits['promotion_gate']):
            raise ValueError('E5 disjoint per-generation Promotion Gate windows exceed the split')
        monitoring_ids = _bounded_window(splits['monitoring'], 0, config.e5.monitoring_tasks)
        budget = E5Budget(root / 'budget.json', config, resume=resume)
        checkpoint_root = _run_path(root, config.e5.checkpoint_dir)
        log_root = _run_path(root, config.e5.log_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        requests_path = checkpoint_root / 'request_cache.json'
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
        else:
            state = {'status': 'running', 'generations_completed': [], 'active_adapter_path': None,
                     'active_adapter_sha256': 'none', 'trained': False, 'weights_updated': False,
                     'promoted': False, 'rolled_back': False, 'training_elapsed_seconds': 0.0}
        active_adapter = _checked_adapter(root, state.get('active_adapter_path'),
                                          state.get('active_adapter_sha256', 'none'))
        replay_path = checkpoint_root / 'replay_buffer.jsonl'
        if replay_path.exists():
            _replay_records(replay_path)  # Parse now so corrupt checkpoints fail before model calls.

        sandbox_base: Sandbox
        if config.agent.sandbox == 'fake':
            sandbox_base = FakeSandbox()
        else:
            sandbox_base = DockerSandbox(
                image=sandbox_image_id, timeout=config.agent.timeout_seconds,
                memory_mb=config.agent.memory_mb, cpus=config.agent.cpus, pids=config.agent.pids,
                max_output_bytes=config.agent.max_output_bytes,
                max_code_bytes=config.agent.max_code_bytes, remaining=budget.remaining)
        sandbox = MeteredSandbox(sandbox_base, budget)

        # Verify every prior completed generation before trusting its adapter path.
        for generation in range(config.e5.generations + 1):
            window_start = config.rsi.exploration_offset + generation * config.e5.real_arc_tasks_per_generation
            real_ids = _bounded_window(splits['exploration'], window_start,
                                       config.e5.real_arc_tasks_per_generation)
            if generation < config.e5.generations:
                gate_start = generation * config.e5.promotion_gate_tasks
                gate_ids = _bounded_window(splits['promotion_gate'], gate_start,
                                           config.e5.promotion_gate_tasks)
            else:
                # The terminal generation does not train or promote, so it must
                # not reserve a Promotion Gate window.
                gate_ids = []
            generation_root = root / f'gen-{generation}'
            generation_result_path = generation_root / 'generation.json'
            if generation_result_path.is_file():
                result = json.loads(generation_result_path.read_text(encoding='utf-8'))
                if result.get('status') == 'complete':
                    _verify_generation(generation_root)
                    if generation not in state['generations_completed']:
                        state['generations_completed'].append(generation)
                    if result.get('active_adapter_path'):
                        active_adapter = _checked_adapter(root, result['active_adapter_path'],
                                                          result.get('active_adapter_sha256'))
                    continue
            generation_root.mkdir(parents=True, exist_ok=True)
            state.update(status='running', phase=f'generation_{generation}',
                         active_adapter_path=str(active_adapter) if active_adapter else None,
                         active_adapter_sha256=adapter_sha256(active_adapter) if active_adapter else 'none')
            _json(state_path, state)

            previous_replay = _replay_records(replay_path)
            generation_records: list[dict[str, Any]] = []
            synthetic_generated = 0
            synthetic_valid = 0
            synthetic_self_solved = 0
            synthetic_sft = 0
            synthetic_rejected = 0
            generator_rejections: list[dict[str, Any]] = []
            arc_metrics: dict[str, Any] = {}
            c2_result: dict[str, Any] | None = None
            monitoring_metrics: dict[str, Any] = {}

            # B is the immutable base condition and is measured once on Gen 0.
            # Gen 0 B and C1 use the same fixed agent, prompts, limits, and task batch.
            adapter_for_generation = active_adapter
            use_base_server = generation == 0 or adapter_for_generation is None
            backend_adapter = None if use_base_server else adapter_for_generation
            with _backend(config, root, budget, requests_path, adapter=backend_adapter,
                          label=f'gen-{generation}-c1') as backend:
                if generation == 0 and 'A' in config.e5.conditions and real_ids:
                    backend.scope = 'gen-0:A:real-arc'
                    arc_metrics['A'] = _eval_direct(backend, tasks, verifier, real_ids)
                if generation == 0 and 'B' in config.e5.conditions and real_ids:
                    backend.scope = 'gen-0:B:real-arc'
                    arc_metrics['B'] = _eval_code(backend, sandbox, tasks, verifier, real_ids,
                                                   attempts=config.e5.self_solve_attempts,
                                                   generation=0)[0]
                if 'C1' in config.e5.conditions and real_ids:
                    backend.scope = f'gen-{generation}:C1:real-arc'
                    c1_metric, c1_records = _eval_code(
                        backend, sandbox, tasks, verifier, real_ids,
                        attempts=config.e5.self_solve_attempts, generation=generation,
                        data_source='self_solved_real_arc')
                    arc_metrics['C1'] = c1_metric
                    generation_records.extend(c1_records)

                if 'C1' in config.e5.conditions:
                    synthetic_root = generation_root / 'synthetic'
                    generator_responses_before = backend.response_count('e5_generator')
                    seen_synthetic_ids: set[str] = set()
                    for prior_generation in range(generation):
                        prior_root = root / f'gen-{prior_generation}'
                        prior_state_path = prior_root / 'generation.json'
                        if (not prior_state_path.is_file() or
                                json.loads(prior_state_path.read_text(encoding='utf-8')).get('status') != 'complete'):
                            continue
                        for task_file in (prior_root / 'synthetic').glob('*/task.json'):
                            seen_synthetic_ids.add(json.loads(task_file.read_text(encoding='utf-8')).get('task_id'))
                    for index in range(config.e5.synthetic_tasks_per_generation):
                        backend.scope = f'gen-{generation}:C1:generator-{index}'
                        synthetic_task = generate_synthetic_task(
                            backend, sandbox,
                            seed=config.experiment.seed + generation * 100003 + index,
                            generation=generation, index=index,
                            generator_candidates=config.e5.generator_candidates,
                            train_examples=config.e5.train_examples,
                            hidden_test_inputs=config.e5.hidden_test_inputs,
                            validation_inputs=config.e5.generator_validation_inputs,
                            rejection_log=generator_rejections)
                        if synthetic_task is None:
                            synthetic_rejected += 1
                            continue
                        if synthetic_task.task.id in seen_synthetic_ids:
                            synthetic_rejected += 1
                            generator_rejections.append({'task_index': index, 'reason': 'duplicate_synthetic_task',
                                                         'task_id': synthetic_task.task.id})
                            continue
                        seen_synthetic_ids.add(synthetic_task.task.id)
                        synthetic_valid += 1
                        task_root = synthetic_root / f'{index:03d}-{synthetic_task.task.id}'
                        task_root.mkdir(parents=True, exist_ok=True)
                        _synthetic_task_artifact(task_root, synthetic_task)
                        backend.scope = f'gen-{generation}:C1:self-solve-{synthetic_task.task.id}'
                        solved = solve_task(
                            backend, sandbox, synthetic_task.task,
                            attempts=config.e5.self_solve_attempts,
                            generator_ast_sha256=synthetic_task.generator_ast_sha256,
                            hidden_outputs=synthetic_task.hidden_outputs,
                            validation_inputs=synthetic_task.validation_inputs,
                            validation_outputs=synthetic_task.validation_outputs)
                        if not solved.error and solved.code:
                            synthetic_self_solved += 1
                            record = make_sft_record(
                                synthetic_task.task, solved,
                                data_source='self_solved_synthetic', generation=generation,
                                generator_sha256=synthetic_task.generator_sha256)
                            if record:
                                generation_records.append(record)
                                synthetic_sft += 1
                        _json(task_root / 'self_solve.json', {
                            'status': 'verified' if not solved.error and solved.code else 'rejected',
                            'attempts': len(solved.history), 'solution_code_sha256':
                            hashlib.sha256(solved.code.encode()).hexdigest() if solved.code else None,
                            'history': solved.history,
                        })
                    synthetic_generated = backend.response_count('e5_generator') - generator_responses_before

                if 'C2' in config.e5.conditions:
                    backend.scope = f'gen-{generation}:C2:plugin-proposal'
                    plugin_source = backend.generate(
                        'Implement one function search_strategy(task_context) -> str. It may return a short '
                        'prompt addition for a fixed code agent. Do not access files, imports, network, model '
                        'settings, labels, evaluation, budgets, or training. Return Python source only.',
                        kind='e5_plugin', candidate=0)
                    c2_result = asdict(evaluate_and_store(
                        generation_root, f'proposal_{generation}', plugin_source,
                        FakePluginEvaluator()))
                    c2_result['used_by_c1'] = False

            if generation in config.e5.monitoring_generations and monitoring_ids:
                monitoring_metrics['B'] = _promotion_eval(
                    config, root, budget, requests_path, tasks, verifier, monitoring_ids,
                    None, label=f'gen-{generation}-monitor-b', sandbox_image_id=sandbox_image_id)
                monitoring_metrics['C1'] = _promotion_eval(
                    config, root, budget, requests_path, tasks, verifier, monitoring_ids,
                    active_adapter, label=f'gen-{generation}-monitor-c1', sandbox_image_id=sandbox_image_id)

            generation_records = _dedupe_records(generation_records)
            replay_updated = _dedupe_records(previous_replay + generation_records)[-config.e5.replay_max_records:]
            _write_records(replay_path, replay_updated)
            training_result = {'status': 'final_generation_no_update', 'trained': False,
                               'weights_updated': False, 'promoted': False, 'rolled_back': False,
                               'new_sft_records': len(generation_records),
                               'new_unique_verified_tasks': len({record['task_id'] for record in generation_records}),
                               'active_adapter_sha256': adapter_sha256(active_adapter) if active_adapter else 'none'}
            promotion_result: dict[str, Any] = {'decision': 'not_evaluated'}
            next_active_adapter = active_adapter

            if generation < config.e5.generations:
                training_result, candidate_adapter = _train_generation(
                    config, budget, generation_root, generation_records, previous_replay,
                    active_adapter, generation=generation)
                if training_result.get('trained') and training_result.get('weights_updated') and candidate_adapter:
                    if not gate_ids:
                        promotion_result = {'decision': 'rollback', 'reason': 'Promotion Gate is empty',
                                            'complete': False}
                    else:
                        parent_metric = _promotion_eval(
                            config, root, budget, requests_path, tasks, verifier, gate_ids,
                            active_adapter, label=f'gen-{generation}-gate-parent', sandbox_image_id=sandbox_image_id)
                        candidate_metric = _promotion_eval(
                            config, root, budget, requests_path, tasks, verifier, gate_ids,
                            candidate_adapter, label=f'gen-{generation}-gate-candidate', sandbox_image_id=sandbox_image_id)
                        promotion_result = _promotion_decision(
                            parent_metric, candidate_metric, expected_tasks=len(gate_ids),
                            minimum_delta=config.e5.promotion_min_delta,
                            rollback_on_regression=config.e5.rollback_on_regression)
                        promotion_result.update(parent=parent_metric, candidate=candidate_metric,
                                                task_ids=gate_ids,
                                                adapter_only_difference=True)
                    promoted = promotion_result.get('decision') == 'promote'
                    training_result['promoted'] = promoted
                    training_result['rolled_back'] = not promoted
                    training_result['promotion'] = promotion_result
                    if promoted:
                        next_active_adapter = candidate_adapter
                else:
                    training_result['promoted'] = False
                    training_result['rolled_back'] = False
                    training_result['promotion'] = {'decision': 'not_evaluated'}

            generation_result = {
                'status': 'complete', 'generation': generation,
                'execution_mode': identity['execution_mode'],
                'conditions': config.e5.conditions,
                'arc_metrics': arc_metrics,
                'monitoring_metrics': monitoring_metrics,
                'synthetic': {'generation_candidates': synthetic_generated,
                              'generated_tasks': synthetic_valid,
                              'valid_tasks': synthetic_valid,
                              'self_solved_tasks': synthetic_self_solved,
                              'sft_records': synthetic_sft,
                              'rejected_generators': synthetic_rejected,
                              'generator_rejections': generator_rejections},
                'sft': {'new_records': len(generation_records),
                        'unique_verified_tasks': len({record['task_id'] for record in generation_records}),
                        'sources': {source: sum(item['data_source'] == source for item in generation_records)
                                    for source in ('self_solved_real_arc', 'self_solved_synthetic')}},
                'training': training_result,
                'promotion': promotion_result,
                'c2': c2_result,
                'active_adapter_path': str(next_active_adapter) if next_active_adapter else None,
                'active_adapter_sha256': adapter_sha256(next_active_adapter) if next_active_adapter else 'none',
                'budget': {key: value for key, value in budget.state.items()
                           if key != 'active_started_unix'},
            }
            _json(generation_root / 'generation.json', generation_result)
            _seal_generation(generation_root)
            state['generations_completed'] = sorted(set(state.get('generations_completed', []) + [generation]))
            state.update(status='running', phase='generation_complete',
                         active_adapter_path=str(next_active_adapter) if next_active_adapter else None,
                         active_adapter_sha256=generation_result['active_adapter_sha256'],
                         trained=bool(state.get('trained') or training_result.get('trained') is True),
                         weights_updated=bool(state.get('weights_updated') or training_result.get('weights_updated') is True),
                         promoted=bool(state.get('promoted') or training_result.get('promoted') is True),
                         rolled_back=bool(state.get('rolled_back') or training_result.get('rolled_back') is True),
                         training_elapsed_seconds=float(state.get('training_elapsed_seconds', 0.0)) +
                         float(training_result.get('elapsed_seconds') or 0.0),
                         last_generation=generation)
            active_adapter = next_active_adapter
            _json(state_path, state)

        state.update(status='complete', phase='complete',
                     active_adapter_path=str(active_adapter) if active_adapter else None,
                     active_adapter_sha256=adapter_sha256(active_adapter) if active_adapter else 'none',
                     budget={key: value for key, value in budget.state.items()
                             if key != 'active_started_unix'})
        budget.finish()
        state['budget'] = {key: value for key, value in budget.state.items()
                           if key != 'active_started_unix'}
        state['budget']['token_usage'] = _observed_token_usage(
            _run_path(root, config.e5.log_dir) / 'llm_requests.jsonl')
        _json(state_path, state)
        _seal_root(root, identity)
    return root


