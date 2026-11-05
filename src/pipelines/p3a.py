"""No-training P3-A execution with durable request reservations and results."""
from __future__ import annotations
import csv
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any
import yaml
from src.arc.agent import direct, code_agent
from src.arc.guarded_contract import SOURCE_FILES
from src.arc.data import exploration_window, load_arc, split_tasks
from src.arc.http_backend import OpenAICompatibleBackend, BackendError, verify_ollama_revision, verify_guarded_contract
from src.arc.sandbox import DockerSandbox
from src.arc.scoring import score
from src.arc.task_manifests import load_diagnostic_task_ids
from src.core.config import ExperimentConfig


def write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(obj, sort_keys=True, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    if os.name != 'nt':
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_config(config: ExperimentConfig) -> dict[str, Any]:
    """Keep legacy P3-A identities stable when E2/E3 search is not enabled."""
    value = asdict(config)
    if not config.search.enabled:
        value.pop('search', None)
    return value


@contextmanager
def run_lock(run_dir: Path):
    """Hold a nonblocking OS lock across reservation, dispatch and commit."""
    with (run_dir / '.run.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('another process is already executing this run') from exc
            try:
                yield
            finally:
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('another process is already executing this run') from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class BudgetExceeded(Exception): pass


class Journal:
    def __init__(self, path: Path, config: ExperimentConfig):
        self.path, self.config, self.start = path, config, time.monotonic()
        self.state = json.loads(path.read_text()) if path.exists() else {'calls': 0, 'reserved_tokens': 0, 'candidates': 0, 'containers': 0, 'pending': None, 'cache': {}, 'candidate_keys': [], 'result_hashes': {}, 'elapsed_seconds': 0.0}
        self.elapsed_before = self.state.get('elapsed_seconds', 0.0)
        if self.state.get('active_started_unix') is not None:
            elapsed_since_start = time.time() - self.state['active_started_unix']
            if elapsed_since_start < 0:
                raise RuntimeError('clock moved backwards; cannot safely resume time budget')
            self.elapsed_before += elapsed_since_start
            self.state['elapsed_seconds'] = self.elapsed_before
            self.state['active_started_unix'] = None
            self.save()
        if self.state.get('fatal_error') is not None:
            raise RuntimeError('run halted because server token usage cannot be trusted')
        if self.state['pending'] is not None:
            raise RuntimeError('uncertain prior inference request: inspect journal before any retry')
        self.state['active_started_unix'] = time.time()
        self.save()
    def save(self): write(self.path, self.state)
    def remaining(self):
        left = self.config.budget.max_seconds - self.elapsed_before - (time.monotonic() - self.start)
        if left <= 0: raise BudgetExceeded('wall time budget exhausted')
        return left
    def reserve(self, field: str, amount: int, limit: int):
        self.remaining()
        if self.state[field] + amount > limit: raise BudgetExceeded(f'{field} budget exhausted')
        self.state[field] += amount; self.save()
    def llm(self, tokens: int):
        self.remaining()
        b = self.config.budget
        if self.state['calls'] + 1 > b.max_llm_calls or self.state['reserved_tokens'] + tokens > b.max_tokens:
            raise BudgetExceeded('LLM or token budget exhausted')
        self.state['calls'] += 1; self.state['reserved_tokens'] += tokens; self.save()
    def container(self): self.reserve('containers', 1, self.config.budget.max_container_starts)


class CachedBackend:
    def __init__(self, backend: OpenAICompatibleBackend, journal: Journal):
        self.backend, self.journal, self.scope = backend, journal, ''
    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        key = hashlib.sha256((self.scope + '\0' + kind + '\0' + str(candidate) + '\0' + prompt).encode()).hexdigest()
        state = self.journal.state
        if key in state['cache']: return state['cache'][key]
        self.backend.timeout = min(self.backend.timeout, self.journal.remaining())
        state['pending'] = key; self.journal.save()
        try:
            result = self.backend.generate(prompt, kind=kind, candidate=candidate)
        except BudgetExceeded:
            state['pending'] = None; self.journal.save(); raise
        except BackendError as exc:
            if exc.kind != 'connection_error':
                state['pending'] = None; self.journal.save()
            raise
        except Exception:
            # An attempted request can have reached the server; never repeat automatically.
            raise
        state['cache'][key] = result; state['pending'] = None; self.journal.save()
        return result


def run_real(config: ExperimentConfig, *, resume: bool = False, run_dir: Path | None = None) -> Path:
    if config.rsi.trainer != 'dry_run':
        raise ValueError('P3-A inference cannot run with a training-enabled configuration')
    if config.model.backend in {'openai_compatible', 'ollama_native'} and config.budget.token_enforcement != 'best_effort':
        raise ValueError('external server requires explicit best_effort token policy')
    if config.model.backend == 'guarded_mlx' and config.budget.token_enforcement != 'strict':
        raise ValueError('guarded MLX requires strict token policy')
    # Preflight is read-only and no HTTP inference occurs before the run exists.
    tasks, verifier = load_arc(config.data.challenges, config.data.solutions)
    splits = split_tasks(list(tasks), config.data.split_seed, {x: getattr(config.data, x) for x in ('exploration', 'promotion_gate', 'monitoring')})
    if config.p3c.enabled:
        if (config.experiment.conditions != ['B'] or config.experiment.generations != 0 or
                config.experiment.evaluation_split != 'exploration' or
                not set(config.p3c.task_ids) <= set(splits['exploration'])):
            raise ValueError('P3-C requires a B-only Gen 0 fixed batch from the Exploration split')
        diagnostic_ids, manifest_sha = load_diagnostic_task_ids(config.p3c.diagnostic_task_manifest)
        if not set(diagnostic_ids) <= set(splits['exploration']):
            raise ValueError('P3-C diagnostic manifest references a task outside Exploration')
        if set(config.p3c.task_ids) & set(diagnostic_ids):
            raise ValueError('P3-C task batch contains a previously used diagnostic task')
        selected_task_ids = list(config.p3c.task_ids)
    else:
        selected_task_ids = exploration_window(splits, config.rsi.exploration_offset,
                                               config.rsi.exploration_per_generation)
        manifest_sha = None
    identity = {'config': _identity_config(config), 'data_sha256': {'challenges': digest(Path(config.data.challenges)), 'solutions': digest(Path(config.data.solutions))}, 'splits': splits,
                'p3c_diagnostic_task_manifest_sha256': manifest_sha,
                'implementation_sha256': {name: digest(Path(name)) for name in (('src/pipelines/p3a.py', 'src/core/config.py', 'src/arc/agent.py', 'src/arc/data.py', 'src/arc/task_manifests.py', 'src/arc/http_backend.py', 'src/arc/sandbox.py', 'src/arc/scoring.py', 'sandbox/runner.py', 'uv.lock') + (SOURCE_FILES if config.model.backend == 'guarded_mlx' else ()))}}
    if resume:
        if run_dir is None or not (run_dir/'identity.json').is_file() or json.loads((run_dir/'identity.json').read_text()) != identity:
            raise ValueError('resume config or data differs from original run')
    else:
        if run_dir is not None: raise ValueError('run-dir is only valid with resume')
        run_dir = Path(config.tracking.output_dir)/(config.experiment.name + '-' + uuid.uuid4().hex[:12]); run_dir.mkdir(parents=True)
        write(run_dir/'identity.json', identity); write(run_dir/'data_split.json', splits)
        (run_dir/'config.snapshot.yml').write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        def sysctl(key):
            try: return subprocess.check_output(['sysctl', '-n', key], text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError): return None
        try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError): commit = None
        write(run_dir/'environment.json', {'python': platform.python_version(), 'platform': platform.platform(), 'machine': platform.machine(), 'cpu': sysctl('machdep.cpu.brand_string'), 'memory_bytes': sysctl('hw.memsize'), 'git_commit': commit, 'model_id': config.model.id, 'revision': config.model.revision, 'adapter_sha256': config.model.adapter_sha256, 'revision_verification': 'user_supplied_unverified', 'quantization': config.model.quantization, 'pyyaml_version': yaml.__version__, 'backend': config.model.backend, 'seed': config.experiment.seed, 'data_split_sha256': digest(run_dir/'data_split.json'), 'token_budget_guarantee': 'unverified_server_tokenizer', 'trained': False, 'weights_updated': False})
    assert run_dir is not None
    with run_lock(run_dir):
        state_path = run_dir/'run_state.json'
        if resume and state_path.is_file() and json.loads(state_path.read_text()).get('status') == 'complete':
            saved = json.loads((run_dir/'budget.json').read_text())
            if saved.get('pending') is not None:
                raise ValueError('completed run has an uncertain request')
            hashes = saved.get('result_hashes', {})
            count = len(config.experiment.conditions) * config.rsi.exploration_per_generation * config.agent.candidates
            if not isinstance(hashes, dict) or len(hashes) != count:
                raise ValueError('completed candidate manifest is inconsistent')
            for name, expected_hash in hashes.items():
                if Path(name).name != name or not (run_dir/'predictions'/name).is_file() or digest(run_dir/'predictions'/name) != expected_hash:
                    raise ValueError('candidate prediction integrity check failed')
            aggregate = saved.get('aggregate_hashes')
            if not isinstance(aggregate, dict) or set(aggregate) != {'results.json', 'metrics.csv'}:
                raise ValueError('completed aggregate manifest missing')
            for name, expected_hash in aggregate.items():
                if not (run_dir/name).is_file() or digest(run_dir/name) != expected_hash:
                    raise ValueError('aggregate integrity check failed')
            return run_dir
        journal = Journal(run_dir/'budget.json', config)
        backend = OpenAICompatibleBackend(model=config.model.id, revision=config.model.revision, base_url=config.model.base_url,
            temperature=config.model.temperature, max_tokens=config.model.max_tokens, seed=config.experiment.seed,
            timeout=config.model.timeout_seconds, thinking=config.model.thinking, allow_remote=config.model.allow_remote,
            log_path=run_dir/'llm_requests.jsonl', reserve=journal.llm, api_style='openai_compatible' if config.model.backend == 'guarded_mlx' else config.model.backend, guarded=config.model.backend == 'guarded_mlx', remaining=journal.remaining)
        cached = CachedBackend(backend, journal)
        a = config.agent
        sandbox = DockerSandbox(a.sandbox_image, timeout=a.timeout_seconds, memory_mb=a.memory_mb, cpus=a.cpus, pids=a.pids,
                                max_output_bytes=a.max_output_bytes, max_code_bytes=a.max_code_bytes, reserve=journal.container, remaining=journal.remaining)
        pred_dir = run_dir/'predictions'; pred_dir.mkdir(exist_ok=True)
        try:
            inspection = subprocess.run(['docker', 'image', 'inspect', a.sandbox_image, '--format', '{{.Id}}'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=min(10, journal.remaining()), text=True)
            image_id = inspection.stdout.strip()
            if not image_id: raise RuntimeError('Docker image identity unavailable')
            environment = json.loads((run_dir/'environment.json').read_text())
            if resume and environment.get('sandbox_image_id') != image_id:
                raise ValueError('sandbox image changed since the run began')
            environment['sandbox_image_id'] = image_id
            try:
                environment['docker_client_version'] = subprocess.check_output(['docker', '--version'], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                environment['docker_client_version'] = None
            write(run_dir/'environment.json', environment)
            sandbox.image = image_id
            if config.model.backend == 'ollama_native':
                try:
                    verify_ollama_revision(config.model.base_url, config.model.id, config.model.revision, min(config.model.timeout_seconds, journal.remaining()))
                except BackendError as exc:
                    journal.state['elapsed_seconds'] = journal.elapsed_before + time.monotonic() - journal.start
                    journal.state['active_started_unix'] = None; journal.save()
                    write(run_dir/'run_state.json', {'status': 'model_unavailable', 'trained': False, 'weights_updated': False, 'error': str(exc)})
                    return run_dir
                environment['revision_verification'] = 'server_reported_digest'
                write(run_dir/'environment.json', environment)
            if config.model.backend == 'guarded_mlx':
                try:
                    contract_args = (config.model.base_url, config.model.id, config.model.revision,
                                     config.budget.max_tokens, config.model.max_tokens,
                                     min(config.model.timeout_seconds, journal.remaining()), config.model.thinking)
                    if config.model.adapter_sha256 == 'none':
                        contract = verify_guarded_contract(*contract_args,
                                                           expected_max_prompt_tokens=config.model.max_prompt_tokens)
                    else:
                        contract = verify_guarded_contract(*contract_args,
                                                           expected_adapter_sha256=config.model.adapter_sha256,
                                                           expected_max_prompt_tokens=config.model.max_prompt_tokens)
                    if resume and environment.get('guarded_ledger_id') != contract['ledger_id']:
                        raise BackendError('budget_unverified', 'guarded server ledger changed')
                    if not resume and contract['reserved_tokens'] != 0:
                        raise BackendError('budget_unverified', 'a new run requires an empty guarded ledger')
                except BackendError as exc:
                    journal.state['elapsed_seconds'] = journal.elapsed_before + time.monotonic() - journal.start
                    journal.state['active_started_unix'] = None; journal.save()
                    write(run_dir/'run_state.json', {'status': 'model_unavailable', 'trained': False, 'weights_updated': False, 'error': str(exc)})
                    return run_dir
                environment['guarded_ledger_id'] = contract['ledger_id']
                environment['guarded_driver_type'] = contract['driver_type']
                environment['guarded_driver_source_sha256'] = contract.get('driver_source_sha256')
                environment['guarded_driver_runtime_version'] = contract.get('driver_runtime_version')
                environment['guarded_source_sha256'] = contract['source_sha256']
                environment['token_budget_guarantee'] = ('guarded_reviewed_source_self_reported'
                    if contract['driver_type'] == 'src.arc.mlx_token_driver.MLXTokenDriver'
                    else 'guarded_contract_unverified_runtime')
                write(run_dir/'environment.json', environment)
        except BudgetExceeded as exc:
            journal.state['elapsed_seconds'] = journal.elapsed_before + time.monotonic() - journal.start
            journal.state['active_started_unix'] = None; journal.save()
            write(run_dir/'run_state.json', {'status': 'budget_exhausted', 'trained': False, 'weights_updated': False, 'error': str(exc)})
            return run_dir
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
            journal.state['elapsed_seconds'] = journal.elapsed_before + time.monotonic() - journal.start
            journal.state['active_started_unix'] = None; journal.save()
            write(run_dir/'run_state.json', {'status': 'sandbox_unavailable', 'trained': False, 'weights_updated': False, 'error': str(exc)})
            return run_dir
        predictions = []
        status = 'complete'
        try:
            for condition in config.experiment.conditions:
                for task_id in selected_task_ids:
                    task = tasks[task_id]
                    results = []
                    for i in range(a.candidates):
                        file = pred_dir/f'{condition}-{task_id}-{i}.json'
                        if file.exists():
                            expected = journal.state.get('result_hashes', {}).get(file.name)
                            if expected is None or digest(file) != expected:
                                raise ValueError('candidate prediction integrity check failed')
                            existing = json.loads(file.read_text())
                            if existing.get('error') and existing['error'].get('kind') in ('budget_violation', 'invalid_usage'):
                                journal.state['fatal_error'] = existing['error']['kind']; journal.save()
                                raise BudgetExceeded('saved server token usage cannot be trusted')
                            results.append(existing); continue
                        cached.scope = f'{condition}:{task_id}:{i}'
                        if cached.scope not in journal.state['candidate_keys']:
                            journal.remaining()
                            if journal.state['candidates'] + 1 > config.budget.max_candidates:
                                raise BudgetExceeded('candidates budget exhausted')
                            journal.state['candidates'] += 1
                            journal.state['candidate_keys'].append(cached.scope)
                            journal.save()
                        sandbox.timeout = min(a.timeout_seconds, journal.remaining())
                        start = time.monotonic()
                        try:
                            result = direct(task, cached, i) if condition == 'A' else code_agent(task, cached, sandbox, a.max_repairs, i)
                            item = {'result': asdict(result), 'elapsed_seconds': time.monotonic()-start, 'error': None}
                        except BackendError as exc:
                            item = {'result': None, 'elapsed_seconds': time.monotonic()-start, 'error': {'kind': exc.kind, 'message': str(exc)}}
                        write(file, item)
                        journal.state['result_hashes'][file.name] = digest(file); journal.save()
                        results.append(item)
                        if item['error'] is not None and item['error']['kind'] in ('budget_violation', 'invalid_usage'):
                            journal.state['fatal_error'] = item['error']['kind']; journal.save()
                            raise BudgetExceeded('server token usage cannot be trusted or exceeds the reservation')
                        if journal.state['pending'] is not None:
                            raise BackendError('request_uncertain', 'request outcome requires inspection before resume')
                    failed = any(x['error'] is not None or (x['result'] and x['result']['error_kind'] == 'timeout') for x in results)
                    outcome = None if failed else asdict(score(verifier, task_id, [x['result']['outputs'] for x in results]))
                    if outcome is not None and a.candidates < 2:
                        outcome['pass_at_2'] = None
                    predictions.append({'condition': condition, 'task_id': task_id, 'candidates': results, 'score': outcome})
        except KeyboardInterrupt:
            status = 'interrupted'
            write(run_dir/'stop.json', {'reason': 'interrupted; inspect pending request before resume'})
        except BackendError as exc:
            status = 'request_uncertain'
            write(run_dir/'stop.json', {'reason': str(exc)})
        except BudgetExceeded as exc:
            status = 'budget_exhausted'
            write(run_dir/'stop.json', {'reason': str(exc)})
        write(run_dir/'results.json', predictions)
        with (run_dir/'metrics.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['condition', 'scored_tasks', 'errors', 'pass_at_1', 'pass_at_2']); w.writeheader()
            for condition in config.experiment.conditions:
                items = [x for x in predictions if x['condition'] == condition]; scored = [x['score'] for x in items if x['score'] is not None]
                w.writerow({'condition': condition, 'scored_tasks': len(scored), 'errors': len(items)-len(scored),
                            'pass_at_1': sum(x['pass_at_1'] for x in scored)/len(scored) if scored else None,
                            'pass_at_2': sum(x['pass_at_2'] for x in scored)/len(scored) if scored and a.candidates >= 2 else None})
        if status == 'complete':
            journal.state['aggregate_hashes'] = {name: digest(run_dir/name) for name in ('results.json', 'metrics.csv')}
        journal.state['elapsed_seconds'] = journal.elapsed_before + time.monotonic() - journal.start
        journal.state['active_started_unix'] = None
        journal.save()
        write(run_dir/'run_state.json', {'status': status, 'trained': False, 'weights_updated': False, 'token_budget_guarantee': environment.get('token_budget_guarantee', 'unverified_server_tokenizer'), 'elapsed_seconds': journal.state['elapsed_seconds'], 'budget': {k:v for k,v in journal.state.items() if k not in ('cache', 'result_hashes', 'candidate_keys')}})
        return run_dir
