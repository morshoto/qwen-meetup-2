"""Matched B versus adapter-backed C comparison on unused Exploration tasks."""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
import shutil
from typing import Any

import yaml

from src.arc.data import split_tasks
from src.arc.http_backend import BackendError, verify_guarded_contract
from src.arc.mlx_token_driver import adapter_sha256
from src.arc.task_manifests import load_diagnostic_task_ids
from src.core.config import ExperimentConfig
from src.pipelines.p3a import run_real


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def _verify_artifact(root: Path) -> None:
    manifest_path = root / 'artifact_hashes.json'
    if not manifest_path.is_file():
        raise ValueError('P3-B training artifact integrity manifest is missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('P3-B training artifact integrity manifest is invalid')
    actual_paths = {str(path.relative_to(root)) for path in root.rglob('*')
                    if path.is_file() and path.name != 'artifact_hashes.json'}
    if set(manifest) != actual_paths:
        raise ValueError('P3-B training artifact contains missing or unmanifested files')
    for relative, expected in manifest.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or _sha(target) != expected:
            raise ValueError(f'P3-B training artifact failed integrity check: {relative}')


def _seal_comparison(root: Path, config: ExperimentConfig, source_identity_hash: str | None) -> None:
    project = Path(__file__).resolve().parents[2]
    code = ('src/arc/mlx_lora.py', 'src/arc/mlx_token_driver.py', 'src/arc/training.py',
            'src/arc/guarded_generation.py', 'src/arc/guarded_http.py', 'src/arc/token_ledger.py',
            'src/cli/guarded_mlx_server.py', 'src/cli/p3b.py', 'src/cli/p3b_probe.py',
            'src/arc/task_manifests.py', 'src/cli/p3c.py',
            'src/pipelines/p3a.py', 'src/pipelines/p3b.py', 'src/pipelines/p3b_compare.py',
            'src/core/config.py')
    identity = {'config': asdict(config), 'source_training_identity_sha256': source_identity_hash,
                'implementation_sha256': {name: _sha(project / name) for name in code}}
    _write(root / 'identity.json', identity)
    files = {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob('*'))
             if path.is_file() and path.name != 'artifact_hashes.json'}
    _write(root / 'artifact_hashes.json', files)


def _normalize(config: ExperimentConfig) -> dict[str, Any]:
    value = asdict(config)
    value['experiment']['conditions'] = ['MATCHED_AGENT']
    value['model']['adapter_sha256'] = 'ADAPTER_CONDITION_ONLY'
    return value


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


class GuardedServerProcess:
    def __init__(self, config: ExperimentConfig, run_root: Path, *, adapter: Path | None,
                 expected_adapter_hash: str) -> None:
        self.config, self.run_root, self.adapter = config, run_root, adapter
        self.expected_adapter_hash = expected_adapter_hash
        self.port = _free_port()
        self.process: subprocess.Popen[str] | None = None
        self._log = None
        self.contract: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}/v1'

    def start(self) -> dict[str, Any]:
        model = self.config.p3b.training.model_path
        training = self.config.p3b.training
        ledger = self.run_root / ('server-c.sqlite' if self.adapter else 'server-b.sqlite')
        log_path = self.run_root / ('server-c.log' if self.adapter else 'server-b.log')
        command = [training.python_executable, '-m', 'src.cli.guarded_mlx_server',
                   '--model-dir', model, '--model-id', self.config.model.id,
                   '--revision', self.config.model.revision,
                   '--expected-mlx-lm-version', training.mlx_lm_version,
                   '--expected-generate-step-sha256', '1aee463145a78556a66604ace7b6d85dc10035226aa4f909cbefccfa81b49de6',
                   '--ledger', str(ledger), '--total-tokens', str(max(1, self.config.budget.max_tokens // 2)),
                   '--max-prompt-tokens', str(self.config.model.max_prompt_tokens), '--max-new-tokens', str(self.config.model.max_tokens),
                   '--temperature', str(self.config.model.temperature), '--thinking', self.config.model.thinking,
                   '--max-request-seconds', str(self.config.model.timeout_seconds), '--port', str(self.port)]
        if self.adapter is not None:
            command.extend(['--adapter-path', str(self.adapter), '--expected-adapter-sha256', self.expected_adapter_hash])
        self._log = log_path.open('w', encoding='utf-8')
        self.process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                        stdout=self._log, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + min(self.config.model.timeout_seconds + 60, 180)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'guarded server exited with status {self.process.returncode}; see {log_path}')
            try:
                self.contract = verify_guarded_contract(
                    self.base_url, self.config.model.id, self.config.model.revision,
                    self.config.budget.max_tokens, self.config.model.max_tokens,
                    self.config.model.timeout_seconds, self.config.model.thinking,
                    expected_adapter_sha256=self.expected_adapter_hash,
                    expected_max_prompt_tokens=self.config.model.max_prompt_tokens)
                if self.contract['reserved_tokens'] != 0:
                    raise BackendError('budget_unverified', 'new comparison server ledger is not empty')
                return self.contract
            except BackendError:
                time.sleep(0.5)
        raise TimeoutError(f'guarded server startup deadline exceeded; see {log_path}')

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._log is not None:
            self._log.close()


def _holdout_data(config: ExperimentConfig, root: Path, trained_task_ids: set[str]) -> tuple[dict[str, Any], Path, Path]:
    with Path(config.data.challenges).open(encoding='utf-8') as stream:
        challenges = json.load(stream)
    with Path(config.data.solutions).open(encoding='utf-8') as stream:
        solutions = json.load(stream)
    splits = split_tasks(list(challenges), config.data.split_seed,
                         {'exploration': config.data.exploration,
                          'promotion_gate': config.data.promotion_gate,
                          'monitoring': config.data.monitoring})
    diagnostic_ids: list[str] = []
    diagnostic_manifest_sha = None
    if config.p3b.promotion.exclude_diagnostic_tasks:
        diagnostic_ids, diagnostic_manifest_sha = load_diagnostic_task_ids(
            config.p3b.promotion.diagnostic_task_manifest)
        if not set(diagnostic_ids) <= set(splits['exploration']):
            raise ValueError('diagnostic task manifest references a task outside Exploration')
    start = config.rsi.exploration_offset
    count = config.p3b.promotion.minimum_tasks
    holdout_ids = [task_id for task_id in splits['exploration'][start:]
                   if task_id not in trained_task_ids and task_id not in diagnostic_ids][:count]
    if len(holdout_ids) != count or len(set(holdout_ids) & trained_task_ids):
        raise ValueError('insufficient unused Exploration tasks for B/C comparison')
    holdout = root / 'holdout'
    holdout.mkdir()
    challenge_path, solution_path = holdout / 'challenges.json', holdout / 'solutions.json'
    challenge_path.write_text(json.dumps({task_id: challenges[task_id] for task_id in holdout_ids}, separators=(',', ':')) + '\n', encoding='utf-8')
    solution_path.write_text(json.dumps({task_id: solutions[task_id] for task_id in holdout_ids}, separators=(',', ':')) + '\n', encoding='utf-8')
    manifest = {'protocol': 'p3b_unused_exploration_holdout_v1', 'task_ids': holdout_ids,
                'training_task_ids': sorted(trained_task_ids),
                'diagnostic_task_ids_excluded': sorted(diagnostic_ids),
                'diagnostic_task_manifest_sha256': diagnostic_manifest_sha,
                'disjoint_from_verified_training_records': True,
                'challenge_sha256': _sha(challenge_path), 'solution_sha256': _sha(solution_path)}
    _write(root / 'holdout_manifest.json', manifest)
    return manifest, challenge_path, solution_path


def decide_promotion(metrics: dict[str, dict[str, Any]], *, minimum_tasks: int,
                     enabled: bool, rollback_on_regression: bool) -> dict[str, Any]:
    """Fail closed when either matched condition is incomplete or has errors."""
    sufficient = True
    scores: dict[str, float] = {}
    for label in ('B', 'C'):
        try:
            row = metrics[label]
            if int(row['scored_tasks']) != minimum_tasks or int(row['errors']) != 0:
                sufficient = False
                break
            score = float(row['pass_at_1'])
            if not 0.0 <= score <= 1.0:
                sufficient = False
                break
            scores[label] = score
        except (KeyError, TypeError, ValueError):
            sufficient = False
            break
    regression = not sufficient or scores.get('C', 0.0) < scores.get('B', 1.0)
    promote = bool(enabled and sufficient and (not rollback_on_regression or not regression))
    return {'base_pass_at_1': scores.get('B'), 'candidate_pass_at_1': scores.get('C'),
            'regressed_or_unverifiable': regression,
            'promotion': 'promote_candidate' if promote else 'rollback'}


def compare_adapter(config: ExperimentConfig, training_run: str | Path, output: str | Path) -> dict[str, Any]:
    """Compare base B with C on identical Exploration inputs and settings.

    Evaluation labels are read only by the pipeline verifier. Each condition is
    run against a separate guarded server with an immutable base or adapter.
    """
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f'comparison output already exists: {root}')
    root.mkdir(parents=True)
    training_root = Path(training_run).expanduser().resolve()
    training_result_path = training_root / 'training.json'
    if not training_result_path.is_file():
        raise ValueError('P3-B training result is missing')
    _verify_artifact(training_root)
    training_identity_path = training_root / 'identity.json'
    training_identity_hash = _sha(training_identity_path) if training_identity_path.is_file() else None
    source_copy = root / 'training-source'
    source_copy.mkdir()
    for name in ('training.json', 'source_manifest.json', 'identity.json', 'artifact_hashes.json'):
        if (training_root / name).is_file():
            shutil.copy2(training_root / name, source_copy / name)
    (root / 'config.snapshot.yml').write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding='utf-8')
    training_result = json.loads(training_result_path.read_text(encoding='utf-8'))
    source_manifest = json.loads((training_root / 'source_manifest.json').read_text(encoding='utf-8'))
    if (training_result.get('status') != 'complete' or training_result.get('trained') is not True or
            training_result.get('weights_updated') is not True or training_result.get('source') != 'exploration'):
        result = {'status': 'skipped_no_verified_adapter', 'trained': False, 'weights_updated': False,
                  'promotion': 'keep_previous', 'previous_adapter_preserved': True,
                  'training_status': training_result.get('status'),
                  'verified_sample_count': training_result.get('verified_samples', source_manifest.get('verified_record_count'))}
        _write(root / 'comparison.json', result)
        _write(root / 'run_state.json', result)
        _seal_comparison(root, config, training_identity_hash)
        return result
    if training_result.get('model_id') != config.model.id or training_result.get('model_revision') != config.model.revision:
        raise ValueError('trained adapter base model identity differs from comparison config')
    adapter = Path(training_result['adapter_path']).expanduser().resolve()
    actual_adapter_hash = adapter_sha256(adapter)
    if actual_adapter_hash != training_result.get('adapter_sha256'):
        raise ValueError('trained adapter hash differs from its training manifest')
    training_ids = set(source_manifest.get('evaluated_task_ids', []))
    holdout_manifest, challenge_path, solution_path = _holdout_data(config, root, training_ids)

    base_config = replace(config, experiment=replace(config.experiment, conditions=['B'], name='p3b-gen1-matched'),
                          data=replace(config.data, challenges=str(challenge_path), solutions=str(solution_path),
                                       exploration=len(holdout_manifest['task_ids']), promotion_gate=0, monitoring=0),
                          rsi=replace(config.rsi, trainer='dry_run', exploration_per_generation=len(holdout_manifest['task_ids']),
                                      exploration_offset=0, promotion_rule='not_applicable'),
                          model=replace(config.model, adapter_sha256='none'),
                          tracking=replace(config.tracking, output_dir=str(root / 'runs')))
    candidate_config = replace(base_config,
                               experiment=replace(base_config.experiment, conditions=['C']),
                               model=replace(base_config.model, adapter_sha256=actual_adapter_hash))
    if _normalize(base_config) != _normalize(candidate_config):
        raise ValueError('B/C configs differ beyond the adapter identity')
    (root / 'config-b.snapshot.yml').write_text(yaml.safe_dump(asdict(base_config), sort_keys=False), encoding='utf-8')
    (root / 'config-c.snapshot.yml').write_text(yaml.safe_dump(asdict(candidate_config), sort_keys=False), encoding='utf-8')
    conditions = {}
    servers = []
    try:
        for label, condition_config, condition_adapter, condition_hash in (
                ('B', base_config, None, 'none'), ('C', candidate_config, adapter, actual_adapter_hash)):
            server = GuardedServerProcess(condition_config, root, adapter=condition_adapter,
                                          expected_adapter_hash=condition_hash)
            servers.append(server)
            contract = server.start()
            live_config = replace(condition_config, model=replace(condition_config.model, base_url=server.base_url))
            run_dir = run_real(live_config)
            state = json.loads((run_dir / 'run_state.json').read_text(encoding='utf-8'))
            conditions[label] = {'run_dir': str(run_dir), 'run_status': state.get('status'),
                                 'adapter_sha256': condition_hash, 'server_ledger_id': contract['ledger_id'],
                                 'candidate_hash': _sha(run_dir / 'results.json') if (run_dir / 'results.json').is_file() else None}
            server.stop()
            servers.remove(server)
    except Exception as exc:
        for server in servers:
            server.stop()
        result = {'status': 'comparison_failed', 'trained': True, 'weights_updated': True,
                  'promotion': 'rollback', 'previous_adapter_preserved': True,
                  'error': f'{type(exc).__name__}: {exc}', 'conditions': conditions,
                  'holdout': holdout_manifest}
        _write(root / 'comparison.json', result)
        _write(root / 'run_state.json', result)
        _seal_comparison(root, config, training_identity_hash)
        return result

    metrics: dict[str, dict[str, Any]] = {}
    for label in ('B', 'C'):
        run_dir = Path(conditions[label]['run_dir'])
        rows = list(__import__('csv').DictReader((run_dir / 'metrics.csv').open(newline='')))
        if len(rows) != 1:
            raise ValueError(f'{label} comparison metrics are missing or ambiguous')
        metrics[label] = rows[0]
    promotion = decide_promotion(metrics, minimum_tasks=config.p3b.promotion.minimum_tasks,
                                 enabled=config.p3b.promotion.enabled,
                                 rollback_on_regression=config.p3b.promotion.rollback_on_regression)
    promoted = promotion['promotion'] == 'promote_candidate'
    result = {'status': 'complete', 'trained': True, 'weights_updated': True,
              'model_id': config.model.id, 'model_revision': config.model.revision,
              'adapter_sha256': actual_adapter_hash, 'previous_adapter_sha256': training_result.get('previous_adapter_sha256', 'none'),
              'holdout': holdout_manifest, 'conditions': conditions, 'metrics': metrics,
              'b_c_config_equal_after_masking_identity': True, 'minimum_tasks': config.p3b.promotion.minimum_tasks,
              'b_c_shared_config_sha256': hashlib.sha256(json.dumps(_normalize(base_config), sort_keys=True,
                                                                    separators=(',', ':')).encode()).hexdigest(),
              'promotion_enabled': config.p3b.promotion.enabled,
              'rollback_on_regression': config.p3b.promotion.rollback_on_regression,
              'regressed_or_unverifiable': promotion['regressed_or_unverifiable'],
              'promotion': promotion['promotion'],
              'active_adapter_sha256': actual_adapter_hash if promoted else training_result.get('previous_adapter_sha256', 'none'),
              'base_pass_at_1': promotion['base_pass_at_1'],
              'candidate_pass_at_1': promotion['candidate_pass_at_1']}
    _write(root / 'comparison.json', result)
    _write(root / 'run_state.json', {'status': 'complete', 'trained': True, 'weights_updated': True,
                                     'promotion': result['promotion'], 'adapter_sha256': actual_adapter_hash,
                                     'active_adapter_sha256': result['active_adapter_sha256']})
    _seal_comparison(root, config, training_identity_hash)
    return result
