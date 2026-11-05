"""P3-B provenance checks, verified data collection and training orchestration."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.arc.agent import AgentResult
from src.arc.data import exploration_window, load_arc
from src.arc.mlx_lora import train_adapter
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.sandbox import DockerSandbox
from src.arc.task_manifests import load_diagnostic_task_ids
from src.arc.training import make_record
from src.core.config import ExperimentConfig, load_experiment_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_verified_exploration_records(run_dir: str | Path, *, required_tasks: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-score a completed, B-only real run and rebuild safe SFT records."""
    root = Path(run_dir).expanduser().resolve()
    required = ('run_state.json', 'identity.json', 'config.snapshot.yml', 'data_split.json',
                'budget.json', 'results.json', 'metrics.csv', 'environment.json')
    if any(not (root / name).is_file() for name in required):
        raise ValueError('source run is missing required P3-A artifacts')
    state = json.loads((root / 'run_state.json').read_text(encoding='utf-8'))
    identity = json.loads((root / 'identity.json').read_text(encoding='utf-8'))
    config = load_experiment_config(root / 'config.snapshot.yml')
    if state.get('status') != 'complete' or state.get('trained') is not False or state.get('weights_updated') is not False:
        raise ValueError('source must be a completed no-training P3-A run')
    if config.model.backend != 'guarded_mlx' or config.experiment.generations != 0 or config.experiment.conditions != ['B']:
        raise ValueError('source run must be guarded real inference with B only at Gen 0')
    if (config.data.exploration < required_tasks or config.rsi.exploration_per_generation < required_tasks or
            config.rsi.exploration_offset + required_tasks > config.data.exploration):
        raise ValueError('source run did not execute the required number of Exploration tasks')
    snapshot_document = yaml.safe_load((root / 'config.snapshot.yml').read_text(encoding='utf-8'))
    if identity.get('config') != snapshot_document:
        raise ValueError('source run config does not match its identity manifest')
    challenge_path, solution_path = Path(config.data.challenges), Path(config.data.solutions)
    if not challenge_path.is_file() or not solution_path.is_file():
        raise ValueError('source ARC challenge/solution files are unavailable')
    if identity.get('data_sha256') != {'challenges': _sha(challenge_path), 'solutions': _sha(solution_path)}:
        raise ValueError('source ARC data fingerprints changed')
    split = json.loads((root / 'data_split.json').read_text(encoding='utf-8'))
    if split != identity.get('splits') or len(split.get('exploration', [])) < required_tasks:
        raise ValueError('source data split differs from its identity manifest')
    environment = json.loads((root / 'environment.json').read_text(encoding='utf-8'))
    source_image_id = environment.get('sandbox_image_id')
    docker = shutil.which('docker')
    if not docker or not isinstance(source_image_id, str) or not source_image_id.startswith('sha256:'):
        raise ValueError('source run lacks a verified Docker runner identity')
    try:
        image_id = subprocess.check_output([docker, 'image', 'inspect', config.agent.sandbox_image,
                                            '--format', '{{.Id}}'], text=True, stderr=subprocess.DEVNULL,
                                           timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('cannot verify the original Docker runner image') from exc
    if image_id != source_image_id:
        raise ValueError('Docker runner image changed since the source run')
    budget = json.loads((root / 'budget.json').read_text(encoding='utf-8'))
    expected_result_hash = budget.get('aggregate_hashes', {}).get('results.json')
    if not isinstance(expected_result_hash, str) or expected_result_hash != _sha(root / 'results.json'):
        raise ValueError('source results fail their integrity check')
    tasks, verifier = load_arc(challenge_path, solution_path)
    if config.p3c.enabled:
        if (config.experiment.conditions != ['B'] or config.experiment.generations != 0 or
                len(config.p3c.task_ids) != required_tasks):
            raise ValueError('P3-C source config must be a B-only Gen 0 run over the full fixed batch')
        diagnostic_ids, diagnostic_manifest_sha = load_diagnostic_task_ids(config.p3c.diagnostic_task_manifest)
        if identity.get('p3c_diagnostic_task_manifest_sha256') != diagnostic_manifest_sha:
            raise ValueError('P3-C diagnostic manifest differs from the source run identity')
        if not set(diagnostic_ids) <= set(split.get('exploration', [])):
            raise ValueError('P3-C diagnostic manifest references a task outside Exploration')
        if set(config.p3c.task_ids) & set(diagnostic_ids):
            raise ValueError('P3-C source batch overlaps a diagnostic task')
        expected_tasks = set(config.p3c.task_ids)
    else:
        diagnostic_ids, diagnostic_manifest_sha = [], None
        expected_tasks = set(exploration_window(split, config.rsi.exploration_offset, required_tasks))
    if not expected_tasks <= set(tasks):
        raise ValueError('Exploration split references unknown task IDs')
    results = json.loads((root / 'results.json').read_text(encoding='utf-8'))
    observed = {item.get('task_id') for item in results if item.get('condition') == 'B'}
    if observed != expected_tasks or any(item.get('condition') != 'B' for item in results):
        raise ValueError('source results must cover exactly the configured B Exploration tasks')
    records = []
    deadline = time.monotonic() + config.budget.max_seconds
    containers = 0
    sandbox = DockerSandbox(image_id, timeout=config.agent.timeout_seconds,
                            memory_mb=config.agent.memory_mb, cpus=config.agent.cpus,
                            pids=config.agent.pids, max_output_bytes=config.agent.max_output_bytes,
                            max_code_bytes=config.agent.max_code_bytes,
                            remaining=lambda: max(0.001, deadline - time.monotonic()))
    for item in results:
        task_id = item['task_id']
        for candidate in item.get('candidates', []):
            if candidate.get('error') is not None or not isinstance(candidate.get('result'), dict):
                continue
            try:
                result = AgentResult(**candidate['result'])
            except (TypeError, ValueError):
                continue
            task = tasks[task_id]
            if not result.code or not result.outputs or not verifier.matches(task_id, result.outputs):
                continue
            reproduced = []
            failed_execution = False
            for grid in (*[pair[0] for pair in task.train], *task.test_inputs):
                if containers >= config.budget.max_container_starts or time.monotonic() >= deadline:
                    raise ValueError('P3-B code verification budget exhausted')
                containers += 1
                try:
                    reproduced.append(sandbox.run(result.code, grid))
                except Exception:
                    failed_execution = True
                    break
            if failed_execution:
                continue
            train_outputs = reproduced[:len(task.train)]
            if any(actual != expected for actual, (_, expected) in zip(train_outputs, task.train)):
                continue
            test_outputs = reproduced[len(task.train):]
            if not verifier.matches(task_id, test_outputs):
                continue
            if test_outputs != result.outputs:
                continue
            reproduced_result = AgentResult(test_outputs, result.code, result.history)
            record = make_record(task, reproduced_result, verifier, set(split['exploration']), generation=0)
            if record is not None:
                records.append(record)
    verified_candidate_successes = len(records)
    unique_by_task: dict[str, dict[str, Any]] = {}
    for record in records:
        unique_by_task.setdefault(record['task_id'], record)
    records = [unique_by_task[task_id] for task_id in sorted(unique_by_task)]
    manifest = {
        'protocol': 'p3b_verified_exploration_v1', 'source_run': str(root),
        'source_identity_sha256': _sha(root / 'identity.json'),
        'source_results_sha256': _sha(root / 'results.json'),
        'challenge_sha256': _sha(challenge_path), 'solution_sha256': _sha(solution_path),
        'split_sha256': _sha(root / 'data_split.json'), 'required_tasks': required_tasks,
        'evaluated_task_ids': sorted(expected_tasks), 'verified_record_count': len(records),
        'verified_candidate_success_count': verified_candidate_successes,
        'verified_unique_task_count': len(records), 'verified_unique_task_ids': sorted(unique_by_task),
        'verified_record_ids': sorted(record['id'] for record in records),
        'test_outputs_in_messages': False, 'code_verification_mode': 'docker_reexecution',
        'docker_image_id': image_id, 'code_verification_container_starts': containers,
        'p3c_fixed_batch': config.p3c.enabled,
        'fixed_batch_task_ids': sorted(expected_tasks) if config.p3c.enabled else None,
        'diagnostic_task_ids_excluded': sorted(diagnostic_ids),
        'diagnostic_task_manifest_sha256': diagnostic_manifest_sha,
        'max_code_attempts': config.p3c.max_code_attempts if config.p3c.enabled else config.agent.max_repairs + 1,
        'test_outputs_in_model_context': False,
    }
    return records, manifest


def synthetic_pilot_records() -> list[dict[str, Any]]:
    """Small explicitly non-ARC examples for the P3-B1 technical smoke test."""
    target = 'P3B1_CANARY_7F3A'
    prompts = [
        'For the adapter-loading smoke check, repeat the canary assigned to label ALPHA.',
        'For the adapter-loading smoke check, repeat the canary assigned to label BRAVO.',
        'For the adapter-loading smoke check, repeat the canary assigned to label CHARLIE.',
        'For the adapter-loading smoke check, repeat the canary assigned to label DELTA.',
        'For the adapter-loading smoke check, repeat the canary assigned to label ECHO.',
        'For the adapter-loading smoke check, repeat the canary assigned to label FOXTROT.',
        'For the adapter-loading smoke check, repeat the canary assigned to label GOLF.',
        'For the adapter-loading smoke check, repeat the canary assigned to label HOTEL.',
    ]
    records = []
    for index, prompt in enumerate(prompts):
        record_id = hashlib.sha256((prompt + '\0' + target).encode()).hexdigest()
        records.append({'id': record_id, 'task_id': f'synthetic-p3b1-{index:02d}',
                        'split': 'synthetic_pilot', 'synthetic': True,
                        'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                                     {'role': 'user', 'content': prompt},
                                     {'role': 'assistant', 'content': target}]})
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def _seal_artifact(root: Path, config: ExperimentConfig, manifest: dict[str, Any]) -> None:
    project = Path(__file__).resolve().parents[2]
    code_files = ('src/arc/mlx_lora.py', 'src/arc/mlx_token_driver.py', 'src/arc/mlx_chat_template.py',
                  'src/arc/prompting.py', 'src/arc/training.py', 'src/cli/p3b.py',
                  'src/cli/p3b_probe.py', 'src/cli/p3b_train.py', 'src/cli/p3b_sft_audit.py',
                  'src/arc/response_diagnostics.py', 'src/arc/task_manifests.py',
                  'src/cli/p3b_format_canary.py', 'src/cli/p3b_hardening.py',
                  'src/cli/p3c.py', 'src/pipelines/p3a.py', 'src/pipelines/p3b.py',
                  'src/pipelines/p3b_ablation.py',
                  'src/core/config.py')
    code_hashes = {name: _sha(project / name) for name in code_files}
    identity = {'config': asdict(config), 'source_manifest_sha256': _sha(root / 'source_manifest.json'),
                'training_records_sha256': _sha(root / 'verified_records.jsonl'),
                'implementation_sha256': code_hashes,
                'source_run_identity_sha256': manifest.get('source_identity_sha256')}
    _write_json(root / 'identity.json', identity)
    hashes = {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob('*'))
              if path.is_file() and path.name != 'artifact_hashes.json'}
    _write_json(root / 'artifact_hashes.json', hashes)


def train_p3b(config: ExperimentConfig, records: list[dict[str, Any]], manifest: dict[str, Any],
              output: str | Path, *, source: str, previous_adapter: str | Path | None = None) -> dict[str, Any]:
    training = config.p3b.training
    if training.backend != 'mlx_lora' or config.rsi.trainer != 'mlx_lora':
        raise ValueError('P3-B requires the explicitly enabled MLX-LoRA backend')
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f'P3-B run already exists: {target}')
    if previous_adapter is not None and not training.continue_previous_adapter:
        raise ValueError('previous adapter supplied while continuation is disabled')
    if source == 'exploration':
        p3b_protocol = manifest.get('protocol') == 'p3b_verified_exploration_v1'
        p3c_protocol = (manifest.get('protocol') == 'p3c_verified_solution_collection_v1' and
                        manifest.get('independent_verifier_protocol') == 'all_test_inputs_exact_match_v1' and
                        manifest.get('inference_mode') == 'live' and
                        manifest.get('verified_unique_task_count') == len({record.get('task_id') for record in records}) and
                        manifest.get('verified_unique_task_ids') == sorted({record.get('task_id') for record in records}) and
                        manifest.get('test_outputs_in_sft_messages') is False)
        if (not (p3b_protocol or p3c_protocol) or
                manifest.get('test_outputs_in_messages') is not False or
                manifest.get('code_verification_mode') != 'docker_reexecution'):
            raise ValueError('real training requires a verifier-built Exploration manifest')
        if manifest.get('verified_record_ids') != sorted(record.get('id') for record in records):
            raise ValueError('Exploration manifest does not match training record IDs')
    elif source == 'synthetic_pilot':
        if manifest.get('protocol') != 'p3b_synthetic_pilot_v1' or manifest.get('arc_data_used') is not False:
            raise ValueError('synthetic training must be labeled and isolated from ARC data')
    else:
        raise ValueError('unknown P3-B training source')
    target.mkdir(parents=True)
    (target / 'config.snapshot.yml').write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding='utf-8')
    _write_json(target / 'source_manifest.json', manifest)
    (target / 'verified_records.jsonl').write_text(''.join(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n' for record in records), encoding='utf-8')
    verified_unique_task_count = len({record.get('task_id') for record in records}) if source == 'exploration' else len(records)
    if source == 'exploration' and verified_unique_task_count != len(records):
        raise ValueError('real training records must contain at most one solution per unique task')
    if source == 'exploration' and (len(records) < config.p3b.insufficient_data.min_verified_samples or
            verified_unique_task_count < config.p3b.insufficient_data.min_verified_unique_tasks):
        result = {'status': 'skipped_insufficient_verified_tasks', 'trained': False, 'weights_updated': False,
                  'source': source, 'verified_samples': len(records),
                  'verified_unique_task_count': verified_unique_task_count,
                  'minimum_verified_samples': config.p3b.insufficient_data.min_verified_samples,
                  'minimum_verified_unique_tasks': config.p3b.insufficient_data.min_verified_unique_tasks,
                  'action': config.p3b.insufficient_data.action, 'previous_adapter_preserved': True}
        _write_json(target / 'training.json', result)
        _write_json(target / 'run_state.json', result)
        _seal_artifact(target, config, manifest)
        return result
    if config.budget.max_gpu_hours <= 0:
        raise ValueError('P3-B training requires a positive max_gpu_hours budget')
    timeout = min(training.timeout_seconds, int(config.budget.max_gpu_hours * 3600))
    if timeout <= 0:
        raise ValueError('P3-B wall-clock training budget rounds down to zero seconds')
    params = {'batch_size': training.batch_size, 'learning_rate': training.learning_rate,
              'max_steps': training.max_steps, 'lora_rank': training.lora_rank,
              'lora_alpha': training.lora_alpha, 'max_seq_length': training.max_seq_length,
              'num_layers': training.num_layers}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    _write_json(target / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'git_commit': commit, 'model_id': config.model.id, 'model_revision': config.model.revision,
        'model_path': training.model_path, 'seed': config.experiment.seed,
        'training_backend': training.backend, 'training_method': training.method,
        'expected_mlx_lm_version': training.mlx_lm_version,
        'gpu_time_limit_seconds': timeout, 'trained': False, 'weights_updated': False,
    })
    try:
        result = train_adapter(records, model_path=training.model_path,
                               python_executable=training.python_executable,
                               output=target / 'training-run', parameters=params,
                               seed=config.experiment.seed, timeout_seconds=timeout,
                               source=source, expected_mlx_lm_version=training.mlx_lm_version,
                               thinking_mode=config.model.thinking,
                               previous_adapter=previous_adapter)
    except Exception as exc:
        result = {'status': 'failed', 'trained': False, 'weights_updated': False,
                  'source': source, 'verified_samples': len(records),
                  'verified_unique_task_count': verified_unique_task_count,
                  'error': f'{type(exc).__name__}: {exc}',
                  'previous_adapter_preserved': True, 'elapsed_seconds': None}
    result.update({'model_id': config.model.id, 'model_revision': config.model.revision,
                   'verified_unique_task_count': verified_unique_task_count,
                   'training_backend': training.backend, 'training_method': training.method,
                   'dataset_format': 'chat', 'mask_prompt': True, 'thinking_mode': config.model.thinking,
                   'gpu_time_limit_seconds': timeout})
    _write_json(target / 'training.json', result)
    state = {'status': result['status'], 'trained': result['trained'], 'weights_updated': result['weights_updated'],
             'source': source, 'verified_samples': len(records),
             'verified_unique_task_count': verified_unique_task_count,
             'elapsed_seconds': result.get('elapsed_seconds'),
             'adapter_sha256': result.get('adapter_sha256'), 'previous_adapter_sha256': result.get('previous_adapter_sha256')}
    _write_json(target / 'run_state.json', state)
    environment_path = target / 'environment.json'
    environment = json.loads(environment_path.read_text(encoding='utf-8'))
    environment.update({'trained': result['trained'], 'weights_updated': result['weights_updated'],
                        'mlx_lm_version': result.get('mlx_lm_version'), 'mlx_version': result.get('mlx_version'),
                        'adapter_sha256': result.get('adapter_sha256'),
                        'base_weights_sha256_before': result.get('base_weights_sha256_before'),
                        'base_weights_sha256_after': result.get('base_weights_sha256_after'),
                        'peak_rss_bytes': result.get('peak_rss_bytes')})
    _write_json(environment_path, environment)
    _seal_artifact(target, config, manifest)
    return result
