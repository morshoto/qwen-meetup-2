"""Budgeted single-record QLoRA format ablation with no promotion path."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from src.arc.mlx_token_driver import adapter_sha256
from src.core.config import ExperimentConfig
from src.pipelines.p3b import collect_verified_exploration_records, train_p3b


FIELDS = ('candidate', 'role', 'steps', 'learning_rate', 'dataset_sha256',
          'training_status', 'adapter_sha256', 'adapter_load_success',
          'response_valid_rate', 'thinking_format_error_count',
          'canary_elapsed_seconds', 'training_elapsed_seconds',
          'training_peak_rss_bytes', 'canary_peak_memory_bytes', 'promoted',
          'rollback_reason', 'artifact_path')


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _hash_training_dataset(records: list[dict[str, Any]]) -> str:
    rows = [{'id': record['id'], 'task_id': record['task_id'], 'split': record['split'],
             'synthetic': record.get('synthetic', False), 'messages': record['messages']}
            for record in records]
    raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _canary(config: ExperimentConfig, *, adapter: Path | None, expected_sha: str,
            output: Path, remaining_seconds: float) -> tuple[dict[str, Any] | None, str | None]:
    training = config.p3b.training
    command = [training.python_executable, '-m', 'src.cli.p3b_format_canary',
               '--model', training.model_path, '--thinking-mode', config.model.thinking,
               '--max-tokens', str(config.p3b.ablation.canary_max_tokens),
               '--expected-adapter-sha256', expected_sha, '--output', str(output)]
    if adapter is not None:
        command.extend(['--adapter', str(adapter.expanduser().resolve())])
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True,
            timeout=max(1, min(training.timeout_seconds, int(remaining_seconds))),
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f'{type(exc).__name__}: {exc}'
    if completed.returncode != 0 or not output.is_file():
        return None, completed.stderr[-2000:] or f'canary exited with status {completed.returncode}'
    try:
        return json.loads(output.read_text(encoding='utf-8')), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f'canary report could not be read: {exc}'


def run_ablation(config: ExperimentConfig, source_run: str | Path,
                 output: str | Path | None = None) -> dict[str, Any]:
    if (not config.p3b.ablation.enabled or config.p3b.training.backend != 'mlx_lora' or
            config.rsi.trainer != 'mlx_lora'):
        raise ValueError('P3-B ablation requires its explicit MLX-LoRA diagnostic config')
    if config.p3b.ablation.max_candidates != len(config.p3b.ablation.step_candidates):
        raise ValueError('ablation max_candidates must equal the fixed step-candidate count')
    records, source_manifest = collect_verified_exploration_records(
        source_run, required_tasks=config.p3b.training.collection_tasks)
    if len(records) != 1 or source_manifest.get('verified_unique_task_count') != 1:
        raise ValueError('the low-data ablation requires exactly one verifier-approved unique task')
    dataset_sha = _hash_training_dataset(records)
    old_adapter = Path(config.p3b.ablation.baseline_adapter_path).expanduser().resolve()
    old_adapter_sha = adapter_sha256(old_adapter)
    root = Path(output).expanduser().resolve() if output else (
        Path(config.tracking.output_dir) / f'p3b-hardening-ablation-{uuid.uuid4().hex[:12]}').resolve()
    if root.exists():
        raise FileExistsError(f'ablation artifact already exists: {root}')
    root.mkdir(parents=True)
    project = Path(__file__).resolve().parents[2]
    source_copy = dict(source_manifest)
    _write_json(root / 'source_manifest.json', source_copy)
    (root / 'verified_records.jsonl').write_text(
        ''.join(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n' for record in records), encoding='utf-8')
    _write_json(root / 'identity.json', {
        'protocol': 'p3b_single_record_ablation_v1', 'config': asdict(config),
        'source_identity_sha256': source_manifest['source_identity_sha256'],
        'source_manifest_sha256': hashlib.sha256((root / 'source_manifest.json').read_bytes()).hexdigest(),
        'dataset_sha256': dataset_sha, 'baseline_c_adapter_sha256': old_adapter_sha,
        'candidate_steps': config.p3b.ablation.step_candidates,
        'learning_rate': config.p3b.ablation.learning_rate,
        'promotion_enabled': False,
        'interpretation': 'one-record wiring and format diagnostic only; no generalization claim',
        'implementation_sha256': {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
                                  for name in ('src/pipelines/p3b_ablation.py', 'src/cli/p3b_format_canary.py',
                                               'src/arc/mlx_lora.py', 'src/arc/mlx_chat_template.py',
                                               'src/cli/p3b_sft_audit.py', 'src/core/config.py')},
    })

    deadline = time.monotonic() + config.p3b.ablation.max_wall_seconds
    rows: list[dict[str, Any]] = []

    def remaining() -> float:
        return deadline - time.monotonic()

    def record_canary(candidate: str, role: str, adapter: Path | None,
                      expected_sha: str, artifact: Path, *, steps: int | None = None,
                      learning_rate: float | None = None, training_status: str = 'not_applicable',
                      training_elapsed: float | None = None, training_peak: int | None = None,
                      adapter_load_success: bool = True) -> None:
        canary_started = time.monotonic()
        report = None
        error = None
        if remaining() > 0:
            report, error = _canary(config, adapter=adapter, expected_sha=expected_sha,
                                    output=artifact, remaining_seconds=remaining())
        else:
            error = 'aggregate ablation wall-clock budget exhausted before canary'
        rows.append({
            'candidate': candidate, 'role': role, 'steps': steps,
            'learning_rate': learning_rate, 'dataset_sha256': dataset_sha,
            'training_status': training_status,
            'adapter_sha256': expected_sha if adapter is not None else 'none',
            'adapter_load_success': bool(adapter_load_success and report is not None),
            'response_valid_rate': report.get('response_valid_rate') if report else None,
            'thinking_format_error_count': report.get('thinking_format_error_count') if report else None,
            'canary_elapsed_seconds': report.get('elapsed_seconds') if report else time.monotonic() - canary_started,
            'training_elapsed_seconds': training_elapsed,
            'training_peak_rss_bytes': training_peak,
            'canary_peak_memory_bytes': report.get('peak_memory_bytes') if report else None,
            'promoted': False,
            'rollback_reason': error or 'promotion disabled for single-record diagnostic',
            'artifact_path': str(artifact.parent),
        })
        _write_csv(root / 'ablation_results.csv', rows)

    record_canary('baseline-B', 'base_model', None, 'none', root / 'canary-B/report.json')
    record_canary('historical-C', 'previous_adapter', old_adapter, old_adapter_sha,
                  root / 'canary-historical-C/report.json')

    for steps in config.p3b.ablation.step_candidates:
        candidate_name = f'lr{config.p3b.ablation.learning_rate:g}-steps{steps}'
        candidate_dir = root / f'train-{candidate_name}'
        if remaining() <= 1:
            rows.append({'candidate': candidate_name, 'role': 'candidate', 'steps': steps,
                         'learning_rate': config.p3b.ablation.learning_rate, 'dataset_sha256': dataset_sha,
                         'training_status': 'budget_not_started', 'adapter_sha256': None,
                         'adapter_load_success': False, 'response_valid_rate': None,
                         'thinking_format_error_count': None, 'canary_elapsed_seconds': None,
                         'training_elapsed_seconds': None, 'training_peak_rss_bytes': None,
                         'canary_peak_memory_bytes': None, 'promoted': False,
                         'rollback_reason': 'aggregate ablation wall-clock budget exhausted',
                         'artifact_path': str(candidate_dir)})
            _write_csv(root / 'ablation_results.csv', rows)
            continue
        remaining_seconds = max(1, int(remaining()))
        train_settings = replace(config.p3b.training,
                                 max_steps=steps,
                                 learning_rate=config.p3b.ablation.learning_rate,
                                 timeout_seconds=min(config.p3b.training.timeout_seconds, remaining_seconds),
                                 continue_previous_adapter=False)
        candidate_config = replace(
            config,
            experiment=replace(config.experiment, name=f'{config.experiment.name}-{candidate_name}'),
            p3b=replace(config.p3b, training=train_settings),
            budget=replace(config.budget, max_seconds=min(config.budget.max_seconds, remaining_seconds),
                           max_gpu_hours=min(config.budget.max_gpu_hours, remaining_seconds / 3600)))
        training_started = time.monotonic()
        try:
            training_result = train_p3b(candidate_config, records, source_manifest,
                                        candidate_dir, source='exploration')
        except Exception as exc:
            training_result = {'status': 'failed', 'trained': False, 'weights_updated': False,
                               'error': f'{type(exc).__name__}: {exc}'}
        training_elapsed = time.monotonic() - training_started
        adapter_path = (Path(training_result['adapter_path']).expanduser().resolve()
                        if training_result.get('status') == 'complete' and training_result.get('adapter_path') else None)
        expected_sha = training_result.get('adapter_sha256', 'none')
        if adapter_path is not None:
            try:
                if adapter_sha256(adapter_path) != expected_sha:
                    raise ValueError('candidate adapter hash differs from training record')
            except (OSError, ValueError) as exc:
                training_result = {**training_result, 'status': 'failed', 'trained': False,
                                   'weights_updated': False, 'adapter_error': str(exc)}
                adapter_path = None
        if adapter_path is not None and remaining() > 1:
            record_canary(candidate_name, 'candidate', adapter_path, expected_sha,
                          candidate_dir / 'canary/report.json', steps=steps,
                          learning_rate=config.p3b.ablation.learning_rate,
                          training_status=training_result['status'],
                          training_elapsed=training_elapsed,
                          training_peak=training_result.get('peak_rss_bytes'))
        else:
            rows.append({'candidate': candidate_name, 'role': 'candidate', 'steps': steps,
                         'learning_rate': config.p3b.ablation.learning_rate, 'dataset_sha256': dataset_sha,
                         'training_status': training_result.get('status'), 'adapter_sha256': expected_sha,
                         'adapter_load_success': False, 'response_valid_rate': None,
                         'thinking_format_error_count': None, 'canary_elapsed_seconds': None,
                         'training_elapsed_seconds': training_elapsed,
                         'training_peak_rss_bytes': training_result.get('peak_rss_bytes'),
                         'canary_peak_memory_bytes': None, 'promoted': False,
                         'rollback_reason': training_result.get('error') or 'no time remained for candidate canary',
                         'artifact_path': str(candidate_dir)})
            _write_csv(root / 'ablation_results.csv', rows)
        _write_json(root / f'{candidate_name}.training.json', training_result)

    report = {
        'protocol': 'p3b_single_record_ablation_v1', 'status': 'complete',
        'source_run': str(Path(source_run).expanduser().resolve()),
        'source_identity_sha256': source_manifest['source_identity_sha256'],
        'verified_unique_task_count': source_manifest['verified_unique_task_count'],
        'dataset_sha256': dataset_sha, 'candidate_count': len(config.p3b.ablation.step_candidates),
        'configured_max_candidates': config.p3b.ablation.max_candidates,
        'learning_rate': config.p3b.ablation.learning_rate,
        'step_candidates': config.p3b.ablation.step_candidates,
        'elapsed_seconds': config.p3b.ablation.max_wall_seconds - max(0, remaining()),
        'max_wall_seconds': config.p3b.ablation.max_wall_seconds,
        'promotion_enabled': False, 'promoted_candidates': [],
        'interpretation': 'low-data technical and output-format diagnostic; not P3-C readiness or generalization evidence',
        'results_csv': 'ablation_results.csv', 'results': rows,
    }
    _write_json(root / 'ablation.json', report)
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'artifact_hashes.json'}
    _write_json(root / 'artifact_hashes.json', hashes)
    return report
