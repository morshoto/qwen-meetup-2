"""Offline P3-B diagnostics and preflight checks; no inference calls are made."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.arc.response_diagnostics import diagnose_artifact_tree
from src.core.config import load_experiment_config
from src.pipelines.p3b_ablation import run_ablation


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_sft(run_dir: Path, output: Path) -> dict:
    source = run_dir.expanduser().resolve()
    training_root = source if (source / 'training.json').is_file() else source.parent
    training = json.loads((training_root / 'training.json').read_text(encoding='utf-8'))
    environment_path = training_root / 'environment.json'
    environment = json.loads(environment_path.read_text(encoding='utf-8')) if environment_path.is_file() else {}
    config_path = training_root / 'config.snapshot.yml'
    config = load_experiment_config(config_path) if config_path.is_file() else None
    model_path = training.get('model_path') or environment.get('model_path')
    python = training.get('python_executable') or (config.p3b.training.python_executable if config else None)
    if not model_path or not python:
        raise ValueError('training artifact does not identify its local model and MLX Python runtime')
    dataset_dir = training_root / 'training-run' / 'dataset'
    if not dataset_dir.is_dir() and source.name == 'training-run':
        dataset_dir = source / 'dataset'
    train_file = dataset_dir / 'train.jsonl'
    if not train_file.is_file():
        raise ValueError('training artifact does not contain dataset/train.jsonl')
    source_sha = _sha(train_file)
    records_file = training_root / 'training-run' / 'verified_records.jsonl'
    if not records_file.is_file() and (source / 'verified_records.jsonl').is_file():
        records_file = source / 'verified_records.jsonl'
    records = [json.loads(line) for line in records_file.read_text(encoding='utf-8').splitlines() if line.strip()] if records_file.is_file() else []
    source_kind = training.get('source') or ('synthetic_pilot' if records and records[0].get('synthetic') else 'exploration')
    actual_mode = training.get('thinking_mode') or 'server_default'
    requested_mode = config.model.thinking if config else actual_mode
    project = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='p3b-sft-audit-') as temporary:
        temp_root = Path(temporary)
        temp_dataset = temp_root / 'dataset'
        temp_dataset.mkdir()
        raw_rows = [json.loads(line) for line in train_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        enriched = []
        for index, row in enumerate(raw_rows):
            record = records[index] if index < len(records) else {}
            for key in ('id', 'task_id', 'split', 'synthetic'):
                if key not in row and key in record:
                    row[key] = record[key]
            enriched.append(row)
        (temp_dataset / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in enriched), encoding='utf-8')
        valid_file = dataset_dir / 'valid.jsonl'
        if valid_file.is_file():
            shutil.copyfile(valid_file, temp_dataset / 'valid.jsonl')
        audits = {}
        for mode in dict.fromkeys((actual_mode, requested_mode)):
            audit_path = temp_root / f'audit-{mode}.json'
            command = [str(python), '-m', 'src.cli.p3b_sft_audit', '--model', str(model_path),
                       '--dataset', str(temp_dataset), '--output', str(audit_path),
                       '--thinking-mode', mode, '--source', source_kind, '--allow-misaligned']
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                         cwd=project,
                                         env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                                         check=False)
                audits[mode] = {'returncode': result.returncode,
                                'stderr': result.stderr[-2000:],
                                'audit': json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.is_file() else None}
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                audits[mode] = {'returncode': None, 'error': f'{type(exc).__name__}: {exc}', 'audit': None}
    report = {
        'protocol': 'p3b_historical_sft_alignment_audit_v1',
        'training_artifact': str(training_root), 'dataset_path': str(train_file),
        'source_dataset_sha256': source_sha, 'source': source_kind,
        'actual_training_thinking_mode': actual_mode,
        'inference_config_thinking_mode': requested_mode,
        'actual_training_entrypoint': training.get('training_entrypoint', 'mlx_lm.lora (no template policy wrapper recorded)'),
        'dataset_format': 'chat', 'mask_prompt_configured': bool(training.get('mask_prompt', True)),
        'mode_audits': audits,
    }
    report['actual_training_audit_passed'] = bool(audits.get(actual_mode, {}).get('audit', {}).get('audit_passed'))
    report['inference_alignment_audit_passed'] = bool(audits.get(requested_mode, {}).get('audit', {}).get('audit_passed'))
    report['audit_passed'] = report['actual_training_audit_passed'] and report['inference_alignment_audit_passed']
    _write_report(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='P3-B hardening offline diagnostics')
    parser.add_argument('--config', type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true', help='validate the existing experiment config schema')
    action.add_argument('--diagnose-from-artifacts', type=Path,
                        help='reparse captured responses or report what historical logs cannot establish')
    action.add_argument('--audit-sft', type=Path,
                        help='render the saved training chat data and inspect MLX-LM prompt-loss spans')
    action.add_argument('--run-canary', action='store_true',
                        help='run fixed synthetic format checks in the configured MLX runtime')
    action.add_argument('--ablate', action='store_true',
                        help='run B, historical C and fixed low-step adapter canaries on one verified record')
    parser.add_argument('--output', type=Path, help='diagnostic report path; defaults beside the input artifact')
    parser.add_argument('--adapter', type=Path, help='adapter to load for --run-canary')
    parser.add_argument('--expected-adapter-sha256', default='none')
    parser.add_argument('--source-run', type=Path, help='completed B-only run supplying the verified record for --ablate')
    args = parser.parse_args()
    if args.check:
        if args.config is None:
            parser.error('--check requires --config')
        config = load_experiment_config(args.config)
        print(json.dumps({'status': 'valid', 'experiment': config.experiment.name,
                          'model': config.model.id, 'revision': config.model.revision}, indent=2))
        return
    if args.audit_sft:
        source = args.audit_sft.expanduser().resolve()
        output = args.output.expanduser().resolve() if args.output else source.parent / f'{source.name}-sft-audit.json'
        report = _audit_sft(source, output)
        print(json.dumps({'status': 'complete', 'audit_passed': report['audit_passed'],
                          'actual_training_audit_passed': report['actual_training_audit_passed'],
                          'inference_alignment_audit_passed': report['inference_alignment_audit_passed'],
                          'report': str(output)}, indent=2))
        return
    if args.run_canary:
        if args.config is None:
            parser.error('--run-canary requires --config')
        config = load_experiment_config(args.config)
        training = config.p3b.training
        if training.backend != 'mlx_lora' or not training.python_executable or not training.model_path:
            parser.error('--run-canary requires an explicitly configured local MLX-LoRA runtime')
        if args.adapter is None and args.expected_adapter_sha256 != 'none':
            parser.error('--expected-adapter-sha256 requires --adapter')
        if args.adapter is not None and args.expected_adapter_sha256 == 'none':
            parser.error('--adapter requires its expected SHA-256')
        source = args.config.expanduser().resolve()
        output = args.output.expanduser().resolve() if args.output else Path(config.tracking.output_dir) / 'p3b-synthetic-format-canary.json'
        command = [training.python_executable, '-m', 'src.cli.p3b_format_canary',
                   '--model', training.model_path, '--thinking-mode', config.model.thinking,
                   '--max-tokens', str(config.p3b.ablation.canary_max_tokens),
                   '--expected-adapter-sha256', args.expected_adapter_sha256,
                   '--output', str(output)]
        if args.adapter is not None:
            command.extend(['--adapter', str(args.adapter.expanduser().resolve())])
        result = subprocess.run(command, capture_output=True, text=True,
                                 timeout=min(training.timeout_seconds, config.p3b.ablation.max_wall_seconds),
                                 cwd=Path(__file__).resolve().parents[2],
                                 env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                                 check=False)
        if result.returncode != 0:
            print(json.dumps({'status': 'failed', 'stderr': result.stderr[-4000:]}, indent=2))
            raise SystemExit(result.returncode or 2)
        report = json.loads(output.read_text(encoding='utf-8'))
        print(json.dumps({'status': 'complete', 'output': str(output),
                          'adapter_sha256': report['adapter_sha256'],
                          'response_valid_rate': report['response_valid_rate'],
                          'thinking_format_error_count': report['thinking_format_error_count']}, indent=2))
        return
    if args.ablate:
        if args.config is None or args.source_run is None:
            parser.error('--ablate requires --config and --source-run')
        config = load_experiment_config(args.config)
        report = run_ablation(config, args.source_run, args.output)
        print(json.dumps({'status': report['status'], 'results_csv': report['results_csv'],
                          'verified_unique_task_count': report['verified_unique_task_count'],
                          'promotion_enabled': report['promotion_enabled'],
                          'elapsed_seconds': report['elapsed_seconds']}, indent=2))
        return
    source = args.diagnose_from_artifacts.expanduser().resolve()
    report = diagnose_artifact_tree(source)
    output = args.output.expanduser().resolve() if args.output else source.parent / f'{source.name}-hardening-diagnostics.json'
    _write_report(output, report)
    summary = []
    for run in report['runs']:
        summary.append({'run_dir': run['run_dir'], 'requests': run['request_log_entries'],
                        'raw_responses': run['raw_responses_available'],
                        'taxonomy_counts': run['taxonomy_counts']})
    print(json.dumps({'status': 'complete', 'report': str(output), 'runs': summary}, indent=2))


if __name__ == '__main__':
    main()
