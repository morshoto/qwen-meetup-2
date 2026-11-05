"""Fixed-batch P3-C collection and unique-task-gated training commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from src.core.config import load_experiment_config
from src.pipelines.p3a import run_real
from src.pipelines.p3b import collect_verified_exploration_records, train_p3b


def _canonical_records(records: list[dict]) -> bytes:
    return ''.join(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n'
                   for record in records).encode('utf-8')


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def collect(config_path: Path, output: Path | None, resume_dir: Path | None) -> dict:
    config = load_experiment_config(config_path)
    if not config.p3c.enabled or config.rsi.trainer != 'dry_run' or config.experiment.conditions != ['B']:
        raise ValueError('P3-C collect requires an enabled fixed-batch, B-only, non-training config')
    source_run = run_real(config, resume=resume_dir is not None, run_dir=resume_dir)
    state = json.loads((source_run / 'run_state.json').read_text(encoding='utf-8'))
    if state.get('status') != 'complete' or state.get('trained') is not False:
        raise RuntimeError(f'P3-C inference run is not complete: {state.get("status")}')
    records, manifest = collect_verified_exploration_records(
        source_run, required_tasks=len(config.p3c.task_ids))
    target = output or (Path(config.tracking.output_dir) / f'p3c-collection-{uuid.uuid4().hex[:12]}')
    target = target.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f'P3-C collection artifact already exists: {target}')
    target.mkdir(parents=True)
    records_bytes = _canonical_records(records)
    (target / 'verified_records.jsonl').write_bytes(records_bytes)
    _write_json(target / 'source_manifest.json', manifest)
    _write_json(target / 'collection.json', {
        'protocol': 'p3c_verified_collection_artifact_v1',
        'source_run': str(source_run.resolve()),
        'source_identity_sha256': manifest['source_identity_sha256'],
        'source_manifest_sha256': hashlib.sha256((target / 'source_manifest.json').read_bytes()).hexdigest(),
        'verified_records_sha256': hashlib.sha256(records_bytes).hexdigest(),
        'verified_record_count': len(records),
        'verified_unique_task_count': manifest['verified_unique_task_count'],
        'fixed_batch_task_ids': manifest['fixed_batch_task_ids'],
        'diagnostic_task_ids_excluded': manifest['diagnostic_task_ids_excluded'],
        'test_outputs_in_messages': False,
    })
    return {'status': 'complete', 'source_run': str(source_run), 'collection': str(target),
            'verified_record_count': len(records),
            'verified_unique_task_count': manifest['verified_unique_task_count'],
            'fixed_batch_task_ids': manifest['fixed_batch_task_ids']}


def train(config_path: Path, collection: Path, output: Path | None) -> dict:
    config = load_experiment_config(config_path)
    if not config.p3c.enabled or config.rsi.trainer != 'mlx_lora':
        raise ValueError('P3-C train requires an enabled P3-C config with MLX-LoRA training')
    root = collection.expanduser().resolve()
    metadata_path, manifest_path, records_path = (root / 'collection.json', root / 'source_manifest.json',
                                                   root / 'verified_records.jsonl')
    if any(not path.is_file() for path in (metadata_path, manifest_path, records_path)):
        raise ValueError('P3-C collection artifact is incomplete')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records_bytes = records_path.read_bytes()
    if (metadata.get('protocol') != 'p3c_verified_collection_artifact_v1' or
            hashlib.sha256(records_bytes).hexdigest() != metadata.get('verified_records_sha256') or
            hashlib.sha256(manifest_path.read_bytes()).hexdigest() != metadata.get('source_manifest_sha256') or
            manifest.get('source_identity_sha256') != metadata.get('source_identity_sha256') or
            manifest.get('test_outputs_in_messages') is not False or
            set(manifest.get('fixed_batch_task_ids', [])) != set(config.p3c.task_ids) or
            set(manifest.get('diagnostic_task_ids_excluded', [])) & set(config.p3c.task_ids)):
        raise ValueError('P3-C collection integrity or configured fixed-batch check failed')
    source_run = Path(manifest.get('source_run', '')).expanduser().resolve()
    if (source_run != Path(metadata.get('source_run', '')).expanduser().resolve() or
            not (source_run / 'config.snapshot.yml').is_file() or
            hashlib.sha256((source_run / 'identity.json').read_bytes()).hexdigest() != metadata.get('source_identity_sha256')):
        raise ValueError('P3-C collection source run identity does not match its artifact')
    source_config = load_experiment_config(source_run / 'config.snapshot.yml')
    if (source_config.experiment.conditions != ['B'] or source_config.experiment.generations != 0 or
            not source_config.p3c.enabled or source_config.model.adapter_sha256 != 'none' or
            source_config.model.id != config.model.id or source_config.model.revision != config.model.revision or
            source_config.model.quantization != config.model.quantization or
            source_config.model.thinking != config.model.thinking or
            source_config.data.split_seed != config.data.split_seed or
            source_config.p3c.task_ids != config.p3c.task_ids or
            source_config.p3c.diagnostic_task_manifest != config.p3c.diagnostic_task_manifest):
        raise ValueError('P3-C training config differs from the verified B-only source experiment')
    for manifest_key, path in (('challenge_sha256', source_config.data.challenges),
                               ('solution_sha256', source_config.data.solutions)):
        if hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest() != manifest.get(manifest_key):
            raise ValueError('P3-C training data differs from the verified source dataset')
    for training_path, source_path in ((config.data.challenges, source_config.data.challenges),
                                       (config.data.solutions, source_config.data.solutions)):
        if hashlib.sha256(Path(training_path).expanduser().read_bytes()).hexdigest() != hashlib.sha256(
                Path(source_path).expanduser().read_bytes()).hexdigest():
            raise ValueError('P3-C training config uses a different ARC dataset')
    records = [json.loads(line) for line in records_bytes.decode('utf-8').splitlines() if line.strip()]
    if _canonical_records(records) != records_bytes:
        raise ValueError('P3-C training records are not in canonical JSONL form')
    fresh_records, fresh_manifest = collect_verified_exploration_records(
        manifest['source_run'], required_tasks=len(config.p3c.task_ids))
    fresh_bytes = _canonical_records(fresh_records)
    if (fresh_bytes != records_bytes or fresh_manifest != manifest or
            hashlib.sha256(fresh_bytes).hexdigest() != metadata.get('verified_records_sha256')):
        raise ValueError('P3-C records could not be reproduced from the verified source run')
    target = output or (Path(config.tracking.output_dir) / f'p3c-training-{uuid.uuid4().hex[:12]}')
    result = train_p3b(config, fresh_records, fresh_manifest, target, source='exploration')
    return {'result': result, 'training_artifact': str(target.expanduser().resolve()),
            'minimum_verified_unique_tasks': config.p3b.insufficient_data.min_verified_unique_tasks,
            'promotion': 'not_evaluated'}


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect verifier-approved P3-C Exploration solutions')
    subparsers = parser.add_subparsers(dest='command', required=True)
    collect_parser = subparsers.add_parser('collect', help='run the fixed B-only Exploration collection batch')
    collect_parser.add_argument('--config', type=Path, required=True)
    collect_parser.add_argument('--output', type=Path)
    collect_parser.add_argument('--resume-dir', type=Path)
    train_parser = subparsers.add_parser('train', help='train only from a reproduced verifier-built collection')
    train_parser.add_argument('--config', type=Path, required=True)
    train_parser.add_argument('--collection', type=Path, required=True)
    train_parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'collect':
        result = collect(args.config, args.output, args.resume_dir)
    else:
        result = train(args.config, args.collection, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
