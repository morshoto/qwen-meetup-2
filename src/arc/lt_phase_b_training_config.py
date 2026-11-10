"""Validate private SFT files and propose, but never authorize, real training."""
import hashlib
import json
import os
import re
import stat
from pathlib import Path

from src.arc.lt_phase_b_canary_config import _REVISION
from src.arc.lt_phase_b_dispatch import _private_run_directory
from src.arc.rsi_model_identity import _hash_file, _json_file


def _private_file(path):
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('SFT artifact must be a private regular file')
    return _hash_file(path)


def build_training_config(*, source_root, run_dir, model_dir, expected_source):
    try:
        _, run = _private_run_directory(source_root, run_dir)
        model_path = Path(model_dir).expanduser()
        if model_path.is_symlink():
            raise ValueError('model directory is a symlink')
        model = model_path.resolve(strict=True)
        if (not model.is_dir() or model.name != _REVISION or model == run or
                model in run.parents or run in model.parents):
            raise ValueError('invalid or overlapping model directory')
        if (not isinstance(expected_source, dict) or
                expected_source.get('model_id') != 'mlx-community/Qwen3-14B-4bit' or
                expected_source.get('model_revision') != _REVISION or
                any(not isinstance(expected_source.get(k), str) or
                    not re.fullmatch(r'[0-9a-f]{64}', expected_source[k])
                    for k in ('source_code_sha256', 'run_sha256')) or
                not isinstance(expected_source.get('run_id'), str) or
                run.name != 'lt-phase-b-' + expected_source['run_id']):
            raise ValueError('invalid expected source')
        data = run / 'sft-data'
        info = data.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('SFT directory must be private')
        manifest_path = data / 'manifest.json'
        manifest_hash = _private_file(manifest_path)
        manifest = _json_file(manifest_path, manifest_hash)
        if (manifest.get('status') != 'private_sft_materialized_unverified' or
                manifest.get('training_allowed') is not False or
                manifest.get('token_audit_status') != 'dataset_tokens_consistent_unverified' or
                manifest.get('source') != expected_source or
                type(manifest.get('maximum_tokens')) is not int or
                not 0 < manifest['maximum_tokens'] <= 512):
            raise ValueError('invalid SFT manifest')
        split = manifest.get('split_task_ids')
        if not isinstance(split, dict) or set(split) != {'train', 'valid'}:
            raise ValueError('invalid split')
        train, valid = split['train'], split['valid']
        if (not isinstance(train, list) or not isinstance(valid, list) or
                any(not isinstance(x, str) or not x for x in train + valid)):
            raise ValueError('invalid task IDs')
        ids = sorted(train + valid)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('empty or repeated tasks')
        nvalid = max(1, len(ids) // 5) if len(ids) >= 3 else 0
        status = 'disjoint_split' if nvalid else 'overfitting_only'
        if (manifest.get('yield_status') != status or valid != ids[:nvalid] or train != ids[nvalid:] or
                type(manifest.get('train_count')) is not int or manifest['train_count'] != len(train) or
                type(manifest.get('valid_count')) is not int or manifest['valid_count'] != len(valid)):
            raise ValueError('invalid yield or split')
        lineages = manifest.get('solution_lineage_sha256')
        if not isinstance(lineages, dict) or set(lineages) != set(ids):
            raise ValueError('invalid lineages')
        for task in ids:
            raw = json.dumps({'task_id': task}, sort_keys=True, separators=(',', ':')).encode()
            if lineages[task] != hashlib.sha256(raw).hexdigest():
                raise ValueError('task lineage differs')
        names = {'manifest.json', 'train.jsonl'} | ({'valid.jsonl'} if nvalid else set())
        if {p.name for p in data.iterdir()} != names:
            raise ValueError('unexpected SFT files')
        for name in ('train', 'valid'):
            expected = manifest.get(name + '_jsonl_sha256')
            if name == 'valid' and not nvalid:
                if expected is not None:
                    raise ValueError('unexpected validation hash')
            elif not isinstance(expected, str) or _private_file(data / (name + '.jsonl')) != expected:
                raise ValueError('SFT file hash differs')
        adapter = run / 'gen1-adapter'
        if adapter.exists() or adapter.is_symlink() or _private_file(manifest_path) != manifest_hash:
            raise ValueError('adapter exists or manifest changed')
    except (OSError, TypeError, RuntimeError) as exc:
        raise ValueError('invalid training inputs') from exc
    config = {'model': str(model), 'train': True, 'fine_tune_type': 'lora', 'optimizer': 'adam',
              'optimizer_config': {'adam': {}, 'adamw': {}, 'muon': {}, 'sgd': {}, 'adafactor': {}},
              'data': str(data), 'seed': 20261111, 'num_layers': 8, 'batch_size': 1, 'iters': 20,
              'val_batches': -1 if nvalid else 0, 'learning_rate': 0.0001, 'steps_per_report': 1,
              'steps_per_eval': 21, 'resume_adapter_file': None, 'adapter_path': str(adapter),
              'save_every': 20, 'test': False, 'test_batches': 500, 'max_seq_length': 512,
              'grad_checkpoint': False, 'grad_accumulation_steps': 1, 'lr_schedule': None,
              'lora_parameters': {'rank': 8, 'scale': 2.0, 'dropout': 0.0},
              'mask_prompt': True, 'report_to': None, 'project_name': None}
    return {'status': 'real_training_configuration_proposed_unverified', 'execution_ready': False,
            'training_allowed': False, 'yield_status': status, 'thinking_mode': 'off',
            'checkpoint_policy': 'final_step_only', 'dataset_manifest_sha256': manifest_hash,
            'maximum_wall_seconds': 14400, 'trainer_config': config,
            'remaining_gates': ['frozen_protocol', 'real_14b_canary_and_reload',
                                'collection_provenance', 'resource_monitor_registration']}
