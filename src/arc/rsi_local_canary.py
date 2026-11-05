"""Audit a canary with the local tokenizer and loader without model weights."""
from pathlib import Path

from src.arc.rsi_canary_dataset import audit_canary_loader
from src.arc.rsi_local_tokenizer import _REVISION, _load_components, _sha


def _load_canary_components(model):
    tokenizer, render, version = _load_components(model)
    from mlx_lm.tuner.datasets import load_local_dataset
    return tokenizer, render, load_local_dataset, version


def audit_local_canary(*, root, forbidden_tasks, model_dir, expected_tokenizer_sha256,
                       expected_config_sha256, expected_dataset_sha256):
    model = Path(model_dir).expanduser().resolve()
    if not model.is_dir() or model.name != _REVISION:
        raise ValueError('model path must be the proposed revision directory')
    files = (model / 'tokenizer.json', model / 'tokenizer_config.json')
    before = tuple(_sha(path) for path in files)
    if before != (expected_tokenizer_sha256, expected_config_sha256):
        raise ValueError('local tokenizer digest differs')
    tokenizer, render, loader, version = _load_canary_components(model)
    if version != '0.30.7':
        raise ValueError('MLX-LM version differs from proposed runtime')
    checked = audit_canary_loader(root=root, forbidden_tasks=forbidden_tasks, tokenizer=tokenizer,
        render=render, expected_dataset_sha256=expected_dataset_sha256, load_local_dataset=loader)
    if tuple(_sha(path) for path in files) != before:
        raise ValueError('local tokenizer changed during canary audit')
    return {'status': 'local_canary_loader_consistent', 'training_allowed': False,
            'row_count': checked['row_count'], 'dataset_jsonl_sha256': expected_dataset_sha256,
            'tokenizer_sha256': before[0], 'tokenizer_config_sha256': before[1], 'mlx_lm_version': version,
            'model_revision': _REVISION, 'evidence_type': 'local_tokenizer_loader_and_supplied_inventory'}
