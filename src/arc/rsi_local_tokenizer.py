"""Read-only local tokenizer format audit; does not load model weights or train."""
import hashlib
import importlib.metadata
import os
from pathlib import Path

from src.arc.rsi_sft_format import audit_sft_format

_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


def _sha(path):
    try:
        with path.open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise ValueError('local tokenizer file unavailable') from exc


def _load_components(model):
    version = importlib.metadata.version('mlx-lm')
    if version != '0.30.7':
        raise ValueError('MLX-LM version differs from proposed runtime')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from src.arc.mlx_chat_template import install_training_template_policy
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.utils import load_tokenizer
    install_training_template_policy('off')
    tokenizer = load_tokenizer(model)
    dataset = ChatDataset([], tokenizer, mask_prompt=True)
    return tokenizer, dataset.process, version


def audit_local_tokenizer(plan, *, tasks, model_dir, expected_tokenizer_sha256, expected_config_sha256):
    model = Path(model_dir).expanduser().resolve()
    if not model.is_dir() or model.name != _REVISION:
        raise ValueError('model path must be the proposed revision directory')
    tokenizer_file, config_file = model / 'tokenizer.json', model / 'tokenizer_config.json'
    before = (_sha(tokenizer_file), _sha(config_file))
    if before != (expected_tokenizer_sha256, expected_config_sha256):
        raise ValueError('local tokenizer digest differs')
    tokenizer, render, version = _load_components(model)
    if version != '0.30.7':
        raise ValueError('MLX-LM version differs from proposed runtime')
    checked = audit_sft_format(plan, tasks=tasks, tokenizer=tokenizer, render=render)
    if (_sha(tokenizer_file), _sha(config_file)) != before:
        raise ValueError('local tokenizer changed during audit')
    return {'status': 'local_tokenizer_format_consistent', 'training_allowed': False,
            'row_count': checked['row_count'], 'token_spans': checked['token_spans'],
            'dataset_jsonl_sha256': checked['dataset_jsonl_sha256'], 'mlx_lm_version': version,
            'tokenizer_sha256': before[0], 'tokenizer_config_sha256': before[1],
            'model_revision': _REVISION, 'evidence_type': 'local_tokenizer_only_unverified_provenance'}
