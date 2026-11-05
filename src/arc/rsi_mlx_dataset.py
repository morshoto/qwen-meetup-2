"""Audit MLX dataset loading without loading weights or authorizing training."""
from pathlib import Path
from types import SimpleNamespace

from src.arc.rsi_sft_format import _tokens
from src.arc.rsi_sft_materialize import verify_quarantined_dataset
from src.arc.rsi_transport_audit import _json_object


def audit_mlx_dataset(*, root, preparation_bytes, expected_preparation_sha256,
                      expected_dataset_sha256, tasks, tokenizer, render, load_local_dataset):
    if not callable(load_local_dataset):
        raise ValueError('dataset loader unavailable')
    arguments = dict(root=root, preparation_bytes=preparation_bytes,
                     expected_preparation_sha256=expected_preparation_sha256,
                     expected_dataset_sha256=expected_dataset_sha256, tasks=tasks,
                     tokenizer=tokenizer, render=render)
    checked = verify_quarantined_dataset(**arguments)
    plan = _json_object(preparation_bytes)
    rows = [{'messages': record['messages']} for record in plan['generation_training']['records']]
    train, valid, test = load_local_dataset(Path(root) / 'generation-sft.quarantine', tokenizer,
                                            SimpleNamespace(mask_prompt=True))
    if len(train) != len(rows) or len(valid) or len(test):
        raise ValueError('loaded dataset split differs')
    for index, expected in enumerate(rows):
        actual = train[index]
        if actual != expected:
            raise ValueError('loaded dataset row differs')
        actual_tokens, actual_offset = train.process(actual)
        expected_tokens, expected_offset = render(expected)
        if (_tokens(actual_tokens) != _tokens(expected_tokens) or type(actual_offset) is not int or
                actual_offset != expected_offset):
            raise ValueError('loaded dataset token mask differs')
    verify_quarantined_dataset(**arguments)
    return {'status': 'dataset_loader_consistent_unverified', 'training_allowed': False,
            'row_count': checked['row_count'], 'dataset_jsonl_sha256': expected_dataset_sha256,
            'evidence_type': 'caller_supplied_loader_tokenizer_and_renderer'}
