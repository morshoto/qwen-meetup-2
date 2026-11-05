"""Materialize replayed synthetic rows without authorizing model training."""
import hashlib
import re

from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_generation_drafts import audit_generation_drafts
from src.arc.rsi_sft_materialize import materialize_quarantined_dataset, verify_quarantined_dataset
from src.arc.rsi_slot_lifecycle import _create, _directory


def materialize_replayed_synthetic_dataset(*, expected_stage_sha256, expected_preparation_sha256,
                                            synthetic_tasks, tokenizer, render, **inputs):
    if (not isinstance(expected_preparation_sha256, str) or
            not re.fullmatch(r'[0-9a-f]{64}', expected_preparation_sha256)):
        raise ValueError('invalid preparation digest')
    root = inputs['root']
    with _directory(root) as directory:
        marker_bytes, marker = _read(directory, 'synthetic.training.replay.json')
        preparation_bytes, plan = _read(directory, 'synthetic.training.preparation.json')
    expected_marker = {'status': 'synthetic_preparation_replay_consistent', 'training_allowed': False,
                       'stage_sha256': expected_stage_sha256, 'preparation_sha256': expected_preparation_sha256,
                       'evidence_type': 'caller_supplied_capture_and_sandbox'}
    if (marker != expected_marker or hashlib.sha256(preparation_bytes).hexdigest() != expected_preparation_sha256 or
            plan.get('stage_sha256') != expected_stage_sha256 or
            plan.get('mapping') != 'generated_demos_holdout_first_visible_v1'):
        raise ValueError('synthetic preparation replay binding differs')
    audit_generation_drafts(expected_sha256=expected_stage_sha256, **inputs)
    arguments = dict(root=root, preparation_bytes=preparation_bytes,
                     expected_preparation_sha256=expected_preparation_sha256,
                     tasks=synthetic_tasks, tokenizer=tokenizer, render=render)
    result = materialize_quarantined_dataset(**arguments)
    verify_quarantined_dataset(**arguments, expected_dataset_sha256=result['dataset_jsonl_sha256'])
    audit_generation_drafts(expected_sha256=expected_stage_sha256, **inputs)
    with _directory(root) as directory:
        marker_after, _ = _read(directory, 'synthetic.training.replay.json')
        preparation_after, _ = _read(directory, 'synthetic.training.preparation.json')
        if marker_after != marker_bytes or preparation_after != preparation_bytes:
            raise ValueError('synthetic preparation changed during dataset creation')
        _create(directory, 'synthetic.dataset.replay.json', {
            'status': 'synthetic_dataset_replay_consistent', 'training_allowed': False,
            'stage_sha256': expected_stage_sha256, 'preparation_sha256': expected_preparation_sha256,
            'dataset_jsonl_sha256': result['dataset_jsonl_sha256'],
            'evidence_type': 'caller_supplied_capture_sandbox_and_tokenizer'})
    result['evidence_type'] = 'caller_supplied_capture_sandbox_and_tokenizer'
    return result
