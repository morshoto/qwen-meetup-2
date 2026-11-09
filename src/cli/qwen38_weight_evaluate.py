"""Prepare and run a matched exploratory adapter evaluation after source replay."""
from __future__ import annotations

import json
from pathlib import Path

from src.arc.qwen38_weight_allocation import freeze_registration
from src.arc.qwen38_weight_eval_registration import build_evaluation_descriptor
from src.arc.qwen38_weight_training_receipt import verify_training_evidence
from src.cli.qwen38_weight_prepare_sft import replay_bundle


def _context(**kwargs):
    comparison_dir = Path(kwargs['comparison_dir']).resolve()
    inputs = {k: v for k, v in kwargs.items() if k != 'comparison_dir'}
    root = Path(inputs['run_dir']).resolve()
    if inputs['manifest'].get('run_dir') != str(root):
        raise ValueError('evaluation run does not match frozen collection')
    selection, registration, replay = replay_bundle(**inputs)
    if (selection.get('status') != 'eligible_pending_source_replay' or selection['counts'].get('unique_parents', 0) < 5 or
        replay.get('status') != 'matches_supplied_replay'):
        raise ValueError('SFT source replay or gate failed')
    training = json.loads((root / 'training/training-result-v1.json').read_bytes())
    evidence = verify_training_evidence(training, root, inputs['model_path'], registration, replay['manifest_sha256'])
    if evidence.get('status') != 'source_evidence_matches':
        raise ValueError('training evidence replay failed')
    comparison = json.loads((comparison_dir / 'manifest.json').read_bytes())
    ids = comparison['allocation']['evaluation']; baseline_dir = comparison_dir / 'evaluation/baseline'
    if {p.stem for p in baseline_dir.glob('*.json')} != set(ids):
        raise ValueError('baseline checkpoint inventory differs')
    files = {}
    for task_id in ids:
        path = baseline_dir / (task_id + '.json')
        if path.is_symlink(): raise ValueError('baseline checkpoint cannot be a symlink')
        files[task_id] = path.read_bytes()
    descriptor = build_evaluation_descriptor(comparison, inputs['manifest'], training, evidence,
                                             inputs['challenges'], files, inputs['repo'])
    return descriptor, comparison, training, selection, files


def freeze_evaluation(**kwargs):
    descriptor, _, _, _, _ = _context(**kwargs)
    root = Path(kwargs['run_dir']).resolve() / 'evaluation'
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError('evaluation directory is not private')
    return freeze_registration(root / 'evaluation-v1.json', {k: v for k, v in descriptor.items() if k != 'sha256'})


def verify_evaluation_descriptor(**kwargs):
    descriptor, _, _, _, _ = _context(**kwargs)
    saved = json.loads((Path(kwargs['run_dir']).resolve() / 'evaluation/evaluation-v1.json').read_bytes())
    if saved != descriptor:
        raise ValueError('evaluation descriptor changed')
    return descriptor
