"""Write private immutable provisional SFT bundles, without authorizing training."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_local_runner import clean_task, parse_prediction
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_preflight import PACKAGES


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _contents(selection, registration):
    required = {'execution_sha256', 'fresh_journal_sha256', 'fresh_verification_sha256', 'challenge_sha256',
                'solution_sha256', 'training_code_sha256', 'historical_aggregate_sha256',
                'historical_checkpoint_inventory_sha256', 'source_files_sha256', 'packages',
                'model_files_sha256', 'training_settings'}
    hashes = {'execution_sha256', 'fresh_journal_sha256', 'fresh_verification_sha256',
              'challenge_sha256', 'solution_sha256', 'training_code_sha256'}
    if set(registration) != required or any(not _valid_hash(registration[k]) for k in hashes):
        raise ValueError('invalid SFT source registration')
    sources, model = registration['source_files_sha256'], registration['model_files_sha256']
    if (not isinstance(sources, dict) or not sources or not all(_valid_hash(v) for v in sources.values()) or
        digest(sources) != registration['training_code_sha256'] or not isinstance(model, dict) or not model or
        not all(_valid_hash(v) for v in model.values()) or registration['packages'] != PACKAGES):
        raise ValueError('invalid SFT source, model or package inventory')
    settings = {'rank': 8, 'batch_size': 1, 'learning_rate': 2e-5, 'max_steps': 20, 'seed': 20261111,
                'max_sequence_length': 2048, 'gradient_checkpointing': True, 'completion_only_loss': True}
    if registration['training_settings'] != settings:
        raise ValueError('training settings differ from the frozen protocol')
    historical = (registration['historical_aggregate_sha256'], registration['historical_checkpoint_inventory_sha256'])
    if historical != (None, None) and not all(_valid_hash(v) for v in historical):
        raise ValueError('invalid historical source identity')
    examples = selection['examples']; counts = selection['counts']; mode = selection['mode']
    if (counts['unique_parents'] != len(examples) or len({e['task_id'] for e in examples}) != len(examples) or
        counts['historical_fp8'] != sum(e['source'] == 'historical_fp8' for e in examples) or
        counts['current_gen0_self_generated'] != sum(e['source'] == 'current_gen0_self_generated' for e in examples)):
        raise ValueError('SFT selection count mismatch')
    expected_mode = ('mixed_source' if counts['historical_fp8'] and counts['current_gen0_self_generated'] else
                     'historical_fp8_distillation' if counts['historical_fp8'] else
                     'fresh_only' if counts['current_gen0_self_generated'] else 'none')
    if mode != expected_mode:
        raise ValueError('SFT source mode mismatch')
    if counts['historical_fp8'] and registration['historical_aggregate_sha256'] is None:
        raise ValueError('missing historical source registration')
    parents = []; output_hashes = set(); data = []
    for item in examples:
        messages = item['messages']
        if (not isinstance(messages, list) or len(messages) != 2 or messages[0].get('role') != 'user' or
            messages[1].get('role') != 'assistant' or not isinstance(messages[1].get('content'), str) or
            hashlib.sha256(messages[1]['content'].encode()).hexdigest() != item['raw_sha256'] or
            not _valid_hash(item['checkpoint_sha256']) or not _valid_hash(item['manifest_sha256']) or
            type(item['token_count']) is not int or not 0 <= item['token_count'] <= 2048):
            raise ValueError('invalid SFT example provenance')
        prompt = json.loads(messages[0]['content'])
        if (set(prompt) != {'instruction', 'task'} or prompt['instruction'] != PROMPT or
            clean_task(prompt['task']) != prompt['task'] or any(set(test) != {'input'} for test in prompt['task']['test'])):
            raise ValueError('held-out answers in SFT prompt')
        parsed, _ = parse_prediction(messages[1]['content'], len(prompt['task']['test']))
        if parsed is None or digest(parsed) in output_hashes:
            raise ValueError('invalid or duplicated SFT target')
        output_hashes.add(digest(parsed))
        parents.append({key: item[key] for key in ('task_id', 'source', 'raw_sha256', 'checkpoint_sha256',
                                                   'manifest_sha256', 'candidate_index', 'token_count')})
        data.append(json.dumps({'messages': messages}, ensure_ascii=False, separators=(',', ':')) + '\n')
    enough = counts['unique_parents'] >= 5
    if selection['status'] != ('eligible_pending_source_replay' if enough else 'insufficient_verified_data'):
        raise ValueError('SFT gate status mismatch')
    data_bytes = ''.join(data).encode() if enough else None
    manifest = {'status': 'prepared_pending_source_replay' if enough else 'insufficient_verified_data',
                'mode': mode, 'counts': counts, 'fresh_only_counts': selection['fresh_only_counts'],
                'registration': registration, 'parents': parents, 'train_sha256': hashlib.sha256(data_bytes).hexdigest() if data_bytes else None}
    manifest['sha256'] = digest(manifest)
    return manifest, data_bytes


def _write(path, payload):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def write_bundle(root, selection, registration):
    manifest, data = _contents(selection, registration)
    root = Path(root); root.mkdir(mode=0o700, exist_ok=False)
    if data is not None:
        _write(root / 'train.jsonl', data)
    _write(root / 'manifest.json', (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    fd = os.open(root, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return manifest


def verify_bundle(root, selection, registration):
    """Check exact bytes against caller-supplied replay; does not authenticate the caller."""
    root = Path(root); expected, data = _contents(selection, registration)
    if json.loads((root / 'manifest.json').read_bytes()) != expected:
        raise ValueError('SFT bundle manifest differs from supplied replay')
    path = root / 'train.jsonl'
    if (data is None and path.exists()) or (data is not None and path.read_bytes() != data):
        raise ValueError('SFT data differs from supplied replay')
    return {'status': 'matches_supplied_replay', 'manifest_sha256': expected['sha256']}
