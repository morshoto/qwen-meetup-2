"""Write private immutable provisional SFT bundles, without authorizing training."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from src.arc.qwen38_local_runner import parse_prediction
from src.arc.qwen38_rsi_comparison import digest


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _contents(selection, registration):
    required = {'execution_sha256', 'fresh_journal_sha256', 'fresh_verification_sha256', 'challenge_sha256',
                'solution_sha256', 'training_code_sha256', 'historical_aggregate_sha256'}
    if set(registration) != required or any(not _valid_hash(registration[k]) for k in required - {'historical_aggregate_sha256'}):
        raise ValueError('invalid SFT source registration')
    if registration['historical_aggregate_sha256'] is not None and not _valid_hash(registration['historical_aggregate_sha256']):
        raise ValueError('invalid historical source identity')
    examples = selection['examples']; counts = selection['counts']; mode = selection['mode']
    if (counts['unique_parents'] != len(examples) or len({e['task_id'] for e in examples}) != len(examples) or
        counts['historical_fp8'] != sum(e['source'] == 'historical_fp8' for e in examples) or
        counts['current_gen0_self_generated'] != sum(e['source'] == 'current_gen0_self_generated' for e in examples)):
        raise ValueError('SFT selection count mismatch')
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
        if any(set(test) != {'input'} for test in prompt['task']['test']):
            raise ValueError('held-out answers in SFT prompt')
        parsed, _ = parse_prediction(messages[1]['content'], len(prompt['task']['test']))
        if parsed is None or digest(parsed) in output_hashes:
            raise ValueError('invalid or duplicated SFT target')
        output_hashes.add(digest(parsed))
        parents.append({key: item[key] for key in ('task_id', 'source', 'raw_sha256', 'checkpoint_sha256',
                                                   'manifest_sha256', 'candidate_index', 'token_count')})
        data.append(json.dumps({'messages': messages}, ensure_ascii=False, separators=(',', ':')) + '\n')
    enough = counts['unique_parents'] >= 5
    if (selection['status'] == 'eligible_pending_source_replay') != enough:
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
