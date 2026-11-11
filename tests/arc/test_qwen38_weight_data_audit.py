import copy
import hashlib
import json

import pytest

from src.arc.qwen38_weight_data_audit import extract_fresh_records
from src.arc.qwen38_weight_generation import create_request, capture_response
from src.arc.qwen38_weight_verification_receipt import build_receipt


def fixture():
    tasks = {'a': {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}}
    solutions = {'a': [[[4]]]}
    challenges = json.dumps(tasks, sort_keys=True).encode(); answers = json.dumps(solutions, sort_keys=True).encode()
    model = {'id': 'mlx-community/Qwen3.8-27B-4bit', 'revision': '10c35caafbb80f7dc6a7a432cdd11af10a6d4818',
             'quantization': '4-bit MLX'}
    manifest = {'sha256': 'a' * 64, 'allocation': {'fresh': ['a'], 'validation': [], 'seed': 9}, 'model': model,
                'input_hashes': {'challenges': hashlib.sha256(challenges).hexdigest(), 'solutions': hashlib.sha256(answers).hexdigest()},
                'generation': {'initial_candidates': 2, 'additional_candidates': 6, 'max_tokens_per_candidate': 10,
                               'reserved_token_limit': 20, 'wall_seconds_limit': 100}}
    rows = []
    for index, raw in enumerate(('[[[4]]]', '[[[0]]]')):
        request = create_request('a', tasks['a'], candidate_index=index, seed=10009 + index)
        response = {'raw': raw, 'prompt_tokens': 5, 'completion_tokens': 1, 'finish_reason': 'stop',
                    'seconds': 1, 'peak_memory_gb': 1}
        rows.append({**capture_response(request, response, 10), 'status': 'complete', 'elapsed_seconds': 1,
                     'manifest_sha256': manifest['sha256'], 'model': model})
    receipt = build_receipt(manifest, challenges, answers, rows)
    return manifest, challenges, answers, rows, receipt


def test_extract_requires_terminal_replayed_verification_and_preserves_raw():
    manifest, challenges, answers, rows, receipt = fixture()
    result = extract_fresh_records(manifest, challenges, answers, rows, receipt, wall_elapsed=10)
    assert result['collection_status'] == 'complete'
    assert len(result['records']) == 1 and result['records'][0]['raw'] == '[[[4]]]'
    assert result['records'][0]['source'] == 'current_gen0_self_generated'
    assert result['records'][0]['manifest_sha256'] == manifest['sha256']


def test_extract_rejects_active_or_changed_records_and_resource_stop():
    manifest, challenges, answers, rows, receipt = fixture()
    changed = copy.deepcopy(rows); changed[0]['raw'] = '[[[9]]]'
    with pytest.raises(ValueError): extract_fresh_records(manifest, challenges, answers, changed, receipt, wall_elapsed=10)
    manifest['generation']['reserved_token_limit'] = 80
    active = copy.deepcopy(rows); active[0] = copy.deepcopy(rows[1]); active[0]['candidate_index'] = 0; active[0]['seed'] = 10009
    active_receipt = build_receipt(manifest, challenges, answers, active)
    with pytest.raises(ValueError): extract_fresh_records(manifest, challenges, answers, active, active_receipt, wall_elapsed=10)
    incomplete = [dict(rows[0], status='started')]
    with pytest.raises(ValueError):
        extract_fresh_records(manifest, challenges, answers, incomplete,
                              build_receipt(manifest, challenges, answers, incomplete), wall_elapsed=1000)
