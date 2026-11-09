import copy
import hashlib
import json

import pytest

from src.arc.qwen38_weight_fresh_verify import verify_attempts
from src.arc.qwen38_weight_generation import create_request, capture_response


def fixture():
    tasks = {'a': {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]},
             'held': {'train': [{'input': [[4]], 'output': [[5]]}], 'test': [{'input': [[6]]}]}}
    answers = {'a': [[[7]]], 'held': [[[8]]]}
    challenge = json.dumps(tasks, sort_keys=True).encode(); solutions = json.dumps(answers, sort_keys=True).encode()
    manifest = {'sha256': 'a' * 64, 'allocation': {'fresh': ['a'], 'validation': ['held'], 'seed': 9},
                'input_hashes': {'challenges': hashlib.sha256(challenge).hexdigest(),
                                 'solutions': hashlib.sha256(solutions).hexdigest()},
                'generation': {'max_tokens_per_candidate': 4096},
                'model': {'id': 'mlx-community/Qwen3.8-27B-4bit',
                          'revision': '10c35caafbb80f7dc6a7a432cdd11af10a6d4818', 'quantization': '4-bit MLX'}}
    request = create_request('a', tasks['a'], candidate_index=0, seed=10009)
    response = {'raw': '[[[7]]]', 'prompt_tokens': 23, 'completion_tokens': 5,
                'finish_reason': 'stop', 'seconds': 1.2, 'peak_memory_gb': 4.0}
    captured = capture_response(request, response, 4096)
    row = {**captured, 'status': 'complete', 'elapsed_seconds': 1.2, 'manifest_sha256': manifest['sha256'], 'model': manifest['model']}
    return manifest, challenge, solutions, [row]


def test_verifier_reparses_saved_output_and_never_returns_labels():
    manifest, challenges, solutions, rows = fixture()
    result = verify_attempts(manifest, challenges, solutions, rows)
    assert result['outcomes'] == {('a', 0): True}
    assert result['counts']['verified_correct'] == 1
    assert '[[[7]]]' not in repr(result)


@pytest.mark.parametrize('change', ['source', 'answer', 'raw', 'parsed', 'seed', 'split', 'model', 'duplicate'])
def test_verifier_rejects_changed_evidence(change):
    manifest, challenges, solutions, rows = fixture(); rows = copy.deepcopy(rows)
    if change == 'source': challenges += b' '
    elif change == 'answer': solutions += b' '
    elif change == 'raw': rows[0]['raw'] = '[[[9]]]'
    elif change == 'parsed': rows[0]['outputs'] = [[[9]]]
    elif change == 'seed': rows[0]['seed'] += 1
    elif change == 'split': rows[0]['task_id'] = 'held'
    elif change == 'model': rows[0]['model']['revision'] = 'bad'
    else: rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError): verify_attempts(manifest, challenges, solutions, rows)


def test_verifier_accepts_bound_model_inventory_and_counts_errors():
    manifest, challenges, solutions, rows = fixture()
    manifest['model']['files_sha256'] = {'model.safetensors': 'f' * 64}
    rows[0]['model'] = copy.deepcopy(manifest['model'])
    result = verify_attempts(manifest, challenges, solutions, rows)
    assert result['counts']['verified_correct'] == 1
    error = {'task_id': 'a', 'candidate_index': 1, 'seed': 10010, 'status': 'error',
             'manifest_sha256': manifest['sha256'], 'model': manifest['model']}
    result = verify_attempts(manifest, challenges, solutions, rows + [error])
    assert result['counts']['error'] == 1 and ('a', 1) not in result['outcomes']
