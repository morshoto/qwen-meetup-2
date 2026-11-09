import copy
import hashlib
import json

import pytest

from src.arc.qwen38_weight_verification_receipt import build_receipt, verify_receipt, write_receipt
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


def test_receipt_binds_journal_and_returns_only_outcomes(tmp_path):
    manifest, challenges, solutions, rows = fixture()
    receipt = build_receipt(manifest, challenges, solutions, rows)
    assert receipt['outcomes'] == [{'task_id': 'a', 'candidate_index': 0, 'correct': True}]
    assert '[[[7]]]' not in json.dumps(receipt)
    path = tmp_path / 'receipt.json'
    write_receipt(path, receipt)
    assert verify_receipt(json.loads(path.read_text()), manifest, challenges, solutions, rows) == {('a', 0): True}
    with pytest.raises(FileExistsError): write_receipt(path, receipt)


def test_receipt_replay_rejects_changed_results_journal_and_sources():
    manifest, challenges, solutions, rows = fixture()
    receipt = build_receipt(manifest, challenges, solutions, rows)
    altered = copy.deepcopy(receipt); altered['outcomes'][0]['correct'] = False
    with pytest.raises(ValueError): verify_receipt(altered, manifest, challenges, solutions, rows)
    changed = copy.deepcopy(rows); changed[0]['raw'] = '[[[0]]]'
    with pytest.raises(ValueError): verify_receipt(receipt, manifest, challenges, solutions, changed)
    with pytest.raises(ValueError): verify_receipt(receipt, manifest, challenges, solutions + b' ', rows)
