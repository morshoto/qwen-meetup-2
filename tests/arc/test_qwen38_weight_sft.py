import copy
import hashlib
import json

import pytest

from src.arc.qwen38_weight_sft import prepare_examples


def fixture():
    tasks = {}
    solutions = {}
    records = []
    for i in range(6):
        task_id = str(i)
        tasks[task_id] = {'train': [{'input': [[0]], 'output': [[1]]}], 'test': [{'input': [[2, i]]}]}
        solutions[task_id] = [[[3, i, 4]]]
        raw = json.dumps(solutions[task_id], separators=(',', ':'))
        records.append({'task_id': task_id, 'raw': raw, 'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                        'source': 'historical_fp8', 'candidate_index': 0, 'checkpoint_sha256': 'a' * 64,
                        'manifest_sha256': 'b' * 64})
    return tasks, solutions, records


def test_sft_targets_are_original_verified_responses_without_test_answers_in_prompt():
    tasks, solutions, records = fixture()
    result = prepare_examples(tasks, solutions, records, blocked_ids={'5'}, token_counter=lambda messages: 10)
    assert result['status'] == 'requires_source_replay'
    assert result['counts']['unique_parents'] == 5
    assert result['counts']['excluded_blocked'] == 1
    for item in result['examples']:
        original = records[int(item['task_id'])]
        assert item['messages'][1]['content'] == original['raw']
        prompt = json.loads(item['messages'][0]['content'])
        assert all(set(test) == {'input'} for test in prompt['task']['test'])
        assert item['source'] == 'historical_fp8'


def test_training_gate_uses_distinct_parents_after_length_filtering():
    tasks, solutions, records = fixture()
    report = prepare_examples(tasks, solutions, records, blocked_ids=set(), token_counter=lambda messages: 2049)
    assert report['status'] == 'insufficient_verified_data'
    assert report['counts']['length_excluded'] == 6
    assert report['examples'] == []
    duplicated = records[:4] + [copy.deepcopy(records[0])]
    report = prepare_examples(tasks, solutions, duplicated, blocked_ids=set(), token_counter=lambda messages: 1)
    assert report['status'] == 'insufficient_verified_data'
    assert report['counts']['unique_parents'] == 4


@pytest.mark.parametrize('change', ['raw', 'hash', 'wrong_answer', 'source', 'no_provenance'])
def test_sft_rejects_forged_or_unverified_targets(change):
    tasks, solutions, records = fixture()
    if change == 'raw': records[0]['raw'] = '[[[9]]]'
    elif change == 'hash': records[0]['raw_sha256'] = '0' * 64
    elif change == 'wrong_answer':
        records[0]['raw'] = '[[[9]]]'
        records[0]['raw_sha256'] = hashlib.sha256(records[0]['raw'].encode()).hexdigest()
    elif change == 'source': records[0]['source'] = 'ground_truth'
    else: records[0]['checkpoint_sha256'] = ''
    with pytest.raises(ValueError):
        prepare_examples(tasks, solutions, records, blocked_ids=set(), token_counter=lambda messages: 1)
