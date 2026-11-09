import hashlib
import json

import pytest

from src.arc.qwen38_weight_sft import select_training_examples
from src.arc.qwen38_weight_sft_bundle import write_bundle, verify_bundle


def fixture(count):
    tasks = {}; answers = {}; records = []
    for i in range(count):
        task = str(i); tasks[task] = {'train': [{'input': [[0]], 'output': [[1]]}], 'test': [{'input': [[2, i]]}]}
        answers[task] = [[[3, i, 4]]]; raw = json.dumps(answers[task], separators=(',', ':'))
        records.append({'task_id': task, 'raw': raw, 'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                        'source': 'current_gen0_self_generated', 'candidate_index': 0,
                        'checkpoint_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64})
    selection = select_training_examples(tasks, answers, records, [], blocked_ids=set(),
                                         forbidden_answer_fingerprints=set(), token_counter=lambda _: 10)
    registration = {'execution_sha256': 'b' * 64, 'fresh_journal_sha256': 'c' * 64,
                    'fresh_verification_sha256': 'd' * 64, 'challenge_sha256': 'e' * 64,
                    'solution_sha256': 'f' * 64, 'training_code_sha256': '1' * 64,
                    'historical_aggregate_sha256': None}
    return selection, registration


def test_bundle_is_private_immutable_and_requires_source_replay(tmp_path):
    selection, registration = fixture(5)
    root = tmp_path / 'sft'
    result = write_bundle(root, selection, registration)
    assert result['status'] == 'prepared_pending_source_replay'
    assert (root / 'train.jsonl').stat().st_mode & 0o077 == 0
    lines = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    assert len(lines) == 5 and all(set(row) == {'messages'} for row in lines)
    assert verify_bundle(root, selection, registration)['status'] == 'source_replay_verified'
    with pytest.raises(FileExistsError): write_bundle(root, selection, registration)
    (root / 'train.jsonl').write_text('{}\n')
    with pytest.raises(ValueError): verify_bundle(root, selection, registration)


def test_insufficient_data_never_writes_sft_dataset(tmp_path):
    selection, registration = fixture(3)
    root = tmp_path / 'insufficient'
    result = write_bundle(root, selection, registration)
    assert result['status'] == 'insufficient_verified_data'
    assert not (root / 'train.jsonl').exists()
    assert 'score' not in result
