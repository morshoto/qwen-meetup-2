import copy
import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_history import audit_history


def fixture():
    tasks = {x: {'train': [{'input': [[i]], 'output': [[i]]}], 'test': [{'input': [[i + 4]]}]}
             for i, x in enumerate(('a', 'b', 'c'))}
    answers = {'a': [[[1]]], 'b': [[[2]]], 'c': [[[3]]]}
    manifest = {'protocol': 'qwen38-quick-exploratory-v1', 'model_id': 'Qwen/Qwen3.8-27B-FP8',
                'model_revision': 'c3a462cc3d38cf351f3b26cbd763469eeb74ef81', 'precision': 'fp8',
                'phase_a_task_ids': list(tasks), 'candidate_count': 2, 'challenge_sha256': 'a' * 64,
                'prompt_sha256': 'b' * 64}
    manifest['sha256'] = digest(manifest)
    rows = {}
    for i, task_id in enumerate(tasks):
        candidates = []
        for index in range(2):
            grid = answers[task_id] if index == 0 else [[[9]]]
            raw = json.dumps(grid, separators=(',', ':'))
            candidates.append({'candidate_index': index, 'raw': raw, 'outputs': grid, 'valid': True,
                               'parse_mode': 'strict'})
        rows[task_id] = {'task_id': task_id, 'status': 'complete', 'manifest_sha256': manifest['sha256'],
                         'test_count': 1, 'candidates': candidates}
    aggregate = {'manifest': manifest, 'task_results': rows, 'complete': True}
    raw = json.dumps(aggregate, sort_keys=True).encode()
    return dict(tasks=tasks, solutions=answers, manifest=manifest, aggregate_bytes=raw,
                checkpoints=copy.deepcopy(rows), blocked_ids={'c'}, expected_aggregate_sha256=hashlib.sha256(raw).hexdigest(),
                expected_manifest_sha256=manifest['sha256'], expected_challenge_sha256='a' * 64,
                expected_prompt_sha256='b' * 64)


def test_history_requires_saved_generated_output_and_excludes_protected():
    report = audit_history(**fixture())
    assert report['counts']['eligible_unique_parents'] == 2
    assert report['counts']['blocked_parents'] == 1
    assert all(row['source'] == 'historical_fp8' for row in report['eligible'])
    assert {row['task_id'] for row in report['eligible']} == {'a', 'b'}
    assert all(row['raw'] for row in report['eligible'])


@pytest.mark.parametrize('change', ['aggregate', 'manifest', 'checkpoint', 'answers', 'missing_raw'])
def test_history_rejects_tampered_or_unverifiable_evidence(change):
    args = fixture()
    if change == 'aggregate': args['aggregate_bytes'] += b' '
    elif change == 'manifest': args['expected_manifest_sha256'] = 'f' * 64
    elif change == 'checkpoint': args['checkpoints']['a']['candidates'][0]['raw'] = '[[[9]]]'
    elif change == 'answers': del args['solutions']['a']
    else:
        aggregate = json.loads(args['aggregate_bytes']); del aggregate['task_results']['a']['candidates'][0]['raw']
        args['checkpoints'] = copy.deepcopy(aggregate['task_results'])
        args['aggregate_bytes'] = json.dumps(aggregate, sort_keys=True).encode()
        args['expected_aggregate_sha256'] = hashlib.sha256(args['aggregate_bytes']).hexdigest()
    with pytest.raises(ValueError): audit_history(**args)
