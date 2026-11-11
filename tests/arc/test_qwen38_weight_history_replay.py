import copy
import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import fingerprint
from src.arc.qwen38_weight_review import review_selection
import src.arc.qwen38_weight_history_replay as replay


def fixture(monkeypatch):
    tasks = {k: {'train': [{'input': [[i]], 'output': [[1]]}], 'test': [{'input': [[i + 5]]}]}
             for i, k in enumerate(('a', 'b', 'c', 'd'))}
    solutions = {k: [[[2, i]]] for i, k in enumerate(tasks)}
    official = {'e': {'train': [{'input': [[9, 9]], 'output': [[1]]}], 'test': [{'input': [[9, 8]]}]}}
    challenges = json.dumps(tasks, sort_keys=True).encode(); answers = json.dumps(solutions, sort_keys=True).encode()
    official_bytes = json.dumps(official, sort_keys=True).encode()
    proposal = {'status': 'proposal_not_authorization', 'allocation': {'fresh': ['c'], 'validation': ['d'],
                'challenge_fingerprints': {k: fingerprint(v) for k, v in tasks.items()}},
                'solution_fingerprints': {k: digest(solutions[k]) for k in ('c', 'd')},
                'exclusions': {'prior_phase_a': ['a', 'b'], 'protected': ['a']}, 'known_lineage': {},
                'official_evaluation_ids': ['e'], 'generation': {'prompt_sha256': 'b' * 64},
                'input_hashes': {'challenges': hashlib.sha256(challenges).hexdigest(),
                                 'solutions': hashlib.sha256(answers).hexdigest(),
                                 'official_challenges': digest(official),
                                 'official_challenges_file': hashlib.sha256(official_bytes).hexdigest()}}
    proposal['sha256'] = digest(proposal); monkeypatch.setattr(replay, 'PROPOSAL_SHA256', proposal['sha256'])
    historical = {'protocol': 'qwen38-quick-exploratory-v1', 'model_id': 'Qwen/Qwen3.8-27B-FP8',
                  'model_revision': 'c3a462cc3d38cf351f3b26cbd763469eeb74ef81', 'precision': 'fp8',
                  'candidate_count': 2, 'challenge_sha256': proposal['input_hashes']['challenges'],
                  'prompt_sha256': proposal['generation']['prompt_sha256'], 'phase_a_task_ids': ['a', 'b']}
    historical['sha256'] = digest(historical)
    rows = {}
    for task_id in ('a', 'b'):
        raw = json.dumps(solutions[task_id], separators=(',', ':'))
        candidates = [{'candidate_index': i, 'raw': raw, 'outputs': solutions[task_id], 'valid': True,
                       'parse_mode': 'strict'} for i in range(2)]
        rows[task_id] = {'task_id': task_id, 'manifest_sha256': historical['sha256'], 'status': 'complete',
                         'test_count': 1, 'candidates': candidates}
    aggregate = json.dumps({'manifest': historical, 'complete': True, 'task_results': rows}, sort_keys=True).encode()
    checkpoint_bytes = {k: json.dumps(v, sort_keys=True).encode() for k, v in rows.items()}
    monkeypatch.setattr(replay, 'HISTORY_MANIFEST_SHA256', historical['sha256'])
    monkeypatch.setattr(replay, 'HISTORY_AGGREGATE_SHA256', hashlib.sha256(aggregate).hexdigest())
    review = {'status': 'reviewed_bounded', 'proposal_sha256': proposal['sha256'],
              'selection': review_selection(proposal, tasks, solutions, official),
              'history_inspection': {'checkpoint_inventory_sha256': digest({k: hashlib.sha256(v).hexdigest() for k, v in checkpoint_bytes.items()})}}
    review['sha256'] = digest(review)
    manifest = {'status': 'execution_ready', 'proposal_sha256': proposal['sha256'], 'lineage_review_sha256': review['sha256'],
                'allocation': proposal['allocation'], 'input_hashes': proposal['input_hashes']}
    manifest['sha256'] = digest(manifest)
    return dict(manifest=manifest, proposal=proposal, review=review, challenges=challenges, solutions=answers,
                official_bytes=official_bytes, history_manifest=historical, aggregate_bytes=aggregate,
                checkpoint_bytes=checkpoint_bytes)


def test_historical_replay_excludes_protected_and_preserves_generated_response(monkeypatch):
    args = fixture(monkeypatch)
    report = replay.replay_history(**args)
    assert report['counts']['blocked_parents'] == 1
    assert report['counts']['eligible_unique_parents'] == 1
    assert report['records'][0]['task_id'] == 'b' and report['records'][0]['source'] == 'historical_fp8'
    assert report['records'][0]['raw'] == '[[[2,1]]]'
    assert report['records'][0]['checkpoint_sha256'] == hashlib.sha256(args['checkpoint_bytes']['b']).hexdigest()


@pytest.mark.parametrize('change', ['dataset', 'official', 'proposal', 'review', 'checkpoint', 'aggregate'])
def test_historical_replay_rejects_changed_sources(monkeypatch, change):
    args = fixture(monkeypatch)
    if change == 'dataset': args['solutions'] += b' '
    elif change == 'official': args['official_bytes'] += b' '
    elif change == 'proposal': args['proposal']['allocation']['validation'] = ['b']
    elif change == 'review': args['review']['status'] = 'draft'
    elif change == 'checkpoint': args['checkpoint_bytes']['b'] += b' '
    else: args['aggregate_bytes'] += b' '
    with pytest.raises(ValueError): replay.replay_history(**args)
