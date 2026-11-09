import copy

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import allocate_weight_pool
from src.arc.qwen38_weight_review import review_selection


def fixture():
    tasks = {str(i): {'train': [{'input': [[i]], 'output': [[1]]}],
                      'test': [{'input': [[100 + i]]}]} for i in range(12)}
    solutions = {k: [[[200 + int(k)]]] for k in tasks}
    official = {'official': {'train': [{'input': [[400]], 'output': [[1]]}], 'test': [{'input': [[401]]}]}}
    exclusions = {'protected': ['0', '1']}; lineage = {'0': 'known', '2': 'known'}
    allocation = allocate_weight_pool(tasks, official, exclusions, lineage, pool_size=5, validation_size=3)
    selected = allocation['fresh'] + allocation['validation']
    proposal = {'allocation': allocation, 'exclusions': exclusions, 'known_lineage': lineage,
                'official_evaluation_ids': ['official'],
                'solution_fingerprints': {k: digest(solutions[k]) for k in selected}}
    return proposal, tasks, solutions, official


def test_review_rechecks_protected_inputs_lineage_and_answer_fingerprints():
    proposal, tasks, solutions, official = fixture()
    report = review_selection(proposal, tasks, solutions, official)
    assert report['fresh_parents'] == 5 and report['validation_parents'] == 3
    assert report['selected_parent_overlap'] == report['selected_input_overlap'] == 0
    assert report['unresolved_semantic_exposure'] is True


@pytest.mark.parametrize('change', ['blocked_parent', 'blocked_input', 'lineage', 'answers', 'official'])
def test_review_rejects_overlap_or_drift(change):
    proposal, tasks, solutions, official = fixture()
    proposal = copy.deepcopy(proposal); tasks = copy.deepcopy(tasks)
    selected = proposal['allocation']['fresh'][0]
    if change == 'blocked_parent': proposal['exclusions']['protected'].append(selected)
    elif change == 'blocked_input': tasks[selected]['test'][0]['input'] = tasks['0']['train'][0]['input']
    elif change == 'lineage': proposal['known_lineage'][selected] = 'known'
    elif change == 'answers': solutions[selected] = [[[999]]]
    else: official = {}
    with pytest.raises(ValueError): review_selection(proposal, tasks, solutions, official)


def test_bounded_review_binds_replayed_sources_and_historical_run(monkeypatch):
    import src.arc.qwen38_weight_review as review
    proposal, tasks, solutions, official = fixture()
    proposal['input_hashes'] = {'challenges': 'a' * 64, 'solutions': 'b' * 64}
    proposal['sha256'] = digest(proposal)
    monkeypatch.setattr(review, 'PROPOSAL_SHA256', proposal['sha256'])
    observed = review_selection(proposal, tasks, solutions, official)
    observed['official_challenges'] = 120
    receipt = review.bind_review(proposal, observed, proposal['input_hashes'], 'c' * 64,
                                 review.HISTORY_MANIFEST_SHA256, review.HISTORY_AGGREGATE_SHA256,
                                 review.PHASE_B_HEAD)
    assert receipt['status'] == 'reviewed_bounded' and receipt['unresolved_exposure'] is True
    assert receipt['proposal_sha256'] == proposal['sha256']
    with pytest.raises(ValueError):
        review.bind_review(proposal, observed, {'challenges': 'd' * 64}, 'c' * 64,
                           review.HISTORY_MANIFEST_SHA256, review.HISTORY_AGGREGATE_SHA256, review.PHASE_B_HEAD)
    observed['selected_input_overlap'] = 1
    with pytest.raises(ValueError):
        review.bind_review(proposal, observed, proposal['input_hashes'], 'c' * 64,
                           review.HISTORY_MANIFEST_SHA256, review.HISTORY_AGGREGATE_SHA256, review.PHASE_B_HEAD)


def test_historical_source_review_reparses_all_checkpoint_outputs(tmp_path, monkeypatch):
    import hashlib
    import json
    from src.arc import qwen38_weight_review as review
    from src.cli.qwen38_weight_review import inspect_history
    root = tmp_path / 'history'; root.mkdir(); (root / 'phase_a').mkdir()
    manifest = {'protocol': 'qwen38-quick-exploratory-v1', 'model_id': 'Qwen/Qwen3.8-27B-FP8',
                'model_revision': 'c3a462cc3d38cf351f3b26cbd763469eeb74ef81', 'precision': 'fp8',
                'challenge_sha256': 'a' * 64, 'prompt_sha256': 'b' * 64,
                'candidate_count': 2, 'phase_a_task_ids': ['a']}
    manifest['sha256'] = digest(manifest)
    candidates = [{'candidate_index': i, 'raw': '[[[1]]]', 'outputs': [[[1]]], 'valid': True,
                   'parse_mode': 'strict'} for i in range(2)]
    row = {'task_id': 'a', 'test_count': 1, 'candidates': candidates, 'status': 'complete', 'manifest_sha256': manifest['sha256']}
    aggregate = {'complete': True, 'manifest': manifest, 'task_results': {'a': row}}
    (root / 'manifest.json').write_text(json.dumps(manifest)); (root / 'phase_a_results.json').write_text(json.dumps(aggregate))
    (root / 'phase_a/a.json').write_text(json.dumps(row))
    monkeypatch.setattr(review, 'HISTORY_MANIFEST_SHA256', manifest['sha256'])
    monkeypatch.setattr(review, 'HISTORY_AGGREGATE_SHA256', hashlib.sha256((root / 'phase_a_results.json').read_bytes()).hexdigest())
    assert inspect_history(root, 'a' * 64, 'b' * 64, {'a': {'test': [{}]}})['candidates'] == 2
    row['candidates'][0]['raw'] = '[[[2]]]'; (root / 'phase_a/a.json').write_text(json.dumps(row))
    with pytest.raises(ValueError): inspect_history(root, 'a' * 64, 'b' * 64, {'a': {'test': [{}]}})
