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
