import copy

import pytest

from src.arc.qwen38_weight_allocation import allocate_weight_pool


def task(n):
    return {'train': [{'input': [[n]], 'output': [[n + 1]]}], 'test': [{'input': [[n + 2]]}]}


def test_seeded_allocation_preserves_disjoint_reservations_and_families():
    tasks = {str(i): task(i) for i in range(30)}
    tasks['alias'] = copy.deepcopy(tasks['0'])
    evaluation = {'eval': copy.deepcopy(tasks['1'])}
    exclusions = {'protected': ['0'], 'prior': ['2']}
    lineage = {'3': 'family', '4': 'family'}
    a = allocate_weight_pool(tasks, evaluation, exclusions, lineage, seed=10, pool_size=12, validation_size=5)
    assert a == allocate_weight_pool(tasks, evaluation, exclusions, lineage, seed=10, pool_size=12, validation_size=5)
    selected = a['fresh'] + a['validation']
    assert len(a['fresh']) == 12 and len(a['validation']) == 5
    assert not set(selected) & {'0', '1', '2', 'alias'}
    assert len(set(selected) & {'3', '4'}) <= 1
    assert not set(a['fresh']) & set(a['validation'])


def test_allocation_records_shortfall_and_rejects_unknown_reservations():
    tasks = {str(i): task(i) for i in range(4)}
    a = allocate_weight_pool(tasks, {}, {'protected': ['0']}, {}, seed=1, pool_size=10, validation_size=2)
    assert len(a['fresh']) == 1 and len(a['validation']) == 2
    assert a['shortfall'] == 9
    with pytest.raises(ValueError): allocate_weight_pool(tasks, {}, {'protected': ['missing']}, {})
    with pytest.raises(ValueError): allocate_weight_pool(tasks, {}, {}, {'missing': 'family'})


def test_derived_task_sharing_an_input_with_protected_task_is_excluded():
    tasks = {str(i): task(i) for i in range(12)}
    tasks['variant'] = {'train': [{'input': [[99]], 'output': [[1]]}],
                        'test': [{'input': tasks['0']['train'][0]['input']}]}
    allocation = allocate_weight_pool(tasks, {}, {'protected': ['0']}, {}, pool_size=10, validation_size=1)
    assert 'variant' not in allocation['fresh'] + allocation['validation']
