import copy
import pytest
from src.arc.qwen38_rsi_comparison import allocate, validate_allocation, compare, summarize
from src.arc.data import Verifier


def fixture():
    tasks = {f'{i:08x}': {'train': [{}, {}]} for i in range(260)}
    ids = list(tasks)
    proposal = {'task_ids': ids[:200], 'protected_exclusions': ids[200:210],
                'training_exclusions': ids[210:220], 'known_exposure_exclusions': ids[220:230],
                'development_exclusions': ids[230:240]}
    return proposal, tasks


def test_allocation_is_repeatable_disjoint_and_uses_common_50():
    p, tasks = fixture()
    a = allocate(p, tasks)
    assert a == allocate(p, tasks)
    assert len(a['evaluation']) == 50 and len(a['development']) == len(a['selection']) == 10
    assert not set(a['development']) & (set(p['task_ids']) | set(p['protected_exclusions']))
    assert not set(a['development']) & set(a['selection'])
    assert all(set(order) == set(a['evaluation']) for order in a['orders'].values())
    bad = copy.deepcopy(a); bad['evaluation'][0] = bad['development'][0]
    with pytest.raises(ValueError): validate_allocation(bad, p, tasks)


def test_incomplete_results_use_paired_denominator():
    verifier = Verifier({'a': [[[1]]], 'b': [[[2]]]})
    def row(t, output):
        return {'task_id': t, 'status': 'complete', 'candidates': [{'outputs': [output]}, {'outputs': None}]}
    baseline = summarize(['a', 'b'], [row('a', [[1]]), row('b', [[0]])], verifier)
    other = summarize(['a', 'b'], [row('b', [[2]])], verifier)
    assert compare(baseline, other) == {'paired': 1, 'gained': 1, 'lost': 0}
    with pytest.raises(ValueError): summarize(['a'], [row('a', [[1]]), row('a', [[1]])], verifier)
