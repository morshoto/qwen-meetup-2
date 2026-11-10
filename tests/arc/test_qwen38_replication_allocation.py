import copy

import pytest

from src.arc.qwen38_replication_allocation import audit_reserved


def task(n):
    return {'train': [{'input': [[n]], 'output': [[n + 1]]}], 'test': [{'input': [[n + 100]]}]}


def fixture():
    tasks = {str(i): task(i) for i in range(7)}
    return tasks, ['0', '1'], {'trained': ['2'], 'historical': ['3']}, {'4': 'family', '5': 'family'}, {str(i): str(i) for i in range(7)}


def test_audit_records_only_fingerprints_and_unresolved_exposure():
    tasks, selected, excluded, lineage, answers = fixture()
    audit = audit_reserved(tasks, selected, excluded, lineage, answers, {})
    assert audit['status'] == 'bounded_disjoint'
    assert audit['designation'] == 'exploratory'
    assert audit['selected_count'] == 2
    assert '[[100]]' not in repr(audit)
    assert audit['unresolved_exposure'] is True


@pytest.mark.parametrize('change', ['id', 'input', 'transitive', 'lineage', 'answer', 'missing_fingerprint', 'official'])
def test_audit_rejects_any_known_exposure_or_missing_evidence(change):
    tasks, selected, excluded, lineage, answers = fixture()
    official = {}
    if change == 'id': excluded['trained'].append('0')
    elif change == 'input': tasks['0']['test'][0]['input'] = tasks['2']['train'][0]['input']
    elif change == 'transitive':
        tasks['4']['test'][0]['input'] = tasks['0']['test'][0]['input']
        tasks['5']['test'][0]['input'] = tasks['2']['test'][0]['input']
    elif change == 'lineage': lineage.update({'0': 'family', '2': 'family'})
    elif change == 'answer': answers['0'] = answers['2']
    elif change == 'missing_fingerprint': del answers['2']
    else: official['other'] = copy.deepcopy(tasks['0'])
    with pytest.raises(ValueError): audit_reserved(tasks, selected, excluded, lineage, answers, official)


def test_transitively_blocked_parent_requires_valid_answer_fingerprint():
    tasks, selected, excluded, lineage, answers = fixture()
    tasks['4']['test'][0]['input'] = tasks['2']['test'][0]['input']
    answers['4'] = ''
    with pytest.raises(ValueError):
        audit_reserved(tasks, selected, excluded, lineage, answers, {})
