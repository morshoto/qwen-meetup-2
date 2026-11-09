import pytest

from src.arc.qwen38_weight_schedule import next_action


def manifest():
    return {'allocation': {'fresh': ['a', 'b'], 'seed': 9},
            'generation': {'initial_candidates': 2, 'additional_candidates': 6,
                           'max_tokens_per_candidate': 10, 'reserved_token_limit': 1000,
                           'wall_seconds_limit': 100}}


def row(task, index, status='complete'):
    return {'task_id': task, 'candidate_index': index, 'status': status, 'elapsed_seconds': 1}


def test_initial_attempts_cover_all_parents_breadth_first():
    m = manifest(); journal = []; expected = [('a', 0), ('b', 0), ('a', 1), ('b', 1)]
    for task, index in expected:
        action = next_action(m, journal, {})
        assert (action['task_id'], action['candidate_index']) == (task, index)
        assert action['seed'] == 9 + 10000 + ['a', 'b'].index(task) * 8 + index
        journal.append(row(task, index))
    assert next_action(m, journal, {})['status'] == 'awaiting_verification'


def test_extras_skip_successful_parents_and_remain_breadth_first():
    m = manifest(); journal = [row(t, i) for i in range(2) for t in ('a', 'b')]
    verified = {('a', 0): True, ('a', 1): False, ('b', 0): False, ('b', 1): False}
    assert next_action(m, journal, verified)['task_id'] == 'b'
    for index in range(2, 8):
        action = next_action(m, journal, verified)
        assert (action['task_id'], action['candidate_index']) == ('b', index)
        journal.append(row('b', index)); verified[('b', index)] = False
    assert next_action(m, journal, verified)['status'] == 'complete'


def test_inflight_and_budget_stop_without_repeating_work():
    m = manifest()
    assert next_action(m, [row('a', 0, 'started')], {})['status'] == 'requires_reconciliation'
    m['generation']['reserved_token_limit'] = 10
    assert next_action(m, [row('a', 0)], {})['status'] == 'token_budget_exhausted'
    m['generation']['reserved_token_limit'] = 1000
    m['generation']['wall_seconds_limit'] = 1
    assert next_action(m, [row('a', 0)], {})['status'] == 'time_budget_exhausted'


def test_invalid_or_forged_journal_entries_fail_closed():
    m = manifest()
    for journal, verified in [([row('a', 0), row('a', 0)], {}), ([row('z', 0)], {}),
                              ([row('a', 8)], {}), ([row('b', 0)], {}),
                              ([row('a', 0)], {('a', 0): 'correct'}),
                              ([row('a', 0, 'error')], {('a', 0): True})]:
        with pytest.raises(ValueError): next_action(m, journal, verified)


def test_replay_rejects_overspent_tokens_and_attempts_after_success():
    m = manifest(); m['generation']['reserved_token_limit'] = 10
    with pytest.raises(ValueError): next_action(m, [row('a', 0), row('b', 0)], {})
    m['generation']['reserved_token_limit'] = 1000
    journal = [row(t, i) for i in range(2) for t in ('a', 'b')] + [row('a', 2)]
    results = {('a', 0): True, ('a', 1): False, ('b', 0): False, ('b', 1): False, ('a', 2): False}
    with pytest.raises(ValueError): next_action(m, journal, results)


@pytest.mark.parametrize('reason', ['low_free_memory', 'swap_limit', 'memory_probe_failed', 'remaining_process_group'])
def test_resource_failures_stop_the_collection(reason):
    m = manifest(); failed = {**row('a', 0, 'error'), 'error': reason}
    assert next_action(m, [failed], {})['status'] == 'resource_stop'
