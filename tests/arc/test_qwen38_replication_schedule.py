import pytest

from src.arc.qwen38_replication_schedule import next_replication_action


def descriptor():
    return {'task_ids': ['a', 'b'], 'candidate_count': 2, 'max_tokens': 4096,
            'reserved_completion_tokens': 8 * 4096, 'wall_seconds_limit': 1000}


def row(arm, task, index, status='complete', error=None):
    return {'arm': arm, 'task_id': task, 'candidate_index': index,
            'status': status, 'error': error, 'elapsed_seconds': 1.0}


def test_schedule_runs_same_tasks_and_seeds_for_both_arms():
    d = descriptor(); rows = []
    for arm in ('base', 'adapter'):
        for task in ('a', 'b'):
            for index in (0, 1):
                action = next_replication_action(d, rows)
                assert (action['arm'], action['task_id'], action['candidate_index']) == (arm, task, index)
                assert action['seed'] == 20261111 + index
                rows.append(row(arm, task, index))
    assert next_replication_action(d, rows)['status'] == 'complete'


def test_schedule_never_repeats_ambiguous_start_or_hides_failure():
    d = descriptor()
    assert next_replication_action(d, [row('base', 'a', 0, 'started')])['status'] == 'requires_reconciliation'
    rows = [row('base', 'a', 0, 'error', 'timeout')]
    for arm, task, index in [('base','a',1),('base','b',0),('base','b',1),
                             ('adapter','a',0),('adapter','a',1),('adapter','b',0),('adapter','b',1)]:
        rows.append(row(arm, task, index))
    assert next_replication_action(d, rows)['status'] == 'incomplete'


def test_schedule_rejects_wrong_arm_or_task_and_respects_total_budget():
    d = descriptor()
    with pytest.raises(ValueError): next_replication_action(d, [row('adapter', 'a', 0)])
    with pytest.raises(ValueError): next_replication_action(d, [row('base', 'b', 0)])
    d['reserved_completion_tokens'] = 4096
    assert next_replication_action(d, [row('base', 'a', 0)])['status'] == 'token_budget_exhausted'
