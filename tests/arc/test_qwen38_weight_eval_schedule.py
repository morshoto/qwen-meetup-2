import pytest

from src.arc.qwen38_weight_eval_schedule import next_evaluation_action


def descriptor():
    return {'task_ids': ['a', 'b'], 'candidate_count': 2, 'max_tokens': 10,
            'reserved_completion_tokens': 40, 'wall_seconds_limit': 100}


def row(task, index, status='complete', error=None):
    return {'task_id': task, 'candidate_index': index, 'status': status, 'elapsed_seconds': 1, 'error': error}


def test_evaluation_attempts_exactly_two_per_task_without_score_feedback():
    manifest = descriptor(); journal = []
    for task, index in [('a', 0), ('a', 1), ('b', 0), ('b', 1)]:
        action = next_evaluation_action(manifest, journal)
        assert (action['task_id'], action['candidate_index']) == (task, index)
        assert action['seed'] == 20261111 + index
        journal.append(row(task, index))
    assert next_evaluation_action(manifest, journal)['status'] == 'complete'


def test_evaluation_records_failures_and_rejects_unresolved_or_overspent_runs():
    manifest = descriptor()
    assert next_evaluation_action(manifest, [row('a', 0, 'started')])['status'] == 'requires_reconciliation'
    assert next_evaluation_action(manifest, [row('a', 0, 'error', 'low_free_memory')])['status'] == 'resource_stop'
    rows = [row('a', 0, 'error', 'timeout'), row('a', 1), row('b', 0), row('b', 1)]
    assert next_evaluation_action(manifest, rows)['status'] == 'incomplete'
    manifest['reserved_completion_tokens'] = 10
    with pytest.raises(ValueError): next_evaluation_action(manifest, rows)
    with pytest.raises(ValueError): next_evaluation_action(descriptor(), [row('b', 0)])
