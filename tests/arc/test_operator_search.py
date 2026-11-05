import json
from pathlib import Path

from src.arc.data import Task, validate_grid
from src.arc.operator_search import (GridHypothesis, apply_hypothesis,
                                     search_hypotheses, solve_task)


ROOT = Path(__file__).resolve().parents[2]


def _tasks() -> dict[str, Task]:
    data = json.loads((ROOT / 'tests/fixtures/e4_synthetic_challenges.v1.json').read_text())
    result = {}
    for task_id, item in data.items():
        train = tuple((validate_grid(pair['input']), validate_grid(pair['output']))
                      for pair in item['train'])
        test_inputs = tuple(validate_grid(pair['input']) for pair in item['test'])
        result[task_id] = Task(task_id, train, test_inputs)
    return result


def test_grid_hypothesis_composes_shift_and_color_mapping():
    hypothesis = GridHypothesis('identity', 1, 0, 0, ((2, 7),))
    assert apply_hypothesis([[0, 2, 0], [0, 0, 0]], hypothesis) == [
        [0, 0, 7], [0, 0, 0]]


def test_operator_search_fits_all_training_pairs_on_synthetic_suite():
    tasks = _tasks()
    results = [solve_task(task) for task in tasks.values()]
    assert len(results) == 10
    assert all(result['status'] == 'train_consistent' for result in results)
    assert all(result['train_match_count'] == result['train_example_count']
               for result in results)
    assert all(search_hypotheses(task) for task in tasks.values())


def test_operator_search_only_receives_task_inputs_and_training_outputs():
    task = _tasks()['e4-color-01']
    # Task contains demonstrations and inputs only; the hidden solution file is
    # intentionally not loaded by this test or by search_hypotheses.
    assert len(task.test_inputs) == 2
    assert len(search_hypotheses(task)) > 0
