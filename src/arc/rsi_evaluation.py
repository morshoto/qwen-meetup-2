"""Matched task evaluation primitives; supplied execution is not authenticated."""
from src.arc.data import Task, validate_grid
from src.arc.e5_selfplay import solve_prompt
from src.arc.e6f_repair import _static_failure, classify_failure


class InvalidModelOutput:
    """A captured invalid completion, distinct from a transport failure."""


class _InvalidOutput(Exception):
    pass


def evaluate_task(task, *, generate, sandbox):
    """Collect two independent programs with the same label-free prompt.

    Infrastructure errors propagate and must abort the comparison. The caller
    must separately enforce matched model settings, budget, seeds and identity.
    """
    if not isinstance(task, Task) or not task.train or not task.test_inputs:
        raise ValueError('task requires visible demonstrations and test inputs')
    prompt = solve_prompt(task)
    predictions, statuses, sources = [], [], []
    for sample in range(2):
        code = generate(prompt, sample=sample)
        if isinstance(code, InvalidModelOutput):
            sources.append(None)
            statuses.append('invalid_model_output')
            continue
        if not isinstance(code, str) or len(code.encode('utf-8')) > 65536:
            code = None
        sources.append(code)
        if code is None or not code.strip() or _static_failure(code):
            statuses.append('static_rejected')
            continue
        starts = 0

        def run(grid):
            nonlocal starts
            if starts >= 10:
                raise TimeoutError('per-sample sandbox budget exhausted')
            starts += 1
            result = sandbox.run(code, grid)
            try:
                return validate_grid(result)
            except (ValueError, TypeError) as exc:
                raise _InvalidOutput from exc

        try:
            mismatch = False
            for grid, expected in task.train:
                if run(grid) != validate_grid(expected):
                    mismatch = True
                    break
            if mismatch:
                statuses.append('visible_demonstration_mismatch')
                continue
            outputs = [run(grid) for grid in task.test_inputs]
        except _InvalidOutput:
            statuses.append('invalid_output')
            continue
        except RuntimeError as exc:
            if classify_failure(exc)[0] != 'runtime_exception':
                raise
            statuses.append('runtime_exception')
            continue
        statuses.append('accepted')
        predictions.append(outputs)
    return {'status': 'complete', 'task_id': task.id, 'sample_statuses': statuses,
            'predictions': predictions, 'sample_sources': sources, 'evidence_type': 'caller_supplied_execution'}


def score_task(task, result, verifier):
    """Score up to two guesses per input using the verifier's isolated labels."""
    allowed = {'accepted', 'static_rejected', 'visible_demonstration_mismatch',
               'invalid_output', 'invalid_model_output', 'runtime_exception'}
    if (not isinstance(task, Task) or not isinstance(result, dict) or result.get('status') != 'complete'
            or result.get('task_id') != task.id or not isinstance(result.get('sample_statuses'), list)
            or len(result['sample_statuses']) != 2 or any(status not in allowed for status in result['sample_statuses'])
            or not isinstance(result.get('predictions'), list)
            or len(result['predictions']) != result['sample_statuses'].count('accepted')):
        raise ValueError('task result is incomplete or malformed')
    count = len(task.test_inputs)
    if verifier.output_count(task.id) != count or not count:
        raise ValueError('solution count differs from task inputs')
    for candidate in result['predictions']:
        if not isinstance(candidate, list) or len(candidate) != count:
            raise ValueError('candidate output count differs from task')
        for grid in candidate:
            validate_grid(grid)
    correct = []
    for index in range(count):
        matched = False
        for candidate in result['predictions']:
            one_guess = [None] * count
            one_guess[index] = candidate[index]
            if verifier.correct_outputs(task.id, one_guess) == 1:
                matched = True
                break
        correct.append(matched)
    return {'task_id': task.id, 'solved': all(correct), 'test_inputs_solved': correct,
            'evidence_type': 'caller_supplied_execution', 'independently_verified': False}
