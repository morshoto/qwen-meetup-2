"""Model and agent interfaces; all prompts contain only public task data."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from src.arc.data import Task, validate_grid
from src.arc.sandbox import Sandbox, IDENTITY_CODE
from src.arc.http_backend import BackendError


class LLMBackend(Protocol):
    def generate(self, prompt: str, *, kind: str, candidate: int) -> str: ...


class FakeLLM:
    """Deterministic test backend; does not consume solution labels."""
    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        if kind == 'direct':
            data = json.loads(prompt)
            return json.dumps([entry['input'] for entry in data['task']['test']])
        return IDENTITY_CODE


@dataclass
class AgentResult:
    outputs: list[Any]
    code: str | None
    history: list[dict[str, Any]]
    error: str | None = None
    error_kind: str | None = None
    raw_response: str | None = None
    analysis: str | None = None


def _prompt_json(value: Any) -> str:
    """Serialize task prompts compactly to conserve the model's context window."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _grid_text(grid: list[list[int]]) -> str:
    return '\n'.join(' '.join(str(cell) for cell in row) for row in grid)


def _code_task_text(task: Task) -> str:
    lines = ['Training examples:']
    for index, (input_grid, output_grid) in enumerate(task.train, 1):
        lines.extend((f'Example {index} input:', _grid_text(input_grid),
                      f'Example {index} output:', _grid_text(output_grid)))
    lines.append('Test inputs:')
    for index, input_grid in enumerate(task.test_inputs, 1):
        lines.extend((f'Test {index} input:', _grid_text(input_grid)))
    return '\n'.join(lines)


def _code_history_text(history: list[dict[str, Any]], *, include_inputs: bool = False) -> str:
    lines = ['Previous attempts and training-pair feedback:']
    for attempt, item in enumerate(history, 1):
        lines.extend((f'Attempt {attempt} code:', str(item.get('code', ''))))
        feedback = item.get('feedback')
        if isinstance(feedback, list):
            for failure in feedback:
                if not isinstance(failure, dict):
                    lines.append(str(failure))
                    continue
                example = failure.get('example')
                if type(example) is int:
                    lines.append(f'Training example {example + 1} mismatch:')
                else:
                    lines.append('Training mismatch:')
                input_grid = failure.get('input')
                if (include_inputs and isinstance(input_grid, list) and
                        all(isinstance(row, list) for row in input_grid)):
                    lines.extend(('Mismatched training input:', _grid_text(input_grid)))
                actual, expected = failure.get('actual'), failure.get('expected')
                if isinstance(actual, list) and all(isinstance(row, list) for row in actual):
                    lines.extend(('Your actual output:', _grid_text(actual)))
                else:
                    lines.extend(('Execution feedback:', str(actual)))
                if isinstance(expected, list) and all(isinstance(row, list) for row in expected):
                    lines.extend(('Expected training output:', _grid_text(expected)))
        elif feedback:
            lines.extend(('Execution feedback:', str(feedback)))
    return '\n'.join(lines)


def _code_analysis_prompt(task: Task) -> str:
    instruction = (
        'Inspect every training pair. Compare grid dimensions, changed cells, colors, object shapes, and '
        'positions. Identify which feature predicts each change, then check the proposed rule against every '
        'training pair. State the observed changes, the general rule, and how it accounts for all examples '
        'in three to six short bullet points. Use test inputs only to check whether the rule is plausible. '
        'Do not output code, any grids, or test answers; do not claim a test answer is known.'
    )
    return instruction + '\n\n' + _code_task_text(task)


def direct(task: Task, backend: LLMBackend, candidate: int) -> AgentResult:
    prompt = _prompt_json({'instruction': 'Return a JSON array of output grids, one per test input.', 'task': task.prompt_data()})
    raw = None
    try:
        raw = backend.generate(prompt, kind='direct', candidate=candidate)
        outputs = json.loads(raw)
        if not isinstance(outputs, list) or len(outputs) != len(task.test_inputs):
            raise ValueError('output count does not match test inputs')
        for grid in outputs: validate_grid(grid)
        return AgentResult(outputs, None, [], raw_response=raw)
    except TimeoutError as exc:
        return AgentResult([], None, [], str(exc), 'timeout')
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return AgentResult([], None, [], str(exc), 'invalid', raw)


def code_agent(task: Task, backend: LLMBackend, sandbox: Sandbox, max_repairs: int,
               candidate: int, *, prompt_variant: str = 'legacy') -> AgentResult:
    if prompt_variant not in {'legacy', 'structured'}:
        raise ValueError('unsupported code-agent prompt variant')
    history: list[dict[str, Any]] = []
    timed_out = False
    try:
        analysis = backend.generate(_code_analysis_prompt(task), kind='analysis', candidate=candidate).strip()
    except BackendError:
        raise
    except (ValueError, RuntimeError, TimeoutError) as exc:
        return AgentResult([], None, [], str(exc), 'timeout' if isinstance(exc, TimeoutError) else 'invalid')
    for attempt in range(max_repairs + 1):
        instruction = (
            'The grids below use space-separated color numbers, with one row per line. '
            'Use the transformation hypothesis as a starting point, then check it against every training pair. '
            'Compare grid shapes, '
            'colors, object positions, and repeated patterns. The function must match every training output '
            'exactly and generalize to each test input. Do not copy one training output or hard-code grids. '
            'Write a complete, concise Python function named transform(grid) that returns a rectangular '
            'list of integer rows. Return only source code: no imports, comments, markdown, or explanation.'
        )
        if attempt:
            if prompt_variant == 'structured':
                instruction += (' The previous function failed on training examples. For each mismatch, compare the '
                                'input, actual output, and expected output; decide whether the rule or its Python '
                                'implementation is wrong. Revise the rule and replace the entire function. Check that '
                                'the revised code is semantically different from the previous attempt and matches every '
                                'training pair. Do not return the previous function unchanged.')
            else:
                instruction += (' Compare each actual output with its expected training output, explain the mismatch '
                                'to yourself, revise the transformation rule, and replace the entire previous function. '
                                'Do not repeat unchanged code.')
        prompt = instruction + '\n\nTransformation hypothesis:\n' + analysis + '\n\n' + _code_task_text(task)
        if history:
            prompt += '\n\n' + _code_history_text(
                history, include_inputs=prompt_variant == 'structured')
        try:
            code = backend.generate(prompt, kind='code', candidate=candidate)
            blocks = re.findall(r'```python\s*\n(.*?)\n```', code, flags=re.DOTALL | re.IGNORECASE)
            if len(blocks) == 1:
                code = blocks[0].strip()
            failures = []
            for index, (inp, expected) in enumerate(task.train):
                actual = validate_grid(sandbox.run(code, inp))
                if actual != expected:
                    failure = {'example': index, 'actual': actual, 'expected': expected}
                    if prompt_variant == 'structured':
                        failure['input'] = inp
                    failures.append(failure)
            history.append({'code': code, 'feedback': failures})
            if not failures:
                outputs = [validate_grid(sandbox.run(code, inp)) for inp in task.test_inputs]
                return AgentResult(outputs, code, history, analysis=analysis)
        except BackendError:
            raise
        except (ValueError, RuntimeError, TimeoutError) as exc:
            timed_out = isinstance(exc, TimeoutError)
            history.append({'code': locals().get('code', ''), 'feedback': str(exc)})
    return AgentResult([], None, history, 'repair limit exhausted', 'timeout' if timed_out else 'invalid',
                       analysis=analysis)
