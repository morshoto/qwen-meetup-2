from src.arc.agent import code_agent
from src.arc.data import Task
from src.arc.prompting import (ANALYSIS_SYSTEM_PROMPT,
                               FINAL_ANSWER_SYSTEM_PROMPT,
                               system_prompt_for_kind)
from src.arc.sandbox import FakeSandbox, IDENTITY_CODE, ZERO_CODE
from src.arc.agent import AgentResult
from src.arc.search import StrategyRun
from src.pipelines.exploration import _extract_attempts


def test_analysis_uses_a_system_prompt_that_allows_a_concise_rule_summary():
    assert system_prompt_for_kind('analysis') == FINAL_ANSWER_SYSTEM_PROMPT
    assert system_prompt_for_kind('analysis', analysis_mode=True) == ANALYSIS_SYSTEM_PROMPT
    assert system_prompt_for_kind('code') == FINAL_ANSWER_SYSTEM_PROMPT
    assert 'checkable summary' in ANALYSIS_SYSTEM_PROMPT
    assert 'chain-of-thought' in ANALYSIS_SYSTEM_PROMPT


def test_repair_prompt_contains_failing_input_and_explicit_correction_request():
    class Backend:
        prompts: list[tuple[str, str]] = []

        def generate(self, prompt, *, kind, candidate):
            self.prompts.append((kind, prompt))
            if kind == 'analysis':
                return 'The demonstrated change removes every colored cell.'
            if sum(item[0] == 'code' for item in self.prompts) == 1:
                return IDENTITY_CODE
            return ZERO_CODE

    task = Task('synthetic', (([[1, 0]], [[0, 0]]),), ([[1, 0]],))
    backend = Backend()
    result = code_agent(task, backend, FakeSandbox(), max_repairs=1, candidate=0,
                        prompt_variant='structured')

    assert result.code == ZERO_CODE
    repair_prompt = [prompt for kind, prompt in backend.prompts if kind == 'code'][1]
    assert 'Mismatched training input:' in repair_prompt
    assert 'Actual output:' in repair_prompt or 'Your actual output:' in repair_prompt
    assert 'decide whether the rule or its Python implementation is wrong' in repair_prompt
    assert 'Do not return the previous function unchanged' in repair_prompt


def test_e2_trace_preserves_execution_error_details():
    run = StrategyRun(AgentResult([], None, [{
        'code': 'def transform(grid):\n    return grid',
        'feedback': 'sandbox execution failed\nIndexError: list index out of range'}]),
        strategy='S0')
    trace = _extract_attempts(run, train_count=3)[0]
    assert trace['execution_errors'] == 1
    assert trace['error_kind'] == 'execution_error'
    assert trace['execution_error_detail'] == 'IndexError: list index out of range'
