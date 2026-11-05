"""Run a bounded matched comparison with caller-supplied components.

This orchestration does not establish model, adapter, sandbox, seed support or
registration authenticity. It does not authorize a live experiment.
"""
import json
from pathlib import Path

from src.arc.data import Task
from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_budget_adapters import BudgetedSandbox
from src.arc.rsi_comparison_budget import ComparisonBudget
from src.arc.rsi_comparison_freeze import _SETTINGS, _validate, comparison_coverage, freeze_comparison
from src.arc.rsi_evaluation import evaluate_task, score_task
from src.arc.rsi_slot_lifecycle import _create, _directory


def run_comparison(*, root, registration_sha256, generation, task_ids, tasks, seeds,
                   parent_adapter, candidate_adapter, generate, sandbox, verifier, budget):
    if (not isinstance(budget, ComparisonBudget) or
            budget.identity != {'registration_sha256': registration_sha256, 'generation': generation} or
            Path(budget.root).resolve() != Path(root).resolve()):
        raise ValueError('comparison budget identity differs')
    placeholders = {task: {'solved': False, 'seeds': seeds.get(task)} for task in task_ids}
    _validate({'registration_sha256': registration_sha256, 'settings': dict(_SETTINGS),
               'task_ids': task_ids, 'seeds': seeds, 'parent_adapter': parent_adapter,
               'candidate_adapter': candidate_adapter, 'parent': placeholders, 'candidate': placeholders,
               'evidence_type': 'caller_supplied_outcomes'})
    if (not isinstance(tasks, dict) or set(tasks) != set(task_ids) or
            any(not isinstance(tasks[key], Task) or tasks[key].id != key or not tasks[key].train or
                not tasks[key].test_inputs or verifier.output_count(key) != len(tasks[key].test_inputs)
                for key in task_ids)):
        raise ValueError('task or verifier manifest differs')
    with _directory(root) as directory:
        _create(directory, 'comparison.started.json', {'registration_sha256': registration_sha256,
                                                         'generation': generation})
    outcomes = {'parent': {}, 'candidate': {}}
    metered = BudgetedSandbox(sandbox, budget)
    try:
        for index, task_id in enumerate(task_ids):
            task = tasks[task_id]
            for arm, adapter in (('parent', parent_adapter), ('candidate', candidate_adapter)):
                def call(prompt, *, sample):
                    messages = [{'role': 'system', 'content': system_prompt_for_kind('e5_self_solve')},
                                {'role': 'user', 'content': prompt}]
                    reserved = 512 + len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
                    budget.reserve_model(reserved, kind='evaluation')
                    return generate(prompt, arm=arm, adapter=adapter, task_id=task_id, sample=sample,
                                    seed=seeds[task_id][sample], timeout=budget.remaining())
                result = evaluate_task(task, generate=call, sandbox=metered)
                score = score_task(task, result, verifier)
                value = {'solved': score['solved'], 'seeds': list(seeds[task_id])}
                with _directory(root) as directory:
                    _create(directory, f'comparison-{index:03d}-{arm}.json',
                            {'task_id': task_id, 'result': result, 'score': score})
                outcomes[arm][task_id] = value
        return freeze_comparison(root=root, registration_sha256=registration_sha256, task_ids=task_ids,
                                 seeds=seeds, parent_adapter=parent_adapter, candidate_adapter=candidate_adapter,
                                 parent=outcomes['parent'], candidate=outcomes['candidate'])
    except Exception as exc:
        coverage = comparison_coverage(task_ids, parent=outcomes['parent'], candidate=outcomes['candidate'])
        report = {**coverage, 'status': 'incomplete', 'failure_type': type(exc).__name__,
                  'evidence_type': 'caller_supplied_execution'}
        with _directory(root) as directory:
            _create(directory, 'comparison.incomplete.json', report)
        return report
