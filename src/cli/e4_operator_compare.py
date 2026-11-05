"""Compare the saved E4 baseline with train-only grid-operator search."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.arc.data import Task, validate_grid
from src.arc.operator_search import solve_task


ROOT = Path(__file__).resolve().parents[2]
CHALLENGES = ROOT / 'tests/fixtures/e4_synthetic_challenges.v1.json'
SOLUTIONS = ROOT / 'tests/fixtures/e4_synthetic_solutions.v1.json'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                indent=2) + '\n', encoding='utf-8')


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _load_challenges(path: Path) -> dict[str, Task]:
    data = _json(path)
    tasks = {}
    for task_id, item in data.items():
        train = tuple((validate_grid(example['input']), validate_grid(example['output']))
                      for example in item['train'])
        test_inputs = tuple(validate_grid(example['input']) for example in item['test'])
        tasks[task_id] = Task(task_id, train, test_inputs)
    return tasks


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_matches = sum(int(row['train_match_count']) for row in rows)
    train_total = sum(int(row['train_example_count']) for row in rows)
    heldout_matches = sum(int(row['heldout_match_count']) for row in rows)
    heldout_total = sum(int(row['heldout_input_count']) for row in rows)
    return {'tasks': len(rows),
            'full_train_match_tasks': sum(bool(row['complete_train_match']) for row in rows),
            'train_matches': train_matches, 'train_examples': train_total,
            'train_match_rate': train_matches / train_total if train_total else None,
            'heldout_matches': heldout_matches, 'heldout_inputs': heldout_total,
            'heldout_match_rate': heldout_matches / heldout_total if heldout_total else None}


def _score_baseline(directory: Path, answers: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _json(directory / 'baseline_summary.json')
    rows = _read_jsonl(directory / 'baseline_results.jsonl')
    if summary.get('status') != 'complete' or len(rows) != 10:
        raise ValueError(f'comparison requires ten completed baseline tasks in {directory}')
    manifest = _json(directory / 'manifest.json')
    if manifest.get('challenges_sha256') != _sha(CHALLENGES) or manifest.get('solutions_sha256') != _sha(SOLUTIONS):
        raise ValueError(f'baseline used different synthetic data: {directory}')
    scored = []
    for source in rows:
        row = dict(source)
        expected = answers[row['task_id']]
        actual = row.get('test_outputs', [])
        row['heldout_match_count'] = sum(a == b for a, b in zip(actual, expected))
        row['heldout_input_count'] = len(expected)
        scored.append(row)
    return summary, scored


def compare(baseline_dir: Path, structured_dir: Path,
            output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite E4 comparison: {output_dir}')
    structured_manifest = _json(structured_dir / 'manifest.json')
    if structured_manifest.get('agent_prompt_variant') not in {
            'structured', 'structured_hypothesis_and_repair_v1'}:
        raise ValueError('the prompt-improved arm must record its structured prompt variant')

    tasks = _load_challenges(CHALLENGES)
    # Generate every prediction from demonstrations and test inputs before
    # opening the separate answer file for scoring.
    improved_rows = [solve_task(task) for task in sorted(tasks.values(), key=lambda task: task.id)]
    answers = _json(SOLUTIONS)
    for row in improved_rows:
        expected = answers[row['task_id']]
        actual = row['test_outputs']
        row['heldout_match_count'] = sum(a == b for a, b in zip(actual, expected))
        row['heldout_input_count'] = len(expected)
        row['complete_heldout_match'] = (len(actual) == len(expected) and
                                         row['heldout_match_count'] == len(expected))
    # Score the saved baseline predictions with labels only after model calls
    # and operator predictions are complete. This cannot alter selection.
    baseline_summary, baseline_scored = _score_baseline(baseline_dir, answers)
    structured_summary, structured_scored = _score_baseline(structured_dir, answers)

    output_dir.mkdir(parents=True)
    _write(output_dir / 'improved_results.json', improved_rows)
    baseline_metrics = _rates(baseline_scored)
    improved_metrics = _rates(improved_rows)
    comparison = {'protocol': 'e4_synthetic_baseline_vs_operator_v1',
                  'status': 'complete',
                  'challenge_sha256': _sha(CHALLENGES),
                  'solutions_sha256': _sha(SOLUTIONS),
                  'baseline_results_sha256': _sha(baseline_dir / 'baseline_results.jsonl'),
                  'structured_prompt_results_sha256': _sha(structured_dir / 'baseline_results.jsonl'),
                  'operator_search_sha256': _sha(ROOT / 'src/arc/operator_search.py'),
                  'task_count': len(tasks),
                  'train_examples': sum(len(task.train) for task in tasks.values()),
                  'heldout_inputs': sum(len(task.test_inputs) for task in tasks.values()),
                  'existing_agent': {'name': 'existing_code_agent_legacy_prompt',
                                     'llm_calls': baseline_summary['budget']['calls'],
                                     'reserved_tokens': baseline_summary['budget']['reserved_tokens'],
                                     'sandbox_starts': baseline_summary['budget']['sandbox_starts'],
                                     'metrics': baseline_metrics},
                  'structured_prompt_agent': {'name': 'existing_code_agent_structured_prompt',
                                              'llm_calls': structured_summary['budget']['calls'],
                                              'reserved_tokens': structured_summary['budget']['reserved_tokens'],
                                              'sandbox_starts': structured_summary['budget']['sandbox_starts'],
                                              'metrics': _rates(structured_scored)},
                  'improved': {'name': 'train_only_operator_search', 'llm_calls': 0,
                               'reserved_tokens': 0, 'sandbox_starts': 0,
                               'metrics': improved_metrics,
                               'unsupported_tasks': [row['task_id'] for row in improved_rows
                                                     if row['status'] == 'unsupported']}}
    _write(output_dir / 'comparison.json', comparison)
    lines = ['# E4 synthetic comparison', '',
             'The legacy and structured-prompt arms use the existing code agent. The operator arm searches a fixed, small grid-operation grammar using training pairs only; it does not use heldout labels or call the model.', '',
             '| Method | Full train-match tasks | Train match | Heldout match | LLM calls | Reserved tokens | Sandbox starts |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, entry in (('Existing code agent, legacy prompts', comparison['existing_agent']),
                         ('Code agent, structured prompts', comparison['structured_prompt_agent']),
                         ('Train-only operator search', comparison['improved'])):
        metrics = entry['metrics']
        lines.append(f"| {label} | {metrics['full_train_match_tasks']}/{metrics['tasks']} | "
                     f"{metrics['train_matches']}/{metrics['train_examples']} ({metrics['train_match_rate']:.1%}) | "
                     f"{metrics['heldout_matches']}/{metrics['heldout_inputs']} ({metrics['heldout_match_rate']:.1%}) | "
                     f"{entry['llm_calls']} | {entry['reserved_tokens']} | {entry['sandbox_starts']} |")
    lines.extend(['', '## Per-task results', '',
                  '| Task | Legacy train | Structured train | Operator train | Legacy heldout | Structured heldout | Operator heldout | Selected rule |',
                  '|---|---:|---:|---:|---:|---:|---:|---|'])
    baseline_by_id = {row['task_id']: row for row in baseline_scored}
    structured_by_id = {row['task_id']: row for row in structured_scored}
    operator_by_id = {row['task_id']: row for row in improved_rows}
    for task_id in sorted(tasks):
        base = baseline_by_id[task_id]
        structured = structured_by_id[task_id]
        improved = operator_by_id[task_id]
        descriptor = improved.get('hypothesis') or {}
        rule = descriptor.get('spatial', 'unsupported')
        if descriptor.get('dx') or descriptor.get('dy'):
            rule += f" + shift({descriptor.get('dx')},{descriptor.get('dy')})"
        if descriptor.get('color_map'):
            rule += ' + color map'
        lines.append(f"| {task_id} | {base['train_match_count']}/{base['train_example_count']} | "
                     f"{structured['train_match_count']}/{structured['train_example_count']} | "
                     f"{improved['train_match_count']}/{improved['train_example_count']} | "
                     f"{base['heldout_match_count']}/{base['heldout_input_count']} | "
                     f"{structured['heldout_match_count']}/{structured['heldout_input_count']} | "
                     f"{improved['heldout_match_count']}/{improved['heldout_input_count']} | {rule} |")
    lines.extend(['', '## Limits', '',
                  f"The structured prompt condition reached {comparison['structured_prompt_agent']['metrics']['train_matches']}/{comparison['structured_prompt_agent']['metrics']['train_examples']} train matches versus {comparison['existing_agent']['metrics']['train_matches']}/{comparison['existing_agent']['metrics']['train_examples']} for the legacy prompt. This small run did not show a prompt-only gain.", '',
                  'These ten synthetic tasks were constructed from the operator grammar being evaluated. The result tests whether the agent can identify these simple transformations from demonstrations; it does not show that the method solves arbitrary ARC tasks or improves heldout ARC performance.', ''])
    (output_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--structured-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline_dir, args.structured_dir, args.output_dir),
                      sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
