"""Offline, evidence-preserving analysis of P3-A/B/C artifacts."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import yaml

from src.arc.data import load_arc


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _candidate_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = _read_json(run_dir / 'results.json', [])
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = row.get('task_id')
        task_score = row.get('score')
        candidates = row.get('candidates')
        if not isinstance(candidates, list):
            continue
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            result_obj = candidate.get('result')
            if isinstance(result_obj, dict):
                result_obj = result_obj
            else:
                result_obj = None
            result.append({'task_id': task_id if isinstance(task_id, str) else None,
                           'condition': row.get('condition'), 'candidate_index': index,
                           'elapsed_seconds': candidate.get('elapsed_seconds'),
                           'outer_error': candidate.get('error'), 'score': task_score if index == 0 else None,
                           'result': result_obj})
    return result


def _history_evidence(result: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[int | None], list[str]]:
    history = result.get('history') if result else None
    if not isinstance(history, list):
        return [], [], []
    records: list[dict[str, Any]] = []
    mismatches: list[int | None] = []
    syntax_failures: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        code = item.get('code')
        feedback = item.get('feedback')
        if isinstance(code, str) and code:
            try:
                ast.parse(code)
                syntax_failures.append('valid')
            except (SyntaxError, ValueError, RecursionError):
                syntax_failures.append('invalid')
        mismatch_count: int | None = None
        if isinstance(feedback, list) and all(isinstance(value, dict) for value in feedback):
            mismatch_count = len(feedback)
        records.append(item)
        mismatches.append(mismatch_count)
    return records, mismatches, syntax_failures


def _metric(numerator: int, denominator: int, not_available_count: int) -> dict[str, Any]:
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None,
            'not_available_count': not_available_count}


def diagnose_run(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create a reproducible summary without copying response bodies or secrets."""
    run = Path(run_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    if not run.is_dir() or not (run / 'results.json').is_file():
        raise ValueError('run directory must contain results.json')
    if out.exists():
        raise FileExistsError(f'diagnosis output already exists: {out}')
    out.mkdir(parents=True)
    state = _read_json(run / 'run_state.json', {})
    budget = _read_json(run / 'budget.json', {})
    try:
        config = yaml.safe_load((run / 'config.snapshot.yml').read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        config = {}
    candidates = _candidate_rows(run)
    responses: list[dict[str, Any]] = []
    private_responses = run / 'llm_responses.jsonl'
    if private_responses.is_file():
        try:
            for line in private_responses.read_text(encoding='utf-8').splitlines():
                item = json.loads(line)
                if isinstance(item, dict):
                    responses.append(item)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            responses = []
    request_rows: list[dict[str, Any]] = []
    request_file = run / 'llm_requests.jsonl'
    if request_file.is_file():
        try:
            request_rows = [json.loads(line) for line in request_file.read_text(encoding='utf-8').splitlines()
                            if line.strip()]
            request_rows = [item for item in request_rows if isinstance(item, dict)]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            request_rows = []

    challenge_lengths: dict[str, int] = {}
    test_lengths: dict[str, int] = {}
    config_data = config.get('data', {}) if isinstance(config, dict) else {}
    challenge_path = config_data.get('challenges') if isinstance(config_data, dict) else None
    if challenge_path and Path(challenge_path).is_file():
        try:
            tasks, _ = load_arc(challenge_path, config_data.get('solutions'))
            challenge_lengths = {key: len(task.train) for key, task in tasks.items()}
            test_lengths = {key: len(task.test_inputs) for key, task in tasks.items()}
        except (OSError, ValueError, TypeError):
            challenge_lengths = {}

    diagnostics: list[dict[str, Any]] = []
    parse_observations = [item.get('diagnostic_category') for item in responses
                          if isinstance(item.get('diagnostic_category'), str)]
    response_failure_counts: dict[str, int] = {}
    for category in parse_observations:
        if category in {'thinking_unclosed', 'missing_final_answer', 'invalid_json',
                        'code_extraction_error', 'generation_truncated', 'transport_error',
                        'budget_rejected', 'sandbox_error', 'verifier_error'}:
            response_failure_counts[category] = response_failure_counts.get(category, 0) + 1
    response_parse_ok = sum(value not in {'invalid_json', 'thinking_unclosed', 'missing_final_answer',
                                           'generation_truncated', 'code_extraction_error'}
                            for value in parse_observations)
    valid_codes = code_count = sandbox_successes = sandbox_failures = sandbox_runs = sandbox_not_available = 0
    full_train = full_train_known = repair_improved = repair_known = 0
    heldout_pass = heldout_known = 0
    task_pass_ids: set[str] = set()
    heldout_task_ids: set[str] = set()
    elapsed_by_task: dict[str, float] = {}
    elapsed_known_tasks: set[str] = set()
    failure_counts: dict[str, int] = {}
    all_units = len(candidates)
    for row in candidates:
        result = row['result']
        history, mismatch_counts, syntax = _history_evidence(result)
        task_id = row['task_id']
        labels: list[str] = []
        outer_error = row.get('outer_error')
        if isinstance(outer_error, dict):
            error_kind = outer_error.get('kind')
            if error_kind == 'budget_exhausted':
                labels.append('budget_exhausted')
            elif error_kind == 'malformed_thinking':
                labels.append('thinking_unclosed')
            elif error_kind in {'invalid_response', 'missing_response'}:
                labels.append('response_parse_failure')
            elif error_kind == 'sandbox_unavailable':
                labels.append('sandbox_rejected')
            else:
                labels.append('http_or_generation_failure')
        if result is None:
            if not labels:
                labels.append('unknown_insufficient_evidence')
        else:
            error_kind = result.get('error_kind')
            message = str(result.get('error') or '').lower()
            if error_kind == 'timeout' and not history:
                labels.append('http_or_generation_failure')
            if error_kind == 'invalid' and not history:
                labels.append('response_parse_failure')
            for item in history:
                code = item.get('code')
                if isinstance(code, str) and code:
                    try:
                        ast.parse(code)
                    except (SyntaxError, ValueError, RecursionError):
                        labels.append('python_syntax_failure')
                feedback = item.get('feedback')
                if isinstance(feedback, str):
                    if 'timeout' in feedback.lower() or 'deadline' in feedback.lower():
                        labels.append('sandbox_timeout')
                    else:
                        labels.append('sandbox_runtime_error')
                elif isinstance(feedback, list):
                    full_train_known += 1
                    if feedback:
                        labels.append('train_example_mismatch')
                    elif isinstance(code, str) and code:
                        full_train += 1
            if history:
                known_mismatches = [value for value in mismatch_counts if value is not None]
                if len(known_mismatches) == len(mismatch_counts) and len(known_mismatches) >= 2:
                    repair_known += 1
                    repair_improved += int(known_mismatches[-1] < known_mismatches[0])
                train_count = challenge_lengths.get(task_id or '')
                if type(train_count) is int:
                    for item, mismatch_count in zip(history, mismatch_counts):
                        if mismatch_count is not None:
                            sandbox_runs += train_count
                            sandbox_successes += train_count
                        elif isinstance(item.get('code'), str) and item['code']:
                            # The log proves at least one sandbox failure but
                            # does not say which training example failed.
                            sandbox_failures += 1
                            sandbox_runs += 1
                            sandbox_not_available += max(0, train_count - 1)
                if (isinstance(result.get('code'), str) and isinstance(result.get('outputs'), list) and
                        isinstance(task_id, str) and task_id in test_lengths):
                    known_test_runs = min(len(result['outputs']), test_lengths[task_id])
                    sandbox_successes += known_test_runs
                    sandbox_runs += known_test_runs
            score = row.get('score')
            if isinstance(score, dict) and type(score.get('pass_at_1')) is bool:
                heldout_known += 1
                if isinstance(task_id, str):
                    heldout_task_ids.add(task_id)
                heldout_pass += int(score['pass_at_1'])
                if score['pass_at_1'] and isinstance(task_id, str):
                    task_pass_ids.add(task_id)
                    if 'train_example_mismatch' in labels:
                        labels.append('train_fit_heldout_pass')
                    else:
                        labels.append('heldout_task_pass')
                elif any(item.get('feedback') == [] for item in history):
                    labels.append('train_fit_heldout_mismatch')
        if not labels:
            labels.append('unknown_insufficient_evidence')
        labels = sorted(set(labels))
        for label in labels:
            failure_counts[label] = failure_counts.get(label, 0) + 1
        code_count += len(syntax)
        valid_codes += syntax.count('valid')
        elapsed = row.get('elapsed_seconds')
        if isinstance(task_id, str) and type(elapsed) in (int, float):
            elapsed_by_task[task_id] = elapsed_by_task.get(task_id, 0.0) + elapsed
            elapsed_known_tasks.add(task_id)
        diagnostics.append({'task_id': task_id, 'condition': row.get('condition'),
                            'candidate_index': row['candidate_index'],
                            'classification': labels,
                            'result_recorded': result is not None,
                            'code_attempt_count': len(syntax) if history else None,
                            'syntax_valid_attempts': syntax.count('valid') if history else None,
                            'train_feedback_attempts': len(history) if history else None,
                            'train_match_attempts': sum(value == 0 for value in mismatch_counts if value is not None),
                            'train_match_count_available': sum(value is not None for value in mismatch_counts),
                            'heldout_pass_at_1': row['score'].get('pass_at_1') if isinstance(row.get('score'), dict) else None,
                            'elapsed_seconds': elapsed if type(elapsed) in (int, float) else None})

    total_tasks = len({row['task_id'] for row in candidates if isinstance(row['task_id'], str)})
    def mean_per_task(total_key: str) -> dict[str, Any]:
        value = budget.get(total_key) if isinstance(budget, dict) else None
        if type(value) is int and total_tasks:
            return {'total': value, 'mean_per_task': value / total_tasks,
                    'task_count_denominator': total_tasks, 'not_available_count': 0,
                    'per_task_distribution_recorded': False}
        return {'total': None, 'mean_per_task': None,
                'task_count_denominator': total_tasks, 'not_available_count': total_tasks,
                'per_task_distribution_recorded': False}

    sandbox_budget_total = budget.get('containers') if isinstance(budget, dict) else None
    if type(sandbox_budget_total) is int:
        sandbox_denominator = sandbox_budget_total
        sandbox_na = max(0, sandbox_denominator - sandbox_runs)
    else:
        sandbox_denominator = sandbox_runs
        sandbox_na = sandbox_not_available

    summary = {
        'protocol': 'p3c_exploration_failure_analysis_v1', 'run_dir': str(run),
        'run_status': state.get('status') if isinstance(state, dict) else None,
        'task_count': total_tasks, 'candidate_count': len(candidates),
        'response_count': len(responses), 'request_count': len(request_rows),
        'failure_counts_are_overlapping_candidate_classifications': True,
        'failure_breakdown': failure_counts,
        'response_event_failure_counts': response_failure_counts,
        'metrics': {
            'response_parse_rate': _metric(response_parse_ok, len(parse_observations),
                                           max(0, len(request_rows) - len(parse_observations))),
            'valid_code_rate': _metric(valid_codes, code_count, max(0, len(candidates) - code_count)),
            'sandbox_success_rate': _metric(min(sandbox_successes, sandbox_denominator),
                                            sandbox_denominator, sandbox_na),
            'train_example_full_match_rate': _metric(full_train, full_train_known,
                                                     max(0, len(candidates) - full_train_known)),
            'repair_improvement_rate': _metric(repair_improved, repair_known,
                                                max(0, len(candidates) - repair_known)),
            'held_out_task_pass_at_1': _metric(heldout_pass, heldout_known,
                                               max(0, len(candidates) - heldout_known)),
            'verified_unique_tasks': {'count': len(task_pass_ids),
                                      'task_ids': sorted(task_pass_ids),
                                      'not_available_count': max(0, total_tasks - len(heldout_task_ids))},
            'calls_per_task_mean': mean_per_task('calls'),
            'reserved_tokens_per_task_mean': mean_per_task('reserved_tokens'),
            'sandbox_starts_per_task_mean': mean_per_task('containers'),
            'per_task_latency_seconds': _metric(sum(elapsed_by_task.values()), len(elapsed_known_tasks),
                                                max(0, total_tasks - len(elapsed_known_tasks))),
        },
        'evidence_gaps': {
            'task_scoped_request_counts': 'not_recorded; run totals are reported as means over the candidate task count',
            'per_task_resource_distribution': 'not_recorded in the legacy request and budget schema',
            'successful_sandbox_invocations': 'run-wide starts are available from budget.json; successful output invocations are only inferable where training feedback is complete',
            'test_verification': 'only reported when the run result contains a boolean pass_at_1 from the independent verifier',
            'raw_responses_in_report': False,
        },
    }
    with (out / 'task_diagnostics.jsonl').open('w', encoding='utf-8') as stream:
        for row in diagnostics:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (out / 'failure_breakdown.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['classification', 'count', 'unit', 'denominator', 'rate'])
        writer.writeheader()
        for key, count in sorted(failure_counts.items()):
            writer.writerow({'classification': key, 'count': count, 'unit': 'candidate',
                             'denominator': len(candidates),
                             'rate': count / len(candidates) if candidates else ''})
        for key, count in sorted(response_failure_counts.items()):
            writer.writerow({'classification': key, 'count': count, 'unit': 'response_event',
                             'denominator': len(responses),
                             'rate': count / len(responses) if responses else ''})
    md = ['# Exploration failure diagnosis', '', f'- Run: `{run}`',
          f'- Run status: `{summary["run_status"]}`', f'- Tasks: {total_tasks}',
          f'- Candidates: {len(candidates)}', '',
          'Rates use the numerator and denominator in `summary.json`; classifications may overlap.',
          'Legacy artifacts report run-level resource totals as mean per task; task-level distributions were not recorded.',
          'No raw response bodies or generated reasoning text are copied to this report.', '',
          '| Classification | Candidates |', '|---|---:|']
    md.extend(f'| {key} | {value} |' for key, value in sorted(failure_counts.items()))
    md.extend(('', 'Response event failures:', ''))
    md.extend(f'- `{key}`: {value}/{len(responses)} response events' for key, value in sorted(response_failure_counts.items()))
    (out / 'diagnosis.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return summary


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

