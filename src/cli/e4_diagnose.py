"""Build an offline diagnosis from saved E2/E3 traces and E4 synthetic runs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.arc.data import validate_grid
from src.arc.sandbox import DockerSandbox
from src.core.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _train_pairs(challenges_path: Path) -> dict[str, list[tuple[list[list[int]], list[list[int]]]]]:
    challenges = _json(challenges_path)
    return {task_id: [(validate_grid(pair['input']), validate_grid(pair['output']))
                      for pair in item['train']]
            for task_id, item in challenges.items()}


def _last_line(exc: Exception) -> str:
    return str(exc).splitlines()[-1][:300]


def _replay(code: str | None, pairs: list[tuple[list[list[int]], list[list[int]]]],
            sandbox: DockerSandbox) -> dict[str, Any]:
    if not code:
        return {'matched': 0, 'examples': len(pairs), 'execution_errors': len(pairs),
                'error_lines': ['candidate_code_missing']}
    matched = 0
    errors = []
    for index, (grid, expected) in enumerate(pairs):
        try:
            actual = validate_grid(sandbox.run(code, grid))
            matched += actual == expected
        except Exception as exc:
            errors.append({'example': index, 'error': _last_line(exc)})
    return {'matched': matched, 'examples': len(pairs),
            'execution_errors': len(errors), 'error_lines': errors}


def diagnose(e2_dir: Path, e3_dir: Path, baseline_dir: Path,
             output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite E4 diagnosis: {output_dir}')
    e2_results = _jsonl(e2_dir / 'results.jsonl')
    e2_traces = _jsonl(e2_dir / 'candidate_traces.jsonl')
    e2_attempts = _jsonl(e2_dir / 'candidate_attempts.jsonl')
    e3_summary = _json(e3_dir / 'collection_summary.json')
    e3_rejected = _jsonl(e3_dir / 'rejected_candidates.jsonl')
    e3_traces = _jsonl(e3_dir / 'candidate_traces.jsonl')
    baseline_rows = _jsonl(baseline_dir / 'baseline_results.jsonl')

    config = load_experiment_config(ROOT / 'configs/e2.live.full.yml')
    training = _train_pairs(Path(config.data.challenges))
    sandbox = DockerSandbox(config.agent.sandbox_image, timeout=config.agent.timeout_seconds,
                            memory_mb=config.agent.memory_mb, cpus=config.agent.cpus,
                            pids=config.agent.pids,
                            max_output_bytes=config.agent.max_output_bytes,
                            max_code_bytes=config.agent.max_code_bytes)

    strategy_attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    s0_replays = []
    for trace in e2_traces:
        strategy = trace.get('strategy')
        strategy_attempts[strategy].append(trace)
        if strategy == 'S0':
            replay = _replay(trace.get('code'), training[trace['task_id']], sandbox)
            s0_replays.append({'task_id': trace['task_id'],
                               'candidate_index': trace.get('candidate_index'),
                               'code_sha256': trace.get('code_sha256'), **replay})

    e2_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strategy_attempts['S0']:
        e2_by_task[row['task_id']].append(row)
    s0_repair_repeats = sum(
        len(rows) >= 2 and rows[0].get('code_sha256') == rows[1].get('code_sha256')
        for rows in e2_by_task.values())
    e2_attempt_events: dict[str, Counter] = defaultdict(Counter)
    for event in e2_attempts:
        e2_attempt_events[event.get('strategy', 'unknown')][
            f"{event.get('kind')}:{event.get('finish_reason')}:{event.get('http_diagnostic_category')}"] += 1

    e2_strategy_summary = {}
    for strategy in ('S0', 'S1', 'S2', 'S3'):
        traces = strategy_attempts.get(strategy, [])
        if strategy == 'S0':
            matches = sum(row['matched'] for row in s0_replays)
            examples = sum(row['examples'] for row in s0_replays)
            errors = sum(row['execution_errors'] for row in s0_replays)
        else:
            matches = sum(int(row.get('train_match_count') or 0) for row in traces)
            examples = sum(int(row.get('train_example_count') or 0) for row in traces)
            errors = sum(int(row.get('execution_errors') or 0) for row in traces)
        e2_strategy_summary[strategy] = {
            'candidate_attempts': len(traces),
            'parsed_candidates': sum(row.get('parse_status') == 'parsed' for row in traces),
            'duplicate_candidates': sum(row.get('parse_status') == 'duplicate' for row in traces),
            'static_filter_passes': sum(row.get('static_filter_pass') is True for row in traces),
            'complete_train_match_candidates': sum(row.get('complete_train_match') is True for row in traces),
            'train_matches': matches, 'train_examples': examples,
            'execution_errors': errors,
            'response_events': dict(e2_attempt_events[strategy])}

    e2_results_summary = {}
    for strategy in ('S0', 'S1', 'S2', 'S3'):
        rows = [row for row in e2_results if row.get('strategy') == strategy]
        e2_results_summary[strategy] = {
            'tasks': len(rows),
            'complete_train_match_tasks': sum(row.get('final_train_full_match') is True for row in rows),
            'macro_train_match_rate': (sum(float(row.get('final_train_match_rate') or 0.0) for row in rows) / len(rows)) if rows else None,
            'heldout_verifier_passes': sum(row.get('pass_at_1') is True for row in rows)}

    rejection_reasons = Counter(row.get('rejection_reason') for row in e3_rejected)
    verifier_rejections = [row for row in e3_rejected
                           if row.get('rejection_reason') == 'independent_verifier_failed']
    unselected = [row for row in e3_rejected
                  if row.get('rejection_reason') == 'not_selected_by_train_policy']
    e3_train_matches = Counter((row.get('train_match_count'), row.get('train_example_count'))
                                for row in verifier_rejections)

    synthetic_training = _train_pairs(ROOT / 'tests/fixtures/e4_synthetic_challenges.v1.json')
    baseline_replays = []
    duplicate_repair_tasks = 0
    for row in baseline_rows:
        pairs = synthetic_training[row['task_id']]
        code = row.get('code')
        replay = _replay(code, pairs, sandbox)
        codes = [attempt.get('code') for attempt in row.get('attempt_trace', [])
                 if isinstance(attempt.get('code'), str)]
        code_hashes = [hashlib.sha256(value.encode()).hexdigest() for value in codes]
        repair_repeated = len(code_hashes) >= 2 and code_hashes[0] == code_hashes[1]
        duplicate_repair_tasks += repair_repeated
        baseline_replays.append({'task_id': row['task_id'],
                                 'model_attempt_count': row.get('candidate_attempts'),
                                 'distinct_code_count': len(set(code_hashes)),
                                 'repair_repeated_code': repair_repeated,
                                 'train_replay': replay})

    summary = {
        'protocol': 'e4_saved_trace_diagnosis_v1',
        'sources': {
            'e2_results_sha256': _sha(e2_dir / 'results.jsonl'),
            'e2_candidate_traces_sha256': _sha(e2_dir / 'candidate_traces.jsonl'),
            'e2_candidate_attempts_sha256': _sha(e2_dir / 'candidate_attempts.jsonl'),
            'e3_collection_summary_sha256': _sha(e3_dir / 'collection_summary.json'),
            'e3_rejected_candidates_sha256': _sha(e3_dir / 'rejected_candidates.jsonl'),
            'e3_candidate_traces_sha256': _sha(e3_dir / 'candidate_traces.jsonl'),
            'e4_baseline_results_sha256': _sha(baseline_dir / 'baseline_results.jsonl')},
        'e2': {'tasks': 5, 'strategies': e2_strategy_summary,
               'final_task_results': e2_results_summary,
               's0_repair_repeats_identical_code': s0_repair_repeats,
               's0_replay_by_candidate': s0_replays},
        'e3': {'attempted_tasks': len(e3_summary.get('attempted_task_ids', [])),
               'accepted_solutions': e3_summary.get('verified_unique_tasks', 0),
               'rejected_candidate_count': len(e3_rejected),
               'rejection_reasons': dict(rejection_reasons),
               'verifier_rejected_train_match_counts': {
                   f'{matches}/{examples}': count
                   for (matches, examples), count in sorted(e3_train_matches.items(),
                                                             key=lambda item: str(item[0]))},
               'selected_candidate_train_match_count': sum(
                   int(row.get('train_match_count') or 0) for row in verifier_rejections),
               'selected_candidate_train_examples': sum(
                   int(row.get('train_example_count') or 0) for row in verifier_rejections),
               'candidate_trace_count': len(e3_traces),
               'stop_reason': e3_summary.get('stop_reason')},
        'e4_synthetic_baseline': {'tasks': len(baseline_rows),
                                  'repair_repeats_identical_code': duplicate_repair_tasks,
                                  'train_replay_by_task': baseline_replays}}

    output_dir.mkdir(parents=True)
    (output_dir / 'diagnosis.json').write_text(json.dumps(summary, sort_keys=True,
                                                          ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8')
    lines = ['# E4-A: saved-trace diagnosis', '',
             'This diagnosis uses saved public training traces and isolated replays of generated code. It does not use hidden ARC or synthetic outputs to change candidate selection.', '',
             '## E2', '',
             '| Strategy | Candidate attempts | Unique parsed | Duplicates | Static pass | Train cell matches | Replay errors | Full-train tasks | Heldout passes |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for strategy in ('S0', 'S1', 'S2', 'S3'):
        candidate = e2_strategy_summary[strategy]
        result = e2_results_summary[strategy]
        lines.append(f"| {strategy} | {candidate['candidate_attempts']} | {candidate['parsed_candidates']} | "
                     f"{candidate['duplicate_candidates']} | {candidate['static_filter_passes']} | "
                     f"{candidate['train_matches']}/{candidate['train_examples']} | "
                     f"{candidate['execution_errors']} | {result['complete_train_match_tasks']}/{result['tasks']} | "
                     f"{result['heldout_verifier_passes']}/{result['tasks']} |")
    lines.extend(['', f"S0 repeated identical code after repair on {s0_repair_repeats}/5 tasks. Offline replay found {sum(row['execution_errors'] for row in s0_replays)} train execution errors. The saved E2 S0 candidate trace omitted those error messages because non-list feedback was not serialized; the replay restores the specific error evidence.", '',
                  '| S0 task | Attempt | Train matches | Error |',
                  '|---|---:|---:|---|'])
    for row in s0_replays:
        for error in row['error_lines']:
            lines.append(f"| {row['task_id']} | {row['candidate_index']} | {row['matched']}/{row['examples']} | {error['error']} |")
    lines.extend(['', 'S1 and S3 each produced 12 unique parsed candidates and 3 duplicate candidates. The 12 unique candidates passed static checks and ran on training inputs, but every candidate matched zero training outputs. Their candidate prompts ask for code directly and do not save a separate hypothesis, so the traces cannot tell whether the conceptual rule or its Python encoding was wrong. S0 analysis responses are also not included in candidate traces; no stronger claim about the model internal rule choice is made.', '',
                  '## E3', '',
                  f"Collection attempted {summary['e3']['attempted_tasks']} tasks and accepted {summary['e3']['accepted_solutions']} solutions. Rejections: `{dict(rejection_reasons)}`. Of the verifier-rejected selected candidates, training match counts were `{summary['e3']['verifier_rejected_train_match_counts']}`. S1 selection chooses the first safe candidate that executes over every training input; it does not rank by train exact-match count. The zero-accept result therefore reflects both weak candidate generation and a selector with no exact-match requirement.", '',
                  '## E4-B baseline', '',
                  f"On the fixed synthetic suite, the existing code agent matched {sum(row['train_replay']['matched'] for row in baseline_replays)}/30 training examples and all training examples on {sum(row['train_replay']['matched'] == row['train_replay']['examples'] and row['train_replay']['execution_errors'] == 0 for row in baseline_replays)}/10 tasks. It made {sum(row.get('candidate_attempts') or 0 for row in baseline_rows)} code-generation attempts. Identical repair code was repeated on {duplicate_repair_tasks}/10 tasks.", '',
                  '| Task | Train matches | Execution errors | Error detail |',
                  '|---|---:|---:|---|'])
    for row in baseline_replays:
        replay = row['train_replay']
        error = '; '.join(f"ex{item['example'] + 1}: {item['error']}"
                          for item in replay['error_lines']) or '—'
        lines.append(f"| {row['task_id']} | {replay['matched']}/{replay['examples']} | {replay['execution_errors']} | {error} |")
    lines.extend(['', '## Interpretation', '',
                  'The logs support three separate findings: executable code generation is not the same as learning the transformation; code output-shape/indexing errors occur on simple tasks; and the current repair loop often repeats or fails to correct the candidate after exact train feedback. The operator-search comparison improves this narrow benchmark, but its operator set matches the transformations used to construct the synthetic suite.', ''])
    (output_dir / 'diagnosis.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--e2-dir', type=Path, default=ROOT / 'artifacts/e2-live-full')
    parser.add_argument('--e3-dir', type=Path, default=ROOT / 'artifacts/e3-live-collection')
    parser.add_argument('--baseline-dir', type=Path,
                        default=ROOT / 'artifacts/e4-synthetic-20261101/baseline-v3')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = diagnose(args.e2_dir, args.e3_dir, args.baseline_dir, args.output_dir)
    print(json.dumps({'protocol': summary['protocol'], 'e2_strategies': list(summary['e2']['strategies']),
                      'e3_rejections': summary['e3']['rejection_reasons'],
                      'e4_baseline_tasks': summary['e4_synthetic_baseline']['tasks']},
                     sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
