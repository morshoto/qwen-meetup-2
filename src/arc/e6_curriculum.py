"""E6 oracle provenance, verification, diagnostics, and distillation records.

This module contains data plumbing only. Transformation programs are accepted
only when their Qwen response evidence is complete and their behavior passes
the isolated verifier.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

from src.arc.data import Grid, Task, validate_grid
from src.arc.ast_fingerprint import stable_ast_sha256, stable_shape_sha256
from src.arc.e5_selfplay import _safe_transform_code, _unique_grids, solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.pipelines.e5 import _verify_generation, _verify_root

VERIFIER_VERSION = 'e6-oracle-verifier-v1'
SOURCE_TYPE = 'generator_distillation'


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def grid_fingerprint(grids: Iterable[Grid]) -> str:
    return sha256_bytes(canonical(list(grids)).encode())


def _quality_grids(seed: int, count: int) -> list[Grid]:
    """Sample distinct inputs with at least one same-shape value contrast."""
    if type(count) is not int or count < 5:
        raise ValueError('oracle verification requires at least five inputs')
    rng = random.Random(seed)
    height, width = rng.randint(2, 5), rng.randint(2, 5)

    def sample(h: int, w: int) -> Grid:
        return [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]

    first = sample(height, width)
    second = sample(height, width)
    while second == first:
        second = sample(height, width)
    result = [first, second]
    fingerprints = {canonical(first), canonical(second)}
    while len(result) < count:
        h, w = rng.randint(2, 5), rng.randint(2, 5)
        grid = sample(h, w)
        fingerprint = canonical(grid)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            result.append(grid)
    return result


def task_fingerprint(task: Task) -> str:
    return sha256_bytes(canonical({
        'train': [{'input': source, 'output': target} for source, target in task.train],
        'query': list(task.test_inputs),
    }).encode())


def oracle_shape_sha256(code: str) -> str:
    return stable_shape_sha256(code)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'{path.name}:{index + 1}: expected object')
            rows.append(value)
    return rows


def diagnose_e5_artifact(root: str | Path) -> dict[str, Any]:
    """Read and verify one completed E5 artifact without modifying it.

    The returned report is safe to store in a new E6 artifact: it contains
    hashes and outcome counts, never response text or code.
    """
    artifact = Path(root).expanduser().resolve()
    identity_path = artifact / 'identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    _verify_root(artifact, identity)
    generation = artifact / 'gen-0'
    _verify_generation(generation)
    before_root_seal = sha256_bytes((artifact / 'root_artifact_hashes.json').read_bytes())
    task_dirs = sorted((generation / 'synthetic').glob('*/self_solve.json'))
    request_rows = _jsonl(artifact / 'logs/e5/llm_requests.jsonl')
    raw_path = artifact / 'logs/e5/llm_responses.jsonl'
    raw_rows = _jsonl(raw_path)
    by_request = {row.get('request_id'): row for row in raw_rows if row.get('request_id')}
    request_index = {row.get('request_id'): row for row in request_rows if row.get('request_id')}

    candidate_reports: list[dict[str, Any]] = []
    category_counts = {'format_mismatch': 0, 'execution_failure': 0,
                       'training_mismatch': 0, 'train_matched_heldout_mismatch': 0,
                       'insufficient_evidence': 0}
    for self_solve_path in task_dirs:
        trace = json.loads(self_solve_path.read_text(encoding='utf-8'))
        task_path = self_solve_path.with_name('task.json')
        task_data = json.loads(task_path.read_text(encoding='utf-8'))
        task_id = task_data.get('task_id')
        payload = task_data.get('task', {})
        task = Task(task_id, tuple((validate_grid(item['input']), validate_grid(item['output']))
                                   for item in payload.get('train', [])),
                    tuple(validate_grid(item['input']) for item in payload.get('test', [])))
        histories = trace.get('history') if isinstance(trace.get('history'), list) else []
        candidates = [item for item in histories if isinstance(item, dict)]
        for index, history in enumerate(candidates):
            code = history.get('code')
            request = {}
            raw = {}
            if isinstance(code, str):
                from src.arc.http_backend import final_answer
                for candidate_request in request_rows:
                    if candidate_request.get('kind') != 'e5_self_solve':
                        continue
                    candidate_raw = by_request.get(candidate_request.get('request_id'), {})
                    candidate_text = candidate_raw.get('generated_text') if isinstance(candidate_raw, dict) else None
                    try:
                        exact = final_answer(candidate_text) == code
                    except Exception:
                        exact = False
                    if exact:
                        request, raw = candidate_request, candidate_raw
                        break
            request_id = request.get('request_id')
            if not raw or request.get('error'):
                category = 'insufficient_evidence'
            elif not isinstance(code, str) or not _safe_transform_code(code):
                category = 'format_mismatch'
            elif history.get('error') or history.get('error_kind'):
                category = 'execution_failure'
            else:
                feedback = history.get('feedback')
                train_mismatch = (isinstance(feedback, list) and any(
                    isinstance(item, dict) and item.get('feedback') ==
                    'candidate output differed from the visible training target' for item in feedback))
                if train_mismatch:
                    category = 'training_mismatch'
                elif feedback == [] and trace.get('status') != 'rejected':
                    category = 'train_matched_heldout_mismatch'
                else:
                    category = 'insufficient_evidence'
            category_counts[category] += 1
            thinking_requested = request.get('thinking_mode_requested')
            thinking_effective = request.get('thinking_mode_effective')
            response_text = raw.get('generated_text') if isinstance(raw, dict) else None
            has_thinking_markup = isinstance(response_text, str) and (
                '<think>' in response_text or '</think>' in response_text)
            candidate_reports.append({
                'task_id': task_id,
                'candidate_index': index,
                'candidate_selection': 'rejected' if category != 'train_matched_heldout_mismatch'
                                       else 'selected_before_heldout_check',
                'category': category,
                'code_sha256': sha256_bytes(code.encode()) if isinstance(code, str) else None,
                'response_parsed': bool(isinstance(raw, dict) and isinstance(response_text, str)),
                'code_extracted': bool(isinstance(code, str)),
                'thinking_mode_requested': thinking_requested,
                'thinking_mode_effective': thinking_effective,
                'thinking_markup_present': has_thinking_markup if response_text is not None else None,
                'stop_reason': request.get('stop_reason'),
                'request_id': request_id,
                'raw_response_sha256': request.get('response_sha256'),
                'train_mismatch_pair_indices': [item.get('pair') for item in history.get('feedback', [])
                                                if isinstance(item, dict) and type(item.get('pair')) is int],
                'heldout_evaluated': category == 'train_matched_heldout_mismatch',
                'timeout': bool(isinstance(request.get('error'), dict) and request['error'].get('kind') == 'timeout'),
            })

    config = identity.get('config', {})
    model = config.get('model', {}) if isinstance(config, dict) else {}
    training = config.get('p3b', {}).get('training', {}) if isinstance(config, dict) else {}
    server_logs = sorted((artifact / 'logs/e5').glob('*server.log'))
    server_text = '\n'.join(path.read_text(encoding='utf-8', errors='replace') for path in server_logs)
    report = {
        'protocol': 'e6_e5_failure_diagnostic_v1',
        'source_artifact': str(artifact),
        'source_root_seal_sha256': before_root_seal,
        'source_generation_seal_verified': True,
        'source_root_seal_verified': True,
        'self_solve_candidates': len(candidate_reports),
        'categories': category_counts,
        'candidates': candidate_reports,
        'generator_oracles': len(list((generation / 'synthetic').glob('*/generator.py'))),
        'qwen_generation_requests': sum(row.get('kind') == 'e5_generator' for row in request_rows),
        'model_revision': model.get('revision'),
        'quantization': model.get('quantization'),
        'thinking_mode_configured': model.get('thinking'),
        'thinking_off_canary_recorded': any(
            row.get('thinking_control') == 'tokenizer_template_canary_passed' and
            row.get('thinking_mode_effective') == 'off' for row in request_rows),
        'common_system_prompt': FINAL_ANSWER_SYSTEM_PROMPT,
        'inference_prompt_builder': 'src.arc.e5_selfplay.solve_prompt',
        'training_prompt_source': 'E5 self_solved_synthetic used canonical task JSON; E6 GD uses solve_prompt(task)',
        'legacy_prompt_alignment': 'E5 self_solved_synthetic prompt did not match solve_prompt(task); E6 GD records do.',
        'training_prompt_mask_enabled': False,
        'e6_training_prompt_mask_policy': 'MLX-LM chat SFT with --mask-prompt; verified by token-span audit before training.',
        'training_runtime_version': training.get('mlx_lm_version'),
        'e5_chat_template_sha256': 'not_retained_in_E5_artifact; E6 remeasures the pinned tokenizer template',
        'evidence_limits': [
            'Held-out outputs were not evaluated because both candidates mismatched visible train pairs.',
            'A candidate that did not reach held-out evaluation is not classified as a held-out failure.',
            'Full hardcode detection is impossible; static checks cover only obvious cases.',
        ],
    }
    after_root_seal = sha256_bytes((artifact / 'root_artifact_hashes.json').read_bytes())
    if after_root_seal != before_root_seal:
        raise ValueError('E5 source root seal changed during read-only diagnostics')
    return report


def _raw_response_index(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    requests = {row.get('request_id'): row for row in _jsonl(root / 'logs/e5/llm_requests.jsonl')}
    raw = {row.get('request_id'): row for row in _jsonl(root / 'logs/e5/llm_responses.jsonl')}
    return requests, raw


def import_e5_oracle(root: str | Path, *, sandbox: Any, seed: int = 66001) -> dict[str, Any]:
    """Import the one sealed E5 oracle only after raw response/provenance match."""
    artifact = Path(root).expanduser().resolve()
    identity = json.loads((artifact / 'identity.json').read_text(encoding='utf-8'))
    _verify_root(artifact, identity)
    generation = artifact / 'gen-0'
    _verify_generation(generation)
    requests, responses = _raw_response_index(artifact)
    generator_path = next(iter(sorted((generation / 'synthetic').glob('*/generator.py'))), None)
    task_path = next(iter(sorted((generation / 'synthetic').glob('*/task.json'))), None)
    if generator_path is None or task_path is None:
        raise ValueError('E5 artifact does not contain a verified generator/task pair')
    oracle_code = generator_path.read_text(encoding='utf-8').strip()
    task_data = json.loads(task_path.read_text(encoding='utf-8'))
    oracle_sha = sha256_bytes(oracle_code.encode())
    if oracle_sha != task_data.get('generator_sha256'):
        raise ValueError('E5 oracle code hash differs from task manifest')
    matching = [(request_id, request) for request_id, request in requests.items()
                if request.get('kind') == 'e5_generator' and request.get('response_sha256')]
    match = None
    for request_id, request in matching:
        raw = responses.get(request_id)
        if not raw:
            continue
        body = raw.get('raw_http_body_text')
        raw_sha = request.get('response_sha256')
        if not isinstance(body, str) or sha256_bytes(body.encode()) != raw_sha:
            continue
        text = raw.get('generated_text')
        if not isinstance(text, str):
            continue
        # final_answer is deliberately imported locally: raw response text is
        # inspected only in memory and never copied into an E6 artifact.
        from src.arc.http_backend import final_answer
        try:
            extracted = final_answer(text)
        except Exception:
            continue
        if extracted.strip() == oracle_code:
            match = (request_id, request, raw_sha)
            break
    if match is None:
        raise ValueError('Qwen response evidence does not identify the sealed E5 oracle')
    request_id, request, raw_sha = match
    task_payload = task_data.get('task')
    if not isinstance(task_payload, dict):
        raise ValueError('E5 task payload is missing')
    train_items = task_payload.get('train', [])
    train = tuple((validate_grid(item['input']), validate_grid(item['output'])) for item in train_items)
    heldout_items = task_payload.get('test', [])
    heldout_inputs = tuple(validate_grid(item['input']) for item in heldout_items)
    # E5 keeps its private verifier outputs outside task.json and in a separate
    # verifier file. They are read only to confirm the imported oracle.
    verifier_path = next(iter(sorted((generation / 'synthetic').glob('*/private_verifier.json'))), None)
    verifier = json.loads(verifier_path.read_text(encoding='utf-8')) if verifier_path else {}
    hidden_outputs = tuple(validate_grid(item) for item in verifier.get('hidden_test_outputs', []))
    validation_inputs = tuple(validate_grid(item) for item in verifier.get('validation_inputs', []))
    validation_outputs = tuple(validate_grid(item) for item in verifier.get('validation_outputs', []))
    if len(heldout_inputs) != len(hidden_outputs):
        raise ValueError('E5 private verifier input/output counts do not match')
    heldout_inputs_all = heldout_inputs + validation_inputs
    heldout_outputs_all = hidden_outputs + validation_outputs
    if len(heldout_inputs_all) != len(heldout_outputs_all):
        raise ValueError('E5 held-out inputs and outputs do not match')
    reserved_inputs = {canonical(source) for source, _ in train}
    reserved_inputs.update(canonical(grid) for grid in heldout_inputs_all)
    query_inputs = ()
    for offset in range(32):
        candidate = _unique_grids(seed + offset, 1)[0]
        if canonical(candidate) not in reserved_inputs:
            query_inputs = (candidate,)
            break
    if not query_inputs:
        raise ValueError('could not sample a fresh E6 query input')
    all_inputs = ([source for source, _ in train] + list(query_inputs) + list(heldout_inputs_all))
    actual = verify_oracle(oracle_code, all_inputs, sandbox=sandbox)
    train_actual = actual[:len(train)]
    query_outputs = tuple(actual[len(train):len(train) + len(query_inputs)])
    heldout_actual = actual[len(train) + len(query_inputs):]
    if train_actual != [target for _, target in train]:
        raise ValueError('E5 oracle failed a visible train example')
    if heldout_actual != list(heldout_outputs_all):
        raise ValueError('E5 oracle failed a sealed held-out verifier check')
    task = Task(task_data['task_id'], train, query_inputs)
    return {
        'task': task,
        'oracle_code': oracle_code,
        'oracle_sha256': oracle_sha,
        'oracle_ast_sha256': stable_ast_sha256(oracle_code),
        'oracle_shape_sha256': oracle_shape_sha256(oracle_code),
        'heldout_inputs': heldout_inputs_all,
        'heldout_outputs': heldout_outputs_all,
        'query_outputs': query_outputs,
        'provenance': {
            'source_type': SOURCE_TYPE,
            'source_artifact': str(artifact),
            'source_generation': 0,
            'request_id': request_id,
            'raw_response_sha256': raw_sha,
            'model_revision': request.get('revision'),
            'model_id': request.get('model'),
            'execution_mode': 'live_model',
            'request_log_path': str(artifact / 'logs/e5/llm_requests.jsonl'),
            'response_log_path': str(artifact / 'logs/e5/llm_responses.jsonl'),
            'thinking_mode': request.get('thinking_mode_effective'),
            'generation': 0,
            'seed_requested': request.get('seed_requested'),
        },
    }


def verify_oracle(code: str, inputs: Iterable[Grid], *, sandbox: Any) -> list[Grid]:
    """Run quality checks through an explicit sandbox; never execute on host."""
    if not _safe_transform_code(code):
        raise ValueError('static_safety_rejected')
    tree = ast.parse(code)
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise ValueError('external_dependency_rejected')
    checked_inputs = [validate_grid(grid) for grid in inputs]
    if len(checked_inputs) < 5:
        raise ValueError('insufficient_distinct_inputs')
    input_fps = {canonical(grid) for grid in checked_inputs}
    if len(input_fps) != len(checked_inputs):
        raise ValueError('duplicate_input_rejected')
    outputs: list[Grid] = []
    for grid in checked_inputs:
        first = validate_grid(sandbox.run(code, grid))
        second = validate_grid(sandbox.run(code, grid))
        if first != second:
            raise ValueError('nondeterministic_output_rejected')
        if sum(len(row) for row in first) > 900:
            raise ValueError('output_size_rejected')
        outputs.append(first)
    if len({canonical(output) for output in outputs}) < 2:
        raise ValueError('constant_or_input_ignoring_output_rejected')
    shape_pairs = [(left, right) for left in range(len(checked_inputs))
                   for right in range(left + 1, len(checked_inputs))
                   if len(checked_inputs[left]) == len(checked_inputs[right]) and
                   len(checked_inputs[left][0]) == len(checked_inputs[right][0]) and
                   checked_inputs[left] != checked_inputs[right]]
    if not shape_pairs or not any(outputs[left] != outputs[right] for left, right in shape_pairs):
        raise ValueError('input_ignoring_or_shape_only_output_rejected')
    return outputs


def make_distillation_record(task: Task, oracle_code: str, *, provenance: dict[str, Any],
                             heldout_inputs: Iterable[Grid], heldout_outputs: Iterable[Grid],
                             generation: int) -> dict[str, Any]:
    required = {'request_id', 'raw_response_sha256', 'model_revision', 'source_type'}
    if provenance.get('source_type') != SOURCE_TYPE or not required <= set(provenance):
        raise ValueError('Qwen generator provenance is incomplete')
    if provenance.get('source_type') in {'external_or_human', 'self_solved_synthetic', 'self_solved_real_arc'}:
        raise ValueError('wrong source type for generator distillation')
    for name in ('raw_response_sha256',):
        if not isinstance(provenance.get(name), str) or len(provenance[name]) != 64:
            raise ValueError(f'invalid {name}')
    if not isinstance(provenance.get('request_id'), str) or not provenance['request_id']:
        raise ValueError('missing Qwen request ID')
    if not isinstance(provenance.get('model_revision'), str) or not provenance['model_revision']:
        raise ValueError('missing Qwen model revision')
    prompt = solve_prompt(task)
    oracle_sha = sha256_bytes(oracle_code.encode())
    heldout_in = list(heldout_inputs)
    heldout_out = list(heldout_outputs)
    if not heldout_in or len(heldout_in) != len(heldout_out):
        raise ValueError('held-out verifier evidence is incomplete')
    record_id = sha256_bytes((task.id + '\0' + prompt + '\0' + oracle_sha).encode())
    record = {
        'id': record_id,
        'task_id': task.id,
        'split': 'generator_distillation',
        'generation': generation,
        'data_source': SOURCE_TYPE,
        'source_type': SOURCE_TYPE,
        'synthetic': True,
        'oracle_sha256': oracle_sha,
        'oracle_ast_sha256': stable_ast_sha256(oracle_code),
        'oracle_shape_sha256': oracle_shape_sha256(oracle_code),
        'input_family_fingerprint': task_fingerprint(task),
        'train_input_fingerprint': grid_fingerprint(source for source, _ in task.train),
        'query_input_fingerprint': grid_fingerprint(task.test_inputs),
        'heldout_input_fingerprint': grid_fingerprint(heldout_in),
        'heldout_output_fingerprint': grid_fingerprint(heldout_out),
        'verifier_version': VERIFIER_VERSION,
        'qwen_provenance': dict(provenance),
        'base_revision': provenance['model_revision'],
        'adapter_revision': 'none',
        '_verifier_only': {
            'inputs': heldout_in,
            'outputs': heldout_out,
        },
        'visible_task': {
            'id': task.id,
            'train': [{'input': source, 'output': target} for source, target in task.train],
            'test_inputs': list(task.test_inputs),
        },
        'messages': [
            {'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': oracle_code},
        ],
    }
    return record


def collection_rows(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Deduplicate by task identity and program AST, retaining source labels."""
    seen_tasks: set[str] = set()
    seen_oracles: set[str] = set()
    seen_shapes: set[str] = set()
    accepted = []
    rejected = []
    for record in records:
        task_id = record.get('task_id')
        oracle = record.get('oracle_ast_sha256')
        shape = record.get('oracle_shape_sha256')
        if task_id in seen_tasks:
            rejected.append({'task_id': task_id, 'reason': 'duplicate_task_id'})
        elif oracle in seen_oracles:
            rejected.append({'task_id': task_id, 'reason': 'duplicate_oracle_fingerprint'})
        elif shape in seen_shapes:
            rejected.append({'task_id': task_id, 'reason': 'similar_oracle_shape'})
        elif record.get('data_source') != SOURCE_TYPE:
            rejected.append({'task_id': task_id, 'reason': 'source_type_rejected'})
        else:
            seen_tasks.add(task_id)
            seen_oracles.add(oracle)
            seen_shapes.add(shape)
            accepted.append(record)
    return {'accepted': accepted, 'rejected': rejected}


def write_private_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')


def synthesize_fake_task(seed: int, task_id: str, *, code: str, sandbox: Any,
                         train_examples: int = 3, query_inputs: int = 1,
                         heldout_inputs: int = 3) -> dict[str, Any]:
    """Create a deterministic fixture task; sandbox remains the only executor."""
    grids = _quality_grids(seed, train_examples + query_inputs + heldout_inputs)
    outputs = [validate_grid(sandbox.run(code, grid)) for grid in grids]
    train = tuple((grids[index], outputs[index]) for index in range(train_examples))
    q_start = train_examples
    query = tuple(grids[q_start:q_start + query_inputs])
    h_start = q_start + query_inputs
    heldout_in = tuple(grids[h_start:])
    heldout_out = tuple(outputs[h_start:])
    task = Task(task_id, train, query)
    return {'task': task, 'heldout_inputs': heldout_in, 'heldout_outputs': heldout_out,
            'query_outputs': tuple(outputs[q_start:q_start + query_inputs])}

