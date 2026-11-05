"""Bounded E6 self-generated curriculum and QLoRA experiment pipeline."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

import yaml

from src.arc.data import Task, load_arc, split_tasks, validate_grid
from src.arc.e5_selfplay import _generator_prompt, _safe_transform_code, solve_prompt
from src.arc.e6_curriculum import (SOURCE_TYPE, VERIFIER_VERSION, collection_rows,
                                   _quality_grids, diagnose_e5_artifact, grid_fingerprint,
                                   import_e5_oracle, make_distillation_record,
                                   sha256_bytes, synthesize_fake_task,
                                   verify_oracle, write_private_jsonl)
from src.arc.http_backend import final_answer
from src.arc.mlx_lora import train_adapter
from src.arc.mlx_token_driver import adapter_sha256
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.sandbox import DockerSandbox
from src.core.config import E5Settings, ExperimentConfig
from src.pipelines.e5 import (E5Budget, E5CachedBackend, FakeE5Backend,
                              MeteredSandbox, _backend, _docker_image_id,
                              _promotion_decision, _promotion_eval,
                              _verify_mutable_file, _seal_mutable_file)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _json(path: Path, value: Any, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if private:
        os.chmod(path, 0o600)


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, private: bool = True) -> None:
    write_private_jsonl(path, rows)
    if not private:
        os.chmod(path, 0o644)


def _secure_private_dirs(root: Path) -> None:
    for relative in ('logs/e6', 'checkpoints/e6', 'training', 'training/audit-input'):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha(path)
            for path in sorted(root.rglob('*')) if path.is_file() and
            path.name not in {'artifact_hashes.json', '.run.lock'}}


def _observed_token_usage(path: Path) -> dict[str, Any]:
    prompt = completion = total = known = calls = 0
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                usage = row.get('usage')
                if not isinstance(usage, dict):
                    continue
                calls += 1
                p, c, t = usage.get('prompt_tokens'), usage.get('completion_tokens'), usage.get('total_tokens')
                if type(p) is int and p >= 0:
                    prompt += p
                if type(c) is int and c >= 0:
                    completion += c
                if type(t) is int and t >= 0:
                    total += t
                    known += 1
            except (TypeError, ValueError):
                continue
    return {'observed_prompt_tokens': prompt if calls else None,
            'observed_completion_tokens': completion if calls else None,
            'observed_total_tokens': total if known else None,
            'calls_with_usage': calls}


def _seal(root: Path) -> None:
    _json(root / 'artifact_hashes.json', _tree_hashes(root))


def _verify_seal(root: Path) -> None:
    manifest = root / 'artifact_hashes.json'
    if not manifest.is_file() or json.loads(manifest.read_text()) != _tree_hashes(root):
        raise ValueError('E6 artifact seal mismatch')


@contextmanager
def _run_lock(root: Path) -> Iterator[None]:
    with (root / '.run.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another E6 process is using this run directory') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _resolve(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _compat_e5(config: ExperimentConfig) -> ExperimentConfig:
    """Reuse E5's already-audited guarded server and metering with E6 caps."""
    e6 = config.e6
    settings = E5Settings(
        enabled=True, profile='cpu_fake' if e6.profile == 'cpu_fake' else 'live_smoke',
        conditions=['B', 'C1'], generations=0, real_arc_tasks_per_generation=0,
        synthetic_tasks_per_generation=1, generator_candidates=1,
        self_solve_attempts=1, train_examples=e6.train_examples,
        hidden_test_inputs=e6.heldout_inputs, generator_validation_inputs=0,
        minimum_successful_tasks=e6.min_unique_verified_tasks,
        max_model_calls=e6.max_model_calls, max_reserved_tokens=e6.max_reserved_tokens,
        max_sandbox_runs=e6.max_sandbox_runs, max_training_steps=e6.max_training_steps,
        max_wall_seconds=e6.max_wall_seconds, max_training_seconds=e6.max_training_seconds,
        checkpoint_dir=e6.checkpoint_dir, log_dir=e6.log_dir)
    return replace(config, e5=settings)


def _identity(config: ExperimentConfig, project: Path, docker_id: str) -> dict[str, Any]:
    challenge = _resolve(project, config.data.challenges)
    solutions = _resolve(project, config.data.solutions)
    e5_root = _resolve(project, config.e6.e5_artifact)
    implementation = ('src/core/config.py', 'src/arc/ast_fingerprint.py', 'src/arc/e5_selfplay.py',
                      'src/arc/e6_curriculum.py', 'src/arc/e6_fake.py',
                      'src/arc/mlx_lora.py', 'src/arc/mlx_chat_template.py',
                      'src/arc/mlx_token_driver.py', 'src/arc/http_backend.py',
                      'src/arc/prompting.py', 'src/arc/sandbox.py',
                      'src/arc/guarded_contract.py', 'src/cli/guarded_mlx_server.py',
                      'src/cli/p3b_sft_audit.py', 'src/cli/p3b_train.py',
                      'src/cli/p3b_probe.py', 'src/pipelines/e5.py', 'src/pipelines/e6.py')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=project, capture_output=True,
                          text=True, check=False).stdout.strip() or 'unknown'
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=project, capture_output=True,
                           text=True, check=False).stdout.splitlines()
    return {
        'protocol': 'e6_generator_distillation_v1',
        'execution_mode': 'fake_fixture' if config.e6.profile == 'cpu_fake' else 'live_model',
        'profile': config.e6.profile,
        'config': asdict(config),
        'config_sha256': sha256_bytes(_canonical(asdict(config)).encode()),
        'data_sha256': {'challenges': _sha(challenge), 'solutions': _sha(solutions)},
        'e5_source_artifact': str(e5_root),
        'e5_source_root_seal_sha256': _sha(e5_root / 'root_artifact_hashes.json'),
        'sandbox_image_id': docker_id,
        'implementation_sha256': {name: _sha(project / name) for name in implementation},
        'git_head': head,
        'git_dirty': dirty,
        'operator_search_used': False,
        'public_evaluation_accessed': False,
    }


def _claim_live_attempt(config: ExperimentConfig, root: Path, *, resume: bool,
                        identity: dict[str, Any]) -> None:
    """Enforce one live smoke and at most one bootstrap per artifact namespace."""
    project = Path(__file__).resolve().parents[2]
    registry = _resolve(project, config.tracking.output_dir) / 'e6'
    registry.mkdir(parents=True, exist_ok=True)
    os.chmod(registry, 0o700)
    marker = registry / f'.{config.e6.profile}-attempt.json'
    if marker.is_file():
        value = json.loads(marker.read_text(encoding='utf-8'))
        if value.get('run_dir') != str(root.resolve()):
            raise RuntimeError(f'only one {config.e6.profile} live run is allowed in this artifact namespace')
        if not resume:
            raise FileExistsError(f'{config.e6.profile} live attempt already exists; resume the same run')
        if value.get('identity_sha256') != identity['config_sha256']:
            raise ValueError('E6 live attempt marker identity mismatch')
        return
    if resume:
        raise ValueError('cannot resume E6 Live without its single-attempt marker')
    _json(marker, {'profile': config.e6.profile, 'run_dir': str(root.resolve()),
                   'identity_sha256': identity['config_sha256'],
                   'started_unix': time.time()}, private=True)


def _diagnostic_write(root: Path, source_root: Path) -> dict[str, Any]:
    report = diagnose_e5_artifact(source_root)
    path = root / 'diagnostics/e5-self-solve.json'
    _json(path, report)
    markdown = [
        '# E5 self-solve diagnostic', '',
        f"- Source artifact seal: `{report['source_root_seal_sha256']}`",
        f"- Self-solve candidates: {report['self_solve_candidates']}",
        f"- Training mismatches: {report['categories']['training_mismatch']}",
        f"- Held-out evaluations: {report['categories']['train_matched_heldout_mismatch']}",
        f"- Insufficient evidence: {report['categories']['insufficient_evidence']}",
        f"- Model revision / quantization: `{report['model_revision']}` / `{report['quantization']}`",
        f"- Thinking mode: `{report['thinking_mode_configured']}`",
        '', 'Both saved candidates were parsed and executed, then disagreed with visible training examples. '
        'Neither reached hidden-input evaluation. The original failed attempts remain classified as '
        '`self_solved_synthetic` failures; generator distillation does not change that outcome.', '',
        'The original raw response sidecar remains at its sealed E5 path and is not copied here.', '',
    ]
    (root / 'diagnostics/e5-self-solve.md').write_text('\n'.join(markdown), encoding='utf-8')
    return report


def _run_sft_mask_audit(config: ExperimentConfig, root: Path,
                        records: list[dict[str, Any]], *, timeout_seconds: int) -> dict[str, Any]:
    """Run the installed MLX-LM tokenizer audit even when training is skipped."""
    from src.arc.mlx_lora import _training_safe_record
    training = config.p3b.training
    if not records:
        return {'status': 'skipped_empty_data', 'audit_passed': False,
                'reason': 'no_generator_distillation_records'}
    if not training.python_executable or not training.model_path:
        return {'status': 'unavailable', 'audit_passed': False,
                'reason': 'MLX tokenizer audit requires configured local runtime and model'}
    dataset = root / 'training/audit-input'
    dataset.mkdir(parents=True, exist_ok=True)
    os.chmod(dataset, 0o700)
    rows = [{key: value for key, value in record.items() if key != '_verifier_only'}
            for record in records]
    for record in rows:
        _training_safe_record(record, 'e6_generator_distillation')
    train_path = dataset / 'train.jsonl'
    train_path.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n'
                                    for row in rows), encoding='utf-8')
    os.chmod(train_path, 0o600)
    diagnostics = root / 'diagnostics'
    diagnostics.mkdir(parents=True, exist_ok=True)
    os.chmod(diagnostics, 0o700)
    output = diagnostics / 'e6-sft-mask-audit.json'
    command = [training.python_executable, '-m', 'src.cli.p3b_sft_audit',
               '--model', training.model_path, '--dataset', str(dataset),
               '--output', str(output), '--thinking-mode', config.model.thinking,
               '--source', 'e6_generator_distillation']
    try:
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=max(1, min(120, timeout_seconds)),
            env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}, check=False)
    except subprocess.TimeoutExpired as exc:
        completed = None
        error = 'tokenizer_audit_timeout'
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or '')
    else:
        error = None if completed.returncode == 0 else 'tokenizer_audit_failed'
        stdout, stderr = completed.stdout, completed.stderr
    (root / 'logs/e6/sft-audit.stdout.log').write_text(stdout, encoding='utf-8')
    (root / 'logs/e6/sft-audit.stderr.log').write_text(stderr, encoding='utf-8')
    os.chmod(root / 'logs/e6/sft-audit.stdout.log', 0o600)
    os.chmod(root / 'logs/e6/sft-audit.stderr.log', 0o600)
    report = json.loads(output.read_text()) if output.is_file() else {}
    passed = bool(error is None and completed is not None and report.get('audit_passed') is True)
    summary = {'status': 'passed' if passed else 'failed', 'audit_passed': passed,
               'error': error, 'example_count': report.get('example_count'),
               'unique_task_count': report.get('unique_task_count'),
               'dataset_sha256': report.get('dataset_sha256'),
               'chat_template_sha256': report.get('chat_template_sha256'),
               'tokenizer_config_sha256': report.get('tokenizer_config_sha256'),
               'thinking_mode': report.get('thinking_mode'),
               'mask_prompt': report.get('mask_prompt'),
               'examples': report.get('examples', [])}
    _json(output, summary, private=True)
    return summary


def _fake_records(config: ExperimentConfig, generation: int, *, start: int = 0) -> list[dict[str, Any]]:
    from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
    sandbox = FakeE6Sandbox()
    provenance_base = {'source_type': SOURCE_TYPE, 'model_revision': 'fake-qwen-revision',
                       'model_id': 'fake-qwen', 'generation': generation,
                       'thinking_mode': 'off', 'execution_mode': 'fake_fixture'}
    records = []
    for index in range(start, min(len(FAKE_PROGRAMS), config.e6.max_collection_attempts)):
        code = FAKE_PROGRAMS[index]
        task_data = synthesize_fake_task(6000 + generation * 100 + index,
            f'e6fake-g{generation}-task-{index}', code=code, sandbox=sandbox,
            train_examples=config.e6.train_examples, query_inputs=config.e6.query_inputs,
            heldout_inputs=config.e6.heldout_inputs)
        task = task_data['task']
        all_inputs = ([source for source, _ in task.train] + list(task.test_inputs) +
                      list(task_data['heldout_inputs']))
        outputs = verify_oracle(code, all_inputs, sandbox=sandbox)
        train_count = len(task.train)
        query_count = len(task.test_inputs)
        heldout_inputs = list(task_data['heldout_inputs'])
        heldout_outputs = outputs[train_count + query_count:]
        provenance = {**provenance_base, 'request_id': f'fake-request-{generation}-{index}',
                      'raw_response_sha256': hashlib.sha256(
                          f'fake-raw-response-{generation}-{index}'.encode()).hexdigest()}
        records.append(make_distillation_record(task, code, provenance=provenance,
            heldout_inputs=heldout_inputs, heldout_outputs=heldout_outputs, generation=generation))
    return collection_rows(records)['accepted']


def _write_collection(root: Path, records: list[dict[str, Any]], rejections: list[dict[str, Any]],
                      *, generation: int, execution_mode: str,
                      min_unique_verified_tasks: int = 5) -> None:
    gen_root = root / f'gen-{generation}'
    gen_root.mkdir(parents=True, exist_ok=True)
    os.chmod(gen_root, 0o700)
    (gen_root / 'private').mkdir(exist_ok=True)
    os.chmod(gen_root / 'private', 0o700)
    tasks = []
    oracles = []
    sft = []
    heldout = []
    heldout_verifier = []
    for record in records:
        visible = record['visible_task']
        tasks.append({'task_id': record['task_id'], 'generation': generation,
                      'train': visible['train'], 'query_inputs': visible['test_inputs'],
                      'input_family_fingerprint': record['input_family_fingerprint']})
        oracle = {'task_id': record['task_id'], 'generation': generation,
                  'oracle_code': record['messages'][2]['content'],
                  'oracle_sha256': record['oracle_sha256'],
                  'oracle_ast_sha256': record['oracle_ast_sha256'],
                  'oracle_shape_sha256': record['oracle_shape_sha256'],
                  'source_type': record['data_source'], 'qwen_provenance': record['qwen_provenance'],
                  'verifier_version': record['verifier_version']}
        oracles.append(oracle)
        sft.append({key: value for key, value in record.items() if key != '_verifier_only'})
        # The verifier labels are not embedded in SFT messages or public task rows.
        heldout.append({'task_id': record['task_id'],
                        'input_fingerprint': record['heldout_input_fingerprint'],
                        'output_fingerprint': record['heldout_output_fingerprint']})
        private_labels = record.get('_verifier_only')
        if isinstance(private_labels, dict):
            heldout_verifier.append({'task_id': record['task_id'],
                                     'inputs': private_labels['inputs'],
                                     'outputs': private_labels['outputs']})
    _write_jsonl(gen_root / 'synthetic_tasks.jsonl', tasks)
    _write_jsonl(gen_root / 'verified_oracles.jsonl', oracles)
    _write_jsonl(gen_root / 'sft_generator_distillation.jsonl', sft)
    _write_jsonl(gen_root / 'sft_self_solved.jsonl', [])
    _write_jsonl(gen_root / 'private/heldout_fingerprints.jsonl', heldout)
    _write_jsonl(gen_root / 'private/heldout_verifier.jsonl', heldout_verifier)
    _write_jsonl(gen_root / 'rejections.jsonl', rejections)
    def public_rejection_reason(value: Any) -> str:
        """Keep dynamic exception text in the private rejection log only."""
        if not isinstance(value, str):
            return 'unknown'
        fixed = {
            'model_request_failed', 'missing_or_mismatched_qwen_provenance',
            'static_safety_rejected', 'duplicate_task_id',
            'duplicate_oracle_fingerprint', 'similar_oracle_shape',
        }
        if value in fixed:
            return value
        if value.startswith('sandbox execution failed:'):
            return 'sandbox_execution_failed'
        if value.startswith('sandbox deadline exceeded'):
            return 'sandbox_deadline_exceeded'
        if value.startswith('nondeterministic oracle'):
            return 'nondeterministic_oracle'
        return 'other_rejection'

    rejection_categories = [public_rejection_reason(item.get('reason')) for item in rejections]
    _json(gen_root / 'collection_manifest.json', {
        'protocol': 'e6_collection_manifest_v1', 'generation': generation,
        'execution_mode': execution_mode,
        'synthetic_task_count': len(tasks), 'verified_oracle_count': len(oracles),
        'sft_generator_distillation_count': len(sft), 'sft_self_solved_count': 0,
        'unique_task_count': len({row['task_id'] for row in tasks}),
        'unique_oracle_count': len({row['oracle_ast_sha256'] for row in oracles}),
        'unique_oracle_shape_count': len({row['oracle_shape_sha256'] for row in oracles}),
        'source_counts': {SOURCE_TYPE: len(sft), 'self_solved_synthetic': 0,
                          'self_solved_real_arc': 0, 'external_or_human': 0},
        'rejection_reason_counts': {reason: rejection_categories.count(reason)
                                    for reason in sorted(set(rejection_categories))},
        'min_unique_verified_tasks': min_unique_verified_tasks,
        'training_gate_met': (len({row['task_id'] for row in tasks}) >= min_unique_verified_tasks and
                              len({row['oracle_shape_sha256'] for row in oracles}) >= min_unique_verified_tasks),
        'hardcode_detection_limit': 'Full hardcode detection is impossible; only static checks and sampled execution apply.',
    })
    _seal(gen_root)


def _fake_run(config: ExperimentConfig, root: Path, *, resume: bool) -> Path:
    from src.arc.e6_fake import FakeE6Trainer
    state_path = root / 'run_state.json'
    if state_path.is_file() and resume:
        state = json.loads(state_path.read_text())
        if state.get('status') == 'complete':
            _verify_seal(root)
            return root
    trainer = FakeE6Trainer()
    generations = list(range(config.e6.max_generations + 1))
    baseline_data: list[dict[str, Any]] | None = None
    gen_results = []
    for generation in generations:
        records = _fake_records(config, generation)
        rejections = []
        _write_collection(root, records, rejections, generation=generation,
                          execution_mode='fake_fixture',
                          min_unique_verified_tasks=config.e6.min_unique_verified_tasks)
        if generation == 0:
            baseline_data = records
        eligible = (len({row['task_id'] for row in records}) >= config.e6.min_unique_verified_tasks and
                    len({row['oracle_shape_sha256'] for row in records}) >= config.e6.min_unique_verified_tasks)
        training = {'condition': 'C1-GD', 'status': 'skipped_insufficient_data',
                    'trained': False, 'weights_updated': False, 'candidate_loaded': False,
                    'promoted': False, 'rolled_back': False, 'active_adapter_hash': 'none',
                    'reason': 'min_unique_verified_tasks_not_met'}
        if eligible and generation < config.e6.max_generations:
            training = trainer.train(records, steps=config.e6.max_training_steps)
            training.update(condition='C1-GD', promoted=False, rolled_back=not training['weights_updated'],
                            candidate_loaded=False,
                            active_adapter_hash=training['active_adapter_hash'],
                            min_unique_verified_tasks=config.e6.min_unique_verified_tasks,
                            source_type=SOURCE_TYPE)
        gen_results.append({'generation': generation, 'unique_verified_tasks': len(records),
                            'training': training, 'new_generation_data_connection':
                            'ready_for_next_generation' if generation > 0 else 'seed_dataset'})
    _write_collection(root, baseline_data or [], [], generation=0, execution_mode='fake_fixture',
                      min_unique_verified_tasks=config.e6.min_unique_verified_tasks)
    _json(root / 'comparison.json', {
        'protocol': 'e6_condition_comparison_v1',
        'conditions': {'B': {'adapter': 'none', 'source': 'none'},
                       'C1-SS': {'status': 'skipped_empty_self_solved_dataset',
                                 'adapter': 'none', 'source': 'self_solved_synthetic'},
                       'C1-GD': {'adapter': 'fake_only', 'source': SOURCE_TYPE},
                       'D': {'source': 'fixed_gen0_data_only',
                             'sample_count': len(baseline_data or []),
                             'training_steps': config.p3b.training.max_steps}},
        'same_fixed_evaluation_tasks': True,
        'recursive_effect_evaluation_starts_at_generation': 2,
        'fake_generations': generations,
    })
    _json(state_path, {'status': 'complete', 'execution_mode': 'fake_fixture',
                       'generations_completed': generations, 'generations': gen_results,
                       'trained': any(item['training'].get('trained') for item in gen_results),
                       'live_technical_demo': False,
                       'reason': 'fake results are interface evidence only'})
    _seal(root)
    return root


def _request_provenance(root: Path, prompt: str, candidate: int) -> dict[str, Any] | None:
    request_path = root / 'logs/e6/llm_requests.jsonl'
    response_path = root / 'logs/e6/llm_responses.jsonl'
    if not request_path.is_file() or not response_path.is_file():
        return None
    prompt_sha = hashlib.sha256(json.dumps(
        [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
         {'role': 'user', 'content': prompt}],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    requests = [json.loads(line) for line in request_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    responses = {row.get('request_id'): row for row in
                 (json.loads(line) for line in response_path.read_text(encoding='utf-8').splitlines() if line.strip())}
    for request in reversed(requests):
        if (request.get('kind') != 'e5_generator' or request.get('candidate') != candidate or
                request.get('prompt_sha256') != prompt_sha or request.get('error')):
            continue
        response = responses.get(request.get('request_id'))
        if not response or not isinstance(response.get('raw_http_body_text'), str):
            continue
        raw_sha = sha256_bytes(response['raw_http_body_text'].encode())
        if raw_sha != request.get('response_sha256'):
            continue
        text = response.get('generated_text')
        if not isinstance(text, str):
            continue
        try:
            extracted = final_answer(text)
        except Exception:
            continue
        return {'source_type': SOURCE_TYPE, 'request_id': request['request_id'],
                'raw_response_sha256': raw_sha, 'model_revision': request.get('revision'),
                'model_id': request.get('model'), 'generation': 0,
                'execution_mode': 'live_model',
                'request_log_path': str(request_path.resolve()),
                'response_log_path': str(response_path.resolve()),
                'thinking_mode': request.get('thinking_mode_effective'),
                'seed_requested': request.get('seed_requested'),
                'response_matches_code': extracted.strip()}
    return None


def _live_generated_records(config: ExperimentConfig, root: Path, backend: E5CachedBackend,
                            sandbox: MeteredSandbox, imported: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = list(imported)
    rejected: list[dict[str, Any]] = []
    existing = collection_rows(records)
    seen_task_ids = {item['task_id'] for item in existing['accepted']}
    seen_oracle_asts = {item['oracle_ast_sha256'] for item in existing['accepted']}
    seen_shapes = {item['oracle_shape_sha256'] for item in existing['accepted']}
    log_root = root / 'logs/e6'
    for attempt in range(config.e6.max_collection_attempts):
        if len(seen_shapes) >= config.e6.min_unique_verified_tasks:
            break
        prompt = _generator_prompt()
        backend.scope = f'e6-generator-attempt-{attempt}'
        try:
            code = backend.generate(prompt, kind='e5_generator', candidate=attempt).strip()
        except Exception as exc:
            rejected.append({'attempt': attempt, 'reason': 'model_request_failed',
                             'error_type': type(exc).__name__})
            continue
        provenance = _request_provenance(root, prompt, attempt)
        if not provenance or provenance.pop('response_matches_code', None) != code:
            rejected.append({'attempt': attempt, 'reason': 'missing_or_mismatched_qwen_provenance',
                             'code_sha256': sha256_bytes(code.encode())})
            continue
        if not _safe_transform_code(code):
            rejected.append({'attempt': attempt, 'reason': 'static_safety_rejected',
                             'code_sha256': sha256_bytes(code.encode())})
            continue
        inputs = _quality_grids(
            66000 + attempt, config.e6.train_examples + config.e6.query_inputs + config.e6.heldout_inputs)
        try:
            outputs = verify_oracle(code, inputs, sandbox=sandbox)
        except Exception as exc:
            rejected.append({'attempt': attempt, 'reason': str(exc)[:128],
                             'code_sha256': sha256_bytes(code.encode())})
            continue
        train_count = config.e6.train_examples
        query_end = train_count + config.e6.query_inputs
        train = tuple((inputs[index], outputs[index]) for index in range(train_count))
        query_inputs = tuple(inputs[train_count:query_end])
        heldout_inputs = tuple(inputs[query_end:])
        heldout_outputs = tuple(outputs[query_end:])
        task_body = {'train': [{'input': source, 'output': target} for source, target in train],
                     'query': list(query_inputs)}
        task_id = 'e6syn-' + sha256_bytes(_canonical(task_body).encode())[:24]
        task = Task(task_id, train, query_inputs)
        record = make_distillation_record(task, code, provenance=provenance,
            heldout_inputs=heldout_inputs, heldout_outputs=heldout_outputs, generation=0)
        if task_id in seen_task_ids:
            rejected.append({'attempt': attempt, 'reason': 'duplicate_task_id', 'task_id': task_id})
            continue
        if record['oracle_ast_sha256'] in seen_oracle_asts:
            rejected.append({'attempt': attempt, 'reason': 'duplicate_oracle_fingerprint',
                             'task_id': task_id,
                             'oracle_ast_sha256': record['oracle_ast_sha256']})
            continue
        if record['oracle_shape_sha256'] in seen_shapes:
            rejected.append({'attempt': attempt, 'reason': 'similar_oracle_shape',
                             'task_id': task_id,
                             'oracle_shape_sha256': record['oracle_shape_sha256']})
            continue
        seen_task_ids.add(task_id)
        seen_oracle_asts.add(record['oracle_ast_sha256'])
        seen_shapes.add(record['oracle_shape_sha256'])
        records.append(record)
    return collection_rows(records)['accepted'], rejected


def _load_live_tasks(config: ExperimentConfig, project: Path) -> tuple[dict[str, Task], Any, dict[str, list[str]]]:
    challenge = _resolve(project, config.data.challenges)
    solution = _resolve(project, config.data.solutions)
    if not all('training' in path.name.lower() for path in (challenge, solution)):
        raise ValueError('E6 Live may read only ARC Training challenge and solution files')
    if any(term in path.name.lower() for path in (challenge, solution) for term in ('test', 'evaluation')):
        raise ValueError('E6 Live refuses ARC test or public evaluation files')
    tasks, verifier = load_arc(challenge, solution)
    splits = split_tasks(list(tasks), config.data.split_seed,
                         {'exploration': config.data.exploration,
                          'promotion_gate': config.data.promotion_gate,
                          'monitoring': config.data.monitoring})
    return tasks, verifier, splits


def _live_eval_synthetic(config: ExperimentConfig, root: Path, budget: E5Budget,
                         record: dict[str, Any], adapter: Path | None, *, label: str,
                         sandbox_image_id: str) -> dict[str, Any]:
    from src.arc.e5_selfplay import solve_task
    from src.arc.data import Verifier
    from src.pipelines.e5 import _backend
    visible = record['visible_task']
    task = Task(visible['id'], tuple((item['input'], item['output']) for item in visible['train']),
                tuple(visible['test_inputs']))
    # Use a deterministic query that is not part of train, SFT, or held-out.
    from src.arc.e5_selfplay import _unique_grids
    query = tuple(_unique_grids(88101, len(task.test_inputs)))
    eval_task = Task(task.id, task.train, query)
    oracle = record['messages'][2]['content']
    expected = []
    sandbox = MeteredSandbox(DockerSandbox(
        image=sandbox_image_id if sandbox_image_id != 'fake' else config.agent.sandbox_image,
        timeout=config.agent.timeout_seconds, memory_mb=config.agent.memory_mb,
        cpus=config.agent.cpus, pids=config.agent.pids,
        max_output_bytes=config.agent.max_output_bytes,
        max_code_bytes=config.agent.max_code_bytes, remaining=budget.remaining), budget)
    for grid in query:
        expected.append(sandbox.run(oracle, grid))
    with _backend(_compat_e5(config), root, budget,
                  root / 'checkpoints/e6' / f'{label}-cache.json',
                  adapter=adapter, label=label) as backend:
        backend.scope = label
        result = solve_task(backend, sandbox, eval_task, attempts=1)
    if result.error:
        return {'task_id': task.id, 'success': False, 'error_kind': result.error_kind,
                'code_sha256': sha256_bytes(result.code.encode()) if result.code else None}
    return {'task_id': task.id, 'success': result.outputs == expected,
            'code_sha256': sha256_bytes(result.code.encode()) if result.code else None,
            'oracle_code_sha256': record['oracle_sha256'], 'query_input_fingerprint': grid_fingerprint(query)}


def _live_training_and_eval(config: ExperimentConfig, root: Path, budget: E5Budget,
                            records: list[dict[str, Any]], *, sandbox_id: str,
                            mask_audit: dict[str, Any]) -> dict[str, Any]:
    e6 = config.e6
    eligible = (len({row['task_id'] for row in records}) >= e6.min_unique_verified_tasks and
                len({row['oracle_shape_sha256'] for row in records}) >= e6.min_unique_verified_tasks)
    active_hash = 'none'
    result: dict[str, Any] = {
        'condition': 'C1-GD', 'source_type': SOURCE_TYPE,
        'eligible_unique_tasks': len({row['task_id'] for row in records}),
        'minimum_unique_tasks': e6.min_unique_verified_tasks,
        'trained': False, 'weights_updated': False, 'candidate_loaded': False,
        'promoted': False, 'rolled_back': False, 'active_adapter_hash': active_hash,
        'reason': 'skipped_insufficient_data',
    }
    if not eligible:
        result['status'] = 'skipped_insufficient_data'
        result['previous_adapter_preserved'] = True
        return result
    if mask_audit.get('audit_passed') is not True:
        result.update(status='failed_sft_mask_audit', reason='training_prompt_mask_audit_failed',
                      previous_adapter_preserved=True)
        return result
    training = config.p3b.training
    steps = min(training.max_steps, e6.max_training_steps)
    if steps < 1:
        result.update(status='skipped_training_disabled', reason='zero_training_steps')
        return result
    result['training_steps_reserved'] = steps
    budget.reserve_training(steps)
    output = root / 'training/c1-gd'
    try:
        training_records = [{key: value for key, value in record.items()
                             if key != '_verifier_only'} for record in records]
        trained = train_adapter(training_records, model_path=training.model_path,
            python_executable=training.python_executable, output=output,
            parameters={'batch_size': training.batch_size, 'learning_rate': training.learning_rate,
                        'max_steps': steps, 'lora_rank': training.lora_rank,
                        'lora_alpha': training.lora_alpha,
                        'max_seq_length': training.max_seq_length,
                        'num_layers': training.num_layers},
            seed=config.experiment.seed, timeout_seconds=min(e6.max_training_seconds,
                training.timeout_seconds, int(budget.remaining())),
            source='e6_generator_distillation', expected_mlx_lm_version=training.mlx_lm_version,
            thinking_mode=config.model.thinking, previous_adapter=None)
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}',
                      rolled_back=True, previous_adapter_preserved=True)
        return result
    result.update({key: trained.get(key) for key in ('status', 'trained', 'weights_updated',
                    'adapter_sha256', 'adapter_path', 'base_weights_sha256_before',
                    'base_weights_sha256_after', 'adapter_probe', 'sft_audit_path',
                    'training_record_count', 'record_sha256') if key in trained})
    result['trained'] = trained.get('trained') is True
    result['weights_updated'] = trained.get('weights_updated') is True
    result['base_weights_unchanged'] = (
        trained.get('base_weights_sha256_before') == trained.get('base_weights_sha256_after'))
    if not (result['trained'] and result['weights_updated'] and result['base_weights_unchanged']):
        result.update(rolled_back=True, active_adapter_hash='none', reason=trained.get('error', 'weight_delta_unverified'))
        return result
    candidate_path = Path(trained['adapter_path']).resolve()
    candidate_hash = adapter_sha256(candidate_path)
    if candidate_hash != trained.get('adapter_sha256'):
        raise ValueError('E6 candidate adapter hash differs from training manifest')

    # Loading and a format canary happen through the guarded inference server,
    # which verifies the adapter hash before serving any request.
    compat = _compat_e5(config)
    canary_path = root / 'checkpoints/e6/canary-cache.json'
    try:
        with _backend(compat, root, budget, canary_path, adapter=candidate_path,
                      label='e6-c1-gd-canary') as canary:
            canary.scope = 'e6-c1-gd-canary'
            code = canary.generate(
                'Return a syntactically valid Python function named probe(value) with exactly one argument. '
                'Use source code only.', kind='code', candidate=0)
            try:
                import ast
                parsed_code = ast.parse(code)
                code_valid = any(isinstance(node, ast.FunctionDef) and node.name == 'probe'
                                 for node in parsed_code.body)
            except (SyntaxError, ValueError, RecursionError):
                code_valid = False
            if not code_valid:
                result.update(candidate_loaded=True, rolled_back=True, reason='code_canary_invalid',
                              active_adapter_hash='none')
                return result
            json_canary = canary.generate('Return exactly the JSON object {"ok": true}.',
                                          kind='direct', candidate=0)
            try:
                json_valid = json.loads(json_canary) == {'ok': True}
            except (TypeError, ValueError):
                json_valid = False
            if not json_valid:
                result.update(candidate_loaded=True, rolled_back=True, reason='json_canary_invalid',
                              active_adapter_hash='none')
                return result
    except Exception as exc:
        result.update(status='candidate_load_failed', candidate_loaded=False,
                      rolled_back=True, active_adapter_hash='none',
                      reason=f'{type(exc).__name__}: adapter_load_or_canary_failed',
                      previous_adapter_preserved=True)
        return result
    result['candidate_loaded'] = True
    result['candidate_adapter_hash'] = candidate_hash

    tasks, verifier, splits = _load_live_tasks(config, Path(__file__).resolve().parents[2])
    gate_ids = splits['promotion_gate'][:e6.promotion_gate_tasks]
    if len(gate_ids) < e6.promotion_gate_tasks:
        result.update(rolled_back=True, reason='promotion_gate_too_small', active_adapter_hash='none')
        return result
    try:
        b_metric = _promotion_eval(compat, root, budget, root / 'checkpoints/e6/gate-b-cache.json',
            tasks, verifier, gate_ids, None, label='e6-gate-b', sandbox_image_id=sandbox_id)
        c_metric = _promotion_eval(compat, root, budget, root / 'checkpoints/e6/gate-c1-gd-cache.json',
            tasks, verifier, gate_ids, candidate_path, label='e6-gate-c1-gd', sandbox_image_id=sandbox_id)
        synthetic_eval = _live_eval_synthetic(config, root, budget, records[0], None,
            label='e6-synthetic-b', sandbox_image_id=sandbox_id)
        synthetic_candidate = _live_eval_synthetic(config, root, budget, records[0], candidate_path,
            label='e6-synthetic-c1-gd', sandbox_image_id=sandbox_id)
    except Exception as exc:
        result.update(status='candidate_evaluation_failed', rolled_back=True,
                      active_adapter_hash='none', reason=f'{type(exc).__name__}: evaluation_failed',
                      previous_adapter_preserved=True)
        return result
    decision = _promotion_decision(b_metric, c_metric, expected_tasks=len(gate_ids),
                                   minimum_delta=0.0, rollback_on_regression=True)
    promoted = decision['decision'] == 'promote'
    result.update(status='complete', canary={'code_valid': True, 'json_returned': True},
        promotion_gate={'task_ids': gate_ids, 'B': b_metric, 'C1-GD': c_metric,
                        'decision': decision},
        synthetic_unseen_input={'B': synthetic_eval, 'C1-GD': synthetic_candidate},
        promoted=promoted, rolled_back=not promoted,
        active_adapter_hash=candidate_hash if promoted else 'none',
        reason='non_regression_gate_passed' if promoted else 'non_regression_gate_failed',
        monitoring_feedback_used=False, public_evaluation_accessed=False,
        small_sample_warning='Five synthetic tasks are a technical check, not evidence of RSI improvement.')
    return result


def _run_live(config: ExperimentConfig, root: Path, *, resume: bool, allow_training: bool) -> Path:
    if not config.e6.enabled or config.e6.profile == 'cpu_fake':
        raise ValueError('Live E6 requires an enabled live_smoke or local_bootstrap config')
    if config.e6.profile == 'local_bootstrap' and not allow_training:
        raise ValueError('E6 bootstrap requires explicit --allow-training')
    if config.e6.profile == 'live_smoke' and allow_training:
        raise ValueError('E6 live smoke is training-disabled')
    e5_root = _resolve(Path(__file__).resolve().parents[2], config.e6.e5_artifact)
    compat = _compat_e5(config)
    project = Path(__file__).resolve().parents[2]
    with _run_lock(root):
        _secure_private_dirs(root)
        sandbox_id = _docker_image_id(compat)
        identity = _identity(config, project, sandbox_id)
        identity_path = root / 'identity.json'
        if identity_path.is_file():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('E6 resume config, data, source artifact, model, or implementation hash mismatch')
            if not resume:
                raise FileExistsError('E6 run directory exists; use --resume')
        elif resume:
            raise ValueError('cannot resume E6 without identity.json')
        _claim_live_attempt(config, root, resume=resume, identity=identity)
        if not identity_path.is_file():
            _json(identity_path, identity)
        state_path = root / 'run_state.json'
        if state_path.is_file() and resume and json.loads(state_path.read_text()).get('status') == 'complete':
            _verify_seal(root)
            diagnose_e5_artifact(e5_root)
            return root
        (root / 'config.snapshot.yml').write_text(
            yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True), encoding='utf-8')
        diagnostic = _diagnostic_write(root, e5_root)
        budget = E5Budget(root / 'budget.json', compat, resume=resume)
        sandbox = MeteredSandbox(DockerSandbox(
            image=compat.agent.sandbox_image, timeout=compat.agent.timeout_seconds,
            memory_mb=compat.agent.memory_mb, cpus=compat.agent.cpus,
            pids=compat.agent.pids, max_output_bytes=compat.agent.max_output_bytes,
            max_code_bytes=compat.agent.max_code_bytes, remaining=budget.remaining), budget)
        cache = root / 'checkpoints/e6/request_cache.json'
        try:
            # The E5 oracle is evidence-backed and read-only; it is imported as
            # generator_distillation, never re-labelled as independent solve.
            seed_oracle = import_e5_oracle(e5_root, sandbox=sandbox)
            imported = [make_distillation_record(seed_oracle['task'], seed_oracle['oracle_code'],
                provenance=seed_oracle['provenance'],
                heldout_inputs=seed_oracle['heldout_inputs'],
                heldout_outputs=seed_oracle['heldout_outputs'], generation=0)]
            with _backend(compat, root, budget, cache, adapter=None, label='e6-gen0') as backend:
                backend.scope = 'e6-gen0'
                records, rejections = _live_generated_records(config, root, backend, sandbox, imported)
            unique = collection_rows(records)['accepted']
            rejection_rows = list(rejections)
            if len(unique) < config.e6.min_unique_verified_tasks:
                reason = 'skipped_insufficient_data'
            else:
                reason = 'data_gate_met'
            _write_collection(root, unique, rejection_rows, generation=0, execution_mode='live_model',
                              min_unique_verified_tasks=config.e6.min_unique_verified_tasks)
            mask_audit = _run_sft_mask_audit(config, root, unique,
                                              timeout_seconds=int(budget.remaining()))
            if config.e6.profile == 'local_bootstrap':
                training = _live_training_and_eval(config, root, budget, unique,
                    sandbox_id=sandbox_id, mask_audit=mask_audit)
            else:
                training = {'condition': 'C1-GD', 'status': 'skipped_training_disabled',
                            'trained': False, 'weights_updated': False, 'candidate_loaded': False,
                            'promoted': False, 'rolled_back': False, 'active_adapter_hash': 'none',
                            'reason': 'live_smoke_training_disabled'}
            _, _, live_splits = _load_live_tasks(config, project)
            split = {'seed': config.data.split_seed,
                     'promotion_gate_task_ids': live_splits['promotion_gate'][:config.e6.promotion_gate_tasks],
                     'monitoring_task_ids': live_splits['monitoring'],
                     'training_data_task_ids': [row['task_id'] for row in unique]}
            _json(root / 'split_task_manifest.json', split)
            evaluation = training.get('promotion_gate', {})
            comparison = {
                'protocol': 'e6_condition_comparison_v1',
                'conditions': {
                    'B': {'adapter': 'none', 'source': 'none',
                          'status': 'evaluated' if evaluation.get('B') else 'not_run'},
                    'C1-SS': {'adapter': 'none', 'source': 'self_solved_synthetic',
                              'status': 'skipped_empty_self_solved_dataset', 'sft_records': 0},
                    'C1-GD': {'adapter': training.get('active_adapter_hash', 'none'),
                              'source': SOURCE_TYPE, 'status': training.get('status'),
                              'sft_records': len(unique)},
                    'D': {'source': 'fixed_gen0_data_only',
                          'sample_count': len(unique),
                          'training_steps': training.get('training_steps_reserved', 0),
                          'gen1_recursive_training_performed': False},
                },
                'same_fixed_evaluation_tasks': list(evaluation.get('task_ids', [])),
                'recursive_effect_evaluation_starts_at_generation': 2,
            }
            _json(root / 'comparison.json', comparison)
            _json(root / 'evaluation_report.json', {
                'B': evaluation.get('B', 'not_run'),
                'C1-GD': evaluation.get('C1-GD', 'not_run'),
                'training': training, 'e5_diagnostic_categories': diagnostic['categories'],
                'sft_mask_audit': mask_audit,
                'monitoring_used_for_decision': False, 'public_evaluation_accessed': False})
            _json(root / 'decision.json', {
                'status': training.get('status'), 'trained': training.get('trained', False),
                'weights_updated': training.get('weights_updated', False),
                'candidate_loaded': training.get('candidate_loaded', False),
                'promoted': training.get('promoted', False),
                'rolled_back': training.get('rolled_back', False),
                'active_adapter_hash': training.get('active_adapter_hash', 'none'),
                'reason': training.get('reason', reason),
                'unique_verified_tasks': len({row['task_id'] for row in unique}),
                'min_unique_verified_tasks': config.e6.min_unique_verified_tasks,
                'live_technical_demo': bool(training.get('trained') and training.get('weights_updated') and
                                            training.get('candidate_loaded')),
                'accuracy_improvement_required': False,
                'recursive_self_improvement_proven': False})
            # Sensitive request/cache/dataset logs are owner-only; public
            # summaries contain hashes and counts only.
            for path in (root / 'logs/e6').glob('*'):
                if path.is_file():
                    os.chmod(path, 0o600)
            for path in (root / 'checkpoints/e6').glob('*'):
                if path.is_file():
                    os.chmod(path, 0o600)
            budget.finish()
            ledger = {**budget.state,
                      'reserved_tokens': budget.state['reserved_tokens'],
                      **_observed_token_usage(root / 'logs/e6/llm_requests.jsonl'),
                      'max_model_calls': config.e6.max_model_calls,
                      'max_reserved_tokens': config.e6.max_reserved_tokens,
                      'max_sandbox_runs': config.e6.max_sandbox_runs,
                      'max_training_steps': config.e6.max_training_steps,
                      'max_generations': config.e6.max_generations,
                      'max_wall_seconds': config.e6.max_wall_seconds}
            _json(root / 'budget_ledger.json', ledger)
            state = {'status': 'complete', 'execution_mode': 'live_model',
                     'profile': config.e6.profile, 'collection_status': reason,
                     'unique_verified_tasks': len({row['task_id'] for row in unique}),
                     'unique_verified_oracles': len({row['oracle_shape_sha256'] for row in unique}),
                     'training': training, 'live_technical_demo': bool(training.get('trained') and
                         training.get('weights_updated') and training.get('candidate_loaded')),
                     'generations_completed': [0], 'candidate_generation': 1
                         if training.get('trained') else None,
                     'budget': ledger}
            _json(state_path, state)
            _verify_root_seal_unchanged(e5_root, identity['e5_source_root_seal_sha256'])
            _seal(root)
            return root
        except BaseException:
            budget.finish()
            raise


def _verify_root_seal_unchanged(source_root: Path, expected_sha: str) -> None:
    actual = _sha(source_root / 'root_artifact_hashes.json')
    if actual != expected_sha:
        raise ValueError('E5 source root seal changed during E6')


def run_e6(config: ExperimentConfig, *, resume: bool = False,
           run_dir: str | Path | None = None, live: bool = False,
           allow_training: bool = False) -> Path:
    """Run E6 fake protocol checks or one bounded Live collection/bootstrap."""
    if not config.e6.enabled:
        raise ValueError('E6 is disabled in this configuration')
    if config.e6.profile == 'cpu_fake':
        if live or allow_training:
            raise ValueError('CPU Fake E6 cannot use live or training authorization flags')
    elif not live:
        raise ValueError('Live E6 requires explicit --live')
    project = Path(__file__).resolve().parents[2]
    root = (Path(run_dir).expanduser().resolve() if run_dir else
            _resolve(project, config.tracking.output_dir) / 'e6' /
            f'{config.experiment.name}-{uuid.uuid4().hex[:10]}')
    if root.exists() and not resume and any(root.iterdir()):
        raise FileExistsError(f'E6 run directory is not empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    if config.e6.profile == 'cpu_fake':
        identity = _identity(config, project, 'fake')
        identity_path = root / 'identity.json'
        if identity_path.is_file() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('E6 Fake resume identity mismatch')
        if resume and not identity_path.is_file():
            raise ValueError('cannot resume E6 Fake without identity.json')
        if not identity_path.is_file():
            _json(identity_path, identity)
        if not resume and len(list(root.iterdir())) > 1:
            raise FileExistsError('E6 Fake run directory exists; use --resume')
        (root / 'config.snapshot.yml').write_text(
            yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True), encoding='utf-8')
        source_root = _resolve(project, config.e6.e5_artifact)
        if source_root.is_dir() and (source_root / 'identity.json').is_file():
            _diagnostic_write(root, source_root)
        return _fake_run(config, root, resume=resume)
    if not live:
        raise ValueError('Live E6 requires explicit --live')
    return _run_live(config, root, resume=resume, allow_training=allow_training)

