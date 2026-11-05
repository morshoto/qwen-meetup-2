"""Isolated MLX-LM QLoRA adapter training and empirical verification."""
from __future__ import annotations

import hashlib
import ast
import json
import math
import os
import platform
import re
try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource
    resource = None
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from src.arc.ast_fingerprint import stable_ast_sha256, stable_shape_sha256
from src.arc.mlx_token_driver import adapter_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def base_weights_sha256(model_path: str | Path) -> dict[str, str]:
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('base model must be an existing local directory')
    weights = sorted(root.rglob('*.safetensors'))
    if not weights or any(not path.resolve().is_file() for path in weights):
        raise ValueError('base model must contain local safetensors weights')
    return {str(path.relative_to(root)): file_sha256(path) for path in weights}


def _training_safe_record(record: dict[str, Any], source: str) -> None:
    if not isinstance(record, dict) or not isinstance(record.get('messages'), list) or len(record['messages']) != 3:
        raise ValueError('training record must contain system, user, and assistant messages')
    system, user, assistant = record['messages']
    from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
    if (not all(isinstance(item, dict) for item in (system, user, assistant)) or
            system.get('role') != 'system' or system.get('content') != FINAL_ANSWER_SYSTEM_PROMPT or
            user.get('role') != 'user' or assistant.get('role') != 'assistant' or
            not isinstance(user.get('content'), str) or not isinstance(assistant.get('content'), str)):
        raise ValueError('invalid training message roles or content')
    if not assistant['content'].strip() or len(assistant['content'].encode()) > 65536:
        raise ValueError('invalid or oversized assistant target')
    if source == 'exploration':
        if record.get('split') != 'exploration' or not record.get('task_id'):
            raise ValueError('real RSI records must be verifier-approved Exploration records')
        try:
            task_prompt = json.loads(user['content'])
        except json.JSONDecodeError as exc:
            raise ValueError('Exploration prompt is not the canonical ARC task JSON') from exc
        if not isinstance(task_prompt, dict) or not isinstance(task_prompt.get('test'), list):
            raise ValueError('Exploration prompt is not an ARC task')
        if any(not isinstance(item, dict) or set(item) != {'input'} for item in task_prompt['test']):
            raise ValueError('held-out test outputs must never enter training messages')
        code = assistant['content']
        expected_id = hashlib.sha256((user['content'] + '\0' + code).encode()).hexdigest()
        if record.get('id') != expected_id:
            raise ValueError('Exploration record fingerprint mismatch')
    elif source == 'synthetic_pilot':
        if record.get('split') != 'synthetic_pilot' or not record.get('synthetic'):
            raise ValueError('synthetic pilot accepts only explicitly marked artificial records')
    elif source == 'e5_selfplay':
        data_source = record.get('data_source')
        if data_source == 'self_solved_real_arc':
            if record.get('split') != 'exploration' or record.get('synthetic') is not False or not record.get('task_id'):
                raise ValueError('E5 real records must come from Exploration')
            try:
                task_prompt = json.loads(user['content'])
            except json.JSONDecodeError as exc:
                raise ValueError('E5 real prompt is not canonical ARC task JSON') from exc
            if (not isinstance(task_prompt, dict) or not isinstance(task_prompt.get('train'), list) or
                    not isinstance(task_prompt.get('test'), list) or
                    any(not isinstance(item, dict) or set(item) != {'input'} for item in task_prompt['test'])):
                raise ValueError('E5 real prompt contains invalid examples or held-out labels')
            expected_id = hashlib.sha256(('self_solved_real_arc\0' + user['content'] + '\0' + assistant['content']).encode()).hexdigest()
            if record.get('id') != expected_id:
                raise ValueError('E5 real record fingerprint mismatch')
        elif data_source == 'self_solved_synthetic':
            if (record.get('split') != 'synthetic' or record.get('synthetic') is not True or
                    not record.get('task_id') or not re.fullmatch(r'[0-9a-f]{64}', str(record.get('generator_sha256', '')))):
                raise ValueError('E5 synthetic records require an explicit self-solved provenance marker')
            try:
                task_prompt = json.loads(user['content'])
            except json.JSONDecodeError as exc:
                raise ValueError('E5 synthetic prompt is not canonical task JSON') from exc
            if (not isinstance(task_prompt, dict) or not isinstance(task_prompt.get('train'), list) or
                    not isinstance(task_prompt.get('test'), list) or
                    any(not isinstance(item, dict) or set(item) != {'input'} for item in task_prompt['test'])):
                raise ValueError('E5 synthetic prompt contains invalid examples or hidden labels')
            expected_id = hashlib.sha256(('self_solved_synthetic\0' + user['content'] + '\0' + assistant['content']).encode()).hexdigest()
            if record.get('id') != expected_id:
                raise ValueError('E5 synthetic record fingerprint mismatch')
        else:
            raise ValueError('E5 training rejects generator_distillation and unknown data sources')
    elif source == 'e6_generator_distillation':
        from src.arc.data import Task, validate_grid
        from src.arc.e5_selfplay import solve_prompt
        if '_verifier_only' in record:
            raise ValueError('E6 held-out verifier labels cannot enter SFT records')
        if (record.get('data_source') != 'generator_distillation' or
                record.get('source_type') != 'generator_distillation' or
                record.get('split') != 'generator_distillation' or
                record.get('synthetic') is not True or
                not isinstance(record.get('task_id'), str) or not record['task_id']):
            raise ValueError('E6 records require explicit generator_distillation provenance')
        provenance = record.get('qwen_provenance')
        if not isinstance(provenance, dict) or provenance.get('source_type') != 'generator_distillation':
            raise ValueError('E6 records require Qwen-authored provenance')
        for key in ('request_id', 'model_revision', 'raw_response_sha256'):
            if not isinstance(provenance.get(key), str) or not provenance[key]:
                raise ValueError(f'E6 provenance is missing {key}')
        if not re.fullmatch(r'[0-9a-f]{64}', provenance['raw_response_sha256']):
            raise ValueError('E6 raw response hash is invalid')
        execution_mode = provenance.get('execution_mode')
        if execution_mode == 'live_model':
            _verify_qwen_generator_evidence(provenance, assistant['content'])
        elif execution_mode != 'fake_fixture':
            raise ValueError('E6 provenance execution mode is invalid')
        visible = record.get('visible_task')
        if not isinstance(visible, dict) or visible.get('id') != record['task_id']:
            raise ValueError('E6 visible task payload is missing or mismatched')
        try:
            train = tuple((validate_grid(item['input']), validate_grid(item['output']))
                          for item in visible['train'])
            test_inputs = tuple(validate_grid(item) for item in visible['test_inputs'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('E6 visible task payload is invalid') from exc
        task = Task(record['task_id'], train, test_inputs)
        if user['content'] != solve_prompt(task):
            raise ValueError('E6 SFT user message differs from the inference prompt')
        oracle_sha = hashlib.sha256(assistant['content'].encode()).hexdigest()
        oracle_ast_sha = stable_ast_sha256(assistant['content'])
        if (record.get('oracle_sha256') != oracle_sha or
                record.get('oracle_ast_sha256') != oracle_ast_sha):
            raise ValueError('E6 oracle fingerprint mismatch')
        if record.get('oracle_shape_sha256') != stable_shape_sha256(assistant['content']):
            raise ValueError('E6 normalized oracle shape fingerprint is missing')
        expected_id = hashlib.sha256((record['task_id'] + '\0' + user['content'] + '\0' + oracle_sha).encode()).hexdigest()
        if record.get('id') != expected_id:
            raise ValueError('E6 SFT record fingerprint mismatch')
        if not isinstance(record.get('input_family_fingerprint'), str) or not re.fullmatch(
                r'[0-9a-f]{64}', record['input_family_fingerprint']):
            raise ValueError('E6 input family fingerprint is missing')
        if not isinstance(record.get('heldout_input_fingerprint'), str) or not re.fullmatch(
                r'[0-9a-f]{64}', record['heldout_input_fingerprint']):
            raise ValueError('E6 held-out fingerprint is missing')
        expected_family = {
            'train': [{'input': source_grid, 'output': target_grid}
                      for source_grid, target_grid in task.train],
            'query': list(task.test_inputs),
        }
        family_sha = hashlib.sha256(json.dumps(expected_family, sort_keys=True,
                                               separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        if record.get('input_family_fingerprint') != family_sha:
            raise ValueError('E6 input family fingerprint mismatch')
        for key in ('train_input_fingerprint', 'query_input_fingerprint',
                    'heldout_output_fingerprint'):
            if not isinstance(record.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', record[key]):
                raise ValueError(f'E6 {key} is missing or invalid')
        if record.get('verifier_version') != 'e6-oracle-verifier-v1':
            raise ValueError('E6 verifier version is missing')
        if any(key in user['content'] for key in ('heldout_outputs', 'validation_outputs')):
            raise ValueError('E6 held-out labels cannot enter the solver prompt')
    else:
        raise ValueError('unknown training source')


def _verify_qwen_generator_evidence(provenance: dict[str, Any], code: str) -> None:
    request_path = Path(str(provenance.get('request_log_path', ''))).expanduser().resolve()
    response_path = Path(str(provenance.get('response_log_path', ''))).expanduser().resolve()
    if not request_path.is_file() or not response_path.is_file():
        raise ValueError('E6 Qwen request or raw response evidence is missing')
    request = None
    for line in request_path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('request_id') == provenance.get('request_id'):
            request = row
            break
    if (not isinstance(request, dict) or request.get('kind') != 'e5_generator' or
            request.get('error') or request.get('response_sha256') != provenance.get('raw_response_sha256') or
            request.get('revision') != provenance.get('model_revision') or
            request.get('model') != provenance.get('model_id') or
            request.get('thinking_mode_effective') != 'off' or provenance.get('thinking_mode') != 'off'):
        raise ValueError('E6 Qwen request metadata does not match provenance')
    from src.arc.http_backend import final_answer
    response = None
    with response_path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('request_id') == provenance.get('request_id'):
                response = row
                break
    if not isinstance(response, dict) or not isinstance(response.get('raw_http_body_text'), str):
        raise ValueError('E6 raw Qwen response evidence is missing')
    raw_hash = hashlib.sha256(response['raw_http_body_text'].encode()).hexdigest()
    if raw_hash != provenance.get('raw_response_sha256'):
        raise ValueError('E6 raw Qwen response hash mismatch')
    text = response.get('generated_text')
    if not isinstance(text, str) or final_answer(text).strip() != code:
        raise ValueError('E6 oracle code differs from its Qwen response')


def train_adapter(records: list[dict[str, Any]], *, model_path: str | Path,
                  python_executable: str | Path, output: str | Path,
                  parameters: dict[str, Any], seed: int, timeout_seconds: int,
                  source: str, expected_mlx_lm_version: str,
                  thinking_mode: str = 'server_default',
                  previous_adapter: str | Path | None = None) -> dict[str, Any]:
    """Train one adapter into a new immutable artifact directory.

    The official MLX-LM QLoRA command runs in a separate native-ARM Python
    environment. This process records inputs and hashes without importing MLX.
    """
    target = Path(output).expanduser().resolve()
    # Keep the venv's python symlink: resolving it can escape the environment
    # and select the unprovisioned base interpreter instead.
    python = Path(os.path.abspath(Path(python_executable).expanduser()))
    model = Path(model_path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f'training output already exists: {target}')
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('configured MLX Python executable is unavailable')
    actual_mlx_lm = _version(python, 'mlx-lm')
    if actual_mlx_lm != expected_mlx_lm_version:
        raise ValueError(f'expected mlx-lm {expected_mlx_lm_version}, found {actual_mlx_lm}')
    if not records or len({record.get('id') for record in records}) != len(records):
        raise ValueError('training records must be nonempty and unique')
    if source == 'exploration' and len({record.get('task_id') for record in records}) != len(records):
        raise ValueError('real training records must be unique by task ID')
    if source == 'e6_generator_distillation' and (
            len({record.get('task_id') for record in records}) != len(records) or
            len({record.get('oracle_ast_sha256') for record in records}) != len(records)):
        raise ValueError('E6 training records must be unique by task and oracle fingerprint')
    if source == 'e6_generator_distillation' and any(
            record.get('qwen_provenance', {}).get('execution_mode') != 'live_model'
            for record in records):
        raise ValueError('E6 MLX training refuses Fake or non-live oracle data')
    if thinking_mode not in {'server_default', 'off', 'on'}:
        raise ValueError('unsupported training thinking mode')
    for record in records:
        _training_safe_record(record, source)
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError('training timeout must be a positive integer')
    required = {'batch_size', 'learning_rate', 'max_steps', 'lora_rank', 'lora_alpha', 'max_seq_length', 'num_layers'}
    if set(parameters) != required:
        raise ValueError('unexpected or missing MLX-LoRA parameters')
    if any(type(parameters[name]) is not int or parameters[name] <= 0 for name in required - {'learning_rate'}):
        raise ValueError('invalid integer MLX-LoRA parameter')
    if type(parameters['learning_rate']) not in (int, float) or not math.isfinite(parameters['learning_rate']) or parameters['learning_rate'] <= 0:
        raise ValueError('invalid MLX-LoRA learning rate')
    base_before = base_weights_sha256(model)
    previous = Path(previous_adapter).expanduser().resolve() if previous_adapter else None
    if previous is not None:
        previous_hash = adapter_sha256(previous)
    else:
        previous_hash = 'none'

    target.parent.mkdir(parents=True, exist_ok=True)
    work = target.parent / f'.{target.name}.inprogress-{uuid.uuid4().hex}'
    work.mkdir()
    data_dir = work / 'dataset'
    adapter_dir = work / 'adapter'
    data_dir.mkdir()
    adapter_dir.mkdir()
    if source == 'e6_generator_distillation':
        os.chmod(work, 0o700)
        os.chmod(data_dir, 0o700)
        os.chmod(adapter_dir, 0o700)
    training_rows = [{'id': record['id'], 'task_id': record['task_id'], 'split': record['split'],
                      'synthetic': record.get('synthetic', False),
                      **({'data_source': record['data_source'], 'generator_sha256': record.get('generator_sha256')}
                         if source == 'e5_selfplay' else {}),
                      **({key: record[key] for key in ('data_source', 'source_type', 'qwen_provenance',
                                                       'oracle_sha256', 'oracle_ast_sha256',
                                                       'oracle_shape_sha256',
                                                       'input_family_fingerprint', 'train_input_fingerprint',
                                                       'query_input_fingerprint', 'heldout_input_fingerprint',
                                                       'heldout_output_fingerprint', 'verifier_version',
                                                       'base_revision', 'adapter_revision', 'visible_task')}
                         if source == 'e6_generator_distillation' else {}),
                      'messages': record['messages']}
                     for record in records]
    (data_dir / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in training_rows), encoding='utf-8')
    (work / 'verified_records.jsonl').write_text(''.join(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n' for record in records), encoding='utf-8')
    if source == 'e6_generator_distillation':
        for path in (data_dir / 'train.jsonl', work / 'verified_records.jsonl'):
            os.chmod(path, 0o600)
    mlx_config = {
        'model': str(model), 'train': True, 'fine_tune_type': 'lora', 'optimizer': 'adam',
        'data': str(data_dir), 'seed': seed, 'num_layers': parameters['num_layers'],
        'batch_size': parameters['batch_size'], 'iters': parameters['max_steps'],
        'val_batches': 0, 'learning_rate': parameters['learning_rate'],
        'steps_per_report': 1, 'steps_per_eval': parameters['max_steps'] + 1,
        'resume_adapter_file': str(previous / 'adapters.safetensors') if previous else None,
        'adapter_path': str(adapter_dir), 'save_every': parameters['max_steps'],
        'max_seq_length': parameters['max_seq_length'], 'grad_checkpoint': False,
        'lora_parameters': {'rank': parameters['lora_rank'], 'scale': parameters['lora_alpha'] / parameters['lora_rank'], 'dropout': 0.0},
        'mask_prompt': True, 'report_to': None,
    }
    config_path = work / 'mlx_lora.yml'
    config_path.write_text(yaml.safe_dump(mlx_config, sort_keys=False), encoding='utf-8')
    audit_path = work / 'sft_audit.json'
    audit_command = [str(python), '-m', 'src.cli.p3b_sft_audit', '--model', str(model),
                     '--dataset', str(data_dir), '--output', str(audit_path),
                     '--thinking-mode', thinking_mode, '--source', source]
    try:
        audited = subprocess.run(audit_command, capture_output=True, text=True,
                                 timeout=min(timeout_seconds, 120),
                                 cwd=Path(__file__).resolve().parents[2],
                                 env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}, check=False)
    except subprocess.TimeoutExpired as exc:
        (work / 'sft_audit.stdout.log').write_text(exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or '', encoding='utf-8')
        (work / 'sft_audit.stderr.log').write_text(exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or '', encoding='utf-8')
        audited = None
    if audited is not None:
        (work / 'sft_audit.stdout.log').write_text(audited.stdout, encoding='utf-8')
        (work / 'sft_audit.stderr.log').write_text(audited.stderr, encoding='utf-8')
    if audited is None or audited.returncode != 0 or not audit_path.is_file():
        result = {'status': 'failed', 'trained': False, 'weights_updated': False,
                  'error': 'sft_chat_template_or_mask_audit_failed', 'source': source,
                  'sample_count': len(records), 'dataset_format': 'chat', 'mask_prompt': True,
                  'thinking_mode': thinking_mode, 'previous_adapter_sha256': previous_hash,
                  'audit_returncode': audited.returncode if audited is not None else None}
        (work / 'training.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        os.replace(work, target)
        return result
    started = time.monotonic()
    env = os.environ.copy()
    env.update({'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    command = [str(python), '-m', 'src.cli.p3b_train', '--config', str(config_path),
               '--thinking-mode', thinking_mode]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds,
                                   cwd=Path(__file__).resolve().parents[2], env=env, check=False)
    except subprocess.TimeoutExpired as exc:
        (work / 'training.stdout.log').write_text(exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or '', encoding='utf-8')
        (work / 'training.stderr.log').write_text(exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or '', encoding='utf-8')
        result = {'status': 'failed', 'trained': False, 'weights_updated': False, 'error': 'training_timeout',
                  'elapsed_seconds': time.monotonic() - started, 'source': source, 'sample_count': len(records),
                  'base_weights_sha256_before': base_before, 'base_weights_sha256_after': base_weights_sha256(model)}
        (work / 'training.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        os.replace(work, target)
        return result
    elapsed = time.monotonic() - started
    peak_rss_bytes = _peak_child_rss_bytes()
    (work / 'training.stdout.log').write_text(completed.stdout, encoding='utf-8')
    (work / 'training.stderr.log').write_text(completed.stderr, encoding='utf-8')
    base_after = base_weights_sha256(model)
    losses = [float(value) for value in re.findall(r'Train loss\s+([0-9]+(?:\.[0-9]+)?)', completed.stdout)]
    adapter_dir_exists = (adapter_dir / 'adapter_config.json').is_file() and (adapter_dir / 'adapters.safetensors').is_file()
    if completed.returncode != 0 or not adapter_dir_exists or not losses or base_before != base_after:
        result = {'status': 'failed', 'trained': False, 'weights_updated': False,
                  'error': 'mlx_training_failed_or_unverified', 'returncode': completed.returncode,
                  'elapsed_seconds': elapsed, 'source': source, 'sample_count': len(records),
                  'losses': losses, 'peak_rss_bytes': peak_rss_bytes, 'base_weights_sha256_before': base_before,
                  'base_weights_sha256_after': base_after, 'previous_adapter_sha256': previous_hash}
        (work / 'training.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        os.replace(work, target)
        return result

    current_adapter_hash = adapter_sha256(adapter_dir)
    probe_path = work / 'adapter_probe.json'
    probe_command = [str(python), '-m', 'src.cli.p3b_probe', '--model', str(model), '--adapter', str(adapter_dir),
                     '--messages', str(data_dir / 'train.jsonl'), '--output', str(probe_path)]
    try:
        probe = subprocess.run(probe_command, capture_output=True, text=True, timeout=timeout_seconds,
                               cwd=Path(__file__).resolve().parents[2], env=env, check=False)
    except subprocess.TimeoutExpired:
        probe = None
    if probe is not None:
        (work / 'probe.stdout.log').write_text(probe.stdout, encoding='utf-8')
        (work / 'probe.stderr.log').write_text(probe.stderr, encoding='utf-8')
    measured_probe = json.loads(probe_path.read_text()) if probe is not None and probe.returncode == 0 and probe_path.is_file() else None
    updated = bool(measured_probe and measured_probe.get('lora_b_max_abs', 0) > 0 and
                   measured_probe.get('max_abs_logit_delta', 0) > 0 and
                   current_adapter_hash != previous_hash and base_before == base_after)
    result = {'status': 'complete' if updated else 'failed', 'trained': updated, 'weights_updated': updated,
              'source': source, 'sample_count': len(records), 'losses': losses,
              'elapsed_seconds': elapsed, 'model_path': str(model), 'base_weights_sha256_before': base_before,
              'base_weights_sha256_after': base_after, 'previous_adapter_sha256': previous_hash,
              'adapter_sha256': current_adapter_hash, 'adapter_path': 'adapter', 'adapter_probe': measured_probe,
              'parameters': parameters, 'seed': seed, 'python_executable': str(python),
              'dataset_format': 'chat', 'mask_prompt': True, 'thinking_mode': thinking_mode,
              'training_entrypoint': 'src.cli.p3b_train', 'prompt_loss_policy': 'assistant_completion_only',
              'sft_audit_path': 'sft_audit.json',
              'peak_rss_bytes': peak_rss_bytes,
              'mlx_lm_version': actual_mlx_lm, 'mlx_version': _version(python, 'mlx'),
              'record_sha256': file_sha256(work / 'verified_records.jsonl')}
    (work / 'training.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    os.replace(work, target)
    if updated:
        result['adapter_path'] = str(target / 'adapter')
    return result


def _version(python: Path, package: str) -> str | None:
    script = 'import importlib.metadata,sys; print(importlib.metadata.version(sys.argv[1]))'
    try:
        completed = subprocess.run([str(python), '-c', script, package], capture_output=True, text=True, timeout=10, check=True)
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _peak_child_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        value = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        return value if platform.system() == 'Darwin' else value * 1024
    except (OSError, ValueError):
        return None
