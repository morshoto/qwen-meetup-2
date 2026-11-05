"""Render an MLX-LM 0.30.7 chat dataset and its loss-mask token spans."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.arc.mlx_chat_template import install_training_template_policy
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_target(value: str) -> bool:
    try:
        tree = ast.parse(value)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'transform'
               for node in tree.body) == 1


def _validate_row(row: Any, index: int, source: str) -> tuple[str, str, str]:
    if not isinstance(row, dict) or not isinstance(row.get('messages'), list) or len(row['messages']) not in (2, 3):
        raise ValueError(f'row {index}: expected user/assistant or system/user/assistant chat messages')
    messages = row['messages']
    if len(messages) == 3:
        system, user, assistant = messages
    else:
        system, user, assistant = None, messages[0], messages[1]
    roles = [item.get('role') if isinstance(item, dict) else None for item in row['messages']]
    if roles not in (['system', 'user', 'assistant'], ['user', 'assistant']):
        raise ValueError(f'row {index}: invalid message roles')
    if not all(isinstance(item.get('content'), str) for item in row['messages']):
        raise ValueError(f'row {index}: message content must be text')
    if system is not None and not system['content'].strip() or not assistant['content'].strip():
        raise ValueError(f'row {index}: empty system or assistant completion')
    if source == 'exploration':
        if row.get('split') not in (None, 'exploration'):
            raise ValueError(f'row {index}: non-Exploration example is forbidden')
        task_id = row.get('task_id')
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f'row {index}: missing task ID')
        try:
            task = json.loads(user['content'])
        except json.JSONDecodeError as exc:
            raise ValueError(f'row {index}: user content is not canonical ARC JSON') from exc
        if not isinstance(task, dict) or not isinstance(task.get('test'), list) or not task['test']:
            raise ValueError(f'row {index}: invalid ARC task prompt')
        if any(not isinstance(item, dict) or set(item) != {'input'} for item in task['test']):
            raise ValueError(f'row {index}: test solution fields are forbidden')
        if not _code_target(assistant['content']):
            raise ValueError(f'row {index}: assistant completion must be a transform function')
        return task_id, assistant['content'], user['content']
    if source == 'synthetic_pilot':
        if not row.get('synthetic'):
            raise ValueError(f'row {index}: synthetic source marker is required')
        return str(row.get('task_id', f'synthetic-{index}')), assistant['content'], user['content']
    if source == 'e5_selfplay':
        data_source = row.get('data_source')
        task_id = row.get('task_id')
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f'row {index}: missing E5 task ID')
        if not _code_target(assistant['content']):
            raise ValueError(f'row {index}: assistant completion must be a transform function')
        if data_source == 'self_solved_real_arc':
            if row.get('split') != 'exploration' or row.get('synthetic') is not False:
                raise ValueError(f'row {index}: real E5 source must be Exploration')
        elif data_source == 'self_solved_synthetic':
            if (row.get('split') != 'synthetic' or row.get('synthetic') is not True or
                    not isinstance(row.get('generator_sha256'), str) or
                    not re.fullmatch(r'[0-9a-f]{64}', row['generator_sha256'])):
                raise ValueError(f'row {index}: synthetic E5 source marker is invalid')
        else:
            raise ValueError(f'row {index}: generator_distillation is forbidden in E5 SFT')
        try:
            task = json.loads(user['content'])
        except json.JSONDecodeError as exc:
            raise ValueError(f'row {index}: E5 user content is not canonical task JSON') from exc
        if (not isinstance(task, dict) or not isinstance(task.get('train'), list) or
                not isinstance(task.get('test'), list) or
                any(not isinstance(item, dict) or set(item) != {'input'} for item in task['test'])):
            raise ValueError(f'row {index}: E5 prompt contains invalid examples or hidden test labels')
        return task_id, assistant['content'], user['content']
    if source == 'e6_generator_distillation':
        from src.arc.mlx_lora import _training_safe_record
        _training_safe_record(row, source)
        return str(row['task_id']), assistant['content'], user['content']
    raise ValueError('unknown SFT data source')


def audit_dataset(model_path: str | Path, dataset_path: str | Path, output_path: str | Path,
                  *, thinking_mode: str, source: str, allow_misaligned: bool = False) -> dict[str, Any]:
    """Run inside the configured MLX-LM runtime; this loads only the tokenizer."""
    model_root, dataset_root, output = map(lambda value: Path(value).expanduser().resolve(),
                                           (model_path, dataset_path, output_path))
    if thinking_mode not in {'server_default', 'off', 'on'}:
        raise ValueError('unsupported thinking mode')
    train_file = dataset_root / 'train.jsonl'
    if not train_file.is_file():
        raise ValueError('dataset directory lacks train.jsonl')
    raw_rows = [json.loads(line) for line in train_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not raw_rows:
        raise ValueError('empty SFT training dataset')
    task_ids = []
    for index, row in enumerate(raw_rows):
        task_id, _, _ = _validate_row(row, index, source)
        task_ids.append(task_id)
    if source in {'exploration', 'e5_selfplay', 'e6_generator_distillation'} and len(set(task_ids)) != len(task_ids):
        raise ValueError('same task ID appears more than once in SFT dataset')
    if source == 'e6_generator_distillation':
        oracle_hashes = [row.get('oracle_ast_sha256') for row in raw_rows]
        if len(set(oracle_hashes)) != len(oracle_hashes):
            raise ValueError('same E6 oracle fingerprint appears more than once in SFT dataset')
        shape_hashes = [row.get('oracle_shape_sha256') for row in raw_rows]
        if len(set(shape_hashes)) != len(shape_hashes):
            raise ValueError('similar E6 oracle shapes appear more than once in SFT dataset')

    install_training_template_policy(thinking_mode)
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.utils import load_tokenizer
    tokenizer = load_tokenizer(model_root)
    template = getattr(tokenizer, 'chat_template', None)
    if not isinstance(template, str):
        raise ValueError('model tokenizer does not expose a string chat template')
    template_sha = hashlib.sha256(template.encode('utf-8')).hexdigest()
    config_path = model_root / 'tokenizer_config.json'
    if not config_path.is_file():
        raise ValueError('tokenizer_config.json is missing')
    config_sha = _sha(config_path)
    reports = []
    for index, row in enumerate(raw_rows):
        messages = row['messages']
        token_ids, offset = ChatDataset([row], tokenizer, mask_prompt=True).process(row)
        prefix_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
            add_generation_prompt=True, return_dict=False)
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        if hasattr(prefix_ids, 'tolist'):
            prefix_ids = prefix_ids.tolist()
        if type(offset) is not int or not isinstance(token_ids, list) or not isinstance(prefix_ids, list):
            raise ValueError(f'row {index}: MLX-LM returned an invalid token offset')
        inference_messages = messages[:-1]
        if messages[0].get('role') != 'system':
            inference_messages = [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT}, *inference_messages]
        inference_prefix_ids = tokenizer.apply_chat_template(inference_messages, tokenize=True,
            add_generation_prompt=True, return_dict=False)
        if hasattr(inference_prefix_ids, 'tolist'):
            inference_prefix_ids = inference_prefix_ids.tolist()
        prefix_match = token_ids[:offset] == prefix_ids and len(prefix_ids) == offset
        off_mode_prefix_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
            add_generation_prompt=True, return_dict=False, enable_thinking=False)
        if hasattr(off_mode_prefix_ids, 'tolist'):
            off_mode_prefix_ids = off_mode_prefix_ids.tolist()
        off_prefix_match = token_ids[:len(off_mode_prefix_ids)] == off_mode_prefix_ids and offset == len(off_mode_prefix_ids)
        inference_match = token_ids[:len(inference_prefix_ids)] == inference_prefix_ids
        if thinking_mode in {'off', 'on'}:
            # The installed MLX-LM trainer invokes the wrapper with the same
            # policy that MLXTokenDriver uses for inference.
            inference_prefix_ids = tokenizer.apply_chat_template(inference_messages, tokenize=True,
                add_generation_prompt=True, return_dict=False,
                enable_thinking=thinking_mode == 'on')
            if hasattr(inference_prefix_ids, 'tolist'):
                inference_prefix_ids = inference_prefix_ids.tolist()
            inference_match = token_ids[:len(inference_prefix_ids)] == inference_prefix_ids
        inference_match = inference_match and offset == len(inference_prefix_ids)
        completion_indices = list(range(max(1, offset), len(token_ids)))
        token_objects = tokenizer._tokenizer
        token_rows = []
        for token_index, token_id in enumerate(token_ids):
            try:
                token_text = token_objects.convert_ids_to_tokens(token_id)
            except (AttributeError, TypeError):
                token_text = tokenizer.decode([token_id])
            token_rows.append({'index': token_index, 'token_id': int(token_id), 'token': token_text,
                               'loss': token_index >= offset,
                               'region': 'assistant_completion' if token_index >= offset else 'masked_prompt'})
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
        prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
            add_generation_prompt=True, return_dict=False)
        reports.append({
            'task_id': task_ids[index], 'record_id': row.get('id'),
            'messages_roles': [message['role'] for message in messages],
            'system_prompt_matches_inference': messages[0].get('role') == 'system' and messages[0].get('content') == FINAL_ANSWER_SYSTEM_PROMPT,
            'assistant_target_sha256': hashlib.sha256(messages[-1]['content'].encode()).hexdigest(),
            'assistant_target_characters': len(messages[-1]['content']),
            'input_token_count': len(token_ids), 'prompt_mask_end_exclusive': offset,
            'masked_prompt_token_count': offset,
            'loss_target_token_range': [max(1, offset), len(token_ids)],
            'loss_target_token_count': len(completion_indices),
            'prefix_token_ids_match_mask_offset': prefix_match,
            'off_mode_prefix_token_count': len(off_mode_prefix_ids),
            'off_mode_prefix_matches_mask_offset': off_prefix_match,
            'template_prefix_matches_inference': inference_match,
            'prompt_mask_valid': prefix_match,
            'test_outputs_in_user_message': False,
            'rendered_chat_template': rendered,
            'rendered_prompt_prefix': prefix_text,
            'token_regions': token_rows,
        })
        if not completion_indices:
            raise ValueError(f'row {index}: assistant completion has no loss-target tokens')

    valid_file = dataset_root / 'valid.jsonl'
    valid_count = sum(1 for line in valid_file.read_text(encoding='utf-8').splitlines() if line.strip()) if valid_file.is_file() else 0
    report = {
        'protocol': 'mlx_lm_chat_sft_mask_audit_v1',
        'mlx_lm_version': __import__('importlib.metadata', fromlist=['version']).version('mlx-lm'),
        'mlx_version': __import__('importlib.metadata', fromlist=['version']).version('mlx'),
        'model_path': str(model_root), 'dataset_path': str(dataset_root),
        'dataset_sha256': _sha(train_file), 'dataset_format': 'chat', 'mask_prompt': True,
        'thinking_mode': thinking_mode, 'tokenizer_config_sha256': config_sha,
        'chat_template_sha256': template_sha, 'example_count': len(raw_rows),
        'unique_task_count': len(set(task_ids)), 'task_ids': task_ids,
        'valid_example_count': valid_count,
        'validation_loss': 'not_measured_no_valid.jsonl' if valid_count == 0 else 'not_measured_by_audit',
        'audit_passed': all(row['prompt_mask_valid'] and row['template_prefix_matches_inference'] and
                            row['system_prompt_matches_inference'] for row in reports),
        'examples': reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if not report['audit_passed'] and not allow_misaligned:
        raise ValueError('SFT prompt mask or inference template alignment failed; report was saved')
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thinking-mode', choices=('server_default', 'off', 'on'), required=True)
    parser.add_argument('--source', choices=('exploration', 'synthetic_pilot', 'e5_selfplay',
                                              'e6_generator_distillation'), required=True)
    parser.add_argument('--allow-misaligned', action='store_true',
                        help='save a failed historical audit instead of returning an error')
    args = parser.parse_args()
    audit_dataset(args.model, args.dataset, args.output, thinking_mode=args.thinking_mode,
                  source=args.source, allow_misaligned=args.allow_misaligned)


if __name__ == '__main__':
    main()
