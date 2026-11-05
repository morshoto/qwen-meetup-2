"""Run fixed artificial code/JSON format checks with a local MLX model."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.arc.mlx_token_driver import adapter_sha256
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.response_diagnostics import diagnose_raw_response


CANARIES = (
    ('python_code', 'Return only Python source code defining transform(grid) that returns grid unchanged. No markdown or explanation.', 'code'),
    ('json_grid', 'Return only the JSON value [[1,2],[3,4]]. No markdown or explanation.', 'json'),
)


def run_canaries(model_path: str | Path, *, adapter_path: str | Path | None,
                 expected_adapter_sha256: str, thinking_mode: str, max_tokens: int) -> dict:
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    model_root = Path(model_path).expanduser().resolve()
    actual_adapter_sha = adapter_sha256(adapter_path) if adapter_path else 'none'
    if expected_adapter_sha256 != actual_adapter_sha:
        raise ValueError('canary adapter hash differs from the expected value')
    kwargs = {'tokenizer_config': {'trust_remote_code': False}}
    if adapter_path is not None:
        kwargs['adapter_path'] = str(Path(adapter_path).expanduser().resolve())
    started = time.monotonic()
    model, tokenizer = load(str(model_root), **kwargs)
    template = {'tokenize': True, 'add_generation_prompt': True, 'return_dict': False}
    if thinking_mode == 'off':
        if 'enable_thinking' not in (getattr(tokenizer, 'chat_template', '') or ''):
            raise ValueError('tokenizer template does not support thinking mode control')
        template['enable_thinking'] = False
    elif thinking_mode != 'server_default':
        raise ValueError('unsupported canary thinking mode')
    stop_ids = getattr(tokenizer, 'eos_token_ids', None)
    if stop_ids is None:
        stop_id = getattr(tokenizer, 'eos_token_id', None)
        stop_ids = set() if stop_id is None else {stop_id}
    else:
        stop_ids = set(stop_ids)
    results = []
    for name, prompt, output_format in CANARIES:
        messages = [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}]
        prompt_ids = tokenizer.apply_chat_template(messages, **template)
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()
        sample_started = time.monotonic()
        generated_ids = []
        finish_reason = 'generator_exhausted'
        stop_token_id = None
        generated_token_count = 0
        for token, _ in generate_step(mx.array(prompt_ids), model,
                                      max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
            token_id = int(token)
            generated_token_count += 1
            if token_id in stop_ids:
                finish_reason = 'stop'
                stop_token_id = token_id
                break
            generated_ids.append(token_id)
            if len(generated_ids) >= max_tokens:
                finish_reason = 'length'
                break
        raw_text = tokenizer.decode(generated_ids)
        elapsed = time.monotonic() - sample_started
        response = {'model': 'local-mlx', 'choices': [{'message': {'role': 'assistant', 'content': raw_text},
                    'finish_reason': finish_reason}], 'usage': {'prompt_tokens': len(prompt_ids),
                    'completion_tokens': generated_token_count, 'total_tokens': len(prompt_ids) + generated_token_count}}
        raw_http_body = json.dumps(response, ensure_ascii=False)
        diagnostic = diagnose_raw_response(raw_http_body, output_format=output_format,
                                           max_tokens=max_tokens)
        results.append({'canary': name, 'output_format': output_format,
                        'prompt_token_count': len(prompt_ids), 'generated_token_count': generated_token_count,
                        'content_token_count': len(generated_ids), 'finish_reason': finish_reason,
                        'stop_token_id': stop_token_id, 'elapsed_seconds': elapsed,
                        'raw_generation_text': raw_text, 'diagnostic': diagnostic,
                        'response_valid': diagnostic.get('category') == 'unknown' and not diagnostic.get('is_error')})
    peak_memory_bytes = int(mx.get_peak_memory()) if hasattr(mx, 'get_peak_memory') else None
    del model, tokenizer
    mx.clear_cache()
    return {'protocol': 'p3b_synthetic_format_canary_v1', 'model_path': str(model_root),
            'thinking_mode': thinking_mode, 'adapter_loaded': adapter_path is not None,
            'adapter_sha256': actual_adapter_sha, 'weights_updated': False, 'promoted': False,
            'max_tokens': max_tokens, 'elapsed_seconds': time.monotonic() - started,
            'peak_memory_bytes': peak_memory_bytes, 'response_valid_rate': sum(row['response_valid'] for row in results) / len(results),
            'thinking_format_error_count': sum(row['diagnostic']['category'] == 'thinking_unclosed' for row in results),
            'results': results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--expected-adapter-sha256', default='none')
    parser.add_argument('--thinking-mode', choices=('server_default', 'off'), required=True)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error('--max-tokens must be positive')
    result = run_canaries(args.model, adapter_path=args.adapter,
                          expected_adapter_sha256=args.expected_adapter_sha256,
                          thinking_mode=args.thinking_mode, max_tokens=args.max_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'response_valid_rate': result['response_valid_rate'],
                      'thinking_format_error_count': result['thinking_format_error_count'],
                      'adapter_sha256': result['adapter_sha256']}, indent=2))


if __name__ == '__main__':
    main()
