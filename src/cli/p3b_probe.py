"""Load a base model and MLX adapter, then measure real inference effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _run(model_path: str, adapter_path: str | None, messages: list[dict[str, str]], max_tokens: int) -> tuple[list[int], list[int], str, float, float]:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    kwargs = {'tokenizer_config': {'trust_remote_code': False}}
    if adapter_path is not None:
        kwargs['adapter_path'] = adapter_path
    model, tokenizer = load(model_path, **kwargs)
    template = {'tokenize': True, 'add_generation_prompt': True, 'return_dict': False}
    if 'enable_thinking' in (getattr(tokenizer, 'chat_template', '') or ''):
        template['enable_thinking'] = False
    token_ids = tokenizer.apply_chat_template(messages, **template)
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list) or not token_ids or any(type(item) is not int for item in token_ids):
        raise ValueError('chat template returned invalid token IDs')
    logits = model(mx.array(token_ids)[None, :])
    if len(logits.shape) == 3:
        logits = logits[0, -1]
    elif len(logits.shape) == 2:
        logits = logits[-1]
    mx.eval(logits)
    last_logits = logits.astype(mx.float32)
    max_abs_logit = float(mx.max(mx.abs(last_logits)).item())
    lora_b_values = [value for name, value in tree_flatten(model.parameters()) if name.endswith('.lora_b')]
    lora_b_max_abs = max((float(mx.max(mx.abs(value)).item()) for value in lora_b_values), default=0.0)
    output_ids = []
    eos_ids = getattr(tokenizer, 'eos_token_ids', None)
    if eos_ids is None:
        eos = getattr(tokenizer, 'eos_token_id', None)
        eos_ids = [] if eos is None else [eos]
    for token, _ in generate_step(mx.array(token_ids), model, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
        token_id = int(token)
        if token_id in eos_ids:
            break
        output_ids.append(token_id)
    response = tokenizer.decode(output_ids)
    del model, tokenizer, logits, last_logits
    mx.clear_cache()
    return token_ids, output_ids, response, max_abs_logit, lora_b_max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--messages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=32)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error('--max-tokens must be positive')
    first = json.loads(args.messages.read_text(encoding='utf-8').splitlines()[0])
    messages = first.get('messages')
    if not isinstance(messages, list):
        parser.error('first training sample does not contain chat messages')
    base_tokens, base_output_ids, base_response, base_logit_max, _ = _run(str(args.model), None, messages, args.max_tokens)
    adapter_tokens, adapter_output_ids, adapter_response, adapter_logit_max, lora_b_max = _run(
        str(args.model), str(args.adapter) if args.adapter else None, messages, args.max_tokens)
    if base_tokens != adapter_tokens:
        raise RuntimeError('base and adapter chat token IDs differ')
    # A second deterministic forward pass captures the actual adapter effect.
    import mlx.core as mx
    from mlx_lm.utils import load
    base_model, _ = load(str(args.model))
    adapter_model, _ = load(str(args.model), adapter_path=str(args.adapter))
    base_logits = base_model(mx.array(base_tokens)[None, :])
    adapter_logits = adapter_model(mx.array(adapter_tokens)[None, :])
    if len(base_logits.shape) == 3:
        base_logits, adapter_logits = base_logits[0, -1], adapter_logits[0, -1]
    else:
        base_logits, adapter_logits = base_logits[-1], adapter_logits[-1]
    delta = mx.max(mx.abs(base_logits.astype(mx.float32) - adapter_logits.astype(mx.float32)))
    mx.eval(delta)
    max_abs_logit_delta = float(delta.item())
    del base_model, adapter_model, base_logits, adapter_logits
    mx.clear_cache()
    result = {'adapter_loaded': args.adapter is not None, 'prompt_token_count': len(base_tokens),
              'base_generated_token_count': len(base_output_ids), 'adapter_generated_token_count': len(adapter_output_ids),
              'base_response': base_response, 'adapter_response': adapter_response,
              'base_max_abs_logit': base_logit_max, 'adapter_max_abs_logit': adapter_logit_max,
              'max_abs_logit_delta': max_abs_logit_delta, 'lora_b_max_abs': lora_b_max,
              'adapter_output_changed': base_output_ids != adapter_output_ids}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
