"""Keep MLX-LM's chat data mask on the same Qwen thinking mode as inference."""
from __future__ import annotations

from typing import Any


def install_training_template_policy(mode: str, tokenizer_wrapper: type | None = None) -> None:
    if mode not in {'server_default', 'off', 'on'}:
        raise ValueError('unsupported chat-template thinking mode')
    if tokenizer_wrapper is None:
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        tokenizer_wrapper = TokenizerWrapper
    original = tokenizer_wrapper.apply_chat_template
    if getattr(original, '_p3b_thinking_mode', None) == mode:
        return

    def apply_chat_template(self: Any, *args: Any, **kwargs: Any) -> Any:
        if mode in {'off', 'on'}:
            template = getattr(self, 'chat_template', None)
            if not isinstance(template, str) or 'enable_thinking' not in template:
                raise ValueError('tokenizer chat template does not support the Qwen thinking switch')
            kwargs.setdefault('enable_thinking', mode == 'on')
        return original(self, *args, **kwargs)

    apply_chat_template._p3b_thinking_mode = mode  # type: ignore[attr-defined]
    tokenizer_wrapper.apply_chat_template = apply_chat_template
