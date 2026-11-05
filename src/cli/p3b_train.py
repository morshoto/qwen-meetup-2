"""Run the pinned MLX-LM LoRA CLI with an explicit Qwen chat-template mode."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.arc.mlx_chat_template import install_training_template_policy


def main() -> None:
    parser = argparse.ArgumentParser(description='MLX-LM chat-template-aligned P3-B trainer')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--thinking-mode', choices=('server_default', 'off'), required=True)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error('MLX-LM config file does not exist')
    install_training_template_policy(args.thinking_mode)
    # MLX-LM 0.30.7's ChatDataset reads mask_prompt from args and computes the
    # prompt-token offset using this tokenizer wrapper. Keep its native chat
    # dataset and loss implementation, explicitly selecting the verified flag.
    sys.argv = ['mlx_lm.lora', '--config', str(args.config), '--mask-prompt']
    from mlx_lm.lora import main as mlx_lora_main
    mlx_lora_main()


if __name__ == '__main__':
    main()
