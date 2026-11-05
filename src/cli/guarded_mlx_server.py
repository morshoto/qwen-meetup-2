"""Explicitly start the optional loopback-only budgeted MLX service."""
import argparse
import importlib.metadata
import re
from pathlib import Path
from src.arc.guarded_generation import GuardedGenerator
from src.arc.guarded_contract import AUDITED_GENERATE_STEP
from src.arc.guarded_http import GuardedHTTPServer
from src.arc.mlx_token_driver import MLXTokenDriver
from src.arc.token_ledger import TokenLedger



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--adapter-path', type=Path)
    parser.add_argument('--expected-adapter-sha256', default='none')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--expected-mlx-lm-version', required=True)
    parser.add_argument('--expected-generate-step-sha256', required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--total-tokens', type=int, required=True)
    parser.add_argument('--max-prompt-tokens', type=int, required=True)
    parser.add_argument('--max-new-tokens', type=int, required=True)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--thinking', choices=('server_default', 'off', 'on'), default='server_default')
    parser.add_argument('--max-request-seconds', type=float, default=120.0)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', args.revision) or args.model_dir.expanduser().resolve().name != args.revision:
        parser.error('model directory must be a local snapshot named by its full revision')
    if min(args.total_tokens, args.max_prompt_tokens, args.max_new_tokens) <= 0:
        parser.error('token limits must be positive')
    try:
        version = importlib.metadata.version('mlx-lm')
    except importlib.metadata.PackageNotFoundError:
        parser.error('mlx-lm is not installed')
    if version != args.expected_mlx_lm_version:
        parser.error('installed mlx-lm version differs from the specified version')
    if AUDITED_GENERATE_STEP.get(version) != args.expected_generate_step_sha256:
        parser.error('MLX-LM generation source is not in the reviewed allowlist')
    ledger = TokenLedger(args.ledger, args.total_tokens)
    if args.adapter_path is None and args.expected_adapter_sha256 != 'none':
        parser.error('--expected-adapter-sha256 requires --adapter-path')
    if args.adapter_path is not None and not re.fullmatch(r'[0-9a-f]{64}', args.expected_adapter_sha256):
        parser.error('adapter loading requires a full expected SHA-256')
    driver = MLXTokenDriver(args.model_dir, temperature=args.temperature, seed=0, thinking=args.thinking,
                            adapter_path=args.adapter_path, expected_adapter_sha256=args.expected_adapter_sha256)
    if driver.generate_step_sha256 != args.expected_generate_step_sha256:
        parser.error('MLX-LM generation source differs from the specified hash')
    driver.runtime_version = version
    guard = GuardedGenerator(ledger, driver, max_prompt_tokens=args.max_prompt_tokens,
                             stop_token_ids=driver.stop_token_ids)
    server = GuardedHTTPServer(('127.0.0.1', args.port), guard, model=args.model_id,
                               revision=args.revision, temperature=args.temperature,
                               max_new_tokens=args.max_new_tokens, max_request_seconds=args.max_request_seconds)
    try:
        server.serve_forever()
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
