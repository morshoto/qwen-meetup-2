"""Source identities advertised by the optional guarded MLX server."""
from __future__ import annotations
import hashlib
from pathlib import Path

# This exact MLX-LM source was inspected for one-token lookahead.
AUDITED_GENERATE_STEP = {'0.30.7': '1aee463145a78556a66604ace7b6d85dc10035226aa4f909cbefccfa81b49de6'}

SOURCE_FILES = (
    'src/arc/guarded_contract.py',
    'src/arc/deadline_watchdog.py',
    'src/arc/guarded_generation.py',
    'src/arc/guarded_http.py',
    'src/arc/mlx_token_driver.py',
    'src/arc/token_ledger.py',
    'src/cli/guarded_mlx_server.py',
)


def source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
