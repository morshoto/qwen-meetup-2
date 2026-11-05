"""Bind extracted Qwen code to the actual sealed HTTP response body."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.arc.mlx_lora import _verify_qwen_generator_evidence


def verify_response_body(provenance: dict[str, Any], code: str) -> None:
    """Require sidecar text to be the content in the hashed response body."""
    _verify_qwen_generator_evidence(provenance, code)
    path = Path(provenance['response_log_path'])
    match = None
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('request_id') == provenance['request_id']:
                match = row
                break
    if not isinstance(match, dict) or not isinstance(match.get('raw_http_body_text'), str):
        raise ValueError('Qwen response body evidence is missing')
    body = match['raw_http_body_text']
    if hashlib.sha256(body.encode()).hexdigest() != provenance['raw_response_sha256']:
        raise ValueError('Qwen response body hash mismatch')
    try:
        content = json.loads(body)['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError('Qwen response body has no supported content') from exc
    if not isinstance(content, str) or content != match.get('generated_text'):
        raise ValueError('Qwen generated text differs from the hashed HTTP response body')
