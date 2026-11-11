"""Single-request 27B worker; it never opens ARC solution files."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_local_runner import Backend, MODEL_REVISION
from src.arc.qwen38_weight_generation import capture_response, create_request


def run_request(payload, model_path, *, backend_factory=Backend):
    model_path = Path(model_path).resolve()
    if not model_path.is_dir() or model_path.name != MODEL_REVISION:
        raise ValueError('worker requires the pinned local model snapshot')
    if not isinstance(payload, dict) or set(payload) != {'request', 'max_tokens'}:
        raise ValueError('unexpected worker input')
    request = payload['request']; max_tokens = payload['max_tokens']
    if not isinstance(request, dict) or type(max_tokens) is not int or not 0 < max_tokens <= 4096:
        raise ValueError('invalid worker request')
    try:
        visible = json.loads(request['content'])
        if not isinstance(visible, dict) or set(visible) != {'instruction', 'task'} or visible['instruction'] != PROMPT:
            raise ValueError('unexpected prompt')
        expected = create_request(request['task_id'], visible['task'],
                                  candidate_index=request['candidate_index'], seed=request['seed'])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('invalid visible task') from error
    if request != expected:
        raise ValueError('request identity differs or contains hidden fields')
    backend = backend_factory(model_path)
    response = backend.call(request['content'], request['seed'], max_tokens)
    return capture_response(request, response, max_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('worker input too large')
    payload = json.loads(raw)
    result = run_request(payload, args.model_path)
    encoded = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
