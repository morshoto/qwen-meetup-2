"""Validate answer-free matched worker requests before loading the model."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION
from src.arc.qwen38_replication_freeze import ADAPTER_SHA256
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import fingerprint
from src.arc.qwen38_weight_generation import create_request
from src.cli.qwen38_weight_eval_worker import run_adapter_request
from src.cli.qwen38_weight_worker import run_request


def run_replication_request(descriptor, arm, payload, model_path, adapter_path):
    """Call the pinned arm; caller must separately verify preregistration sources."""
    try:
        request = payload['request']
        index = request['candidate_index']
        task_id = request['task_id']
        visible = json.loads(request['content'])
        model = descriptor['model']
        identity = {k: descriptor[k] for k in ('adapter_sha256', 'adapter_config_sha256')}
        if (descriptor.get('status') != 'evaluation_ready' or descriptor.get('inference_authorized') is not True or
            descriptor.get('sha256') != digest({k: v for k, v in descriptor.items() if k != 'sha256'}) or
            arm not in ('base', 'adapter') or set(payload) != {'request', 'max_tokens'} or
            model.get('id') != MODEL_ID or model.get('revision') != MODEL_REVISION or model.get('quantization') != '4-bit MLX' or
            descriptor.get('candidate_count') != 2 or descriptor.get('temperature') != .6 or
            descriptor.get('max_context') != 18432 or descriptor.get('max_tokens') != 4096 or payload['max_tokens'] != 4096 or
            identity['adapter_sha256'] != ADAPTER_SHA256 or
            any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None for v in identity.values()) or
            task_id not in descriptor['task_ids'] or type(index) is not int or index not in (0, 1) or
            request['seed'] != 20261111 + index or request['prompt_sha256'] != descriptor['prompt_sha256'] or
            fingerprint(visible['task']) != descriptor['task_fingerprints'][task_id] or
            request != create_request(task_id, visible['task'], candidate_index=index, seed=request['seed'])):
            raise ValueError('replication request does not match registration')
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('invalid replication request') from error
    if arm == 'base':
        return run_request(payload, model_path)
    return run_adapter_request(payload, model_path, adapter_path, identity)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for flag in ('registration', 'registration-sha256', 'arm', 'model-path', 'adapter-path', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv)
    registration = json.loads(Path(args.registration).read_bytes())
    if (registration.get('sha256') != args.registration_sha256 or
        registration.get('sha256') != digest({k: v for k, v in registration.items() if k != 'sha256'})):
        raise ValueError('worker registration identity mismatch')
    if Path(args.output).exists() or Path(args.output).is_symlink():
        raise FileExistsError(args.output)
    raw = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('worker input too large')
    result = run_replication_request(registration, args.arm, json.loads(raw), args.model_path, args.adapter_path)
    encoded = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
