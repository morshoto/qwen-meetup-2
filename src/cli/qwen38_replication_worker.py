"""Validate answer-free matched worker requests before loading the model."""
import json
import re

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
