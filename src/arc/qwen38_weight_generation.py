"""Format full-task requests and preserve ungraded generated responses.

This module never reads solution files or authorizes model execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_local_runner import clean_task, parse_prediction
from src.arc.qwen38_rsi_comparison import digest


def create_request(task_id, task, *, candidate_index, seed):
    if (not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', task_id) or
        type(candidate_index) is not int or not 0 <= candidate_index < 8 or type(seed) is not int or seed < 0):
        raise ValueError('invalid request identity')
    visible = clean_task(task)
    content = json.dumps({'instruction': PROMPT, 'task': visible}, sort_keys=True, separators=(',', ':'))
    return {'task_id': task_id, 'task_sha256': digest(visible), 'candidate_index': candidate_index,
            'seed': seed, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
            'content_sha256': hashlib.sha256(content.encode()).hexdigest(), 'content': content,
            'test_count': len(visible['test'])}


def capture_response(request, response, max_tokens):
    raw = response.get('raw')
    prompt_tokens, completion_tokens = response.get('prompt_tokens'), response.get('completion_tokens')
    seconds, peak = response.get('seconds'), response.get('peak_memory_gb')
    finish = response.get('finish_reason')
    if (not isinstance(raw, str) or len(raw.encode()) > 1024 * 1024 or
        type(max_tokens) is not int or max_tokens <= 0 or type(prompt_tokens) is not int or prompt_tokens < 0 or
        type(completion_tokens) is not int or not 0 <= completion_tokens <= max_tokens or
        isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0 or
        isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak < 0 or
        not isinstance(finish, str) or not finish):
        raise ValueError('invalid backend response metadata')
    parsed, mode = parse_prediction(raw, request['test_count'])
    return {key: request[key] for key in ('task_id', 'task_sha256', 'candidate_index', 'seed', 'prompt_sha256', 'content_sha256')} | {
        'raw': raw, 'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'outputs': parsed,
        'valid': parsed is not None, 'parse_mode': mode, 'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens, 'finish_reason': finish, 'seconds': seconds, 'peak_memory_gb': peak}
