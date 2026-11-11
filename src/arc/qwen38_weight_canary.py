"""Fail-closed evidence checks for the separate 27B technical canary.

This validates a receipt; it does not run training or establish ARC performance.
"""
from __future__ import annotations

import math
import re


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_evidence(record):
    before = record.get('base_before')
    after = record.get('base_after')
    if not isinstance(before, dict) or not before or before != after or not all(
        isinstance(name, str) and name and _hash(value) for name, value in before.items()
    ):
        raise ValueError('missing or changed base-weight hashes')
    if record.get('exit_code') != 0 or record.get('timed_out') is not False or record.get('memory_limit_exceeded') is not False or record.get('process_group_gone') is not True:
        raise ValueError('canary execution or cleanup failed')
    for field in ('loss', 'max_abs_logit_delta'):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'invalid {field}')
    if record['max_abs_logit_delta'] <= 0 or not _hash(record.get('adapter_sha256')):
        raise ValueError('adapter reload evidence missing')
    changes = record.get('changed_lora_tensors')
    if isinstance(changes, bool) or not isinstance(changes, int) or changes < 1:
        raise ValueError('no measured LoRA tensor changes')
    return 'passed'


def parse_loss(log):
    clean = re.sub(r'\x1b\[[0-9;]*m', '', log)
    matches = re.findall(r'Iter\s+1:\s*Train loss\s+([^,\s]+)', clean)
    if len(matches) != 1:
        raise ValueError('expected one step loss')
    try:
        value = float(matches[0])
    except ValueError as exc:
        raise ValueError('invalid step loss') from exc
    if not math.isfinite(value):
        raise ValueError('invalid step loss')
    return value
