"""Provisional novelty screening; pool provenance needs an independent audit."""
import hashlib
import re

from src.arc.ast_fingerprint import stable_ast_sha256

_HASH = re.compile(r'^[0-9a-f]{64}$')
_FIELDS = ('source_sha256', 'ast_sha256', 'behavior_sha256', 'probe_set_sha256')


def screen_novelty(code: str, behavior_sha256: str, probe_set_sha256: str,
                   reference_pool: list[dict]) -> dict:
    if (not isinstance(code, str) or not code.strip() or
            not isinstance(behavior_sha256, str) or not _HASH.fullmatch(behavior_sha256) or
            not isinstance(probe_set_sha256, str) or not _HASH.fullmatch(probe_set_sha256) or
            not isinstance(reference_pool, list)):
        raise ValueError('invalid candidate fingerprint or reference pool')
    for item in reference_pool:
        if (not isinstance(item, dict) or any(not isinstance(item.get(key), str) or
                not _HASH.fullmatch(item[key]) for key in _FIELDS)):
            raise ValueError('invalid reference fingerprint')
        if item['probe_set_sha256'] != probe_set_sha256:
            raise ValueError('reference probe set differs from candidate')
    try:
        source_hash = hashlib.sha256(code.strip().encode()).hexdigest()
        ast_hash = stable_ast_sha256(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError('candidate source is not parseable') from exc
    if any(item['source_sha256'] == source_hash for item in reference_pool):
        status = 'duplicate_source'
    elif any(item['ast_sha256'] == ast_hash for item in reference_pool):
        status = 'duplicate_ast'
    elif any(item['behavior_sha256'] == behavior_sha256 for item in reference_pool):
        status = 'sampled_behavior_match'
    else:
        status = 'novel_unverified'
    return {'status': status, 'eligible_for_training': False}
