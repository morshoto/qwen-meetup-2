"""Build duplicate-only fingerprints from caller-supplied historical sources.

Origins, ancestry completeness and sandbox identity are not authenticated by
this module. No source from this pool is eligible for training by virtue of it.
"""
import hashlib
import json
import re

from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.e6_curriculum import canonical, verify_oracle
from src.arc.e6f_repair import _static_failure
from src.arc.rsi_lineage_audit import probes_sha256

_HASH = re.compile(r'[0-9a-f]{64}')
_FIELDS = {'record_id', 'source', 'source_sha256', 'origin_sha256', 'ancestor_task_ids'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_reference_pool(*, entries, private_probes, expected_probe_sha256, generation_task_ids,
                         development_task_ids, sandbox, max_sandbox_starts):
    if (not isinstance(entries, list) or not isinstance(generation_task_ids, set) or not generation_task_ids or
            not isinstance(development_task_ids, set) or not development_task_ids or
            any(not isinstance(key, str) or not key for key in generation_task_ids | development_task_ids) or
            generation_task_ids & development_task_ids or type(max_sandbox_starts) is not int or
            not 1 <= max_sandbox_starts <= 8000 or not callable(getattr(sandbox, 'run', None))):
        raise ValueError('invalid historical reference settings')
    if (probes_sha256(private_probes) != expected_probe_sha256 or
            len({canonical(grid) for grid in private_probes}) != len(private_probes)):
        raise ValueError('private reference probes differ or contain duplicates')
    if len(entries) * len(private_probes) * 2 > max_sandbox_starts:
        raise ValueError('reference pool sandbox budget is insufficient')
    seen, manifest = set(), []
    for item in entries:
        if not isinstance(item, dict) or set(item) != _FIELDS:
            raise ValueError('invalid historical reference record')
        record_id, code, ancestry = item['record_id'], item['source'], item['ancestor_task_ids']
        if (not isinstance(record_id, str) or not record_id or len(record_id) > 200 or record_id in seen or
                not isinstance(code, str) or not code.strip() or len(code.encode()) > 65536 or
                not isinstance(item['source_sha256'], str) or not _HASH.fullmatch(item['source_sha256']) or
                hashlib.sha256(code.strip().encode()).hexdigest() != item['source_sha256'] or
                not isinstance(item['origin_sha256'], str) or not _HASH.fullmatch(item['origin_sha256']) or
                not isinstance(ancestry, list) or not ancestry or
                any(not isinstance(key, str) or not key for key in ancestry) or len(set(ancestry)) != len(ancestry) or
                not set(ancestry) <= generation_task_ids or _static_failure(code)):
            raise ValueError('historical source or ancestry differs')
        seen.add(record_id)
        manifest.append({'record_id': record_id, 'source_sha256': item['source_sha256'],
                         'origin_sha256': item['origin_sha256'], 'ancestor_task_ids': list(ancestry)})
    starts = 0

    class MeteredSandbox:
        def run(self, code, grid):
            nonlocal starts
            if starts >= max_sandbox_starts:
                raise TimeoutError('reference sandbox budget exhausted')
            starts += 1
            return sandbox.run(code, grid)

    pool = []
    metered = MeteredSandbox()
    for item in entries:
        outputs = verify_oracle(item['source'], private_probes, sandbox=metered)
        pool.append({'source_sha256': item['source_sha256'], 'ast_sha256': stable_ast_sha256(item['source']),
                     'behavior_sha256': hashlib.sha256(canonical(outputs).encode()).hexdigest(),
                     'probe_set_sha256': expected_probe_sha256})
    return {'status': 'reference_pool_unverified', 'pool': pool, 'pool_sha256': _digest(pool),
            'reference_manifest': manifest, 'reference_manifest_sha256': _digest(manifest),
            'sandbox_starts': starts, 'eligible_for_training': False,
            'source_origins_verified': False, 'history_complete': False}
