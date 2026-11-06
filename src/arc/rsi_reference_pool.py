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


def build_legacy_duplicate_reference(*, model_key, source, metadata_bytes, expected_metadata_sha256,
                                     private_probes, expected_probe_sha256, sandbox, max_sandbox_starts):
    """Build a model-specific duplicate reference without assigning ARC ancestry.

    The supplied metadata and sandbox are not authenticated here; the result
    cannot authorize collection or make a historical candidate trainable.
    """
    from src.arc.rsi_model_identity import _MODELS
    from src.arc.rsi_transport_audit import _json_object

    if (not isinstance(model_key, str) or model_key not in _MODELS or not isinstance(metadata_bytes, bytes) or
            not 1 <= len(metadata_bytes) <= 65536 or not isinstance(expected_metadata_sha256, str) or
            not _HASH.fullmatch(expected_metadata_sha256) or
            hashlib.sha256(metadata_bytes).hexdigest() != expected_metadata_sha256 or
            not isinstance(source, str) or not source.strip() or len(source.encode()) > 65536 or
            not callable(getattr(sandbox, 'run', None)) or type(max_sandbox_starts) is not int or
            not 1 <= max_sandbox_starts <= 300):
        raise ValueError('invalid legacy duplicate reference inputs')
    metadata = _json_object(metadata_bytes)
    model_id, revision = _MODELS[model_key]
    if (metadata.get('status') != 'historical_static_reference_unverified' or
            metadata.get('model_key') != model_key or metadata.get('model_id') != model_id or
            metadata.get('model_revision') != revision or metadata.get('eligible_for_training') is not False or
            metadata.get('independently_verified_for_new_protocol') is not False or
            metadata.get('candidate_code_executed') is not False or
            metadata.get('private_behavior_for_proposed_probes') != 'not_evaluated' or
            not isinstance(metadata.get('record_task_id'), str) or
            not re.fullmatch(r'e6f-[0-9a-f]{24}', metadata['record_task_id']) or
            any(not isinstance(metadata.get(key), str) or not _HASH.fullmatch(metadata[key])
                for key in ('source_sha256', 'ast_sha256', 'candidate_index_sha256',
                            'e8_slot_result_sha256', 'e9_existing_oracles_sha256')) or
            hashlib.sha256(source.strip().encode()).hexdigest() != metadata['source_sha256'] or
            _static_failure(source)):
        raise ValueError('legacy reference identity differs')
    try:
        ast_sha = stable_ast_sha256(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError('legacy reference source invalid') from exc
    if (ast_sha != metadata['ast_sha256'] or probes_sha256(private_probes) != expected_probe_sha256 or
            len({canonical(grid) for grid in private_probes}) != len(private_probes) or
            len(private_probes) * 2 > max_sandbox_starts):
        raise ValueError('legacy reference probes or budget differ')
    starts = 0

    class MeteredSandbox:
        def run(self, code, grid):
            nonlocal starts
            if starts >= max_sandbox_starts:
                raise TimeoutError('legacy reference sandbox budget exhausted')
            starts += 1
            return sandbox.run(code, grid)

    outputs = verify_oracle(source, private_probes, sandbox=MeteredSandbox())
    pool = [{'source_sha256':metadata['source_sha256'], 'ast_sha256':ast_sha,
             'behavior_sha256':hashlib.sha256(canonical(outputs).encode()).hexdigest(),
             'probe_set_sha256':expected_probe_sha256}]
    return {'status':'legacy_duplicate_reference_unverified', 'model_key':model_key,
            'historical_task_id':metadata['record_task_id'], 'metadata_sha256':expected_metadata_sha256,
            'pool':pool, 'pool_sha256':_digest(pool), 'sandbox_starts':starts,
            'eligible_for_training':False, 'source_origins_verified':False, 'history_complete':False,
            'arc_ancestry_assumed':False, 'evidence_type':'caller_supplied_metadata_and_sandbox'}
