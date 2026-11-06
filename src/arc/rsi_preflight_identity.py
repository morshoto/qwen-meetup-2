"""Compare a preflight proposal with local files, without authorizing execution."""
import hashlib
from pathlib import Path

from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_comparison_freeze import _hash
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_model_identity import _hash_file, verify_model_manifest
from src.arc.rsi_runtime_identity import verify_runtime_identity
from src.arc.rsi_slot_lifecycle import _directory
from src.arc.rsi_source_inventory import verify_source_inventory
from src.arc.rsi_transport_audit import _json_object

_MANIFESTS = {'model_manifest', 'runtime_manifest', 'environment_manifest', 'source_inventory'}


def verify_preflight_local_identities(*, root, expected_spec_sha256, manifest_bytes, source_root, python_executable):
    if (not _hash(expected_spec_sha256) or not isinstance(manifest_bytes, dict) or
            set(manifest_bytes) != _MANIFESTS or any(not isinstance(value, bytes) or
            not 1 <= len(value) <= 8 * 1024 * 1024 for value in manifest_bytes.values())):
        raise ValueError('invalid preflight identity inputs')
    with _directory(root) as directory:
        spec_raw, spec = _read(directory, 'preflight.proposal.json')
    if (hashlib.sha256(spec_raw).hexdigest() != expected_spec_sha256 or
            spec.get('status') != 'preflight_proposed_unverified' or spec.get('execution_ready') is not False or
            spec.get('training_allowed') is not False or not isinstance(spec.get('identity_hashes'), dict) or
            not isinstance(spec.get('config'), dict) or not isinstance(spec['config'].get('model'), str)):
        raise ValueError('preflight proposal differs')
    claimed = spec['identity_hashes']
    if any(hashlib.sha256(raw).hexdigest() != claimed.get(key) for key, raw in manifest_bytes.items()):
        raise ValueError('preflight identity manifest digest differs')
    manifests = {key: _json_object(value) for key, value in manifest_bytes.items()}
    runtime, environment = manifests['runtime_manifest'], manifests['environment_manifest']
    if any(not isinstance(runtime.get(key), str) or runtime.get(key) != environment.get(key)
           for key in ('interpreter_realpath', 'interpreter_sha256')):
        raise ValueError('runtime and environment interpreter identities differ')
    model = Path(spec['config']['model'])
    try:
        token_paths = [(model / name).resolve(strict=True) for name in ('tokenizer.json', 'tokenizer_config.json')]
        token_before = [_hash_file(path) for path in token_paths]
    except (OSError, RuntimeError) as exc:
        raise ValueError('preflight tokenizer files unavailable') from exc
    if token_before != [claimed.get('tokenizer'), claimed.get('tokenizer_config')]:
        raise ValueError('preflight tokenizer digest differs')
    verify_model_manifest(model, manifests['model_manifest'])
    verify_runtime_identity(python_executable, runtime)
    verify_environment(python_executable, environment)
    verify_source_inventory(source_root, manifests['source_inventory'])
    if ([(model / name).resolve(strict=True) for name in ('tokenizer.json', 'tokenizer_config.json')] != token_paths or
            [_hash_file(path) for path in token_paths] != token_before):
        raise ValueError('preflight tokenizer changed during verification')
    with _directory(root) as directory:
        after, _ = _read(directory, 'preflight.proposal.json')
    if after != spec_raw:
        raise ValueError('preflight proposal changed during verification')
    return {'status': 'preflight_local_identities_consistent', 'execution_ready': False,
            'independently_verified': False, 'proposal_sha256': expected_spec_sha256,
            'evidence_type': 'local_files_and_caller_supplied_manifests'}
