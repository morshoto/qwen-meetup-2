"""Compare a draft runtime manifest with a local interpreter observation."""
from pathlib import Path

from src.arc.rsi_registration import _read_inputs, verify_draft_registration
from src.arc.rsi_runtime_identity import verify_runtime_identity


def verify_registered_runtime(root, *, registration, python_executable):
    root = Path(root).resolve()
    verify_draft_registration(root, registration)
    inputs, hashes = _read_inputs(root, registration['input_paths'])
    if hashes != registration['input_sha256']:
        raise ValueError('registration inputs changed during runtime check')
    manifest = inputs['runtime_manifest']
    if set(manifest) == {'runtime_identity', 'environment_inventory'}:
        manifest = manifest['runtime_identity']
    verify_runtime_identity(python_executable, manifest)
    verify_draft_registration(root, registration)
    return {'status': 'draft_runtime_consistent', 'execution_ready': False,
            'runtime_manifest_sha256': hashes['runtime_manifest'],
            'evidence_type': 'local_runtime_and_unwitnessed_draft'}
