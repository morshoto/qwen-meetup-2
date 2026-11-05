"""Compare draft environment inventory with a local virtual environment."""
from pathlib import Path

from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_registration import _read_inputs, verify_draft_registration


def verify_registered_environment(root, *, registration, python_executable):
    root = Path(root).resolve()
    verify_draft_registration(root, registration)
    inputs, hashes = _read_inputs(root, registration['input_paths'])
    if hashes != registration['input_sha256']:
        raise ValueError('registration inputs changed during environment check')
    manifest = inputs['runtime_manifest']
    if (not isinstance(manifest, dict) or set(manifest) != {'runtime_identity', 'environment_inventory'} or
            not isinstance(manifest['runtime_identity'], dict) or
            not isinstance(manifest['environment_inventory'], dict)):
        raise ValueError('runtime manifest lacks environment inventory')
    identity, environment = manifest['runtime_identity'], manifest['environment_inventory']
    if any(not isinstance(identity.get(key), str) or identity.get(key) != environment.get(key)
           for key in ('interpreter_realpath', 'interpreter_sha256')):
        raise ValueError('runtime and environment interpreter identities differ')
    verify_environment(python_executable, environment)
    verify_draft_registration(root, registration)
    return {'status': 'draft_environment_consistent', 'execution_ready': False,
            'runtime_manifest_sha256': hashes['runtime_manifest'],
            'evidence_type': 'local_venv_and_unwitnessed_draft'}
