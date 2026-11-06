"""Compare draft registration with local model files; never authorize a run."""
from pathlib import Path

from src.arc.rsi_model_identity import verify_model_manifest
from src.arc.rsi_registration import _read_inputs, verify_draft_registration


def verify_registered_model(root, *, registration, model_dir):
    root = Path(root).resolve()
    verify_draft_registration(root, registration)
    inputs, hashes = _read_inputs(root, registration['input_paths'])
    if hashes != registration['input_sha256']:
        raise ValueError('registration inputs changed during model check')
    verify_model_manifest(model_dir, inputs['model_manifest'])
    verify_draft_registration(root, registration)
    return {'status': 'draft_model_files_consistent', 'execution_ready': False,
            'model_manifest_sha256': hashes['model_manifest'],
            'evidence_type': 'local_snapshot_and_unwitnessed_draft'}
