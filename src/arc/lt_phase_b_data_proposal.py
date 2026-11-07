"""Propose a draft answer-file identity without selecting any task answers.

Hashing local files and checking their registration is not independent
attestation, a frozen protocol, or authorization to prepare training data.
"""
import hashlib
from pathlib import Path

from src.arc.lt_attestation import _object
from src.arc.lt_phase_b_collection_preflight import preflight_collection
from src.arc.lt_phase_b_registered_worker import _digest


def _read_protocol(root):
    return _object((Path(root) / 'configs/lt.phase-b.protocol.json').read_bytes())


def propose_data_registration(*, collection_spec, expected_collection_sha256, source_root, source_inventory,
                              allocation_receipt, receipt, export, challenge_bytes, solution_bytes,
                              official_challenge_bytes, python_executable, model_path):
    if not isinstance(challenge_bytes, bytes) or not isinstance(solution_bytes, bytes):
        raise ValueError('invalid local data bytes')
    protocol = _read_protocol(source_root)
    data = protocol.get('data')
    if (not isinstance(data, dict) or data.get('challenge_sha256') != hashlib.sha256(challenge_bytes).hexdigest() or
            data.get('solution_sha256') != hashlib.sha256(solution_bytes).hexdigest()):
        raise ValueError('data hashes differ from protocol')
    def preflight():
        return preflight_collection(source_root=source_root, spec=collection_spec, expected_run_sha256=expected_collection_sha256,
            source_inventory=source_inventory, allocation_receipt=allocation_receipt, export=export, receipt=receipt,
            challenge_bytes=challenge_bytes, official_challenge_bytes=official_challenge_bytes,
            python_executable=python_executable, model_path=model_path)
    checked = preflight()
    if checked != {'status': 'preflight_consistent_unverified', 'execution_ready': False,
                   'run_sha256': expected_collection_sha256, 'selected_tasks': len(collection_spec['selected_task_sha256']),
                   'evidence_type': 'local_files_and_caller_supplied_registration'}:
        raise ValueError('collection preflight differs')
    spec = {'protocol': 'lt-phase-b-data-v1', 'status': 'draft', 'run_id': collection_spec['run_id'],
            'collection_sha256': expected_collection_sha256, 'source_code_sha256': collection_spec['source_code_sha256'],
            'solution_sha256': data['solution_sha256']}
    if _read_protocol(source_root) != protocol or preflight() != checked:
        raise ValueError('data proposal inputs changed')
    return {'status': 'data_registration_proposed_unverified', 'execution_ready': False,
            'training_allowed': False, 'spec': spec, 'spec_sha256': _digest(spec)}
