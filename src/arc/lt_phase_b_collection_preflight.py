"""Recompute local collection identities before any worker is dispatched.

This is a local consistency check, not proof of preregistration, independent
review, complete historical non-exposure, isolation, or authorization.
"""
from src.arc.lt_attestation import _object
from src.arc.lt_phase_b_allocation_receipt import derive_allocation
from src.arc.lt_phase_b_registered_worker import validate_descriptor
from src.arc.lt_phase_b_protocol_gate import validate_frozen_protocol
from src.arc.lt_public import build_public_export, load_public_export
from src.arc.lt_source import verify_source_tree
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_model_identity import verify_model_manifest
from src.arc.rsi_runtime_identity import verify_runtime_identity


def preflight_collection(*, source_root, spec, expected_run_sha256, source_inventory, allocation_receipt,
                         export, receipt, challenge_bytes, official_challenge_bytes, python_executable, model_path):
    validate_descriptor(spec=spec, expected_run_sha256=expected_run_sha256, source_inventory=source_inventory,
                        receipt=receipt, allocation_receipt=allocation_receipt)
    verify_source_tree(source_root, source_inventory)
    protocol = validate_frozen_protocol(source_root=source_root, spec=spec, source_inventory=source_inventory,
                                        allocation_receipt=allocation_receipt, challenge_bytes=challenge_bytes)
    if protocol != {'status': 'frozen_protocol_consistent_unverified', 'execution_ready': False}:
        raise ValueError('frozen protocol check differs')
    verify_runtime_identity(python_executable, spec['runtime_manifest'])
    verify_environment(python_executable, spec['environment_manifest'])
    verify_model_manifest(model_path, spec['model_manifest'], model_key='14b')
    checked = derive_allocation(source_root, challenge_bytes, official_challenge_bytes,
                                sorted(spec['selected_task_sha256']))
    if checked != allocation_receipt:
        raise ValueError('allocation receipt differs from local inventories')
    challenges = _object(challenge_bytes)
    expected_export = build_public_export(challenges, receipt)
    if export != expected_export or set(load_public_export(export, receipt)) != set(spec['selected_task_sha256']):
        raise ValueError('public task export differs')
    verify_source_tree(source_root, source_inventory)
    return {'status': 'preflight_consistent_unverified', 'execution_ready': False,
            'run_sha256': expected_run_sha256, 'selected_tasks': len(spec['selected_task_sha256']),
            'evidence_type': 'local_files_and_caller_supplied_registration'}
