"""Assemble a draft-only collection registration from local inventories.

No files are written and no model is loaded. This is not an independent
preregistration, allocation decision, authorization or frozen run.
"""
import copy
import hashlib
import re

from src.arc.lt_attestation import _object
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_allocation_receipt import derive_allocation
from src.arc.lt_phase_b_registered_worker import _SETTINGS, _digest, validate_descriptor
from src.arc.lt_public import build_public_export
from src.arc.lt_source import capture_source_tree
from src.arc.rsi_environment_identity import snapshot_environment
from src.arc.rsi_model_identity import snapshot_model_manifest
from src.arc.rsi_runtime_identity import snapshot_runtime_identity


def propose_collection_registration(*, source_root, challenge_bytes, official_challenge_bytes, training_task_ids,
                                    run_id, python_executable, model_path):
    if (not isinstance(training_task_ids, list) or len(training_task_ids) != 100 or
            any(not isinstance(x, str) or not re.fullmatch(r'[0-9a-f]{8}', x) for x in training_task_ids) or
            len(set(training_task_ids)) != 100 or
            not isinstance(run_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', run_id) or
            not isinstance(challenge_bytes, bytes) or not isinstance(official_challenge_bytes, bytes)):
        raise ValueError('invalid proposed collection inputs')
    ids = sorted(training_task_ids)
    allocation = derive_allocation(source_root, challenge_bytes, official_challenge_bytes, ids)
    if (not isinstance(allocation, dict) or allocation.get('status') != 'allocation_consistent_unverified' or
            allocation.get('execution_ready') is not False or
            not isinstance(allocation.get('training_task_sha256'), dict) or
            set(allocation['training_task_sha256']) != set(ids)):
        raise ValueError('allocation proposal differs')
    inventory = capture_source_tree(source_root)
    runtime = snapshot_runtime_identity(python_executable)
    environment = snapshot_environment(python_executable)
    model = snapshot_model_manifest(model_path, model_key='14b')
    spec = {'protocol': 'lt-phase-b-collection-v1', 'status': 'draft', 'run_id': run_id,
            'source_code_sha256': inventory['code_sha256'], 'allocation_audit_sha256': _digest(allocation),
            'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
            'selected_task_sha256': allocation['training_task_sha256'], 'model_manifest': model,
            'settings': copy.deepcopy(_SETTINGS), 'runtime_manifest': runtime, 'environment_manifest': environment}
    candidate_sha = _digest(spec)
    receipt = {'comparison_sha256': candidate_sha, 'selected_task_sha256': spec['selected_task_sha256']}
    export = build_public_export(_object(challenge_bytes), receipt)
    # Shape-check the future frozen form without returning or persisting it.
    hypothetical = {**spec, 'status': 'frozen'}
    validate_descriptor(spec=hypothetical, expected_run_sha256=_digest(hypothetical), source_inventory=inventory,
        receipt={'comparison_sha256': _digest(hypothetical), 'selected_task_sha256': spec['selected_task_sha256']},
        allocation_receipt=allocation)
    if capture_source_tree(source_root) != inventory:
        raise ValueError('source tree changed during proposal')
    return {'status': 'collection_registration_proposed_unverified', 'execution_ready': False,
            'spec': spec, 'spec_sha256': candidate_sha, 'inventory': inventory, 'allocation': allocation,
            'receipt': receipt, 'export': export}
