"""Propose a draft public-only matched comparison without scoring or inference.

Local consistency is not independent attestation, a frozen evaluation, or
permission to run one. Comparison answers are not selected here.
"""
import copy
import hashlib
import re

from src.arc.lt_attestation import _object
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_real_reload_dispatch import review_real_reload_attempt
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_training_preflight import preflight_training_data
from src.arc.lt_public import build_public_export


def propose_comparison_registration(*, collection_spec, expected_collection_sha256, training_spec, expected_training_sha256,
                                    canary_spec, expected_canary_sha256, source_root, run_dir, source_inventory,
                                    allocation_receipt, collection_receipt, collection_export, challenge_bytes,
                                    solution_bytes, official_challenge_bytes, python_executable, model_path):
    settings, _ = validate_canary_identity(canary_spec=canary_spec, expected_canary_sha256=expected_canary_sha256,
        collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
        source_inventory=source_inventory, allocation_receipt=allocation_receipt, receipt=collection_receipt)
    if (not isinstance(training_spec, dict) or training_spec.get('protocol') != 'lt-phase-b-real-training-v1' or
            training_spec.get('status') != 'frozen' or _digest(training_spec) != expected_training_sha256 or
            training_spec.get('collection_sha256') != expected_collection_sha256 or
            training_spec.get('canary_sha256') != expected_canary_sha256 or not isinstance(solution_bytes, bytes) or
            hashlib.sha256(solution_bytes).hexdigest() != training_spec.get('solution_sha256')):
        raise ValueError('training descriptor or answer file differs')
    def preflight():
        return preflight_training_data(training_spec=training_spec, expected_training_sha256=expected_training_sha256,
            collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
            source_root=source_root, run_dir=run_dir, source_inventory=source_inventory, allocation_receipt=allocation_receipt,
            receipt=collection_receipt, export=collection_export, challenge_bytes=challenge_bytes, solution_bytes=solution_bytes,
            official_challenge_bytes=official_challenge_bytes, python_executable=python_executable, model_path=model_path)
    def reload():
        return review_real_reload_attempt(run_dir, training=training_spec, canary=canary_spec,
            expected_training_sha256=expected_training_sha256, expected_canary_sha256=expected_canary_sha256,
            expected_collection_sha256=expected_collection_sha256)
    checked, observed = preflight(), reload()
    if (not isinstance(checked, dict) or checked.get('status') != 'training_data_consistent_unverified' or
            checked.get('training_allowed') is not False or checked.get('training_sha256') != expected_training_sha256 or
            checked.get('solution_sha256') != training_spec.get('solution_sha256') or
            checked.get('dataset_manifest_sha256') != training_spec.get('dataset_manifest_sha256') or
            checked.get('attempted_tasks') != 100 or type(checked.get('eligible_tasks')) is not int or checked['eligible_tasks'] <= 0 or
            checked.get('yield_status') != training_spec.get('yield_status') or not isinstance(observed, dict) or
            observed.get('status') != 'real_adapter_reload_observed_unverified' or observed.get('reload_verified') is not False or
            observed.get('promotion_allowed') is not False or not isinstance(observed.get('reload'), dict)):
        raise ValueError('training or adapter reload evidence differs')
    adapter = observed['reload'].get('adapter_sha256')
    tasks = allocation_receipt.get('comparison_task_sha256')
    if (not isinstance(adapter, str) or not re.fullmatch(r'[0-9a-f]{64}', adapter) or not isinstance(tasks, dict) or
            len(tasks) != 50 or any(not isinstance(k, str) or not re.fullmatch(r'[0-9a-f]{8}', k) or
                not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for k, v in tasks.items()) or
            set(tasks) & set(collection_spec['selected_task_sha256'])):
        raise ValueError('comparison allocation or adapter differs')
    spec = {'protocol': 'lt-phase-b-comparison-v1', 'status': 'draft', 'run_id': collection_spec['run_id'],
            'collection_sha256': expected_collection_sha256, 'training_sha256': expected_training_sha256,
            'canary_sha256': expected_canary_sha256, 'source_code_sha256': collection_spec['source_code_sha256'],
            'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(), 'selected_task_sha256': copy.deepcopy(tasks),
            'adapter_sha256': adapter, 'settings': copy.deepcopy(settings)}
    digest = _digest(spec)
    receipt = {'comparison_sha256': digest, 'selected_task_sha256': spec['selected_task_sha256']}
    export = build_public_export(_object(challenge_bytes), receipt)
    if preflight() != checked or reload() != observed:
        raise ValueError('comparison proposal evidence changed')
    return {'status': 'comparison_registration_proposed_unverified', 'execution_ready': False,
            'spec': spec, 'spec_sha256': digest, 'receipt': receipt, 'export': export}
