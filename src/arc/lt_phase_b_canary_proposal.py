"""Build a draft synthetic-canary registration; never start or authorize it."""
import copy
import hashlib
import json
import re
from pathlib import Path

from src.arc.lt_phase_b_canary_config import build_canary_config
from src.arc.lt_phase_b_canary_sample import RELOAD_SETTINGS
from src.arc.lt_phase_b_canary_trainer import _validate_dataset
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest, validate_descriptor
from src.arc.lt_source import verify_source_tree
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_model_identity import verify_model_manifest
from src.arc.rsi_runtime_identity import verify_runtime_identity


def propose_canary_registration(*, collection_spec, expected_collection_sha256, source_inventory, source_root,
                                allocation_receipt, receipt, model_dir, data_dir, adapter_dir,
                                python_executable, canary_run_id):
    settings, model = validate_descriptor(spec=collection_spec, expected_run_sha256=expected_collection_sha256,
        source_inventory=source_inventory, receipt=receipt, allocation_receipt=allocation_receipt)
    if not isinstance(canary_run_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', canary_run_id):
        raise ValueError('invalid canary run identity')
    def verify():
        verify_runtime_identity(python_executable, collection_spec['runtime_manifest'])
        verify_environment(python_executable, collection_spec['environment_manifest'])
        verify_model_manifest(model_dir, model, model_key='14b')
        verify_source_tree(source_root, source_inventory)
    def plan():
        value = build_canary_config(model_dir=model_dir, data_dir=data_dir, adapter_dir=adapter_dir)
        if any(Path(data_dir).iterdir()):
            _validate_dataset(Path(data_dir), value['dataset_jsonl'])
        return value
    proposed = plan()
    verify()
    config = proposed['trainer_config']
    config_bytes = (json.dumps(config, sort_keys=True, allow_nan=False) + '\n').encode()
    spec = {'protocol': 'lt-phase-b-canary-v1', 'status': 'draft', 'run_id': canary_run_id,
            'collection_sha256': expected_collection_sha256, 'source_code_sha256': collection_spec['source_code_sha256'],
            'runtime_manifest': copy.deepcopy(collection_spec['runtime_manifest']),
            'environment_manifest': copy.deepcopy(collection_spec['environment_manifest']),
            'model_manifest': copy.deepcopy(collection_spec['model_manifest']),
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'dataset_sha256': proposed['dataset_sha256'],
            'thinking_mode': 'off', 'timeout_seconds': settings['task_timeout_seconds'],
            'resource_limits': copy.deepcopy(settings['resource_limits']), 'reload_settings': copy.deepcopy(RELOAD_SETTINGS)}
    hypothetical = {**spec, 'status': 'frozen'}
    validate_canary_identity(canary_spec=hypothetical, expected_canary_sha256=_digest(hypothetical),
        collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
        source_inventory=source_inventory, allocation_receipt=allocation_receipt, receipt=receipt)
    verify()
    if plan() != proposed:
        raise ValueError('canary proposal inputs changed')
    return {'status': 'canary_registration_proposed_unverified', 'execution_ready': False, 'promotion_allowed': False,
            'spec': spec, 'spec_sha256': _digest(spec), 'trainer_config': config, 'dataset_jsonl': proposed['dataset_jsonl']}
