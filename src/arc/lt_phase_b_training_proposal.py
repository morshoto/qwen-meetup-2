"""Propose a draft real-training descriptor from verified local artifacts.

No configuration is frozen, no trainer is started, and local consistency does
not establish independent attestation or authorization for training.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_reload_dispatch import review_reload_attempt
from src.arc.lt_phase_b_training_config import build_training_config
from src.arc.lt_phase_b_training_preflight import preflight_training_data
from src.arc.rsi_model_identity import _json_file


def propose_training_registration(*, collection_spec, expected_collection_sha256, canary_spec, expected_canary_sha256,
                                  data_spec, expected_data_sha256, source_root, run_dir, source_inventory,
                                  allocation_receipt, receipt, export, challenge_bytes, solution_bytes,
                                  official_challenge_bytes, python_executable, model_path):
    settings, _ = validate_canary_identity(canary_spec=canary_spec, expected_canary_sha256=expected_canary_sha256,
        collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
        source_inventory=source_inventory, allocation_receipt=allocation_receipt, receipt=receipt)
    keys = {'protocol', 'status', 'run_id', 'collection_sha256', 'source_code_sha256', 'solution_sha256'}
    valid_hash = lambda x: isinstance(x, str) and bool(re.fullmatch(r'[0-9a-f]{64}', x))
    if (not isinstance(data_spec, dict) or set(data_spec) != keys or not valid_hash(expected_data_sha256) or
            _digest(data_spec) != expected_data_sha256 or data_spec.get('protocol') != 'lt-phase-b-data-v1' or
            data_spec.get('status') != 'frozen' or data_spec.get('run_id') != collection_spec['run_id'] or
            data_spec.get('collection_sha256') != expected_collection_sha256 or
            data_spec.get('source_code_sha256') != collection_spec['source_code_sha256'] or
            not isinstance(solution_bytes, bytes) or hashlib.sha256(solution_bytes).hexdigest() != data_spec.get('solution_sha256')):
        raise ValueError('training data descriptor differs')
    source = {'model_id': 'mlx-community/Qwen3-14B-4bit',
              'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
              'source_code_sha256': collection_spec['source_code_sha256'],
              'run_sha256': expected_collection_sha256, 'run_id': collection_spec['run_id']}
    def config_plan():
        return build_training_config(source_root=source_root, run_dir=run_dir, model_dir=model_path, expected_source=source)
    plan = config_plan()
    config = plan['trainer_config']
    config_bytes = (json.dumps(config, sort_keys=True, allow_nan=False) + '\n').encode()
    manifest = _json_file(Path(config['data']) / 'manifest.json', plan['dataset_manifest_sha256'])
    count = manifest['train_count'] + manifest['valid_count']
    spec = {'protocol': 'lt-phase-b-real-training-v1', 'status': 'draft', 'run_id': collection_spec['run_id'],
            'collection_sha256': expected_collection_sha256, 'canary_sha256': expected_canary_sha256,
            'source_code_sha256': collection_spec['source_code_sha256'], 'dataset_manifest_sha256': plan['dataset_manifest_sha256'],
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'solution_sha256': data_spec['solution_sha256'],
            'yield_status': plan['yield_status'], 'checkpoint_policy': 'final_step_only',
            'timeout_seconds': settings['task_timeout_seconds'], 'resource_limits': copy.deepcopy(settings['resource_limits'])}
    hypothetical = {**spec, 'status': 'frozen'}
    def preflight():
        return preflight_training_data(training_spec=hypothetical, expected_training_sha256=_digest(hypothetical),
            collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
            source_root=source_root, run_dir=run_dir, source_inventory=source_inventory, allocation_receipt=allocation_receipt,
            receipt=receipt, export=export, challenge_bytes=challenge_bytes, solution_bytes=solution_bytes,
            official_challenge_bytes=official_challenge_bytes, python_executable=python_executable, model_path=model_path)
    def canary():
        return review_reload_attempt(run_dir, canary=canary_spec, expected_canary_sha256=expected_canary_sha256,
                                     expected_collection_sha256=expected_collection_sha256)
    checked, observed = preflight(), canary()
    if (not isinstance(checked, dict) or checked.get('status') != 'training_data_consistent_unverified' or
            checked.get('training_allowed') is not False or checked.get('training_sha256') != _digest(hypothetical) or
            checked.get('solution_sha256') != data_spec['solution_sha256'] or
            checked.get('dataset_manifest_sha256') != plan['dataset_manifest_sha256'] or
            checked.get('attempted_tasks') != len(collection_spec['selected_task_sha256']) or
            type(checked.get('eligible_tasks')) is not int or checked['eligible_tasks'] != count or count <= 0 or
            checked.get('yield_status') != plan['yield_status'] or not isinstance(observed, dict) or
            observed.get('status') != 'canary_reload_observed_unverified' or
            observed.get('reload_verified') is not False or observed.get('promotion_allowed') is not False):
        raise ValueError('training data or canary evidence differs')
    if config_plan() != plan or preflight() != checked or canary() != observed:
        raise ValueError('training proposal evidence changed')
    return {'status': 'real_training_registration_proposed_unverified', 'execution_ready': False,
            'training_allowed': False, 'spec': spec, 'spec_sha256': _digest(spec), 'trainer_config': config,
            'eligible_tasks': count}
