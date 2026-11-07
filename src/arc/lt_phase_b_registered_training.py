"""Bind a real-data trainer callback to local registrations and canary evidence.

This worker is not a dispatcher: the parent must establish collection and
eligibility provenance, supervise resources and preserve one-shot failures.
Local file consistency does not attest training or authorize a live run.
"""
import json
import re
import sys
from pathlib import Path

from src.arc.lt_phase_b_canary_trainer import _private_file
from src.arc.lt_phase_b_registered_canary import _trainer_runtime, validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_reload_dispatch import review_reload_attempt
from src.arc.lt_phase_b_training_config import build_training_config
from src.arc.lt_phase_b_training_trainer import run_real_instrumented_cli
from src.arc.lt_source import verify_source_tree
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_model_identity import _json_file, verify_model_manifest
from src.arc.rsi_runtime_identity import verify_runtime_identity


def run_registered_training(*, training_spec, expected_training_sha256, canary_spec, expected_canary_sha256,
                            collection_spec, expected_collection_sha256, source_inventory, source_root, run_dir,
                            model_path, allocation_receipt, receipt, config_path, evidence_dir):
    settings, model = validate_canary_identity(canary_spec=canary_spec, expected_canary_sha256=expected_canary_sha256,
        collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
        source_inventory=source_inventory, allocation_receipt=allocation_receipt, receipt=receipt)
    keys = {'protocol', 'status', 'run_id', 'collection_sha256', 'canary_sha256', 'source_code_sha256',
            'dataset_manifest_sha256', 'config_sha256', 'solution_sha256', 'yield_status', 'checkpoint_policy',
            'timeout_seconds', 'resource_limits'}
    valid_hash = lambda x: isinstance(x, str) and bool(re.fullmatch(r'[0-9a-f]{64}', x))
    if (not isinstance(training_spec, dict) or set(training_spec) != keys or
            not valid_hash(expected_training_sha256) or _digest(training_spec) != expected_training_sha256 or
            training_spec.get('protocol') != 'lt-phase-b-real-training-v1' or training_spec.get('status') != 'frozen' or
            training_spec.get('run_id') != collection_spec['run_id'] or
            training_spec.get('collection_sha256') != expected_collection_sha256 or
            training_spec.get('canary_sha256') != expected_canary_sha256 or
            training_spec.get('source_code_sha256') != collection_spec['source_code_sha256'] or
            not valid_hash(training_spec.get('config_sha256')) or not valid_hash(training_spec.get('dataset_manifest_sha256')) or
            not valid_hash(training_spec.get('solution_sha256')) or
            training_spec.get('checkpoint_policy') != 'final_step_only' or
            type(training_spec.get('timeout_seconds')) is not int or
            training_spec['timeout_seconds'] != settings['task_timeout_seconds'] or
            json.dumps(training_spec.get('resource_limits'), sort_keys=True, separators=(',', ':')) !=
            json.dumps(settings['resource_limits'], sort_keys=True, separators=(',', ':'))):
        raise ValueError('invalid real training descriptor')
    source = {'model_id': 'mlx-community/Qwen3-14B-4bit',
              'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
              'source_code_sha256': collection_spec['source_code_sha256'],
              'run_sha256': expected_collection_sha256, 'run_id': collection_spec['run_id']}
    plan = build_training_config(source_root=source_root, run_dir=run_dir, model_dir=model_path, expected_source=source)
    config_path = Path(config_path)
    _private_file(config_path)
    config = _json_file(config_path, training_spec['config_sha256'])
    canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (canonical(config) != canonical(plan['trainer_config']) or
            training_spec['dataset_manifest_sha256'] != plan['dataset_manifest_sha256'] or
            training_spec['yield_status'] != plan['yield_status']):
        raise ValueError('registered training config or dataset differs')
    manifest = _json_file(Path(plan['trainer_config']['data']) / 'manifest.json', plan['dataset_manifest_sha256'])
    task_ids = set(manifest['split_task_ids']['train'] + manifest['split_task_ids']['valid'])
    if not task_ids <= set(collection_spec['selected_task_sha256']):
        raise ValueError('dataset task not in registered training pool')
    def replay():
        return review_reload_attempt(run_dir, canary=canary_spec, expected_canary_sha256=expected_canary_sha256,
                                     expected_collection_sha256=expected_collection_sha256)
    canary = replay()
    if (not isinstance(canary, dict) or canary.get('status') != 'canary_reload_observed_unverified' or
            canary.get('reload_verified') is not False or canary.get('promotion_allowed') is not False):
        raise ValueError('canary reload evidence differs')
    def verify():
        verify_runtime_identity(sys.executable, collection_spec['runtime_manifest'])
        verify_environment(sys.executable, collection_spec['environment_manifest'])
        verify_model_manifest(model_path, model, model_key='14b')
        verify_source_tree(source_root, source_inventory)
    verify()
    trainer, policy, flatten, evaluate = _trainer_runtime()
    observed = run_real_instrumented_cli(source_root=source_root, run_dir=run_dir, model_dir=model_path,
        expected_source=source, config_path=config_path, expected_config_sha256=training_spec['config_sha256'],
        evidence_dir=evidence_dir, trainer_module=trainer, install_policy=policy, flatten=flatten, evaluate=evaluate)
    if (not isinstance(observed, dict) or observed.get('status') != 'tensor_change_consistent_unverified' or
            observed.get('training_verified') is not False):
        raise ValueError('real training tensor evidence differs')
    verify()
    if replay() != canary:
        raise ValueError('canary evidence changed during training')
    return {**observed, 'training_sha256': expected_training_sha256, 'collection_sha256': expected_collection_sha256,
            'canary_sha256': expected_canary_sha256, 'training_run_id': training_spec['run_id'],
            'yield_status': plan['yield_status'], 'promotion_allowed': False}
