"""Execute a locally bound synthetic 14B canary in a supervised child.

Local descriptor consistency is not independent preregistration, process
isolation, resource supervision, or authorization. The parent must enforce the
full Phase B protocol and retain loss, cleanup and separate reload evidence.
"""
import json
import re
import sys
from pathlib import Path

from src.arc.lt_phase_b_canary_config import build_canary_config
from src.arc.lt_phase_b_canary_trainer import _private_file, _validate_dataset, run_instrumented_cli
from src.arc.lt_phase_b_registered_worker import _digest, validate_descriptor
from src.arc.lt_source import verify_source_tree
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_model_identity import _json_file, verify_model_manifest
from src.arc.rsi_runtime_identity import verify_runtime_identity


def _trainer_runtime():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import lora
    from src.arc.mlx_chat_template import install_training_template_policy
    mx.set_memory_limit(48 * 1024**3)
    return lora, install_training_template_policy, tree_flatten, mx.eval


def validate_canary_descriptor(*, canary_spec, expected_canary_sha256, collection_spec, expected_collection_sha256,
                               source_inventory, allocation_receipt, receipt, config_path):
    settings, model = validate_descriptor(spec=collection_spec, expected_run_sha256=expected_collection_sha256,
        source_inventory=source_inventory, receipt=receipt, allocation_receipt=allocation_receipt)
    keys = {'protocol', 'status', 'run_id', 'collection_sha256', 'source_code_sha256', 'runtime_manifest',
            'environment_manifest', 'model_manifest', 'config_sha256', 'dataset_sha256', 'thinking_mode',
            'timeout_seconds', 'resource_limits'}
    valid_hash = lambda x: isinstance(x, str) and bool(re.fullmatch(r'[0-9a-f]{64}', x))
    canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (not isinstance(canary_spec, dict) or set(canary_spec) != keys or
            canary_spec.get('protocol') != 'lt-phase-b-canary-v1' or canary_spec.get('status') != 'frozen' or
            not isinstance(canary_spec.get('run_id'), str) or
            not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', canary_spec['run_id']) or
            not valid_hash(expected_canary_sha256) or _digest(canary_spec) != expected_canary_sha256 or
            canary_spec.get('collection_sha256') != expected_collection_sha256 or
            canary_spec.get('source_code_sha256') != collection_spec['source_code_sha256'] or
            canary_spec.get('thinking_mode') != 'off' or type(canary_spec.get('timeout_seconds')) is not int or
            canary_spec['timeout_seconds'] != settings['task_timeout_seconds'] or
            canonical(canary_spec.get('resource_limits')) != canonical(settings['resource_limits']) or
            any(canonical(canary_spec.get(k)) != canonical(collection_spec[k])
                for k in ('runtime_manifest', 'environment_manifest', 'model_manifest')) or
            not valid_hash(canary_spec.get('config_sha256')) or not valid_hash(canary_spec.get('dataset_sha256'))):
        raise ValueError('invalid canary descriptor')
    config_path = Path(config_path)
    _private_file(config_path)
    config = _json_file(config_path, canary_spec['config_sha256'])
    try:
        plan = build_canary_config(model_dir=config['model'], data_dir=config['data'], adapter_dir=config['adapter_path'])
    except KeyError as exc:
        raise ValueError('canary config incomplete') from exc
    if canonical(config) != canonical(plan['trainer_config']) or plan['dataset_sha256'] != canary_spec['dataset_sha256']:
        raise ValueError('canary config or dataset identity differs')
    _validate_dataset(Path(config['data']), plan['dataset_jsonl'])
    return settings, model, config


def run_registered_canary(*, canary_spec, expected_canary_sha256, collection_spec, expected_collection_sha256,
                          source_inventory, source_root, allocation_receipt, receipt, config_path, evidence_dir):
    settings, model, config = validate_canary_descriptor(canary_spec=canary_spec,
        expected_canary_sha256=expected_canary_sha256, collection_spec=collection_spec,
        expected_collection_sha256=expected_collection_sha256, source_inventory=source_inventory,
        allocation_receipt=allocation_receipt, receipt=receipt, config_path=config_path)
    verify_runtime_identity(sys.executable, canary_spec['runtime_manifest'])
    verify_environment(sys.executable, canary_spec['environment_manifest'])
    verify_model_manifest(config['model'], model, model_key='14b')
    verify_source_tree(source_root, source_inventory)
    trainer, policy, flatten, evaluate = _trainer_runtime()
    observed = run_instrumented_cli(config_path=config_path, expected_config_sha256=canary_spec['config_sha256'],
        evidence_dir=evidence_dir, trainer_module=trainer, install_policy=policy, flatten=flatten, evaluate=evaluate)
    if (not isinstance(observed, dict) or observed.get('status') != 'tensor_change_consistent_unverified' or
            observed.get('training_verified') is not False):
        raise ValueError('canary tensor evidence differs')
    verify_runtime_identity(sys.executable, canary_spec['runtime_manifest'])
    verify_environment(sys.executable, canary_spec['environment_manifest'])
    verify_model_manifest(config['model'], model, model_key='14b')
    verify_source_tree(source_root, source_inventory)
    return {**observed, 'canary_sha256': expected_canary_sha256, 'collection_sha256': expected_collection_sha256,
            'canary_run_id': canary_spec['run_id'], 'promotion_allowed': False}
