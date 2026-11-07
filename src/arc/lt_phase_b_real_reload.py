"""Reload a real-data adapter with an unscored sample under parent supervision.

Local observations do not independently attest training, prove process cleanup,
authorize promotion or establish ARC improvement.
"""
import hashlib
import math
import re
import sys
from pathlib import Path

from src.arc.lt_direct_format import normalize_direct
from src.arc.lt_phase_b_backend import PhaseBBackend
from src.arc.lt_phase_b_canary_sample import SAMPLE_PROMPT
from src.arc.lt_phase_b_dispatch import _private_run_directory, _read_object
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_reload_dispatch import review_reload_attempt
from src.arc.lt_phase_b_training_config import _private_file
from src.arc.lt_phase_b_training_dispatch import review_real_training_attempt
from src.arc.lt_source import verify_source_tree
from src.arc.rsi_environment_identity import verify_environment
from src.arc.rsi_model_identity import _json_file, verify_model_manifest
from src.arc.rsi_runtime_identity import verify_runtime_identity


def run_real_reload(*, training_spec, expected_training_sha256, canary_spec, expected_canary_sha256,
                    collection_spec, expected_collection_sha256, source_inventory, source_root, run_dir,
                    allocation_receipt, receipt, config_path, backend_factory=PhaseBBackend):
    _, directory = _private_run_directory(source_root, run_dir)
    _, model = validate_canary_identity(canary_spec=canary_spec, expected_canary_sha256=expected_canary_sha256,
        collection_spec=collection_spec, expected_collection_sha256=expected_collection_sha256,
        source_inventory=source_inventory, allocation_receipt=allocation_receipt, receipt=receipt)
    if (not isinstance(training_spec, dict) or not isinstance(expected_training_sha256, str) or
            not re.fullmatch(r'[0-9a-f]{64}', expected_training_sha256) or
            _digest(training_spec) != expected_training_sha256 or training_spec.get('status') != 'frozen' or
            training_spec.get('protocol') != 'lt-phase-b-real-training-v1' or
            training_spec.get('run_id') != collection_spec['run_id'] or
            directory.name != 'lt-phase-b-' + collection_spec['run_id'] or
            training_spec.get('collection_sha256') != expected_collection_sha256 or
            training_spec.get('canary_sha256') != expected_canary_sha256 or
            training_spec.get('source_code_sha256') != collection_spec['source_code_sha256'] or
            not callable(backend_factory)):
        raise ValueError('real reload identity differs')
    def review():
        training = review_real_training_attempt(directory, training=training_spec,
                                                expected_training_sha256=expected_training_sha256)
        canary = review_reload_attempt(directory, canary=canary_spec, expected_canary_sha256=expected_canary_sha256,
                                       expected_collection_sha256=expected_collection_sha256)
        if (_read_object(directory / 'training-summary.json') != training or
                training.get('status') != 'real_training_observed_reload_pending' or
                training.get('training_verified') is not False or training.get('promotion_allowed') is not False or
                canary.get('status') != 'canary_reload_observed_unverified' or
                canary.get('reload_verified') is not False or canary.get('promotion_allowed') is not False):
            raise ValueError('prior training or canary evidence differs')
        return training, canary
    training, canary = review()
    config_path = Path(config_path)
    if config_path.resolve() != (directory / 'training-config.json').resolve() or _private_file(config_path) != training_spec.get('config_sha256'):
        raise ValueError('training config identity differs')
    config = _json_file(config_path, training_spec['config_sha256'])
    try:
        model_path, adapter = Path(config['model']), Path(config['adapter_path'])
        if adapter.is_symlink() or adapter.resolve(strict=True) != (directory / 'gen1-adapter').resolve(strict=True):
            raise ValueError('real adapter path differs')
        digest = training['adapter']['adapter_sha256']
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('real adapter hash invalid')
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('real reload paths or artifact unavailable') from exc
    def verify():
        verify_runtime_identity(sys.executable, collection_spec['runtime_manifest'])
        verify_environment(sys.executable, collection_spec['environment_manifest'])
        verify_model_manifest(model_path, model, model_key='14b')
        verify_source_tree(source_root, source_inventory)
    verify()
    settings = canary_spec['reload_settings']
    backend = backend_factory(model_path=model_path, model_manifest=model, adapter_state='on', adapter_path=adapter,
        adapter_sha256=digest, seed=settings['seed'], max_tokens=settings['max_tokens'], temperature=settings['temperature'])
    if getattr(backend, 'adapter_state', None) != 'on' or getattr(backend, 'adapter_sha256', None) != digest:
        raise ValueError('loaded adapter identity differs')
    backend.generate(SAMPLE_PROMPT, kind='direct', candidate=0)
    raw, usage = getattr(backend, 'raw', None), getattr(backend, 'stats', None)
    if (not isinstance(raw, str) or not raw or len(raw.encode()) > 32768 or not isinstance(usage, dict) or
            type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] < 0 or
            type(usage.get('completion_tokens')) is not int or not 0 < usage['completion_tokens'] <= settings['max_tokens'] or
            type(usage.get('peak_memory_gb')) not in (int, float) or not math.isfinite(usage['peak_memory_gb']) or
            not 0 <= usage['peak_memory_gb'] <= 48 * 1024**3 / 1e9 or usage.get('finish_reason') not in ('stop', 'length')):
        raise ValueError('invalid real adapter sample')
    try:
        normalize_direct(raw, 1)
        valid = True
    except ValueError:
        valid = False
    verify()
    if review() != (training, canary) or _private_file(config_path) != training_spec['config_sha256']:
        raise ValueError('training or canary evidence changed during reload')
    return {'status': 'real_reload_observed_unverified', 'reload_verified': False, 'promotion_allowed': False,
            'training_sha256': expected_training_sha256, 'collection_sha256': expected_collection_sha256,
            'canary_sha256': expected_canary_sha256, 'adapter_sha256': digest,
            'sample_prompt_sha256': hashlib.sha256(SAMPLE_PROMPT.encode()).hexdigest(),
            'sample_raw': raw, 'sample_format_valid': valid, 'usage': usage}
