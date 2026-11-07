"""Instrument a real-data trainer callback without registering or launching a run.

Caller-owned supervision must bind the allocation, canary, model and runtime,
monitor resources and retain failure evidence. Local tensor observations are not
independent attestation, adapter reload evidence, or training authorization.
"""
import json
import os
import stat
import sys
from pathlib import Path

from src.arc.lt_phase_b_canary_trainer import _private_file as _private_config
from src.arc.lt_phase_b_tensor_step import observe_training
from src.arc.lt_phase_b_training_config import build_training_config, _private_file
from src.arc.rsi_model_identity import _hash_file, _json_file


def run_real_instrumented_cli(*, source_root, run_dir, model_dir, expected_source, config_path,
                              expected_config_sha256, evidence_dir, trainer_module, install_policy,
                              flatten, evaluate):
    config_path = Path(config_path)
    _private_config(config_path)
    config = _json_file(config_path, expected_config_sha256)
    plan = build_training_config(source_root=source_root, run_dir=run_dir, model_dir=model_dir,
                                 expected_source=expected_source)
    canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (canonical(config) != canonical(plan['trainer_config']) or
            not callable(getattr(trainer_module, 'main', None)) or
            not callable(getattr(trainer_module, 'train', None)) or not callable(install_policy)):
        raise ValueError('training configuration or trainer differs')
    data, model, adapter = (Path(config[k]) for k in ('data', 'model', 'adapter_path'))
    evidence = Path(evidence_dir)
    if evidence.is_symlink():
        raise ValueError('training evidence directory is a symlink')
    evidence = evidence.resolve(strict=True)
    info = evidence.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or
            any(evidence == p or evidence in p.parents or p in evidence.parents for p in (data, model, adapter)) or
            any(evidence.iterdir())):
        raise ValueError('invalid training evidence directory')
    manifest_path = data / 'manifest.json'
    manifest = _json_file(manifest_path, plan['dataset_manifest_sha256'])
    hashes = {'manifest.json': plan['dataset_manifest_sha256'], 'train.jsonl': manifest['train_jsonl_sha256']}
    if manifest['valid_count']:
        hashes['valid.jsonl'] = manifest['valid_jsonl_sha256']
    original_train, original_argv = trainer_module.train, sys.argv
    calls, observed = 0, []

    def instrumented(*args, **kwargs):
        nonlocal calls
        calls += 1
        if args or calls != 1 or 'model' not in kwargs or 'args' not in kwargs:
            raise ValueError('unexpected trainer invocation')
        settings = kwargs['args']
        required = {'iters': 20, 'batch_size': 1, 'max_seq_length': 512, 'steps_per_report': 1,
                    'steps_per_save': 20, 'grad_accumulation_steps': 1, 'val_batches': config['val_batches']}
        if (any(type(getattr(settings, k, None)) is not int or getattr(settings, k) != v
                for k, v in required.items()) or
                Path(getattr(settings, 'adapter_file', '')) != adapter / 'adapters.safetensors' or
                kwargs.get('train_dataset') is None or len(kwargs['train_dataset']) != manifest['train_count'] or
                kwargs.get('val_dataset') is None or len(kwargs['val_dataset']) != manifest['valid_count']):
            raise ValueError('trainer settings or dataset lengths differ')
        result = observe_training(kwargs['model'], output_dir=evidence,
            run_step=lambda: original_train(**kwargs), flatten=flatten, evaluate=evaluate)
        observed.append(result)

    try:
        install_policy('off')
        trainer_module.train = instrumented
        sys.argv = ['mlx_lm.lora', '--config', str(config_path)]
        trainer_module.main()
    finally:
        trainer_module.train = original_train
        sys.argv = original_argv
    if calls != 1 or len(observed) != 1:
        raise ValueError('trainer did not finish the observed callback')
    if _hash_file(config_path) != expected_config_sha256 or set(os.listdir(data)) != set(hashes):
        raise ValueError('training configuration or dataset files changed')
    if any(_private_file(data / name) != digest for name, digest in hashes.items()):
        raise ValueError('training dataset changed')
    return observed[0]
