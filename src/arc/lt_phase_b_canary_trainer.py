"""Instrument the installed MLX-LM CLI around a fixed synthetic canary step.

This does not verify a frozen registration, supervise a process, attest model or
runtime identity, or authorize execution. Run only under a separately verified
parent that records resources, cleanup, loss, and an independent adapter reload.
"""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from src.arc.lt_phase_b_canary_config import build_canary_config
from src.arc.lt_phase_b_tensor_step import observe_one_step
from src.arc.rsi_model_identity import _hash_file, _json_file


def _private_file(path):
    if path.is_symlink():
        raise ValueError('canary file may not be a symlink')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('canary file must be private')


def _validate_dataset(data_dir, expected):
    if set(os.listdir(data_dir)) != {'train.jsonl'}:
        raise ValueError('canary dataset directory differs')
    path = data_dir / 'train.jsonl'
    _private_file(path)
    digest = _hash_file(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest or raw != expected.encode():
        raise ValueError('canary dataset differs')


def run_instrumented_cli(*, config_path, expected_config_sha256, evidence_dir, trainer_module,
                         install_policy, flatten, evaluate):
    config_path = Path(config_path)
    _private_file(config_path)
    config = _json_file(config_path, expected_config_sha256)
    try:
        plan = build_canary_config(model_dir=config['model'], data_dir=config['data'], adapter_dir=config['adapter_path'])
    except KeyError as exc:
        raise ValueError('canary config is incomplete') from exc
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (canonical(config) != canonical(plan['trainer_config']) or not callable(getattr(trainer_module, 'main', None)) or
            not callable(getattr(trainer_module, 'train', None)) or not callable(install_policy)):
        raise ValueError('canary configuration or trainer differs')
    if Path(evidence_dir).is_symlink():
        raise ValueError('canary evidence directory may not be a symlink')
    model, data, adapter, evidence = (Path(config['model']), Path(config['data']),
                                      Path(config['adapter_path']), Path(evidence_dir).resolve(strict=True))
    if any(evidence == p or evidence in p.parents or p in evidence.parents for p in (model, data, adapter)):
        raise ValueError('canary evidence path overlaps inputs or output')
    _validate_dataset(data, plan['dataset_jsonl'])
    original_train, original_argv = trainer_module.train, sys.argv
    calls, observed = 0, []

    def instrumented(*args, **kwargs):
        nonlocal calls
        calls += 1
        if args or calls != 1 or 'model' not in kwargs or 'args' not in kwargs:
            raise ValueError('unexpected trainer invocation')
        settings = kwargs['args']
        required = {'iters': 1, 'batch_size': 1, 'max_seq_length': 512, 'steps_per_report': 1,
                    'steps_per_save': 1, 'grad_accumulation_steps': 1, 'val_batches': 0}
        if (any(type(getattr(settings, k, None)) is not int or getattr(settings, k) != v
                for k, v in required.items()) or
                Path(getattr(settings, 'adapter_file', '')) != adapter / 'adapters.safetensors' or
                kwargs.get('val_dataset') is None or len(kwargs['val_dataset']) != 0):
            raise ValueError('trainer step settings differ')
        result = observe_one_step(kwargs['model'], output_dir=evidence,
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
        raise ValueError('canary trainer did not finish one observed step')
    if _hash_file(config_path) != expected_config_sha256:
        raise ValueError('canary config changed during training')
    _validate_dataset(data, plan['dataset_jsonl'])
    return observed[0]
