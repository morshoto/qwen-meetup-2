"""Create non-executing review material for the proposed MLX preflight."""
import hashlib
import os
from pathlib import Path

import yaml

from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_comparison_freeze import _hash
from src.arc.rsi_model_identity import _hash_file
from src.arc.rsi_preflight_identity import verify_preflight_local_identities
from src.arc.rsi_slot_lifecycle import _create, _directory


def _write_config(directory, raw):
    fd = os.open('mlx_lora.yml', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(directory)


def prepare_preflight_bundle(*, root, expected_spec_sha256, manifest_bytes, source_root, python_executable):
    if not _hash(expected_spec_sha256):
        raise ValueError('invalid preflight proposal digest')
    with _directory(root) as directory:
        spec_raw, spec = _read(directory, 'preflight.proposal.json')
    if (hashlib.sha256(spec_raw).hexdigest() != expected_spec_sha256 or
            spec.get('status') != 'preflight_proposed_unverified' or spec.get('execution_ready') is not False or
            spec.get('training_allowed') is not False or not isinstance(spec.get('config'), dict) or
            type(spec['config'].get('iters')) is not int or spec['config']['iters'] != 1 or
            spec['config'].get('mask_prompt') is not True or
            spec['config'].get('data') != str((Path(root) / 'preflight-canary.quarantine').resolve())):
        raise ValueError('invalid preflight proposal')
    checked = verify_preflight_local_identities(root=root, expected_spec_sha256=expected_spec_sha256,
        manifest_bytes=manifest_bytes, source_root=source_root, python_executable=python_executable)
    if (checked.get('status') != 'preflight_local_identities_consistent' or checked.get('execution_ready') is not False or
            checked.get('independently_verified') is not False or checked.get('proposal_sha256') != expected_spec_sha256):
        raise ValueError('local preflight identity verification differs')
    source = Path(source_root).resolve()
    python = Path(os.path.abspath(Path(python_executable).expanduser()))
    try:
        interpreter = python.resolve(strict=True)
        if not interpreter.is_file() or not os.access(python, os.X_OK):
            raise ValueError('preflight interpreter unavailable')
        python_hash = _hash_file(interpreter)
        wrapper = source / 'src' / 'cli' / 'p3b_train.py'
        wrapper_hash = _hash_file(wrapper)
    except OSError as exc:
        raise ValueError('preflight executable or wrapper unavailable') from exc
    config_raw = yaml.safe_dump(spec['config'], sort_keys=True).encode()
    if len(config_raw) > 65536 or yaml.safe_load(config_raw) != spec['config']:
        raise ValueError('preflight configuration cannot be frozen')
    bundle_path = (Path(root) / 'preflight.launch.quarantine').resolve()
    launch = {'status': 'preflight_launch_proposed_unverified', 'execution_ready': False,
              'training_allowed': False, 'proposal_sha256': expected_spec_sha256,
              'config_sha256': hashlib.sha256(config_raw).hexdigest(), 'python_sha256': python_hash,
              'wrapper_sha256': wrapper_hash, 'working_directory': str(source),
              'argv': [str(python), '-u', '-B', '-m', 'src.cli.p3b_train', '--config',
                       str(bundle_path / 'mlx_lora.yml'), '--thinking-mode', 'off'],
              'environment': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                              'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'},
              'evidence_type': 'local_files_and_caller_supplied_manifests'}
    with _directory(root) as parent:
        current, _ = _read(parent, 'preflight.proposal.json')
        if current != spec_raw or python.resolve(strict=True) != interpreter or _hash_file(interpreter) != python_hash or _hash_file(wrapper) != wrapper_hash:
            raise ValueError('preflight launch inputs changed')
        os.mkdir('preflight.launch.quarantine', 0o700, dir_fd=parent)
        os.fsync(parent)
        directory = os.open('preflight.launch.quarantine', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            os.fchmod(directory, 0o700)
            _write_config(directory, config_raw)
            _create(directory, 'launch.json', launch)
        finally:
            os.close(directory)
    return {'status': launch['status'], 'execution_ready': False, 'training_allowed': False,
            'config_sha256': launch['config_sha256']}
