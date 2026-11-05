"""Create non-executing, model-specific synthetic canary review material."""
import copy
import hashlib
import json
import os
from pathlib import Path

from src.arc.rsi_model_identity import _hash_file
from src.arc.rsi_canary_containment import fork_restricted_command
from src.arc.rsi_slot_lifecycle import _create, _directory
from src.arc.rsi_technical_canary import verify_technical_canary_identities


def prepare_technical_canary_bundle(*, root, proposal, manifest_bytes, source_root, python_executable,
                                    forbidden_tasks, tokenizer, render, load_local_dataset):
    """Freeze local proposal material for review; never launch a process."""
    if not isinstance(proposal, dict) or proposal.get('model_key') not in ('14b', '32b'):
        raise ValueError('invalid technical canary proposal')
    original = copy.deepcopy(proposal)
    key = original['model_key']
    verify_args = dict(proposal=proposal, manifest_bytes=manifest_bytes, source_root=source_root,
                       python_executable=python_executable, root=root, forbidden_tasks=forbidden_tasks,
                       tokenizer=tokenizer, render=render, load_local_dataset=load_local_dataset)

    def verify():
        result = verify_technical_canary_identities(**verify_args)
        if (result.get('status') != 'technical_canary_local_identities_consistent' or
                result.get('model_key') != key or result.get('execution_ready') is not False or
                result.get('training_allowed') is not False or result.get('promotion_allowed') is not False or
                result.get('independently_verified') is not False or proposal != original):
            raise ValueError('technical canary identity review differs')
    verify()
    source = Path(source_root).resolve(strict=True)
    python = Path(os.path.abspath(Path(python_executable).expanduser()))
    try:
        interpreter = python.resolve(strict=True)
        wrapper = source / 'src' / 'cli' / 'p3b_train.py'
        if not interpreter.is_file() or not os.access(python, os.X_OK):
            raise ValueError('technical canary interpreter unavailable')
        python_hash, wrapper_hash = _hash_file(interpreter), _hash_file(wrapper)
    except OSError as exc:
        raise ValueError('technical canary executable unavailable') from exc
    raw = json.dumps(original['config'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > 65536:
        raise ValueError('technical canary config too large')
    # JSON is a YAML subset accepted by the pinned MLX-LM config parser.
    bundle = (Path(root) / f'technical-canary-{key}.launch.quarantine').resolve()
    launch = {'status': 'technical_canary_launch_proposed_unverified', 'model_key': key,
              'execution_ready': False, 'training_allowed': False, 'promotion_allowed': False,
              'independently_verified': False, 'config_sha256': hashlib.sha256(raw).hexdigest(),
              'proposal_sha256': hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'),
                                                       allow_nan=False).encode()).hexdigest(),
              'python_sha256': python_hash, 'wrapper_sha256': wrapper_hash, 'working_directory': str(source),
              'argv': [str(python), '-u', '-B', '-m', 'src.cli.p3b_train', '--config',
                       str(bundle / 'mlx_lora.json'), '--thinking-mode', 'off'],
              'environment': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                              'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'},
              'evidence_type': 'local_files_and_caller_supplied_manifests'}
    launch['fork_restricted_argv'] = fork_restricted_command(launch['argv'])['argv']
    launch['containment_verified'] = False
    verify()
    if python.resolve(strict=True) != interpreter or _hash_file(interpreter) != python_hash or _hash_file(wrapper) != wrapper_hash:
        raise ValueError('technical canary launch inputs changed')
    with _directory(root) as parent:
        name = f'technical-canary-{key}.launch.quarantine'
        os.mkdir(name, 0o700, dir_fd=parent)
        os.fsync(parent)
        directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            os.fchmod(directory, 0o700)
            # Existing helper uses a fixed filename; write the JSON config explicitly.
            fd = os.open('mlx_lora.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, 'wb', closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                os.fsync(fd)
            finally:
                os.close(fd)
            for manifest_name, manifest_raw in manifest_bytes.items():
                fd = os.open(manifest_name + '.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                try:
                    with os.fdopen(fd, 'wb', closefd=False) as stream:
                        stream.write(manifest_raw)
                        stream.flush()
                    os.fsync(fd)
                finally:
                    os.close(fd)
            _create(directory, 'proposal.json', original)
            _create(directory, 'launch.json', launch)
            os.fsync(directory)
        finally:
            os.close(directory)
    return {'status': launch['status'], 'execution_ready': False, 'training_allowed': False,
            'model_key': key, 'config_sha256': launch['config_sha256']}


def verify_technical_canary_bundle(*, root, proposal, manifest_bytes, source_root, python_executable,
                                   forbidden_tasks, tokenizer, render, load_local_dataset, expected_launch_sha256):
    """Recheck saved launch material without starting or authorizing a process."""
    import re
    from src.arc.rsi_comparison_audit import _read

    if (not isinstance(expected_launch_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_launch_sha256) or
            not isinstance(proposal, dict) or proposal.get('model_key') not in ('14b', '32b') or
            not isinstance(manifest_bytes, dict) or set(manifest_bytes) !=
            {'model_manifest', 'runtime_manifest', 'environment_manifest', 'source_inventory'}):
        raise ValueError('invalid canary bundle verification inputs')
    original = copy.deepcopy(proposal)
    key = original['model_key']
    root = Path(root).expanduser().resolve()
    bundle = root / f'technical-canary-{key}.launch.quarantine'
    source = Path(source_root).resolve(strict=True)
    python = Path(os.path.abspath(Path(python_executable).expanduser()))
    verify_args = dict(proposal=proposal, manifest_bytes=manifest_bytes, source_root=source_root,
                       python_executable=python_executable, root=root, forbidden_tasks=forbidden_tasks,
                       tokenizer=tokenizer, render=render, load_local_dataset=load_local_dataset)
    with _directory(root), _directory(bundle) as directory:
        before = {}
        for name in ['launch.json', 'proposal.json', 'mlx_lora.json',
                     *(item + '.json' for item in sorted(manifest_bytes))]:
            before[name], _ = _read(directory, name)
        launch = json.loads(before['launch.json'])
        saved_proposal = json.loads(before['proposal.json'])
        config_raw = json.dumps(original['config'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if (hashlib.sha256(before['launch.json']).hexdigest() != expected_launch_sha256 or
                saved_proposal != original or before['mlx_lora.json'] != config_raw or
                any(before[name + '.json'] != raw for name, raw in manifest_bytes.items())):
            raise ValueError('canary bundle contents differ')
        interpreter = python.resolve(strict=True)
        if not interpreter.is_file() or not os.access(python, os.X_OK):
            raise ValueError('canary interpreter unavailable')
        wrapper = source / 'src' / 'cli' / 'p3b_train.py'
        argv = [str(python), '-u', '-B', '-m', 'src.cli.p3b_train', '--config',
                str(bundle / 'mlx_lora.json'), '--thinking-mode', 'off']
        expected = {'status':'technical_canary_launch_proposed_unverified', 'model_key':key,
                    'execution_ready':False, 'training_allowed':False, 'promotion_allowed':False,
                    'independently_verified':False, 'config_sha256':hashlib.sha256(config_raw).hexdigest(),
                    'proposal_sha256':hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'),
                                                               allow_nan=False).encode()).hexdigest(),
                    'python_sha256':_hash_file(interpreter), 'wrapper_sha256':_hash_file(wrapper),
                    'working_directory':str(source), 'argv':argv,
                    'environment':{'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                                   'TOKENIZERS_PARALLELISM':'false', 'PYTHONDONTWRITEBYTECODE':'1'},
                    'fork_restricted_argv':fork_restricted_command(argv)['argv'], 'containment_verified':False,
                    'evidence_type':'local_files_and_caller_supplied_manifests'}
        if launch != expected or any(launch.get(k) is not False for k in
                ('execution_ready','training_allowed','promotion_allowed','independently_verified','containment_verified')):
            raise ValueError('canary launch proposal differs')
        checked = verify_technical_canary_identities(**verify_args)
        if (checked.get('status') != 'technical_canary_local_identities_consistent' or
                checked.get('model_key') != key or any(checked.get(k) is not False for k in
                ('execution_ready','training_allowed','promotion_allowed','independently_verified'))):
            raise ValueError('canary identity verification differs')
        if (proposal != original or python.resolve(strict=True) != interpreter or
                _hash_file(interpreter) != expected['python_sha256'] or _hash_file(wrapper) != expected['wrapper_sha256'] or
                any(_read(directory, name)[0] != raw for name, raw in before.items())):
            raise ValueError('canary bundle changed during verification')
    return {'status':'technical_canary_bundle_consistent_unverified', 'model_key':key,
            'launch_sha256':expected_launch_sha256, 'execution_ready':False, 'training_allowed':False,
            'promotion_allowed':False, 'independently_verified':False}
