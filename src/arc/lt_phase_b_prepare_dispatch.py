"""Prepare registered private SFT inputs with the installed MLX tokenizer.

No model weights are loaded. Local consistency is not independent attestation
or authorization for training.
"""
import fcntl
import hashlib
import os
import sys
import zipfile
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_collection_preflight import preflight_collection
from src.arc.lt_phase_b_dispatch import _private_run_directory, _read_object
from src.arc.lt_phase_b_prepare_sft import prepare_sft
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_training_config import _private_file
from src.arc.rsi_model_identity import _json_file


def _tokenizer_runtime(model_path):
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.utils import load_tokenizer
    from src.arc.mlx_chat_template import install_training_template_policy
    install_training_template_policy('off')
    return load_tokenizer(model_path), lambda example, tokenizer: ChatDataset([example], tokenizer, mask_prompt=True)


def require_preparation_receipt(directory, *, data_sha256, collection_sha256):
    """Replay private preparation receipts without claiming attestation."""
    directory = Path(directory)
    def read(name):
        path = directory / name
        try:
            return _json_file(path, _private_file(path))
        except OSError as exc:
            raise ValueError('SFT preparation artifact unavailable') from exc
    marker = {'data_sha256': data_sha256, 'collection_sha256': collection_sha256}
    if read('sft-dispatch-started.json') != marker:
        raise ValueError('SFT preparation start identity differs')
    summary = read('sft-preparation.json')
    eligibility = read('eligibility.json')
    if (summary.get('data_sha256') != data_sha256 or summary.get('training_allowed') is not False or
            summary.get('status') not in ('private_sft_prepared_unverified', 'insufficient_verified_self_solutions') or
            eligibility.get('data_sha256') != data_sha256 or eligibility.get('collection_sha256') != collection_sha256 or
            not isinstance(eligibility.get('records'), list)):
        raise ValueError('SFT preparation artifacts differ')
    manifest = directory / 'sft-data/manifest.json'
    if summary['status'] == 'private_sft_prepared_unverified':
        try:
            manifest_sha = _private_file(manifest)
        except OSError as exc:
            raise ValueError('SFT manifest unavailable') from exc
    else:
        if manifest.exists() or manifest.is_symlink(): raise ValueError('unexpected zero-yield manifest')
        manifest_sha = None
    expected = {'marker': marker, 'preparation_sha256': _private_file(directory / 'sft-preparation.json'),
                'eligibility_sha256': _private_file(directory / 'eligibility.json'), 'manifest_sha256': manifest_sha}
    if read('sft-dispatch-complete.json') != expected:
        raise ValueError('SFT preparation completion differs')
    return summary


def run_prepare_sft(*, source_root, run_dir, expected_data_sha256, expected_collection_sha256,
                    challenge_path, solution_path, official_archive, python_executable, model_path):
    root, directory = _private_run_directory(source_root, run_dir)
    try:
        if not os.path.samefile(sys.executable, python_executable):
            raise ValueError('preparation must use the registered interpreter')
    except OSError as exc:
        raise ValueError('registered interpreter unavailable') from exc
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('sft-dispatch-started.json', 'sft-dispatch-complete.json'):
            if (directory / name).exists() or (directory / name).is_symlink():
                raise ValueError('SFT dispatch requires reconciliation')
        names = ('data-spec', 'spec', 'inventory', 'allocation', 'receipt', 'export')
        def registrations():
            return {name: _read_object(directory / (name + '.json')) for name in names}
        inputs = registrations()
        data, collection = inputs['data-spec'], inputs['spec']
        if (data.get('protocol') != 'lt-phase-b-data-v1' or data.get('status') != 'frozen' or
                _digest(data) != expected_data_sha256 or data.get('collection_sha256') != expected_collection_sha256 or
                data.get('run_id') != collection.get('run_id') or
                data.get('source_code_sha256') != collection.get('source_code_sha256')):
            raise ValueError('data registration differs')
        def read_data():
            with zipfile.ZipFile(official_archive) as archive:
                info = archive.getinfo('arc-agi_evaluation_challenges.json')
                if info.file_size > 16 * 1024**2: raise ValueError('official challenge inventory too large')
                official = archive.read(info)
            return Path(challenge_path).read_bytes(), Path(solution_path).read_bytes(), official
        challenge, solutions, official = read_data()
        if hashlib.sha256(solutions).hexdigest() != data.get('solution_sha256'):
            raise ValueError('registered solutions differ')
        checked = preflight_collection(source_root=root, spec=collection, expected_run_sha256=expected_collection_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'], receipt=inputs['receipt'],
            export=inputs['export'], challenge_bytes=challenge, official_challenge_bytes=official,
            python_executable=python_executable, model_path=model_path)
        if checked != {'status': 'preflight_consistent_unverified', 'execution_ready': False,
                       'run_sha256': expected_collection_sha256, 'selected_tasks': len(collection['selected_task_sha256']),
                       'evidence_type': 'local_files_and_caller_supplied_registration'}:
            raise ValueError('collection preflight differs')
        marker = {'data_sha256': expected_data_sha256, 'collection_sha256': expected_collection_sha256}
        _create(directory / 'sft-dispatch-started.json', marker)
        tokenizer, factory = _tokenizer_runtime(model_path)
        result = prepare_sft(data_spec=data, expected_data_sha256=expected_data_sha256, collection_spec=collection,
            expected_collection_sha256=expected_collection_sha256, source_root=root, run_dir=directory,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'], receipt=inputs['receipt'],
            export=inputs['export'], challenge_bytes=challenge, solution_bytes=solutions, official_challenge_bytes=official,
            python_executable=python_executable, model_path=model_path, tokenizer=tokenizer, dataset_factory=factory, _locked_fd=lockfd)
        if registrations() != inputs or read_data() != (challenge, solutions, official):
            raise ValueError('SFT preparation inputs changed; reconcile artifacts')
        final_check = preflight_collection(source_root=root, spec=collection, expected_run_sha256=expected_collection_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'], receipt=inputs['receipt'],
            export=inputs['export'], challenge_bytes=challenge, official_challenge_bytes=official,
            python_executable=python_executable, model_path=model_path)
        if final_check != checked:
            raise ValueError('SFT preparation postflight differs; reconcile artifacts')
        manifest = directory / 'sft-data/manifest.json'
        completion = {'marker': marker, 'preparation_sha256': _private_file(directory / 'sft-preparation.json'),
                      'eligibility_sha256': _private_file(directory / 'eligibility.json'),
                      'manifest_sha256': _private_file(manifest) if result['status'] == 'private_sft_prepared_unverified' else None}
        _create(directory / 'sft-dispatch-complete.json', completion)
        return result
    finally:
        os.close(lockfd)
