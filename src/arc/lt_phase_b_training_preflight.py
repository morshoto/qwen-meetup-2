"""Replay private training eligibility against registered local inputs.

This performs local consistency checks only. The caller must preserve the
frozen registration, supervise the process and establish any independent
attestation required by the protocol; no execution is authorized here.
"""
import hashlib

from src.arc.data import Verifier, validate_grid
from src.arc.lt_attestation import _object
from src.arc.lt_phase_b_collection_preflight import preflight_collection
from src.arc.lt_phase_b_protocol_gate import require_protocol_solutions
from src.arc.lt_phase_b_prepare_dispatch import require_preparation_receipt
from src.arc.lt_phase_b_collection_replay import replay_collection_dataset
from src.arc.lt_phase_b_sft_preflight import _json
from src.arc.lt_phase_b_dispatch import _private_run_directory
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_training_config import _private_file
from src.arc.lt_public import build_public_export, load_public_export
from src.arc.rsi_model_identity import _json_file


def preflight_training_data(*, training_spec, expected_training_sha256, collection_spec, expected_collection_sha256,
                            source_root, run_dir, source_inventory, allocation_receipt, receipt, export,
                            challenge_bytes, solution_bytes, official_challenge_bytes, python_executable, model_path):
    if (not isinstance(training_spec, dict) or training_spec.get('protocol') != 'lt-phase-b-real-training-v1' or
            training_spec.get('status') != 'frozen' or _digest(training_spec) != expected_training_sha256 or
            training_spec.get('collection_sha256') != expected_collection_sha256 or
            not isinstance(solution_bytes, bytes) or
            hashlib.sha256(solution_bytes).hexdigest() != training_spec.get('solution_sha256')):
        raise ValueError('registered training data identity differs')
    _, directory = _private_run_directory(source_root, run_dir)
    if directory.name != 'lt-phase-b-' + str(collection_spec.get('run_id')):
        raise ValueError('run directory identity differs')
    checked = preflight_collection(source_root=source_root, spec=collection_spec, expected_run_sha256=expected_collection_sha256,
        source_inventory=source_inventory, allocation_receipt=allocation_receipt, export=export, receipt=receipt,
        challenge_bytes=challenge_bytes, official_challenge_bytes=official_challenge_bytes,
        python_executable=python_executable, model_path=model_path)
    expected = {'status': 'preflight_consistent_unverified', 'execution_ready': False,
                'run_sha256': expected_collection_sha256, 'selected_tasks': len(collection_spec['selected_task_sha256']),
                'evidence_type': 'local_files_and_caller_supplied_registration'}
    if checked != expected:
        raise ValueError('collection preflight differs')
    require_protocol_solutions(source_root, source_inventory, solution_bytes)
    data_path = directory / 'data-spec.json'
    data_spec = _json_file(data_path, _private_file(data_path))
    if (data_spec.get('protocol') != 'lt-phase-b-data-v1' or data_spec.get('status') != 'frozen' or
            data_spec.get('run_id') != collection_spec['run_id'] or
            data_spec.get('collection_sha256') != expected_collection_sha256 or
            data_spec.get('source_code_sha256') != collection_spec['source_code_sha256'] or
            data_spec.get('solution_sha256') != training_spec['solution_sha256']):
        raise ValueError('SFT data descriptor differs')
    preparation = require_preparation_receipt(directory, data_sha256=_digest(data_spec),
                                              collection_sha256=expected_collection_sha256)
    if not isinstance(preparation, dict) or preparation.get('status') != 'private_sft_prepared_unverified':
        raise ValueError('SFT preparation is not trainable')
    tasks = load_public_export(export, receipt)
    challenges = _object(challenge_bytes)
    comparison_receipt = {'comparison_sha256': expected_collection_sha256,
                          'selected_task_sha256': allocation_receipt['comparison_task_sha256']}
    comparison = load_public_export(build_public_export(challenges, comparison_receipt), comparison_receipt)
    solutions = _object(solution_bytes)
    selected = {}
    for task_id, task in tasks.items():
        answers = solutions.get(task_id)
        if not isinstance(answers, list) or len(answers) != len(task.test_inputs):
            raise ValueError('selected training solution count differs')
        selected[task_id] = [validate_grid(grid) for grid in answers]
    source = {'model_id': 'mlx-community/Qwen3-14B-4bit',
              'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
              'source_code_sha256': collection_spec['source_code_sha256'],
              'run_sha256': expected_collection_sha256, 'run_id': collection_spec['run_id']}
    manifest_path = directory / 'sft-data' / 'manifest.json'
    digest = _private_file(manifest_path)
    if digest != training_spec.get('dataset_manifest_sha256'):
        raise ValueError('registered dataset manifest differs')
    manifest = _json_file(manifest_path, digest)
    result = replay_collection_dataset(journal_dir=directory / 'collection', tasks=tasks, comparison_tasks=comparison,
        verifier=Verifier(selected), source=source, manifest=manifest,
        resource_limits=collection_spec['settings']['resource_limits'])
    eligibility_path = directory / 'eligibility.json'
    eligibility = _json_file(eligibility_path, _private_file(eligibility_path))
    if (not isinstance(eligibility, dict) or eligibility.get('data_sha256') != _digest(data_spec) or
            eligibility.get('collection_sha256') != expected_collection_sha256 or
            not isinstance(eligibility.get('records'), list) or
            hashlib.sha256(_json(eligibility['records']).encode()).hexdigest() != result['eligibility_sha256']):
        raise ValueError('retained eligibility differs from collection replay')
    return {'status': 'training_data_consistent_unverified', 'training_allowed': False,
            'training_sha256': expected_training_sha256, 'solution_sha256': training_spec['solution_sha256'],
            'dataset_manifest_sha256': digest, 'attempted_tasks': result['attempted_tasks'],
            'eligible_tasks': result['eligible_tasks'], 'yield_status': result['yield_status']}
