"""Prepare private SFT examples from a replayed collection without training.

Local source and registration consistency is not independent attestation or
authorization. Eligibility records and JSONL contain sensitive predictions.
"""
import fcntl
import hashlib
import os
import re

from src.arc.data import Verifier, validate_grid
from src.arc.lt_attestation import _object
from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_collection_preflight import preflight_collection
from src.arc.lt_phase_b_protocol_gate import require_protocol_solutions
from src.arc.lt_phase_b_collection_replay import reconstruct_collection_dataset
from src.arc.lt_phase_b_dispatch import _private_run_directory
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_sft_materialize import materialize_sft
from src.arc.lt_public import build_public_export, load_public_export


def prepare_sft(*, data_spec, expected_data_sha256, collection_spec, expected_collection_sha256,
                source_root, run_dir, source_inventory, allocation_receipt, receipt, export,
                challenge_bytes, solution_bytes, official_challenge_bytes, python_executable, model_path,
                tokenizer, dataset_factory):
    root, directory = _private_run_directory(source_root, run_dir)
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('eligibility.json', 'sft-data', 'sft-preparation.json'):
            path = directory / name
            if path.exists() or path.is_symlink(): raise ValueError('SFT preparation requires reconciliation')
        keys = {'protocol', 'status', 'run_id', 'collection_sha256', 'source_code_sha256', 'solution_sha256'}
        if (not isinstance(data_spec, dict) or set(data_spec) != keys or
                not isinstance(expected_data_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_data_sha256) or
                _digest(data_spec) != expected_data_sha256 or data_spec.get('protocol') != 'lt-phase-b-data-v1' or
                data_spec.get('status') != 'frozen' or data_spec.get('run_id') != collection_spec.get('run_id') or
                directory.name != 'lt-phase-b-' + str(collection_spec.get('run_id')) or
                data_spec.get('collection_sha256') != expected_collection_sha256 or
                data_spec.get('source_code_sha256') != collection_spec.get('source_code_sha256') or
                not isinstance(solution_bytes, bytes) or hashlib.sha256(solution_bytes).hexdigest() != data_spec.get('solution_sha256')):
            raise ValueError('SFT data registration differs')
        checked = preflight_collection(source_root=root, spec=collection_spec, expected_run_sha256=expected_collection_sha256,
            source_inventory=source_inventory, allocation_receipt=allocation_receipt, export=export, receipt=receipt,
            challenge_bytes=challenge_bytes, official_challenge_bytes=official_challenge_bytes,
            python_executable=python_executable, model_path=model_path)
        if checked != {'status': 'preflight_consistent_unverified', 'execution_ready': False,
                       'run_sha256': expected_collection_sha256, 'selected_tasks': len(collection_spec['selected_task_sha256']),
                       'evidence_type': 'local_files_and_caller_supplied_registration'}:
            raise ValueError('collection preflight differs')
        require_protocol_solutions(source_root, source_inventory, solution_bytes)
        tasks = load_public_export(export, receipt)
        comparison_receipt = {'comparison_sha256': expected_collection_sha256,
                              'selected_task_sha256': allocation_receipt['comparison_task_sha256']}
        comparison = load_public_export(build_public_export(_object(challenge_bytes), comparison_receipt), comparison_receipt)
        solutions = _object(solution_bytes)
        selected = {}
        for task_id, task in tasks.items():
            answers = solutions.get(task_id)
            if not isinstance(answers, list) or len(answers) != len(task.test_inputs):
                raise ValueError('selected training solution count differs')
            selected[task_id] = [validate_grid(grid) for grid in answers]
        verifier = Verifier(selected)
        source = {'model_id': 'mlx-community/Qwen3-14B-4bit',
                  'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                  'source_code_sha256': collection_spec['source_code_sha256'],
                  'run_sha256': expected_collection_sha256, 'run_id': collection_spec['run_id']}
        dataset = reconstruct_collection_dataset(journal_dir=directory / 'collection', tasks=tasks,
            comparison_tasks=comparison, verifier=verifier, source=source,
            resource_limits=collection_spec['settings']['resource_limits'])
        if dataset['yield_status'] not in ('insufficient_verified_self_solutions', 'overfitting_only', 'disjoint_split'):
            raise ValueError('collection yield incomplete')
        _create(directory / 'eligibility.json', {'data_sha256': expected_data_sha256,
            'collection_sha256': expected_collection_sha256, 'records': dataset['journal']})
        if dataset['yield_status'] == 'insufficient_verified_self_solutions':
            result = {'status': 'insufficient_verified_self_solutions', 'training_allowed': False,
                      'data_sha256': expected_data_sha256, 'attempted_tasks': len(tasks), 'eligible_tasks': 0}
        else:
            manifest = materialize_sft(source_root=root, run_dir=directory, dataset=dataset, tasks=tasks,
                verifier=verifier, source=source, tokenizer=tokenizer, dataset_factory=dataset_factory, _locked_fd=lockfd)
            result = {'status': 'private_sft_prepared_unverified', 'training_allowed': False,
                      'data_sha256': expected_data_sha256, 'attempted_tasks': len(tasks),
                      'eligible_tasks': dataset['eligible_task_count'], 'yield_status': dataset['yield_status'],
                      'train_count': manifest['train_count'], 'valid_count': manifest['valid_count']}
        _create(directory / 'sft-preparation.json', result)
        return result
    finally:
        os.close(lockfd)
