"""Replay completed private collection and offline self-solution eligibility.

Callers must validate the frozen allocation and authoritative task/answer files.
This is local consistency evidence, not process attestation or authorization.
"""
import hashlib
import math
import os
import stat
from pathlib import Path

from src.arc.lt_phase_b_dataset import build_dataset
from src.arc.lt_phase_b_sft_preflight import _json
from src.arc.lt_phase_b_training_config import _private_file
from src.arc.rsi_model_identity import _json_file
from src.arc.rsi_training_resources import training_resource_decision, valid_resource_limits


def reconstruct_collection_dataset(*, journal_dir, tasks, comparison_tasks, verifier, source, resource_limits):
    """Return sensitive reconstructed records only to the trusted caller."""
    if (not isinstance(tasks, dict) or not tasks or not isinstance(source, dict) or
            not isinstance(source.get('run_sha256'), str) or not valid_resource_limits(resource_limits)):
        raise ValueError('invalid collection replay inputs')
    directory = Path(journal_dir)
    try:
        info = directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('collection journal must be private')
        ids = sorted(tasks)
        identity = source['run_sha256'] + ':14b:collection'
        def read(name):
            path = directory / name
            return _json_file(path, _private_file(path))
        if read('identity.json') != {'identity': identity, 'task_ids': ids}:
            raise ValueError('collection identity differs')
        names = {'identity.json'} | {task + suffix for task in ids for suffix in
                 ('.started.json', '.process.json', '.resources.json', '.worker.json', '.result.json')}
        if {path.name for path in directory.iterdir()} != names:
            raise ValueError('incomplete or unexpected collection records')
        rows = []
        for task in ids:
            if read(task + '.started.json') != {'task_id': task, 'identity': identity}:
                raise ValueError('collection start differs')
            process = read(task + '.process.json')
            if (set(process) != {'task_id', 'identity', 'pid'} or process['task_id'] != task or
                    process['identity'] != identity or type(process['pid']) is not int or process['pid'] <= 0):
                raise ValueError('collection process differs')
            resources = read(task + '.resources.json')
            observation = resources.get('observation')
            if (resources.get('status') != 'complete' or resources.get('leader_reaped') is not True or
                    resources.get('group_absent') is not True or type(resources.get('returncode')) is not int or
                    resources['returncode'] != 0 or not isinstance(observation, dict) or
                    observation.get('stop_reason') != 'process_exited' or type(observation.get('exit_code')) is not int or
                    observation['exit_code'] != 0 or observation.get('stream_complete') is not True or
                    observation.get('post_reap_group_absent') is not True or
                    not isinstance(observation.get('cleanup_observation'), dict) or
                    training_resource_decision(resources.get('baseline'), observation.get('samples'),
                                               limits=resource_limits) is not None):
                raise ValueError('collection resource evidence differs')
            elapsed = observation.get('elapsed_seconds', resources.get('elapsed_seconds'))
            if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or
                    not 0 <= elapsed <= resource_limits['timeout_seconds'] or
                    observation['samples'][-1]['monotonic_seconds'] - resources['baseline']['monotonic_seconds'] > resource_limits['timeout_seconds']):
                raise ValueError('collection time limit differs')
            worker = read(task + '.worker.json')
            expected = {**worker, 'worker_diagnostics': {'stdout': resources.get('stdout'), 'stderr': resources.get('stderr')},
                        'cleanup': {'leader_reaped': True, 'group_absent': True, 'returncode': resources['returncode']},
                        'identity': identity}
            row = read(task + '.result.json')
            if (_json(row) != _json(expected) or worker.get('task_id') != task or worker.get('status') != 'complete' or
                    worker.get('model') != '14b'):
                raise ValueError('collection worker and result differ')
            rows.append(row)
        dataset = build_dataset(tasks, rows, verifier, source=source, comparison_tasks=comparison_tasks)
        return dataset
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('collection replay unavailable') from exc


def replay_collection_dataset(*, journal_dir, tasks, comparison_tasks, verifier, source, manifest, resource_limits):
    if not isinstance(manifest, dict):
        raise ValueError('invalid collection manifest')
    dataset = reconstruct_collection_dataset(journal_dir=journal_dir, tasks=tasks, comparison_tasks=comparison_tasks,
                                             verifier=verifier, source=source, resource_limits=resource_limits)
    ids = sorted(tasks)
    hashes = {key: hashlib.sha256(''.join(_json(example) + '\n' for example in dataset[key]).encode()).hexdigest()
              for key in ('train', 'valid')}
    eligible = {r['task_id']: r['solution_lineage_sha256'] for r in dataset['journal']
                if r['verification_status'] == 'eligible'}
    if (dataset['yield_status'] not in ('disjoint_split', 'overfitting_only') or
            manifest.get('status') != 'private_sft_materialized_unverified' or manifest.get('training_allowed') is not False or
            manifest.get('token_audit_status') != 'dataset_tokens_consistent_unverified' or
            manifest.get('source') != source or manifest.get('yield_status') != dataset['yield_status'] or
            manifest.get('split_task_ids') != dataset['split_task_ids'] or
            manifest.get('solution_lineage_sha256') != eligible or
            type(manifest.get('train_count')) is not int or manifest['train_count'] != len(dataset['train']) or
            type(manifest.get('valid_count')) is not int or manifest['valid_count'] != len(dataset['valid']) or
            manifest.get('train_jsonl_sha256') != hashes['train'] or
            manifest.get('valid_jsonl_sha256') != (hashes['valid'] if dataset['valid'] else None)):
        raise ValueError('materialized dataset differs from collection')
    return {'status': 'collection_dataset_consistent_unverified', 'training_allowed': False,
            'attempted_tasks': len(ids), 'eligible_tasks': len(eligible), 'yield_status': dataset['yield_status'],
            'eligibility_sha256': hashlib.sha256(_json(dataset['journal']).encode()).hexdigest(),
            'train_jsonl_sha256': hashes['train'], 'valid_jsonl_sha256': hashes['valid'] if dataset['valid'] else None}
