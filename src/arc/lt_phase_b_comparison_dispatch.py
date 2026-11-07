"""Dispatch sequential matched public comparison workers under local supervision.

No evaluation solutions are sent to workers. Local checks do not prove
independent attestation, escaped-process cleanup, or authorize evaluation.
"""
import fcntl
import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path

from src.arc.data import Verifier, validate_grid
from src.arc.lt_attestation import _object
from src.arc.lt_batch_journal import BatchJournal, _create
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_comparison_journal import compare_batch, _replay, validate_comparison_journals
from src.arc.lt_phase_b_comparison_score import score_completed_comparison
from src.arc.lt_phase_b_dispatch import _private_run_directory, _read_object
from src.arc.lt_phase_b_registered_canary import validate_canary_identity
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_real_reload_dispatch import review_real_reload_attempt
from src.arc.lt_phase_b_training_preflight import preflight_training_data
from src.arc.lt_phase_b_worker_supervisor import run_monitored_worker
from src.arc.lt_public import build_public_export, load_public_export
from src.arc.rsi_model_identity import _hash_file

_BOOTSTRAP = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
              'from src.cli.lt_phase_b_comparison_worker import main; main()')


def run_comparison(*, source_root, run_dir, expected_comparison_sha256, expected_training_sha256,
                   expected_canary_sha256, expected_collection_sha256, challenge_path, solution_path,
                   official_archive, python_executable, model_path, limit, score_only=False):
    root, directory = _private_run_directory(source_root, run_dir)
    if type(limit) is not int or not 0 < limit <= 100 or type(score_only) is not bool:
        raise ValueError('invalid comparison batch size')
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ('spec', 'inventory', 'allocation', 'export', 'receipt', 'canary-spec', 'training-spec',
                 'comparison-spec', 'comparison-export', 'comparison-receipt')
        def read_registrations():
            return {name: _read_object(directory / (name + '.json')) for name in names}
        inputs = read_registrations()
        spec, canary, training, comparison = (inputs[k] for k in ('spec', 'canary-spec', 'training-spec', 'comparison-spec'))
        def read_data():
            with zipfile.ZipFile(official_archive) as archive:
                info = archive.getinfo('arc-agi_evaluation_challenges.json')
                if info.file_size > 16 * 1024**2: raise ValueError('official challenge inventory too large')
                official = archive.read(info)
            return Path(challenge_path).read_bytes(), Path(solution_path).read_bytes(), official
        challenge, solutions, official = read_data()
        def preflight():
            return preflight_training_data(training_spec=training, expected_training_sha256=expected_training_sha256,
                collection_spec=spec, expected_collection_sha256=expected_collection_sha256,
                source_root=root, run_dir=directory, source_inventory=inputs['inventory'],
                allocation_receipt=inputs['allocation'], receipt=inputs['receipt'], export=inputs['export'],
                challenge_bytes=challenge, solution_bytes=solutions, official_challenge_bytes=official,
                python_executable=python_executable, model_path=model_path)
        data = preflight()
        if (data.get('status') != 'training_data_consistent_unverified' or data.get('training_allowed') is not False or
                data.get('training_sha256') != expected_training_sha256 or data.get('attempted_tasks') != 100 or
                data.get('solution_sha256') != training.get('solution_sha256') or
                data.get('dataset_manifest_sha256') != training.get('dataset_manifest_sha256') or
                data.get('yield_status') != training.get('yield_status')):
            raise ValueError('training data preflight differs')
        settings, _ = validate_canary_identity(canary_spec=canary, expected_canary_sha256=expected_canary_sha256,
            collection_spec=spec, expected_collection_sha256=expected_collection_sha256,
            source_inventory=inputs['inventory'], allocation_receipt=inputs['allocation'], receipt=inputs['receipt'])
        keys = {'protocol', 'status', 'run_id', 'collection_sha256', 'training_sha256', 'canary_sha256',
                'source_code_sha256', 'prompt_sha256', 'selected_task_sha256', 'adapter_sha256', 'settings'}
        valid_hash = lambda x: isinstance(x, str) and bool(re.fullmatch(r'[0-9a-f]{64}', x))
        canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
        tasks = comparison.get('selected_task_sha256')
        if (set(comparison) != keys or not valid_hash(expected_comparison_sha256) or
                _digest(comparison) != expected_comparison_sha256 or comparison.get('status') != 'frozen' or
                comparison.get('protocol') != 'lt-phase-b-comparison-v1' or comparison.get('run_id') != spec['run_id'] or
                directory.name != 'lt-phase-b-' + spec['run_id'] or
                comparison.get('collection_sha256') != expected_collection_sha256 or
                comparison.get('training_sha256') != expected_training_sha256 or comparison.get('canary_sha256') != expected_canary_sha256 or
                comparison.get('source_code_sha256') != spec['source_code_sha256'] or
                comparison.get('prompt_sha256') != hashlib.sha256(PROMPT.encode()).hexdigest() or
                canonical(comparison.get('settings')) != canonical(settings) or not valid_hash(comparison.get('adapter_sha256')) or
                not isinstance(tasks, dict) or len(tasks) != 50 or
                any(not isinstance(k, str) or not re.fullmatch(r'[0-9a-f]{8}', k) or not valid_hash(v) for k, v in tasks.items()) or
                tasks != inputs['allocation'].get('comparison_task_sha256') or set(tasks) & set(spec['selected_task_sha256']) or
                inputs['comparison-receipt'] != {'comparison_sha256': expected_comparison_sha256, 'selected_task_sha256': tasks}):
            raise ValueError('comparison registration differs')
        public = load_public_export(inputs['comparison-export'], inputs['comparison-receipt'])
        if inputs['comparison-export'] != build_public_export(_object(challenge), inputs['comparison-receipt']):
            raise ValueError('comparison public export differs')
        def replay():
            return review_real_reload_attempt(directory, training=training, canary=canary,
                expected_training_sha256=expected_training_sha256, expected_canary_sha256=expected_canary_sha256,
                expected_collection_sha256=expected_collection_sha256)
        previous = replay()
        if (previous.get('status') != 'real_adapter_reload_observed_unverified' or previous.get('reload_verified') is not False or
                previous.get('promotion_allowed') is not False or not isinstance(previous.get('reload'), dict) or
                previous['reload'].get('adapter_sha256') != comparison['adapter_sha256']):
            raise ValueError('real reload evidence differs')
        prompts = {task: hashlib.sha256(canonical({'instruction': PROMPT, 'task': public[task].prompt_data()}).encode()).hexdigest()
                   for task in tasks}
        registration = {'run_sha256': expected_comparison_sha256, 'source_code_sha256': comparison['source_code_sha256'],
                        'adapter_sha256': comparison['adapter_sha256'], 'selected_task_sha256': tasks,
                        'selected_prompt_sha256': prompts}
        for state in ('off', 'on'):
            journal_dir = directory / ('comparison-' + state)
            if journal_dir.is_symlink():
                raise ValueError('comparison journal may not be symlinked')
            if score_only and (not journal_dir.is_dir() or not (journal_dir / 'identity.json').is_file()):
                raise ValueError('comparison journal is missing')
        journals = {state: BatchJournal(directory / ('comparison-' + state), sorted(tasks),
                                       expected_comparison_sha256 + ':14b:' + state) for state in ('off', 'on')}
        batches = directory / 'comparison-batches'
        if batches.is_symlink(): raise ValueError('comparison batch directory is a symlink')
        if score_only and not batches.is_dir(): raise ValueError('comparison batches missing')
        batches.mkdir(mode=0o700, exist_ok=True)
        info = batches.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('comparison batch directory must be private')
        entries = {path.name for path in batches.iterdir()}
        if any(not re.fullmatch(r'[0-9]{8}\.(started|complete)\.json', name) for name in entries):
            raise ValueError('unexpected comparison batch record')
        indices = sorted({int(name[:8]) for name in entries})
        if indices != list(range(len(indices))): raise ValueError('comparison batch sequence differs')
        for index in indices:
            prefix = f'{index:08d}'
            marker = {'comparison_sha256': expected_comparison_sha256, 'index': index}
            start, complete = batches / (prefix + '.started.json'), batches / (prefix + '.complete.json')
            if (not start.exists() or not complete.exists() or _read_object(start) != marker or
                    _read_object(complete).get('marker') != marker):
                raise ValueError('comparison batch requires reconciliation')
        if score_only:
            output = directory / 'comparison-score.json'
            if output.exists() or output.is_symlink() or not indices:
                raise ValueError('score exists or comparison has no completed batches')
            final_batch = _read_object(batches / (f'{indices[-1]:08d}.complete.json')).get('result')
            if (not isinstance(final_batch, dict) or final_batch.get('state') != 'on' or
                    final_batch.get('stop_reason') != 'complete'):
                raise ValueError('comparison batch sequence is incomplete')
            validate_comparison_journals(off_journal=journals['off'], on_journal=journals['on'],
                                         registration=registration, resource_limits=settings['resource_limits'])
            for state in ('off', 'on'):
                if _replay(journals[state], state, registration, settings['resource_limits']) is not None or journals[state].pending(1):
                    raise ValueError('comparison journals are incomplete')
            all_solutions = _object(solutions)
            selected = {}
            for task, public_task in public.items():
                values = all_solutions.get(task)
                if not isinstance(values, list) or len(values) != len(public_task.test_inputs):
                    raise ValueError('comparison solution count differs')
                selected[task] = [validate_grid(grid) for grid in values]
            report = score_completed_comparison(off_journal=journals['off'], on_journal=journals['on'],
                registration=registration, resource_limits=settings['resource_limits'], verifier=Verifier(selected))
            if read_data() != (challenge, solutions, official) or read_registrations() != inputs or preflight() != data or replay() != previous:
                raise ValueError('comparison inputs changed during scoring')
            _create(output, report)
            return {'status': report['status'], 'evaluation_verified': False, 'report_sha256': _hash_file(output),
                    'selected_tasks': len(tasks), 'report_path': str(output)}
        index = len(indices)
        if index >= 100000000: raise ValueError('comparison batch sequence exhausted')
        marker = {'comparison_sha256': expected_comparison_sha256, 'index': index}
        _create(batches / (f'{index:08d}.started.json'), marker)
        def worker(task, adapter_state, output, on_start):
            argv = [str(python_executable), '-I', '-B', '-c', _BOOTSTRAP, str(root),
                '--comparison', str(directory / 'comparison-spec.json'), '--collection', str(directory / 'spec.json'),
                '--training', str(directory / 'training-spec.json'), '--canary', str(directory / 'canary-spec.json'),
                '--inventory', str(directory / 'inventory.json'), '--allocation', str(directory / 'allocation.json'),
                '--collection-receipt', str(directory / 'receipt.json'), '--comparison-receipt', str(directory / 'comparison-receipt.json'),
                '--export', str(directory / 'comparison-export.json'), '--run-dir', str(directory), '--model-path', str(model_path),
                '--output', str(output), '--task', task, '--adapter-state', adapter_state,
                '--expected-comparison-sha256', expected_comparison_sha256, '--expected-training-sha256', expected_training_sha256,
                '--expected-canary-sha256', expected_canary_sha256, '--expected-collection-sha256', expected_collection_sha256]
            return run_monitored_worker(argv, root=directory, cwd=root, timeout=settings['task_timeout_seconds'],
                on_start=on_start, resource_limits=settings['resource_limits'], interval_seconds=settings['monitor_interval_seconds'],
                sample_timeout_seconds=settings['sample_timeout_seconds'], terminate_grace_seconds=settings['termination_grace_seconds'])
        result = compare_batch(off_journal=journals['off'], on_journal=journals['on'], registration=registration,
                               limit=limit, worker=worker, resource_limits=settings['resource_limits'])
        if read_data() != (challenge, solutions, official) or read_registrations() != inputs or preflight() != data or replay() != previous:
            raise ValueError('comparison inputs changed during batch')
        _create(batches / (f'{index:08d}.complete.json'), {'marker': marker, 'result': result})
        return result
    finally:
        os.close(lockfd)
