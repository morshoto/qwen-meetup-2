"""Separate matched evaluation after the pinned pre-inference permission failure."""
from __future__ import annotations

import json
from pathlib import Path

from src.arc.qwen38_weight_allocation import freeze_registration
from src.arc.qwen38_weight_eval_registration import build_evaluation_descriptor
from src.arc.qwen38_weight_training_receipt import verify_training_evidence
from src.cli.qwen38_weight_prepare_sft import replay_bundle


# Exact immutable evidence for the stopped first evaluation; no model was loaded.
FAILURE_HASHES = {'evaluation-v1.json': '6bc6712dfb32662eb889f99bf103a53a898ffcaa3da6ad51f122098938097b98', 'adapter-v1/termination.json': '5672689ee088f90ee03fb877905119681a374e3a05299547836a145da9272edc', 'adapter-v1/journal/000000.start.json': '27c1dd3916cca756ae6ac290feae0abc4365e84da7d20e5850a021724eb5c76c', 'adapter-v1/journal/000000.result.json': '90eaeb43c1713b469a7be26b2416dd737d1157e9090ef8e092dc73ab1db49d31', 'adapter-v1/workers/000000/execution.json': 'e5afda74f4ea76ae829b9e1eef80d0e048ee2b237f7fee90467e4b9cd01ecbac', 'adapter-v1/workers/000000/stderr.log': 'ba95708d550ebc5419cf93160c5b7b34a5a7df349dec02d181a0dbdd624829cb', 'adapter-v1/workers/000000/stdout.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'adapter-v1/workers/000000/request.json': 'eef6dd70aa951ee86789b4f8d7613ec22936b9d107913dd4c707ec7443343e38'}


def verify_preinference_failure(root):
    import hashlib
    root = Path(root).resolve()
    for name, expected in FAILURE_HASHES.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('original evaluation failure evidence changed')
    journal = root / 'adapter-v1/journal'
    if {p.name for p in journal.iterdir()} != {'.lock', '000000.start.json', '000000.result.json'}:
        raise ValueError('original evaluation journal has additional attempts')
    workers = root / 'adapter-v1/workers'
    if {p.name for p in workers.iterdir()} != {'000000'} or (workers / '000000/response.json').exists():
        raise ValueError('original evaluation may have generated a response')
    return {'status': 'preinference_failure_verified', 'artifact_sha256': dict(FAILURE_HASHES)}


def _context(**kwargs):
    comparison_dir = Path(kwargs['comparison_dir']).resolve()
    inputs = {k: v for k, v in kwargs.items() if k != 'comparison_dir'}
    root = Path(inputs['run_dir']).resolve()
    if inputs['manifest'].get('run_dir') != str(root):
        raise ValueError('evaluation run does not match frozen collection')
    selection, registration, replay = replay_bundle(**inputs)
    if (selection.get('status') != 'eligible_pending_source_replay' or selection['counts'].get('unique_parents', 0) < 5 or
        replay.get('status') != 'matches_supplied_replay'):
        raise ValueError('SFT source replay or gate failed')
    training = json.loads((root / 'training/training-result-v1.json').read_bytes())
    evidence = verify_training_evidence(training, root, inputs['model_path'], registration, replay['manifest_sha256'])
    if evidence.get('status') != 'source_evidence_matches':
        raise ValueError('training evidence replay failed')
    comparison = json.loads((comparison_dir / 'manifest.json').read_bytes())
    ids = comparison['allocation']['evaluation']; baseline_dir = comparison_dir / 'evaluation/baseline'
    if {p.stem for p in baseline_dir.glob('*.json')} != set(ids):
        raise ValueError('baseline checkpoint inventory differs')
    files = {}
    for task_id in ids:
        path = baseline_dir / (task_id + '.json')
        if path.is_symlink(): raise ValueError('baseline checkpoint cannot be a symlink')
        files[task_id] = path.read_bytes()
    descriptor = build_evaluation_descriptor(comparison, inputs['manifest'], training, evidence,
                                             inputs['challenges'], files, inputs['repo'])
    from src.arc.qwen38_rsi_comparison import digest
    import hashlib
    failure = verify_preinference_failure(root / 'evaluation')
    descriptor['protocol'] = 'qwen38-weight-paired-evaluation-recovery-v2'
    descriptor['run_dir'] = str((root / 'evaluation/adapter-v2').resolve())
    descriptor['previous_failure'] = failure
    source = Path(__file__).resolve().relative_to(Path(inputs['repo']).resolve()).as_posix()
    descriptor['source_files_sha256'][source] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    descriptor['limitations'].append('Recovery is a new registered run after a permission failure before inference.')
    descriptor['sha256'] = digest({k: v for k, v in descriptor.items() if k != 'sha256'})
    return descriptor, comparison, training, selection, files


def freeze_evaluation(**kwargs):
    descriptor, _, _, _, _ = _context(**kwargs)
    root = Path(kwargs['run_dir']).resolve() / 'evaluation'
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError('evaluation directory is not private')
    return freeze_registration(root / 'evaluation-v2.json', {k: v for k, v in descriptor.items() if k != 'sha256'})


def verify_evaluation_descriptor(**kwargs):
    descriptor, _, _, _, _ = _context(**kwargs)
    saved = json.loads((Path(kwargs['run_dir']).resolve() / 'evaluation/evaluation-v2.json').read_bytes())
    if saved != descriptor:
        raise ValueError('evaluation descriptor changed')
    return descriptor


import sys
from src.arc.qwen38_weight_eval_schedule import next_evaluation_action
from src.arc.qwen38_weight_generation import capture_response, create_request
from src.arc.qwen38_weight_journal import AttemptJournal
from src.arc.qwen38_weight_supervisor import run_supervised
from src.cli.qwen38_weight_collect import _artifact_hashes, _replay_artifacts, _run_start, _validate_worker_execution


def _worker_command(descriptor, root, model_path, index):
    adapter = root / 'training/adapter-run-v1/adapter'
    worker_dir = Path(descriptor['run_dir']) / 'workers' / f'{index:06}'
    return [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_eval_worker',
            '--model-path', str(Path(model_path).resolve()), '--adapter-path', str(adapter),
            '--adapter-sha256', descriptor['adapter_sha256'], '--adapter-config-sha256', descriptor['adapter_config_sha256'],
            '--output', str(worker_dir / 'response.json')]


def _replay_evaluation(rows, descriptor, root, model_path):
    workers = Path(descriptor['run_dir']) / 'workers'
    _replay_artifacts(rows, workers, descriptor['max_tokens'], descriptor['supervision'])
    for index, row in enumerate(rows):
        if row.get('worker_command') != _worker_command(descriptor, root, model_path, index):
            raise ValueError('evaluation worker command differs from registration')


def _finalize_evaluation(eval_root, descriptor, rows, status, wall_elapsed):
    from src.arc.qwen38_rsi_comparison import digest
    path = eval_root / 'termination.json'
    if path.exists():
        value = json.loads(path.read_bytes())
        if (value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}) or
            value.get('evaluation_sha256') != descriptor['sha256'] or value.get('journal_sha256') != digest(rows) or
            value.get('status') != status):
            raise ValueError('evaluation termination receipt mismatch')
        return value
    return freeze_registration(path, {'status': status, 'evaluation_sha256': descriptor['sha256'],
                                      'journal_sha256': digest(rows), 'wall_seconds': wall_elapsed})


def evaluate_step(**kwargs):
    descriptor = verify_evaluation_descriptor(**kwargs)
    root = Path(kwargs['run_dir']).resolve(); eval_root = Path(descriptor['run_dir'])
    if (descriptor.get('status') != 'evaluation_ready' or eval_root != root / 'evaluation/adapter-v2' or
        not eval_root.parent.is_dir() or eval_root.parent.stat().st_mode & 0o077):
        raise ValueError('evaluation destination or descriptor invalid')
    eval_root.mkdir(mode=0o700, exist_ok=True)
    workers = eval_root / 'workers'; workers.mkdir(mode=0o700, exist_ok=True)
    if any(p.is_symlink() or p.stat().st_mode & 0o077 for p in (eval_root, workers)):
        raise ValueError('evaluation evidence directories must be private')
    with AttemptJournal(eval_root / 'journal') as journal:
        wall_elapsed = _run_start(eval_root, descriptor['sha256'])
        rows = journal.read(); _replay_evaluation(rows, descriptor, root, kwargs['model_path'])
        action = next_evaluation_action(descriptor, rows)
        if action['status'] != 'ready':
            if action['status'] != 'requires_reconciliation':
                _finalize_evaluation(eval_root, descriptor, rows, action['status'], wall_elapsed)
            return action
        remaining = descriptor['wall_seconds_limit'] - wall_elapsed
        if remaining <= 0:
            _finalize_evaluation(eval_root, descriptor, rows, 'time_budget_exhausted', wall_elapsed)
            return {'status': 'time_budget_exhausted'}
        tasks = json.loads(kwargs['challenges'])
        request = create_request(action['task_id'], tasks[action['task_id']],
                                 candidate_index=action['candidate_index'], seed=action['seed'])
        if request['prompt_sha256'] != descriptor['prompt_sha256']:
            raise ValueError('evaluation prompt differs from baseline')
        directory = workers / f'{len(rows):06}'
        if directory.exists(): raise ValueError('worker evidence exists without journal start')
        command = _worker_command(descriptor, root, kwargs['model_path'], len(rows))
        identity = {k: v for k, v in request.items() if k not in ('content', 'test_count')}
        identity.update({'manifest_sha256': descriptor['sha256'], 'model': kwargs['manifest']['model'], 'worker_command': command})
        journal.start(identity)
        limits = descriptor['supervision']
        execution = run_supervised(command, {'request': request, 'max_tokens': descriptor['max_tokens']}, directory,
                                   timeout_seconds=min(remaining, descriptor['per_attempt_seconds']),
                                   min_free=limits['minimum_free_percent'], max_swap=limits['maximum_swap_bytes'],
                                   max_output_bytes=limits['max_output_bytes'], cwd=kwargs['repo'])
        if json.loads((directory / 'execution.json').read_bytes()) != execution:
            raise ValueError('evaluation worker execution receipt mismatch')
        if execution.get('process_group_gone') is not True:
            return {'status': 'requires_reconciliation'}
        _validate_worker_execution(execution, command, limits)
        if verify_evaluation_descriptor(**kwargs) != descriptor:
            raise ValueError('evaluation postflight changed')
        artifacts = _artifact_hashes(directory)
        if execution['status'] == 'complete':
            try:
                response = json.loads((directory / 'response.json').read_bytes())
                capture = capture_response(request, response, descriptor['max_tokens'])
                if response != capture: raise ValueError('evaluation worker response mismatch')
                outcome = {k: v for k, v in capture.items() if k not in identity}
                outcome.update({'status': 'complete', 'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts})
            except (ValueError, KeyError, OSError, TypeError):
                outcome = {'status': 'error', 'error': 'worker_output_invalid', 'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts}
        else:
            outcome = {'status': 'error', 'error': execution.get('reason') or 'worker_failed',
                       'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifacts}
        journal.finish(len(rows), outcome)
        return {'status': 'attempt_recorded', 'task_id': action['task_id'], 'candidate_index': action['candidate_index'],
                'attempt_status': outcome['status']}


def report_evaluation(**kwargs):
    """Score saved paired rows offline without passing solutions to a worker."""
    import hashlib
    import math
    import time
    from src.arc.data import Verifier
    from src.arc.qwen38_rsi_comparison import digest
    from src.arc.qwen38_weight_eval_rows import assemble_rows
    from src.arc.qwen38_weight_paired import paired_report

    descriptor = verify_evaluation_descriptor(**kwargs)
    rebuilt, _, training, selection, baseline_files = _context(**kwargs)
    if rebuilt != descriptor or hashlib.sha256(kwargs['solutions']).hexdigest() != descriptor['solution_sha256']:
        raise ValueError('evaluation source or solution identity mismatch')
    root = Path(kwargs['run_dir']).resolve(); eval_root = Path(descriptor['run_dir'])
    if not eval_root.is_dir() or not (eval_root / 'journal').is_dir():
        raise ValueError('evaluation has not started')
    with AttemptJournal(eval_root / 'journal') as journal:
        start = json.loads((eval_root / 'run-start.json').read_bytes())
        if (start.get('manifest_sha256') != descriptor['sha256'] or
            start.get('sha256') != digest({k: v for k, v in start.items() if k != 'sha256'}) or
            isinstance(start.get('started_epoch'), bool) or not isinstance(start.get('started_epoch'), (int, float)) or
            not math.isfinite(start['started_epoch'])):
            raise ValueError('invalid evaluation start identity')
        elapsed = time.time() - start['started_epoch']
        if elapsed < 0: raise ValueError('system clock precedes evaluation start')
        rows = journal.read(); _replay_evaluation(rows, descriptor, root, kwargs['model_path'])
        termination_path = eval_root / 'termination.json'
        termination = None
        if termination_path.exists():
            termination = json.loads(termination_path.read_bytes())
            wall = termination.get('wall_seconds')
            if (termination.get('sha256') != digest({k: v for k, v in termination.items() if k != 'sha256'}) or
                termination.get('evaluation_sha256') != descriptor['sha256'] or termination.get('journal_sha256') != digest(rows) or
                isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0):
                raise ValueError('evaluation termination evidence mismatch')
        assembly = assemble_rows(descriptor, json.loads(kwargs['challenges']), rows)
        if termination is not None:
            status = termination.get('status')
            if (status != assembly['schedule_status'] and not (status == 'time_budget_exhausted' and
                assembly['schedule_status'] == 'ready' and termination['wall_seconds'] >= descriptor['wall_seconds_limit'])):
                raise ValueError('evaluation termination differs from journal state')
        baseline = [json.loads(baseline_files[task_id]) for task_id in descriptor['task_ids']]
        scores = paired_report(descriptor['task_ids'], baseline, assembly['rows'], Verifier(json.loads(kwargs['solutions'])),
                               manifest_sha256=descriptor['comparison_sha256'], prompt_sha256=descriptor['prompt_sha256'],
                               adapter_sha256=descriptor['adapter_sha256'])
        result = {'status': assembly['status'] if termination is not None else 'incomplete',
                  'schedule_status': termination['status'] if termination is not None else assembly['schedule_status'],
                  'journal_sha256': digest(rows), 'evaluation_sha256': descriptor['sha256'],
                  'comparison_sha256': descriptor['comparison_sha256'], 'training_result_sha256': training['sha256'],
                  'adapter_sha256': descriptor['adapter_sha256'], 'training_mode': selection['mode'],
                  'training_source_counts': selection['counts'], 'evaluation_errors': assembly['errors'],
                  'wall_seconds': termination['wall_seconds'] if termination is not None else None, 'paired': scores,
                  'limitations': ['Previously evaluated tasks are exploratory, not an official ARC score.',
                                  'A parameter update and one generation do not establish recursive improvement.']}
        return freeze_registration(eval_root / ('report-' + digest(result) + '.json'), result)


def main(argv=None):
    import argparse
    from zipfile import ZipFile
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'report'))
    for flag in ('manifest', 'proposal', 'review', 'repo', 'model-path', 'challenges', 'solutions',
                 'archive', 'canary-dir', 'run-dir', 'history-dir', 'comparison-dir'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    with ZipFile(args.archive) as archive:
        official_bytes = archive.read('arc-agi_evaluation_challenges.json')
    inputs = {'manifest': json.loads(Path(args.manifest).read_bytes()),
              'proposal': json.loads(Path(args.proposal).read_bytes()), 'review': json.loads(Path(args.review).read_bytes()),
              'repo': Path(args.repo).resolve(), 'model_path': Path(args.model_path).resolve(),
              'challenges': Path(args.challenges).read_bytes(), 'solutions': Path(args.solutions).read_bytes(),
              'official_bytes': official_bytes, 'canary_dir': Path(args.canary_dir).resolve(),
              'run_dir': Path(args.run_dir).resolve(), 'history_dir': Path(args.history_dir).resolve(),
              'comparison_dir': Path(args.comparison_dir).resolve()}
    if args.action == 'prepare':
        result = freeze_evaluation(**inputs)
        print(json.dumps({'status': result['status'], 'sha256': result['sha256']}, sort_keys=True), flush=True)
        return 0
    if args.action == 'report':
        print(json.dumps(report_evaluation(**inputs), sort_keys=True), flush=True)
        return 0
    while True:
        result = evaluate_step(**inputs)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result['status'] != 'attempt_recorded' or args.once:
            return 2 if result['status'] in ('requires_reconciliation', 'resource_stop') else 0


if __name__ == '__main__':
    raise SystemExit(main())
