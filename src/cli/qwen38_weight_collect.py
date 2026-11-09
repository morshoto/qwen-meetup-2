"""Single supervised collection step under a frozen experiment registration.

The model worker receives visible challenge data only. Offline verification runs
in this controller, and an unresolved started record always stops dispatch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_generation import capture_response, create_request
from src.arc.qwen38_weight_journal import AttemptJournal
from src.arc.qwen38_weight_preflight import verify_preflight
from src.arc.qwen38_weight_schedule import next_action
from src.arc.qwen38_weight_supervisor import run_supervised
from src.arc.qwen38_weight_verification_receipt import build_receipt, verify_receipt, write_receipt


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_once(path, value):
    encoded = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    parent = os.open(Path(path).parent, os.O_RDONLY)
    try: os.fsync(parent)
    finally: os.close(parent)


def _run_start(root, manifest_sha):
    path = root / 'run-start.json'
    if not path.exists():
        value = {'manifest_sha256': manifest_sha, 'started_epoch': time.time()}
        value['sha256'] = digest(value)
        _write_once(path, value)
    value = json.loads(path.read_bytes())
    if (value.get('manifest_sha256') != manifest_sha or value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}) or
        isinstance(value.get('started_epoch'), bool) or not isinstance(value.get('started_epoch'), (int, float)) or
        not math.isfinite(value['started_epoch'])):
        raise ValueError('invalid collection start identity')
    elapsed = time.time() - value['started_epoch']
    if elapsed < 0:
        raise ValueError('system clock precedes collection start')
    return elapsed


def _artifact_hashes(directory):
    names = ['request.json', 'stdout.log', 'stderr.log', 'execution.json']
    if (directory / 'response.json').exists(): names.append('response.json')
    return {name: _sha(directory / name) for name in names}


def _validate_worker_execution(execution, command, limits):
    if (execution.get('command') != command or execution.get('status') not in ('complete', 'error') or
        execution.get('process_group_gone') is not True or not isinstance(execution.get('samples'), list) or
        not execution['samples'] or isinstance(execution.get('elapsed_seconds'), bool) or
        not isinstance(execution.get('elapsed_seconds'), (int, float)) or
        not math.isfinite(execution['elapsed_seconds']) or execution['elapsed_seconds'] < 0):
        raise ValueError('invalid worker execution evidence')
    if execution['status'] == 'complete':
        if (execution.get('exit_code') != 0 or execution.get('reason') is not None or
            type(execution.get('pid')) is not int or execution['pid'] <= 0):
            raise ValueError('invalid successful worker evidence')
        for sample in execution['samples']:
            free, swap = sample.get('free_percent'), sample.get('swap_bytes')
            if type(free) is not int or free < limits['minimum_free_percent'] or type(swap) is not int or not 0 <= swap <= limits['maximum_swap_bytes']:
                raise ValueError('successful worker resource limit violated')
    elif not isinstance(execution.get('reason'), str) or not execution['reason']:
        raise ValueError('worker error lacks reason')


def _replay_artifacts(rows, workers, max_tokens, limits):
    for index, row in enumerate(rows):
        if row['status'] == 'started':
            continue
        directory = workers / f'{index:06}'
        recorded = row.get('worker_artifacts')
        if not isinstance(recorded, dict) or recorded != _artifact_hashes(directory):
            raise ValueError('worker artifact changed')
        execution = json.loads((directory / 'execution.json').read_bytes())
        payload = json.loads((directory / 'request.json').read_bytes())
        _validate_worker_execution(execution, row['worker_command'], limits)
        request = payload.get('request', {})
        if (set(payload) != {'request', 'max_tokens'} or payload['max_tokens'] != max_tokens or
            execution.get('process_group_gone') is not True or execution.get('elapsed_seconds') != row['elapsed_seconds'] or
            any(row.get(k) != v for k, v in request.items() if k not in ('content', 'test_count'))):
            raise ValueError('worker execution or request mismatch')
        if row['status'] == 'complete':
            if execution.get('status') != 'complete' or 'response.json' not in recorded:
                raise ValueError('completed attempt lacks successful worker output')
            response = json.loads((directory / 'response.json').read_bytes())
            if any(row.get(k) != v for k, v in response.items()):
                raise ValueError('worker output differs from journal')
        elif row['status'] == 'error':
            if execution.get('status') != 'error' and row.get('error') != 'worker_output_invalid':
                raise ValueError('worker error is not supported by execution record')
        else:
            raise ValueError('invalid journal status')


def collect_step(*, manifest, proposal, review, repo, model_path, challenges, solutions, canary_dir, run_dir):
    if manifest.get('status') != 'execution_ready':
        raise ValueError('collection requires an execution registration')
    if verify_preflight(manifest, proposal, review, repo, model_path, challenges, solutions, canary_dir=canary_dir).get('status') != 'passed':
        raise ValueError('execution preflight failed')
    root = Path(run_dir); root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError('run directory must be private')
    receipts = root / 'receipts'; workers = root / 'workers'
    receipts.mkdir(mode=0o700, exist_ok=True); workers.mkdir(mode=0o700, exist_ok=True)
    if any(p.is_symlink() or p.stat().st_mode & 0o077 for p in (receipts, workers)):
        raise ValueError('run evidence directories must be private')
    with AttemptJournal(root / 'journal') as journal:
        wall_elapsed = _run_start(root, manifest['sha256'])
        rows = journal.read(); max_tokens = manifest['generation']['max_tokens_per_candidate']
        _replay_artifacts(rows, workers, max_tokens, manifest['supervision'])
        receipt = build_receipt(manifest, challenges, solutions, rows)
        receipt_path = receipts / (receipt['journal_sha256'] + '.json')
        if receipt_path.exists():
            outcomes = verify_receipt(json.loads(receipt_path.read_bytes()), manifest, challenges, solutions, rows)
        else:
            write_receipt(receipt_path, receipt)
            outcomes = verify_receipt(receipt, manifest, challenges, solutions, rows)
        action = next_action(manifest, rows, outcomes)
        if action['status'] != 'ready':
            return action
        remaining = manifest['generation']['wall_seconds_limit'] - wall_elapsed
        if remaining <= 0:
            return {'status': 'time_budget_exhausted'}
        tasks = json.loads(challenges)
        request = create_request(action['task_id'], tasks[action['task_id']],
                                 candidate_index=action['candidate_index'], seed=action['seed'])
        identity = {k: v for k, v in request.items() if k not in ('content', 'test_count')}
        identity.update({'manifest_sha256': manifest['sha256'], 'model': manifest['model']})
        directory = workers / f'{len(rows):06}'
        if directory.exists():
            raise ValueError('worker evidence exists without journal start')
        limits = manifest['supervision']
        command = [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_worker', '--model-path', str(model_path),
                   '--output', str(directory / 'response.json')]
        identity['worker_command'] = command
        journal.start(identity)
        execution = run_supervised(command, {'request': request, 'max_tokens': max_tokens}, directory,
                                   timeout_seconds=min(remaining, limits['per_attempt_seconds']),
                                   min_free=limits['minimum_free_percent'], max_swap=limits['maximum_swap_bytes'],
                                   max_output_bytes=limits['max_output_bytes'], cwd=repo)
        if json.loads((directory / 'execution.json').read_bytes()) != execution:
            raise ValueError('worker execution receipt mismatch')
        if execution.get('process_group_gone') is not True:
            return {'status': 'requires_reconciliation'}
        _validate_worker_execution(execution, command, limits)
        if verify_preflight(manifest, proposal, review, repo, model_path, challenges, solutions, canary_dir=canary_dir).get('status') != 'passed':
            raise ValueError('postflight failed')
        artifact_hashes = _artifact_hashes(directory)
        if execution['status'] == 'complete':
            try:
                response = json.loads((directory / 'response.json').read_bytes())
                captured = capture_response(request, response, max_tokens)
                if response != captured:
                    raise ValueError('worker response identity mismatch')
                outcome = {k: v for k, v in captured.items() if k not in identity}
                outcome.update({'status': 'complete', 'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifact_hashes})
            except (ValueError, KeyError, OSError, TypeError):
                outcome = {'status': 'error', 'error': 'worker_output_invalid', 'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifact_hashes}
        else:
            outcome = {'status': 'error', 'error': execution.get('reason') or 'worker_failed',
                       'elapsed_seconds': execution['elapsed_seconds'], 'worker_artifacts': artifact_hashes}
        journal.finish(len(rows), outcome)
        return {'status': 'attempt_recorded', 'task_id': action['task_id'], 'candidate_index': action['candidate_index'],
                'attempt_status': outcome['status']}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    for flag in ('manifest', 'proposal', 'review', 'repo', 'model-path', 'challenges', 'solutions', 'canary-dir', 'run-dir'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    inputs = {'manifest': json.loads(Path(args.manifest).read_bytes()),
              'proposal': json.loads(Path(args.proposal).read_bytes()),
              'review': json.loads(Path(args.review).read_bytes()),
              'repo': Path(args.repo).resolve(), 'model_path': Path(args.model_path).resolve(),
              'challenges': Path(args.challenges).read_bytes(), 'solutions': Path(args.solutions).read_bytes(),
              'canary_dir': Path(args.canary_dir).resolve(), 'run_dir': Path(args.run_dir).resolve()}
    while True:
        result = collect_step(**inputs)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result['status'] != 'attempt_recorded' or args.once:
            return 2 if result['status'] in ('requires_reconciliation', 'resource_stop') else 0


if __name__ == '__main__':
    sys.exit(main())
