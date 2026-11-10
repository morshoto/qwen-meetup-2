import hashlib
import json
from pathlib import Path

import pytest

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import fingerprint
from src.arc.qwen38_weight_generation import capture_response
from src.cli.qwen38_replication import freeze_replication, run_replication_step


def fixture(tmp_path, monkeypatch):
    import src.cli.qwen38_replication as controller
    root = tmp_path / 'run'; root.mkdir(mode=0o700)
    task = {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
    challenges = json.dumps({'a': task}).encode()
    descriptor = {'status': 'evaluation_ready', 'inference_authorized': True, 'run_dir': str(root.resolve()),
                  'task_ids': ['a'], 'task_fingerprints': {'a': fingerprint(task)},
                  'challenge_sha256': hashlib.sha256(challenges).hexdigest(),
                  'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(), 'model': {'id': 'model'},
                  'adapter_sha256': 'a' * 64, 'adapter_config_sha256': 'b' * 64,
                  'candidate_count': 2, 'max_tokens': 4096, 'reserved_completion_tokens': 4 * 4096,
                  'wall_seconds_limit': 1000, 'per_attempt_seconds': 10,
                  'supervision': {'minimum_free_percent': 10, 'maximum_swap_bytes': 100, 'max_output_bytes': 10000}}
    descriptor['sha256'] = digest(descriptor)
    rebuild = lambda: descriptor
    freeze_replication(root, rebuild)
    calls = []
    def worker(command, payload, directory, **kwargs):
        calls.append((command, payload)); assert set(payload) == {'request', 'max_tokens'}
        assert 'solutions' not in json.dumps(payload)
        directory = Path(directory); directory.mkdir(mode=0o700)
        for name in ('stdout.log', 'stderr.log'): (directory / name).write_text('')
        (directory / 'request.json').write_text(json.dumps(payload))
        execution = {'status': 'complete', 'reason': None, 'exit_code': 0, 'process_group_gone': True,
                     'elapsed_seconds': 1, 'command': command, 'pid': 123,
                     'samples': [{'free_percent': 80, 'swap_bytes': 0}]}
        (directory / 'execution.json').write_text(json.dumps(execution))
        response = {'raw': '[[[4]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                    'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}
        (directory / 'response.json').write_text(json.dumps(capture_response(payload['request'], response, payload['max_tokens'])))
        return execution
    monkeypatch.setattr(controller, 'run_supervised', worker)
    return {'root': root, 'rebuild': rebuild, 'challenges': challenges, 'model_path': tmp_path / 'model',
            'adapter_path': tmp_path / 'adapter', 'repo': tmp_path}, descriptor, calls


def test_supervisor_runs_matched_arms_and_writes_terminal_receipt(tmp_path, monkeypatch):
    args, descriptor, calls = fixture(tmp_path, monkeypatch)
    actions = [run_replication_step(**args) for _ in range(4)]
    assert [x['arm'] for x in actions] == ['base', 'base', 'adapter', 'adapter']
    assert [p['request']['seed'] for _, p in calls] == [20261111, 20261112, 20261111, 20261112]
    assert run_replication_step(**args)['status'] == 'complete'
    termination = json.loads((args['root'] / 'termination.json').read_text())
    assert termination['status'] == 'complete' and termination['registration_sha256'] == descriptor['sha256']
    assert run_replication_step(**args)['status'] == 'complete' and len(calls) == 4


def test_supervisor_rejects_source_drift_and_retains_unresolved_start(tmp_path, monkeypatch):
    args, descriptor, calls = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError): run_replication_step(**{**args, 'challenges': args['challenges'] + b' '})
    assert not calls
    from src.arc.qwen38_weight_journal import AttemptJournal
    with AttemptJournal(args['root'] / 'journal') as journal:
        journal.start({'task_id': 'a', 'candidate_index': 0, 'seed': 20261111,
                       'manifest_sha256': descriptor['sha256'], 'arm': 'base'})
    assert run_replication_step(**args)['status'] == 'requires_reconciliation'
    assert not calls


def test_supervisor_rejects_early_forged_time_termination(tmp_path, monkeypatch):
    from src.arc.qwen38_weight_allocation import freeze_registration
    args, descriptor, calls = fixture(tmp_path, monkeypatch)
    freeze_registration(args['root'] / 'termination.json', {'status': 'time_budget_exhausted',
                        'registration_sha256': descriptor['sha256'], 'journal_sha256': digest([]), 'wall_seconds': 0})
    with pytest.raises(ValueError): run_replication_step(**args)
    assert not calls
