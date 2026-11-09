import hashlib
import json
from pathlib import Path

import pytest

from src.arc.qwen38_weight_generation import capture_response
import src.cli.qwen38_weight_collect as collect


def fixture(tmp_path, monkeypatch):
    tasks = {'a': {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}}
    answers = {'a': [[[4]]]}
    challenges = json.dumps(tasks, sort_keys=True).encode(); solutions = json.dumps(answers, sort_keys=True).encode()
    model = {'id': 'mlx-community/Qwen3.8-27B-4bit', 'revision': '10c35caafbb80f7dc6a7a432cdd11af10a6d4818',
             'quantization': '4-bit MLX'}
    manifest = {'status': 'execution_ready', 'sha256': 'a' * 64,
                'allocation': {'fresh': ['a'], 'validation': [], 'seed': 9}, 'model': model,
                'input_hashes': {'challenges': hashlib.sha256(challenges).hexdigest(),
                                 'solutions': hashlib.sha256(solutions).hexdigest()},
                'generation': {'initial_candidates': 2, 'additional_candidates': 6,
                               'max_tokens_per_candidate': 10, 'reserved_token_limit': 80,
                               'wall_seconds_limit': 100},
                'supervision': {'per_attempt_seconds': 10, 'minimum_free_percent': 10,
                                'maximum_swap_bytes': 100, 'max_output_bytes': 10000}}
    monkeypatch.setattr(collect, 'verify_preflight', lambda *a, **k: {'status': 'passed'})
    return manifest, challenges, solutions


def test_collect_step_saves_evidence_and_replays_verifier(tmp_path, monkeypatch):
    manifest, challenges, solutions = fixture(tmp_path, monkeypatch)
    calls = []
    def fake_supervisor(command, payload, directory, **kwargs):
        calls.append(payload)
        assert set(payload) == {'request', 'max_tokens'} and 'solutions' not in json.dumps(payload)
        directory = Path(directory); directory.mkdir()
        for name in ('stdout.log', 'stderr.log'):
            (directory / name).write_text('')
        (directory / 'request.json').write_text(json.dumps(payload))
        execution = {'status': 'complete', 'reason': None, 'exit_code': 0, 'process_group_gone': True, 'elapsed_seconds': 1,
                     'command': command, 'pid': 123, 'samples': [{'free_percent': 80, 'swap_bytes': 0}]}
        (directory / 'execution.json').write_text(json.dumps(execution))
        response = {'raw': '[[[4]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                    'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}
        captured = capture_response(payload['request'], response, payload['max_tokens'])
        (directory / 'response.json').write_text(json.dumps(captured))
        return execution
    monkeypatch.setattr(collect, 'run_supervised', fake_supervisor)
    kwargs = dict(manifest=manifest, proposal={}, review={}, repo=tmp_path, model_path=tmp_path,
                  challenges=challenges, solutions=solutions, canary_dir=tmp_path, run_dir=tmp_path / 'run')
    first = collect.collect_step(**kwargs)
    assert first['status'] == 'attempt_recorded' and len(calls) == 1
    second = collect.collect_step(**kwargs)
    assert second['status'] == 'attempt_recorded' and len(calls) == 2
    final = collect.collect_step(**kwargs)
    assert final['status'] == 'complete' and len(calls) == 2
    assert len(list((tmp_path / 'run/receipts').glob('*.json'))) == 3


def test_unresolved_attempt_never_restarts_and_proposal_cannot_dispatch(tmp_path, monkeypatch):
    manifest, challenges, solutions = fixture(tmp_path, monkeypatch)
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('interrupted')
    monkeypatch.setattr(collect, 'run_supervised', interrupted)
    kwargs = dict(manifest=manifest, proposal={}, review={}, repo=tmp_path, model_path=tmp_path,
                  challenges=challenges, solutions=solutions, canary_dir=tmp_path, run_dir=tmp_path / 'run')
    with pytest.raises(RuntimeError): collect.collect_step(**kwargs)
    assert collect.collect_step(**kwargs)['status'] == 'requires_reconciliation' and len(calls) == 1
    manifest['status'] = 'proposal_not_authorization'
    with pytest.raises(ValueError): collect.collect_step(**kwargs)


def test_worker_cleanup_evidence_must_match_before_commit(tmp_path, monkeypatch):
    manifest, challenges, solutions = fixture(tmp_path, monkeypatch)
    def inconsistent(command, payload, directory, **kwargs):
        directory = Path(directory); directory.mkdir()
        for name in ('stdout.log', 'stderr.log'):
            (directory / name).write_text('')
        (directory / 'request.json').write_text(json.dumps(payload))
        saved = {'status': 'error', 'reason': 'remaining_process_group', 'exit_code': 0,
                 'process_group_gone': False, 'elapsed_seconds': 1}
        (directory / 'execution.json').write_text(json.dumps(saved))
        response = {'raw': '[[[4]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                    'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}
        (directory / 'response.json').write_text(json.dumps(capture_response(payload['request'], response, payload['max_tokens'])))
        return {**saved, 'status': 'complete', 'reason': None, 'process_group_gone': True}
    monkeypatch.setattr(collect, 'run_supervised', inconsistent)
    kwargs = dict(manifest=manifest, proposal={}, review={}, repo=tmp_path, model_path=tmp_path,
                  challenges=challenges, solutions=solutions, canary_dir=tmp_path, run_dir=tmp_path / 'run')
    with pytest.raises(ValueError): collect.collect_step(**kwargs)
    from src.arc.qwen38_weight_journal import AttemptJournal
    with AttemptJournal(tmp_path / 'run/journal') as journal:
        assert journal.read()[-1]['status'] == 'started'


def test_worker_command_must_match_registered_dispatch(tmp_path, monkeypatch):
    manifest, challenges, solutions = fixture(tmp_path, monkeypatch)
    def wrong_command(command, payload, directory, **kwargs):
        directory = Path(directory); directory.mkdir()
        for name in ('stdout.log', 'stderr.log'):
            (directory / name).write_text('')
        (directory / 'request.json').write_text(json.dumps(payload))
        execution = {'status': 'complete', 'reason': None, 'exit_code': 0, 'process_group_gone': True,
                     'elapsed_seconds': 1, 'command': ['unexpected'], 'pid': 123,
                     'samples': [{'free_percent': 80, 'swap_bytes': 0}]}
        (directory / 'execution.json').write_text(json.dumps(execution))
        response = {'raw': '[[[4]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                    'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}
        (directory / 'response.json').write_text(json.dumps(capture_response(payload['request'], response, payload['max_tokens'])))
        return execution
    monkeypatch.setattr(collect, 'run_supervised', wrong_command)
    kwargs = dict(manifest=manifest, proposal={}, review={}, repo=tmp_path, model_path=tmp_path,
                  challenges=challenges, solutions=solutions, canary_dir=tmp_path, run_dir=tmp_path / 'run')
    with pytest.raises(ValueError): collect.collect_step(**kwargs)


def test_collection_cli_loops_until_terminal_without_printing_labels(tmp_path, monkeypatch, capsys):
    manifest, challenges, solutions = fixture(tmp_path, monkeypatch)
    for name, content in [('manifest.json', json.dumps(manifest).encode()), ('proposal.json', b'{}'),
                          ('review.json', b'{}'), ('challenges.json', challenges), ('solutions.json', solutions)]:
        (tmp_path / name).write_bytes(content)
    calls = []
    def fake_step(**kwargs):
        calls.append(kwargs)
        return {'status': 'attempt_recorded', 'task_id': 'a', 'candidate_index': 0} if len(calls) == 1 else {'status': 'complete'}
    monkeypatch.setattr(collect, 'collect_step', fake_step)
    args = ['--manifest', str(tmp_path / 'manifest.json'), '--proposal', str(tmp_path / 'proposal.json'),
            '--review', str(tmp_path / 'review.json'), '--repo', str(tmp_path), '--model-path', str(tmp_path),
            '--challenges', str(tmp_path / 'challenges.json'), '--solutions', str(tmp_path / 'solutions.json'),
            '--canary-dir', str(tmp_path), '--run-dir', str(tmp_path / 'run')]
    assert collect.main(args) == 0
    output = capsys.readouterr().out
    assert len(calls) == 2 and 'complete' in output and '[[[4]]]' not in output


def test_resource_failure_is_recorded_and_prevents_further_dispatch(tmp_path, monkeypatch):
    manifest, challenges, solutions = fixture(tmp_path, monkeypatch); calls = []
    def resource_failure(command, payload, directory, **kwargs):
        calls.append(1); directory = Path(directory); directory.mkdir()
        (directory / 'request.json').write_text(json.dumps(payload))
        for name in ('stdout.log', 'stderr.log'): (directory / name).write_text('')
        execution = {'status': 'error', 'reason': 'low_free_memory', 'exit_code': -15,
                     'process_group_gone': True, 'elapsed_seconds': 1, 'command': command,
                     'pid': 123, 'samples': [{'free_percent': 4, 'swap_bytes': 0}]}
        (directory / 'execution.json').write_text(json.dumps(execution))
        return execution
    monkeypatch.setattr(collect, 'run_supervised', resource_failure)
    kwargs = dict(manifest=manifest, proposal={}, review={}, repo=tmp_path, model_path=tmp_path,
                  challenges=challenges, solutions=solutions, canary_dir=tmp_path, run_dir=tmp_path / 'run')
    assert collect.collect_step(**kwargs)['attempt_status'] == 'error'
    assert collect.collect_step(**kwargs)['status'] == 'resource_stop'
    assert len(calls) == 1
