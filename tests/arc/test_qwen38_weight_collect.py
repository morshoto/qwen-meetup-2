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
        execution = {'status': 'complete', 'reason': None, 'exit_code': 0, 'process_group_gone': True, 'elapsed_seconds': 1}
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
