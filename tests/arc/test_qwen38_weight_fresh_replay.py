import hashlib
import json
import sys

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_generation import capture_response, create_request
from src.arc.qwen38_weight_journal import AttemptJournal
from src.arc.qwen38_weight_verification_receipt import build_receipt, write_receipt
import src.cli.qwen38_weight_fresh_replay as replay


def fixture(tmp_path, monkeypatch):
    tasks = {'a': {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}}
    solutions = {'a': [[[4]]]}
    challenges = json.dumps(tasks, sort_keys=True).encode(); answers = json.dumps(solutions, sort_keys=True).encode()
    model = {'id': 'mlx-community/Qwen3.8-27B-4bit', 'revision': '10c35caafbb80f7dc6a7a432cdd11af10a6d4818',
             'quantization': '4-bit MLX'}
    root = tmp_path / 'run'; root.mkdir(mode=0o700); (root / 'workers').mkdir(mode=0o700); (root / 'receipts').mkdir(mode=0o700)
    model_path = tmp_path / 'model'
    manifest = {'sha256': 'a' * 64, 'run_dir': str(root.resolve()), 'allocation': {'fresh': ['a'], 'validation': [], 'seed': 9},
                'model': model, 'input_hashes': {'challenges': hashlib.sha256(challenges).hexdigest(),
                                                  'solutions': hashlib.sha256(answers).hexdigest()},
                'generation': {'initial_candidates': 2, 'additional_candidates': 6, 'max_tokens_per_candidate': 10,
                               'reserved_token_limit': 80, 'wall_seconds_limit': 100}, 'supervision': {}}
    start = {'manifest_sha256': manifest['sha256'], 'started_epoch': 1000}; start['sha256'] = digest(start)
    (root / 'run-start.json').write_text(json.dumps(start))
    with AttemptJournal(root / 'journal') as journal:
        for index in range(2):
            request = create_request('a', tasks['a'], candidate_index=index, seed=10009 + index)
            command = [sys.executable, '-u', '-m', 'src.cli.qwen38_weight_worker', '--model-path', str(model_path),
                       '--output', str(root / 'workers' / f'{index:06}' / 'response.json')]
            identity = {k: v for k, v in request.items() if k not in ('content', 'test_count')}
            identity.update({'manifest_sha256': manifest['sha256'], 'model': model, 'worker_command': command})
            journal.start(identity)
            response = capture_response(request, {'raw': '[[[4]]]', 'prompt_tokens': 4, 'completion_tokens': 1,
                                                  'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}, 10)
            journal.finish(index, {**{k: v for k, v in response.items() if k not in identity},
                                   'status': 'complete', 'elapsed_seconds': 1})
        rows = journal.read()
    receipt = build_receipt(manifest, challenges, answers, rows)
    write_receipt(root / 'receipts' / (receipt['journal_sha256'] + '.json'), receipt)
    monkeypatch.setattr(replay, 'verify_preflight', lambda *a, **k: {'status': 'passed'})
    monkeypatch.setattr(replay, '_replay_artifacts', lambda *a, **k: None)
    monkeypatch.setattr(replay.time, 'time', lambda: 1010)
    return manifest, challenges, answers, model_path, root


def test_replay_requires_terminal_receipt_and_preserves_source(tmp_path, monkeypatch):
    manifest, challenges, answers, model, root = fixture(tmp_path, monkeypatch)
    result = replay.replay_fresh(manifest, {}, {}, tmp_path, model, challenges, answers, tmp_path, root)
    assert result['collection_status'] == 'complete'
    assert len(result['records']) == 2
    assert all(r['source'] == 'current_gen0_self_generated' for r in result['records'])


def test_replay_rejects_changed_receipt_and_run_identity(tmp_path, monkeypatch):
    manifest, challenges, answers, model, root = fixture(tmp_path, monkeypatch)
    receipt = next((root / 'receipts').glob('*.json')); receipt.write_text('{}')
    with pytest.raises(ValueError): replay.replay_fresh(manifest, {}, {}, tmp_path, model, challenges, answers, tmp_path, root)
    with pytest.raises(ValueError): replay.replay_fresh(manifest, {}, {}, tmp_path, model, challenges, answers, tmp_path, tmp_path / 'other')
