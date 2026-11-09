"""Synthetic integration only: no real model or ARC data."""
import hashlib
import json

from src.arc.qwen38_weight_fresh_verify import verify_attempts
from src.arc.qwen38_weight_generation import capture_response, create_request
from src.arc.qwen38_weight_journal import AttemptJournal
from src.arc.qwen38_weight_schedule import next_action


def test_synthetic_collection_replays_and_uses_offline_verification(tmp_path):
    tasks = {k: {'train': [{'input': [[i]], 'output': [[2]]}], 'test': [{'input': [[3, i]]}]}
             for i, k in enumerate(('a', 'b'))}
    answers = {'a': [[[7]]], 'b': [[[8]]]}
    challenges = json.dumps(tasks, sort_keys=True).encode(); solutions = json.dumps(answers, sort_keys=True).encode()
    model = {'id': 'mlx-community/Qwen3.8-27B-4bit', 'revision': '10c35caafbb80f7dc6a7a432cdd11af10a6d4818',
             'quantization': '4-bit MLX'}
    manifest = {'sha256': 'a' * 64, 'allocation': {'fresh': ['a', 'b'], 'validation': [], 'seed': 9},
                'input_hashes': {'challenges': hashlib.sha256(challenges).hexdigest(),
                                 'solutions': hashlib.sha256(solutions).hexdigest()},
                'model': model, 'generation': {'initial_candidates': 2, 'additional_candidates': 6,
                                               'max_tokens_per_candidate': 10, 'reserved_token_limit': 200,
                                               'wall_seconds_limit': 100}}
    with AttemptJournal(tmp_path / 'journal') as journal:
        for _ in range(4):
            rows = journal.read()
            verified = verify_attempts(manifest, challenges, solutions, rows)['outcomes']
            action = next_action(manifest, rows, verified)
            assert action['status'] == 'ready'
            request = create_request(action['task_id'], tasks[action['task_id']],
                                     candidate_index=action['candidate_index'], seed=action['seed'])
            identity = {k: v for k, v in request.items() if k not in ('content', 'test_count')}
            identity.update({'manifest_sha256': manifest['sha256'], 'model': model})
            journal.start(identity)
            raw = json.dumps(answers[action['task_id']])
            capture = capture_response(request, {'raw': raw, 'prompt_tokens': 5, 'completion_tokens': 1,
                                                'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}, 10)
            outcome = {k: v for k, v in capture.items() if k not in identity}
            journal.finish(len(rows), {**outcome, 'status': 'complete', 'elapsed_seconds': 1})
        rows = journal.read()
        verified = verify_attempts(manifest, challenges, solutions, rows)['outcomes']
        assert next_action(manifest, rows, verified)['status'] == 'complete'
    with AttemptJournal(tmp_path / 'journal') as restarted:
        assert len(restarted.read()) == 4
