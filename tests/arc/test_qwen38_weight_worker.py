import copy
import json

import pytest

from src.arc.qwen38_local_runner import MODEL_REVISION
from src.arc.qwen38_weight_generation import create_request
from src.cli.qwen38_weight_worker import run_request


class FakeBackend:
    calls = []

    def __init__(self, path):
        self.path = path

    def call(self, content, seed, max_tokens):
        self.calls.append((content, seed, max_tokens))
        return {'raw': '[[[5]]]', 'prompt_tokens': 12, 'completion_tokens': 4,
                'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 3}


def fixture(tmp_path):
    model = tmp_path / MODEL_REVISION; model.mkdir()
    task = {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
    request = create_request('a', task, candidate_index=0, seed=42)
    return model, {'request': request, 'max_tokens': 4096}


def test_worker_dispatches_only_registered_visible_request(tmp_path):
    FakeBackend.calls = []
    model, payload = fixture(tmp_path)
    result = run_request(payload, model, backend_factory=FakeBackend)
    assert len(FakeBackend.calls) == 1
    assert FakeBackend.calls[0] == (payload['request']['content'], 42, 4096)
    assert result['raw'] == '[[[5]]]' and 'correct' not in result


@pytest.mark.parametrize('change', ['label', 'prompt', 'seed', 'extra', 'model'])
def test_worker_rejects_changed_or_leaking_request_before_model_load(tmp_path, change):
    FakeBackend.calls = []
    model, payload = fixture(tmp_path); payload = copy.deepcopy(payload)
    if change == 'label':
        content = json.loads(payload['request']['content']); content['task']['test'][0]['output'] = [[9]]
        payload['request']['content'] = json.dumps(content)
    elif change == 'prompt': payload['request']['prompt_sha256'] = 'a' * 64
    elif change == 'seed': payload['request']['seed'] = -1
    elif change == 'extra': payload['solutions'] = {'a': [[[9]]]}
    else: model = tmp_path / 'not-the-pinned-snapshot'
    with pytest.raises(ValueError): run_request(payload, model, backend_factory=FakeBackend)
    assert FakeBackend.calls == []
