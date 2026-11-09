import hashlib

import pytest

from src.arc.qwen38_local_runner import MODEL_REVISION
from src.arc.qwen38_weight_generation import create_request
from src.cli.qwen38_weight_eval_worker import run_adapter_request


class FakeBackend:
    calls = []
    def __init__(self, model_path, *, adapter_path):
        self.calls.append(('load', model_path, adapter_path))
    def call(self, content, seed, max_tokens):
        self.calls.append(('call', content, seed, max_tokens))
        return {'raw': '[[[5]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}


def fixture(tmp_path):
    model = tmp_path / MODEL_REVISION; model.mkdir()
    adapter = tmp_path / 'adapter'; adapter.mkdir(mode=0o700)
    (adapter / 'adapters.safetensors').write_bytes(b'adapter'); (adapter / 'adapter_config.json').write_bytes(b'config')
    identity = {'adapter_sha256': hashlib.sha256(b'adapter').hexdigest(),
                'adapter_config_sha256': hashlib.sha256(b'config').hexdigest()}
    task = {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
    request = create_request('a', task, candidate_index=0, seed=20261111)
    return model, adapter, identity, {'request': request, 'max_tokens': 4096}


def test_adapter_worker_binds_saved_weights_and_baseline_request(tmp_path):
    model, adapter, identity, payload = fixture(tmp_path); FakeBackend.calls = []
    result = run_adapter_request(payload, model, adapter, identity, backend_factory=FakeBackend)
    assert result['raw'] == '[[[5]]]'
    assert FakeBackend.calls[0] == ('load', model, adapter)
    assert FakeBackend.calls[1][2:] == (20261111, 4096)
    assert 'correct' not in result


def test_adapter_worker_rejects_changed_adapter_before_model_load(tmp_path):
    model, adapter, identity, payload = fixture(tmp_path); FakeBackend.calls = []
    (adapter / 'adapter_config.json').write_bytes(b'changed')
    with pytest.raises(ValueError): run_adapter_request(payload, model, adapter, identity, backend_factory=FakeBackend)
    assert not FakeBackend.calls
