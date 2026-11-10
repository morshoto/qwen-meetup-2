import copy
import hashlib
import json

import pytest

from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import fingerprint
from src.arc.qwen38_weight_generation import create_request
from src.cli.qwen38_replication_worker import run_replication_request


def fixture():
    task = {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
    request = create_request('a', task, candidate_index=0, seed=20261111)
    descriptor = {'status': 'evaluation_ready', 'inference_authorized': True, 'task_ids': ['a'],
                  'task_fingerprints': {'a': fingerprint(task)}, 'adapter_sha256': 'a' * 64,
                  'adapter_config_sha256': 'b' * 64, 'prompt_sha256': request['prompt_sha256'],
                  'candidate_count': 2, 'temperature': .6, 'max_context': 18432, 'max_tokens': 4096,
                  'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'quantization': '4-bit MLX'}}
    descriptor['sha256'] = digest(descriptor)
    return descriptor, {'request': request, 'max_tokens': 4096}


def test_worker_matches_both_arms_and_passes_no_answers(monkeypatch):
    import src.cli.qwen38_replication_worker as worker
    d, payload = fixture(); calls = []
    monkeypatch.setattr(worker, 'ADAPTER_SHA256', 'a' * 64)
    def base(value, path): calls.append(('base', value, path)); return {'raw': 'base'}
    def adapter(value, path, adapter_path, identity):
        calls.append(('adapter', value, path, adapter_path, identity)); return {'raw': 'adapter'}
    monkeypatch.setattr(worker, 'run_request', base)
    monkeypatch.setattr(worker, 'run_adapter_request', adapter)
    assert run_replication_request(d, 'base', payload, 'model', 'adapter')['raw'] == 'base'
    assert run_replication_request(d, 'adapter', payload, 'model', 'adapter')['raw'] == 'adapter'
    assert calls[1][-1] == {'adapter_sha256': 'a' * 64, 'adapter_config_sha256': 'b' * 64}
    assert 'solutions' not in json.dumps(calls)


@pytest.mark.parametrize('change', ['pending', 'arm', 'task', 'seed', 'payload', 'model', 'digest'])
def test_worker_rejects_drift_before_loading_model(monkeypatch, change):
    import src.cli.qwen38_replication_worker as worker
    d, payload = fixture(); arm = 'base'; calls = []
    monkeypatch.setattr(worker, 'ADAPTER_SHA256', 'a' * 64)
    monkeypatch.setattr(worker, 'run_request', lambda *a: calls.append(a))
    if change == 'pending': d['status'] = 'answer_audit_pending'
    elif change == 'arm': arm = 'other'
    elif change == 'task': payload['request']['task_id'] = 'z'
    elif change == 'seed': payload['request']['seed'] += 1
    elif change == 'payload': payload['request']['content'] += ' '
    elif change == 'model': d['model']['revision'] = 'other'
    else: d['sha256'] = '0' * 64
    if change in ('pending', 'model'): d['sha256'] = digest({k:v for k,v in d.items() if k != 'sha256'})
    with pytest.raises(ValueError): run_replication_request(d, arm, payload, 'model', 'adapter')
    assert not calls


def test_worker_cli_requires_registered_digest_and_writes_once(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    import src.cli.qwen38_replication_worker as worker
    descriptor, payload = fixture()
    registration = tmp_path / 'registration.json'; registration.write_text(json.dumps(descriptor))
    output = tmp_path / 'response.json'
    calls = []
    monkeypatch.setattr(worker, 'run_replication_request', lambda *args: calls.append(args) or {'raw': 'result'})
    monkeypatch.setattr(worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(json.dumps(payload).encode())))
    args = ['--registration', str(registration), '--registration-sha256', descriptor['sha256'],
            '--arm', 'base', '--model-path', 'model', '--adapter-path', 'adapter', '--output', str(output)]
    worker.main(args)
    assert json.loads(output.read_text()) == {'raw': 'result'} and len(calls) == 1
    monkeypatch.setattr(worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(json.dumps(payload).encode())))
    with pytest.raises(FileExistsError): worker.main(args)
    assert len(calls) == 1
    output.unlink()
    args[3] = '0' * 64
    with pytest.raises(ValueError): worker.main(args)
    assert not output.exists()
