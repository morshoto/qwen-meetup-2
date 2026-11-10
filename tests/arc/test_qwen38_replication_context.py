import json
import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.cli.qwen38_replication import freeze_replication, verify_replication


def ready(root):
    d = {'status': 'evaluation_ready', 'inference_authorized': True, 'run_dir': str(root)}
    d['sha256'] = digest(d)
    return d


def test_registration_freeze_and_replay_detect_drift(tmp_path):
    root = tmp_path / 'run'; root.mkdir(mode=0o700)
    source = ready(root)
    calls = []
    def rebuild(): calls.append(1); return source
    assert freeze_replication(root, rebuild) == source
    assert verify_replication(root, rebuild) == source
    assert len(calls) == 2
    changed = dict(source, inference_authorized=False)
    changed['sha256'] = digest({k:v for k,v in changed.items() if k != 'sha256'})
    with pytest.raises(ValueError): verify_replication(root, lambda: changed)
    assert json.loads((root / 'registration.json').read_text()) == source


def test_registration_fails_closed_for_pending_and_nonprivate_roots(tmp_path):
    root = tmp_path / 'run'; root.mkdir(mode=0o755)
    with pytest.raises(ValueError): freeze_replication(root, lambda: ready(root))
    root.chmod(0o700)
    pending = ready(root); pending['status'] = 'answer_audit_pending'; pending['inference_authorized'] = False
    pending['sha256'] = digest({k:v for k,v in pending.items() if k != 'sha256'})
    with pytest.raises(ValueError): freeze_replication(root, lambda: pending)
    assert not (root / 'registration.json').exists()


def test_runtime_replay_rejects_model_or_package_drift(tmp_path, monkeypatch):
    import hashlib
    import src.cli.qwen38_replication as controller
    repo = tmp_path / 'repo'; repo.mkdir(); (repo / 'source.py').write_bytes(b'source')
    model = tmp_path / 'model'; model.mkdir(); (model / 'weight').write_bytes(b'weight')
    sha = lambda value: hashlib.sha256(value).hexdigest()
    execution = {'source_files_sha256': {'source.py': sha(b'source')},
                 'model': {'files_sha256': {'weight': sha(b'weight')}}, 'packages': {'example': '1'}}
    monkeypatch.setattr(controller.metadata, 'version', lambda name: '1')
    assert controller.verify_runtime(repo, model, execution)['status'] == 'runtime_matches'
    (model / 'weight').write_bytes(b'drift')
    with pytest.raises(ValueError): controller.verify_runtime(repo, model, execution)
    (model / 'weight').write_bytes(b'weight')
    monkeypatch.setattr(controller.metadata, 'version', lambda name: '2')
    with pytest.raises(ValueError): controller.verify_runtime(repo, model, execution)
