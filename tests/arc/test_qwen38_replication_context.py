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
