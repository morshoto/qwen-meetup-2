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


def test_rebuild_replays_audit_without_loading_target_answers(tmp_path, monkeypatch):
    import src.cli.qwen38_replication as controller
    repo = tmp_path / 'repo'; repo.mkdir()
    old = tmp_path / 'old'; (old / 'evaluation/adapter-v2').mkdir(parents=True)
    (old / 'training/sft-v1').mkdir(parents=True); (old / 'training/adapter-run-v1/adapter').mkdir(parents=True)
    (old / 'journal').mkdir()
    for name in ('execution', 'proposal', 'history', 'snapshot', 'audit'):
        (tmp_path / (name + '.json')).write_text('{}')
    (tmp_path / 'snapshot.json').write_text(json.dumps({'source_root': str(old)}))
    (tmp_path / 'history.json').write_text(json.dumps({'phase_a_task_ids': [], 'dev_task_ids': []}))
    (old / 'evaluation/evaluation-v2.json').write_text('{}')
    (old / 'evaluation/adapter-v2/report.json').write_bytes(b'report')
    (old / 'training/adapter-run-v1/adapter/adapters.safetensors').write_bytes(b'adapter')
    (old / 'training/sft-v1/manifest.json').write_text(json.dumps({'parents': [{'task_id': 'x'}]}))
    (old / 'journal/000000.start.json').write_text(json.dumps({'request': {'task_id': 'x'}}))
    (tmp_path / 'challenge').write_text('{}'); (tmp_path / 'official').write_text('{}')
    solutions = tmp_path / 'unreadable-solutions'
    calls = []
    monkeypatch.setattr(controller, 'verify_runtime', lambda *a: calls.append('runtime'))
    monkeypatch.setattr(controller, 'verify_frozen_outcome', lambda *a: calls.append('outcome'))
    monkeypatch.setattr(controller, 'build_audit_receipt', lambda *a: calls.append(('audit', a[-1])) or {})
    monkeypatch.setattr(controller, 'build_replication_registration', lambda *a, **k: {'pending': True})
    monkeypatch.setattr(controller, 'build_ready_registration', lambda *a: {'status': 'evaluation_ready'})
    paths = {name: tmp_path / (name + '.json') for name in ('execution','proposal','history','snapshot','audit')}
    paths.update({'repo': repo, 'model_path': tmp_path / 'model', 'old_root': old,
                  'challenges': tmp_path / 'challenge', 'official': tmp_path / 'official',
                  'solutions': solutions, 'run_dir': repo / 'run', 'report': old / 'evaluation/adapter-v2/report.json'})
    result = controller.rebuild_registration(**paths)
    assert result['status'] == 'evaluation_ready'
    assert ('audit', solutions) in calls and 'runtime' in calls and 'outcome' in calls


def test_cli_routes_prepare_run_and_report_without_reading_solutions(tmp_path, monkeypatch, capsys):
    import src.cli.qwen38_replication as controller
    names = ('repo', 'model-path', 'old-root', 'execution', 'proposal', 'history', 'snapshot', 'audit',
             'challenges', 'official', 'solutions', 'run-dir', 'report', 'adapter-path')
    args = []
    for name in names:
        path = tmp_path / name
        if name == 'challenges': path.write_text('{}')
        args += ['--' + name, str(path)]
    calls = []
    monkeypatch.setattr(controller, 'rebuild_registration', lambda **kwargs: {'ready': True})
    monkeypatch.setattr(controller, 'freeze_replication', lambda root, rebuild: calls.append(('prepare', rebuild())) or {'status': 'ready', 'sha256': 'a' * 64})
    monkeypatch.setattr(controller, 'run_replication_step', lambda **kwargs: calls.append(('run', kwargs)) or {'status': 'complete'})
    monkeypatch.setattr(controller, 'report_replication', lambda **kwargs: calls.append(('report', kwargs)) or {'status': 'complete'})
    assert controller.main(['prepare'] + args) == 0
    assert controller.main(['run', '--once'] + args) == 0
    assert controller.main(['report'] + args) == 0
    assert [c[0] for c in calls] == ['prepare', 'run', 'report']
    assert calls[1][1]['challenges'] == b'{}'
    assert 'solutions_path' not in calls[1][1]
    assert calls[2][1]['solutions_path'] == tmp_path / 'solutions'
    assert capsys.readouterr().out
