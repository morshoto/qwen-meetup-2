import hashlib
import json
from pathlib import Path

import pytest


def test_recovery_requires_exact_pre_inference_failure(tmp_path, monkeypatch):
    from src.cli import qwen38_weight_evaluate_recovery as recovery
    root = tmp_path / 'evaluation'
    worker = root / 'adapter-v1/workers/000000'
    journal = root / 'adapter-v1/journal'
    worker.mkdir(parents=True); journal.mkdir(parents=True)
    (journal / '.lock').touch()
    files = {
        'evaluation-v1.json': b'old descriptor',
        'adapter-v1/termination.json': b'old stop',
        'adapter-v1/journal/000000.start.json': b'old start',
        'adapter-v1/journal/000000.result.json': b'old result',
        'adapter-v1/workers/000000/execution.json': b'old execution',
        'adapter-v1/workers/000000/stderr.log': b'old error',
        'adapter-v1/workers/000000/stdout.log': b'',
    }
    for name, content in files.items(): (root / name).write_bytes(content)
    monkeypatch.setattr(recovery, 'FAILURE_HASHES', {name: hashlib.sha256(content).hexdigest() for name, content in files.items()})
    evidence = recovery.verify_preinference_failure(root)
    assert evidence['status'] == 'preinference_failure_verified'
    (worker / 'response.json').write_text('{}')
    with pytest.raises(ValueError): recovery.verify_preinference_failure(root)
    (worker / 'response.json').unlink()
    (journal / '000001.start.json').write_text('{}')
    with pytest.raises(ValueError): recovery.verify_preinference_failure(root)


def test_recovery_registration_uses_new_path_and_binds_failure(tmp_path, monkeypatch):
    from src.arc.qwen38_rsi_comparison import digest
    from src.cli import qwen38_weight_evaluate_recovery as recovery
    root = tmp_path / 'run'; root.mkdir(); (root / 'training').mkdir()
    old = tmp_path / 'old'; (old / 'evaluation/baseline').mkdir(parents=True)
    (old / 'manifest.json').write_text(json.dumps({'allocation': {'evaluation': ['a']}}))
    (old / 'evaluation/baseline/a.json').write_text('{}')
    (root / 'training/training-result-v1.json').write_text(json.dumps({'status': 'trained', 'sha256': 'a' * 64}))
    monkeypatch.setattr(recovery, 'replay_bundle', lambda **kwargs: (
        {'status': 'eligible_pending_source_replay', 'counts': {'unique_parents': 5}}, {},
        {'status': 'matches_supplied_replay', 'manifest_sha256': 'b' * 64}))
    monkeypatch.setattr(recovery, 'verify_training_evidence', lambda *args: {'status': 'source_evidence_matches'})
    monkeypatch.setattr(recovery, 'verify_preinference_failure', lambda path: {'status': 'preinference_failure_verified'})
    original = {'status': 'evaluation_ready', 'run_dir': str(root / 'evaluation/adapter-v1'),
                'source_files_sha256': {}, 'limitations': [], 'sha256': 'original'}
    monkeypatch.setattr(recovery, 'build_evaluation_descriptor', lambda *args: dict(original))
    repo = Path.cwd()
    descriptor, *_ = recovery._context(run_dir=root, comparison_dir=old, manifest={'run_dir': str(root)},
        repo=repo, challenges=b'{}', model_path=tmp_path)
    assert descriptor['run_dir'] == str(root / 'evaluation/adapter-v2')
    assert descriptor['previous_failure']['status'] == 'preinference_failure_verified'
    assert descriptor['sha256'] == digest({k: v for k, v in descriptor.items() if k != 'sha256'})
    assert 'src/cli/qwen38_weight_evaluate_recovery.py' in descriptor['source_files_sha256']
    assert original['run_dir'].endswith('adapter-v1')
