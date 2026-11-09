import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
import src.cli.qwen38_weight_evaluate as evaluate


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'run'; root.mkdir(mode=0o700); (root / 'training').mkdir(mode=0o700)
    baseline = tmp_path / 'old'; baseline.mkdir(); (baseline / 'evaluation').mkdir(); (baseline / 'evaluation/baseline').mkdir()
    comparison = {'allocation': {'evaluation': ['a']}}; (baseline / 'manifest.json').write_text(json.dumps(comparison))
    (baseline / 'evaluation/baseline/a.json').write_text('{}')
    result = {'status': 'trained', 'sha256': 'a' * 64}; (root / 'training/training-result-v1.json').write_text(json.dumps(result))
    selection = {'mode': 'fresh_only', 'counts': {'unique_parents': 5}}
    registration = {'execution_sha256': 'b' * 64}; receipt = {'status': 'matches_supplied_replay', 'manifest_sha256': 'c' * 64}
    monkeypatch.setattr(evaluate, 'replay_bundle', lambda **kwargs: (selection, registration, receipt))
    monkeypatch.setattr(evaluate, 'verify_training_evidence', lambda *args: {'status': 'source_evidence_matches', 'adapter_sha256': 'd' * 64})
    descriptor = {'status': 'evaluation_ready', 'run_dir': str((root / 'evaluation/adapter-v1').resolve()), 'task_ids': ['a']}
    descriptor['sha256'] = digest(descriptor)
    monkeypatch.setattr(evaluate, 'build_evaluation_descriptor', lambda *args: descriptor)
    kwargs = {'manifest': {'run_dir': str(root)}, 'proposal': {}, 'review': {}, 'repo': tmp_path,
              'model_path': tmp_path / 'model', 'challenges': b'{}', 'solutions': b'{}', 'official_bytes': b'{}',
              'canary_dir': tmp_path, 'run_dir': root, 'history_dir': tmp_path, 'comparison_dir': baseline}
    return kwargs, descriptor


def test_evaluation_registration_requires_replayed_training_and_fixed_path(tmp_path, monkeypatch):
    kwargs, descriptor = fixture(tmp_path, monkeypatch)
    result = evaluate.freeze_evaluation(**kwargs)
    path = kwargs['run_dir'] / 'evaluation/evaluation-v1.json'
    assert result == descriptor and json.loads(path.read_text()) == descriptor
    assert evaluate.verify_evaluation_descriptor(**kwargs) == descriptor


def test_evaluation_refuses_changed_training_or_extra_baseline_rows(tmp_path, monkeypatch):
    kwargs, descriptor = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, 'verify_training_evidence', lambda *args: (_ for _ in ()).throw(ValueError('changed')))
    with pytest.raises(ValueError): evaluate.freeze_evaluation(**kwargs)
    assert not (kwargs['run_dir'] / 'evaluation/evaluation-v1.json').exists()
    monkeypatch.setattr(evaluate, 'verify_training_evidence', lambda *args: {'status': 'source_evidence_matches'})
    (kwargs['comparison_dir'] / 'evaluation/baseline/extra.json').write_text('{}')
    with pytest.raises(ValueError): evaluate.freeze_evaluation(**kwargs)
