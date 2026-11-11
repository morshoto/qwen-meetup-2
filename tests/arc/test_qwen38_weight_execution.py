import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.cli import qwen38_weight_execution as execution


def fixture(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; repo.mkdir(); (repo / 'logic.py').write_text('source'); (repo / 'pins.json').write_text('{}')
    proposal = {'status': 'proposal_not_authorization', 'comparison_sha256': 'c' * 64,
                'allocation': {'fresh': ['a'], 'validation': ['b']}, 'generation': {'order': 'breadth_first'},
                'model': {'id': 'model'}, 'input_hashes': {'challenges': 'a' * 64, 'solutions': 'b' * 64}}
    proposal['sha256'] = digest(proposal)
    review = {'status': 'reviewed_bounded', 'proposal_sha256': proposal['sha256']}
    review['sha256'] = digest(review)
    monkeypatch.setattr(execution.preflight, 'PROPOSAL_SHA256', proposal['sha256'])
    monkeypatch.setattr(execution.preflight, 'EXECUTION_SOURCES', ('logic.py',))
    monkeypatch.setattr(execution.preflight, 'PINS_PATH', 'pins.json')
    return repo, proposal, review


def test_descriptor_freezes_review_source_and_limits(tmp_path, monkeypatch):
    repo, proposal, review = fixture(tmp_path, monkeypatch)
    descriptor = execution.make_descriptor(repo, proposal, review, tmp_path / 'run')
    assert descriptor['status'] == 'execution_ready'
    assert descriptor['run_dir'] == str((tmp_path / 'run').resolve())
    assert descriptor['proposal_sha256'] == proposal['sha256']
    assert descriptor['lineage_review_sha256'] == review['sha256']
    assert descriptor['source_files_sha256'] == {'logic.py': hashlib.sha256(b'source').hexdigest()}
    assert descriptor['supervision'] == execution.preflight.SUPERVISION
    assert descriptor['sha256'] == digest({k: v for k, v in descriptor.items() if k != 'sha256'})


def test_descriptor_rejects_unreviewed_or_changed_inputs(tmp_path, monkeypatch):
    repo, proposal, review = fixture(tmp_path, monkeypatch)
    review['status'] = 'draft'
    with pytest.raises(ValueError): execution.make_descriptor(repo, proposal, review, tmp_path / 'run')
    review['status'] = 'reviewed_bounded'; proposal['allocation']['fresh'] = ['changed']
    with pytest.raises(ValueError): execution.make_descriptor(repo, proposal, review, tmp_path / 'run')


def test_freeze_writes_only_after_preflight(tmp_path, monkeypatch):
    repo, proposal, review = fixture(tmp_path, monkeypatch)
    path = tmp_path / 'execution.json'
    monkeypatch.setattr(execution.preflight, 'verify_preflight', lambda *a, **k: {'status': 'failed'})
    with pytest.raises(ValueError):
        execution.freeze_execution(repo, proposal, review, tmp_path, b'c', b's', tmp_path, tmp_path / 'run', path)
    assert not path.exists()
    monkeypatch.setattr(execution.preflight, 'verify_preflight', lambda *a, **k: {'status': 'passed'})
    result = execution.freeze_execution(repo, proposal, review, tmp_path, b'c', b's', tmp_path, tmp_path / 'run', path)
    assert json.loads(path.read_text()) == result
