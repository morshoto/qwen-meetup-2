import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_replication_freeze import verify_frozen_outcome


def fixture():
    adapter = b'frozen adapter'
    report = {'status': 'complete', 'schedule_status': 'complete',
              'adapter_sha256': hashlib.sha256(adapter).hexdigest(),
              'paired': {'paired': {'complete': 50, 'gained': 4, 'lost': 1,
                                    'both_correct': 1, 'both_incorrect': 44}}}
    report['sha256'] = digest(report)
    return adapter, report


def test_freeze_binds_original_report_and_adapter(monkeypatch):
    import src.arc.qwen38_replication_freeze as freeze
    adapter, report = fixture()
    monkeypatch.setattr(freeze, 'ADAPTER_SHA256', hashlib.sha256(adapter).hexdigest())
    monkeypatch.setattr(freeze, 'REPORT_SHA256', report['sha256'])
    result = verify_frozen_outcome(adapter, json.dumps(report).encode())
    assert result['adapter_sha256'] == report['adapter_sha256']
    assert result['report_sha256'] == report['sha256']
    assert result['status'] == 'frozen_outcome_verified'


def test_freeze_rejects_changed_adapter_or_report(monkeypatch):
    import src.arc.qwen38_replication_freeze as freeze
    adapter, report = fixture()
    monkeypatch.setattr(freeze, 'ADAPTER_SHA256', hashlib.sha256(adapter).hexdigest())
    monkeypatch.setattr(freeze, 'REPORT_SHA256', report['sha256'])
    with pytest.raises(ValueError):
        verify_frozen_outcome(adapter + b'changed', json.dumps(report).encode())
    report['paired']['paired']['gained'] = 5
    with pytest.raises(ValueError):
        verify_frozen_outcome(adapter, json.dumps(report).encode())


def test_frozen_inventory_rejects_missing_changed_and_extra_files(tmp_path):
    from src.arc.qwen38_replication_freeze import inventory_tree, verify_inventory
    root = tmp_path / 'evidence'; root.mkdir()
    (root / 'journal').mkdir()
    (root / 'journal/one.json').write_bytes(b'one')
    recorded = inventory_tree(root)
    assert verify_inventory(root, recorded) == recorded
    (root / 'journal/two.json').write_bytes(b'two')
    with pytest.raises(ValueError): verify_inventory(root, recorded)
    (root / 'journal/two.json').unlink()
    (root / 'journal/one.json').write_bytes(b'changed')
    with pytest.raises(ValueError): verify_inventory(root, recorded)
    (root / 'journal/one.json').unlink()
    with pytest.raises(ValueError): verify_inventory(root, recorded)


def test_frozen_inventory_rejects_symlinks(tmp_path):
    from src.arc.qwen38_replication_freeze import inventory_tree
    root = tmp_path / 'evidence'; root.mkdir()
    target = tmp_path / 'outside'; target.write_text('outside')
    (root / 'alias').symlink_to(target)
    with pytest.raises(ValueError): inventory_tree(root)


def test_snapshot_requires_pinned_files_and_is_write_once(tmp_path):
    from src.arc.qwen38_replication_freeze import freeze_snapshot
    root = tmp_path / 'source'; root.mkdir()
    (root / 'pinned').write_bytes(b'original')
    (root / 'other').write_bytes(b'evidence')
    output = tmp_path / 'snapshot.json'
    expected = {'pinned': hashlib.sha256(b'original').hexdigest()}
    snapshot = freeze_snapshot(root, output, expected)
    assert snapshot['status'] == 'frozen_inventory' and output.is_file()
    assert freeze_snapshot(root, output, expected) == snapshot
    (root / 'other').write_bytes(b'changed')
    with pytest.raises(ValueError): freeze_snapshot(root, output, expected)
    (root / 'pinned').write_bytes(b'changed')
    with pytest.raises(ValueError): freeze_snapshot(root, tmp_path / 'bad.json', expected)
    assert not (tmp_path / 'bad.json').exists()


def test_snapshot_rejects_symlinked_source_directory(tmp_path):
    from src.arc.qwen38_replication_freeze import freeze_snapshot
    root = tmp_path / 'source'; root.mkdir()
    (root / 'pinned').write_bytes(b'original')
    link = tmp_path / 'link'; link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        freeze_snapshot(link, tmp_path / 'out.json', {'pinned': hashlib.sha256(b'original').hexdigest()})


def test_pinned_file_verification_rejects_drift_traversal_and_symlinks(tmp_path):
    from src.arc.qwen38_replication_freeze import verify_pinned_files
    root = tmp_path / 'root'; root.mkdir()
    (root / 'model').mkdir()
    (root / 'model/tokenizer').write_bytes(b'tokenizer')
    digest_value = hashlib.sha256(b'tokenizer').hexdigest()
    assert verify_pinned_files(root, {'model/tokenizer': digest_value}) == {'model/tokenizer': digest_value}
    with pytest.raises(ValueError): verify_pinned_files(root, {'../outside': digest_value})
    (root / 'alias').symlink_to(root / 'model', target_is_directory=True)
    with pytest.raises(ValueError): verify_pinned_files(root, {'alias/tokenizer': digest_value})
    (root / 'model/tokenizer').write_bytes(b'drift')
    with pytest.raises(ValueError): verify_pinned_files(root, {'model/tokenizer': digest_value})
