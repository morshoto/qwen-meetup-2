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
