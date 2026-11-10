import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_replication_fingerprints import permitted_fingerprints


def test_only_allowed_answers_are_decoded_and_fingerprinted(tmp_path, monkeypatch):
    import src.arc.qwen38_replication_fingerprints as module
    data = {'blocked_a': [[[1, 2]]], 'held_out': [['secret-marker']], 'blocked_b': [[[3]]]}
    raw = json.dumps(data, indent=2).encode()
    path = tmp_path / 'solutions.json'; path.write_bytes(raw)
    original = json.loads; decoded = []
    def checked(value, *args, **kwargs):
        assert 'secret-marker' not in (value.decode() if isinstance(value, bytes) else value)
        decoded.append(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(module.json, 'loads', checked)
    result = permitted_fingerprints(path, {'blocked_a', 'blocked_b'}, {'held_out'}, hashlib.sha256(raw).hexdigest())
    assert result == {'blocked_a': digest(data['blocked_a']), 'blocked_b': digest(data['blocked_b'])}
    assert 'secret-marker' not in repr(result)
    assert decoded


def test_extractor_fails_closed_on_source_drift_overlap_or_missing(tmp_path):
    raw = b'{"a": [[[1]]], "withheld": [[[2]]]}'
    path = tmp_path / 'solutions.json'; path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError): permitted_fingerprints(path, {'a'}, {'withheld'}, '0' * 64)
    with pytest.raises(ValueError): permitted_fingerprints(path, {'withheld'}, {'withheld'}, sha)
    with pytest.raises(ValueError): permitted_fingerprints(path, {'missing'}, {'withheld'}, sha)
    with pytest.raises(ValueError): permitted_fingerprints(path, {'a'}, {'absent'}, sha)
