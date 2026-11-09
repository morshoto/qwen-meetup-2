import json

import pytest

from src.arc.qwen38_weight_allocation import freeze_registration


def test_registration_is_immutable_and_rechecks_digest(tmp_path):
    path = tmp_path / 'manifest.json'
    value = {'protocol': 'test', 'fresh': ['a'], 'validation': ['b']}
    frozen = freeze_registration(path, value)
    assert frozen == json.loads(path.read_text())
    assert freeze_registration(path, value) == frozen
    with pytest.raises(ValueError): freeze_registration(path, {'protocol': 'other'})
    path.write_text('{}')
    with pytest.raises(ValueError): freeze_registration(path, value)
