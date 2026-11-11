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


def test_registration_binds_exclusions_prompt_and_schedule():
    from src.arc.qwen38_weight_allocation import build_registration
    def task(n):
        return {'train': [{'input': [[n]], 'output': [[n + 1]]}], 'test': [{'input': [[n + 200]]}]}
    tasks = {str(i): task(i) for i in range(12)}
    allocations = {'protected': ['0'], 'prior': ['1']}
    manifest = build_registration(tasks, {}, allocations, {},
                                  input_hashes={'challenges': 'a' * 64, 'solutions': 'b' * 64},
                                  comparison_sha256='c' * 64, prompt='direct prompt',
                                  pool_size=5, validation_size=3)
    assert manifest['exclusions'] == allocations
    assert len(manifest['allocation']['fresh']) == 5
    assert manifest['generation']['initial_candidates'] == 2
    assert manifest['generation']['additional_candidates'] == 6
    assert manifest['generation']['order'] == 'breadth_first'
    assert manifest['comparison_sha256'] == 'c' * 64
    assert manifest['generation']['prompt_sha256'] != ''
    with pytest.raises(ValueError):
        build_registration(tasks, {}, allocations, {}, input_hashes={}, comparison_sha256='bad', prompt='')
