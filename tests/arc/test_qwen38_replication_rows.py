import hashlib

import pytest

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_generation import create_request, capture_response
from src.arc.qwen38_replication_rows import assemble_replication_rows


def fixture():
    tasks = {'a': {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}}
    d = {'task_ids': ['a'], 'candidate_count': 2, 'max_tokens': 4096, 'max_context': 18432,
         'reserved_completion_tokens': 4*4096, 'wall_seconds_limit': 1000,
         'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(), 'adapter_sha256': 'a'*64}
    d['sha256'] = digest(d)
    rows = []
    for arm in ('base', 'adapter'):
        for i in (0, 1):
            req = create_request('a', tasks['a'], candidate_index=i, seed=20261111+i)
            response = {'raw': '[[[4]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                        'seconds': 1.0, 'peak_memory_gb': 1.0, 'finish_reason': 'stop'}
            rows.append({**capture_response(req, response, 4096), 'arm': arm,
                         'manifest_sha256': d['sha256'], 'status': 'complete', 'elapsed_seconds': 1.0})
    return d, tasks, rows


def test_assemble_both_arms_and_preserve_duplicate_candidates():
    d, tasks, journal = fixture()
    result = assemble_replication_rows(d, tasks, journal)
    assert result['schedule_status'] == 'complete' and not result['errors']
    for arm in ('base', 'adapter'):
        item = result[arm][0]
        assert item['status'] == 'complete' and item['runtime_seconds'] == 2
        assert item['candidates'][0]['raw'] == item['candidates'][1]['raw']
        assert item['manifest_sha256'] == d['sha256']
    assert result['base'][0]['variant_sha256'] is None
    assert result['adapter'][0]['variant_sha256'] == d['adapter_sha256']


def test_assemble_retains_incomplete_and_rejects_response_drift():
    d, tasks, journal = fixture()
    result = assemble_replication_rows(d, tasks, journal[:1])
    assert result['base'][0]['status'] == 'incomplete' and result['adapter'] == []
    journal[0]['raw_sha256'] = 'f'*64
    with pytest.raises(ValueError): assemble_replication_rows(d, tasks, journal)
