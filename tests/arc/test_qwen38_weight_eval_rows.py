import copy
import hashlib

import pytest

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_weight_eval_rows import assemble_rows
from src.arc.qwen38_weight_generation import capture_response, create_request


def fixture():
    tasks = {'a': {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]},
             'b': {'train': [{'input': [[2]], 'output': [[1]]}], 'test': [{'input': [[4]]}]}}
    descriptor = {'sha256': 'a' * 64, 'comparison_sha256': 'b' * 64, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                  'adapter_sha256': 'd' * 64, 'task_ids': ['a', 'b'], 'candidate_count': 2, 'max_tokens': 4096,
                  'reserved_completion_tokens': 16384, 'wall_seconds_limit': 100}
    rows = []
    for task_id, index in [('a', 0), ('a', 1), ('b', 0)]:
        request = create_request(task_id, tasks[task_id], candidate_index=index, seed=20261111 + index)
        response = {'raw': '[[[5]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                    'finish_reason': 'stop', 'seconds': 2, 'peak_memory_gb': 3}
        rows.append({**capture_response(request, response, 4096), 'status': 'complete', 'elapsed_seconds': 2,
                     'manifest_sha256': descriptor['sha256']})
    rows.append({'task_id': 'b', 'candidate_index': 1, 'seed': 20261112, 'status': 'error',
                 'error': 'timeout', 'elapsed_seconds': 3, 'manifest_sha256': descriptor['sha256']})
    return descriptor, tasks, rows


def test_assembly_preserves_complete_pairs_and_marks_failed_tasks_incomplete():
    descriptor, tasks, rows = fixture()
    result = assemble_rows(descriptor, tasks, rows)
    assert result['status'] == 'incomplete'
    assert result['rows'][0]['task_id'] == 'a' and result['rows'][0]['status'] == 'complete'
    assert result['rows'][0]['variant_sha256'] == descriptor['adapter_sha256']
    assert result['rows'][1]['task_id'] == 'b' and result['rows'][1]['status'] == 'incomplete'
    assert result['errors'] == {'timeout': 1}


def test_assembly_rejects_changed_response_or_identity():
    descriptor, tasks, rows = fixture(); changed = copy.deepcopy(rows); changed[0]['outputs'] = [[[9]]]
    with pytest.raises(ValueError): assemble_rows(descriptor, tasks, changed)
    changed = copy.deepcopy(rows); changed[1]['seed'] += 1
    with pytest.raises(ValueError): assemble_rows(descriptor, tasks, changed)
