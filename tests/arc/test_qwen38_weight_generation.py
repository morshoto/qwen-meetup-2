import hashlib
import json

import pytest

from src.arc.qwen38_weight_generation import create_request, capture_response
from src.arc.lt_live_backend import PROMPT


def task():
    return {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]], 'output': [[9]]}]}


def test_request_matches_baseline_format_without_test_answers():
    request = create_request('a', task(), candidate_index=1, seed=101)
    content = json.loads(request['content'])
    assert content['instruction'] == PROMPT
    assert content['task'] == {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
    assert request['task_id'] == 'a' and request['seed'] == 101
    assert request['prompt_sha256'] == hashlib.sha256(PROMPT.encode()).hexdigest()


def test_capture_keeps_original_and_parser_result_without_grading():
    request = create_request('a', task(), candidate_index=0, seed=100)
    result = capture_response(request, {'raw': '[[[4]]]', 'prompt_tokens': 23, 'completion_tokens': 5,
                                         'finish_reason': 'stop', 'seconds': 1.2, 'peak_memory_gb': 4.0}, 4096)
    assert result['raw'] == '[[[4]]]'
    assert result['outputs'] == [[[4]]] and result['valid'] is True
    assert result['raw_sha256'] == hashlib.sha256(b'[[[4]]]').hexdigest()
    assert 'correct' not in result and 'solution' not in result
    assert result['candidate_index'] == 0 and result['seed'] == 100


@pytest.mark.parametrize('field,value', [('raw', None), ('prompt_tokens', -1), ('completion_tokens', 4097),
                                          ('seconds', float('nan')), ('finish_reason', None)])
def test_capture_rejects_unverifiable_backend_metadata(field, value):
    request = create_request('a', task(), candidate_index=0, seed=100)
    response = {'raw': '[[[4]]]', 'prompt_tokens': 23, 'completion_tokens': 5,
                'finish_reason': 'stop', 'seconds': 1.2, 'peak_memory_gb': 4.0}
    response[field] = value
    with pytest.raises(ValueError): capture_response(request, response, 4096)
