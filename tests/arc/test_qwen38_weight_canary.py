import copy

import pytest

from src.arc.qwen38_weight_canary import validate_evidence


def evidence():
    return {
        'exit_code': 0, 'loss': 1.2, 'base_before': {'model.safetensors': 'a' * 64},
        'base_after': {'model.safetensors': 'a' * 64}, 'adapter_sha256': 'b' * 64,
        'changed_lora_tensors': 1, 'max_abs_logit_delta': 0.001,
        'timed_out': False, 'memory_limit_exceeded': False, 'process_group_gone': True,
    }


def test_canary_requires_complete_evidence():
    assert validate_evidence(evidence()) == 'passed'
    for key, bad in [('exit_code', 1), ('loss', float('nan')), ('loss', float('inf')),
                     ('base_after', {}), ('adapter_sha256', ''), ('changed_lora_tensors', 0),
                     ('max_abs_logit_delta', 0), ('max_abs_logit_delta', float('nan')),
                     ('timed_out', True), ('memory_limit_exceeded', True), ('process_group_gone', False)]:
        record = copy.deepcopy(evidence()); record[key] = bad
        with pytest.raises(ValueError):
            validate_evidence(record)


def test_canary_rejects_missing_or_invalid_hashes():
    record = evidence(); record['base_before'] = {}; record['base_after'] = {}
    with pytest.raises(ValueError): validate_evidence(record)
    record = evidence(); record['adapter_sha256'] = 'x' * 64
    with pytest.raises(ValueError): validate_evidence(record)


def test_loss_parser_requires_one_finite_step():
    from src.arc.qwen38_weight_canary import parse_loss
    assert parse_loss('Iter 1: Train loss \x1b[92m0.125\x1b[0m, Learning Rate') == 0.125
    for log in ('', 'Iter 1: Train loss nan, Learning Rate', 'Iter 1: Train loss 0.1,\nIter 1: Train loss 0.2,'):
        with pytest.raises(ValueError): parse_loss(log)
