from src.cli.qwen38_canary import resource_limit


def test_resource_limits_fail_closed():
    assert resource_limit(20, 100, 10, 1000) is None
    assert resource_limit(4, 100, 10, 1000) == 'low_free_memory'
    assert resource_limit(20, 1001, 10, 1000) == 'swap_limit'
    assert resource_limit(None, 100, 10, 1000) == 'memory_probe_failed'
    assert resource_limit(20, None, 10, 1000) == 'memory_probe_failed'


def test_canary_instruments_only_expected_training_site():
    import pytest
    from src.cli.qwen38_canary_worker import instrument
    source = '    model = setup_model_for_training(model, args, args.adapter_path)\n    print_trainable_parameters(model)\n'
    result = instrument(source)
    assert 'save_safetensors' in result
    with pytest.raises(ValueError): instrument('unexpected trainer source')
    with pytest.raises(ValueError): instrument(source + source)
