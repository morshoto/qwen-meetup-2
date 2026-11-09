from src.cli.qwen38_canary import resource_limit


def test_resource_limits_fail_closed():
    assert resource_limit(20, 100, 10, 1000) is None
    assert resource_limit(4, 100, 10, 1000) == 'low_free_memory'
    assert resource_limit(20, 1001, 10, 1000) == 'swap_limit'
    assert resource_limit(None, 100, 10, 1000) == 'memory_probe_failed'
    assert resource_limit(20, None, 10, 1000) == 'memory_probe_failed'
