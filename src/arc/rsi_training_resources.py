"""Evaluate recorded training samples; this does not start or stop a process."""
import math

_GIB = 1024 ** 3
_LIMIT_CONSTRAINTS = {'timeout_seconds': (1, 300), 'minimum_disk_gib': (20, 1000000),
                      'minimum_free_memory_percent': (20, 100), 'maximum_swap_growth_mib': (1, 512),
                      'maximum_process_rss_gib': (1, 32), 'sample_interval_seconds': (1, 5)}
_DEFAULT_LIMITS = {'timeout_seconds': 300, 'minimum_disk_gib': 20, 'minimum_free_memory_percent': 20,
                   'maximum_swap_growth_mib': 512, 'maximum_process_rss_gib': 32, 'sample_interval_seconds': 5}


def valid_resource_limits(limits):
    return (isinstance(limits, dict) and set(limits) == set(_LIMIT_CONSTRAINTS) and
            all(type(limits[key]) is int and lo <= limits[key] <= hi
                for key, (lo, hi) in _LIMIT_CONSTRAINTS.items()))
_KEYS = ('monotonic_seconds', 'disk_free_bytes', 'memory_free_percent',
         'swap_used_mib', 'process_tree_rss_bytes')


def _valid(sample: object) -> bool:
    if not isinstance(sample, dict):
        return False
    for key in _KEYS:
        value = sample.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return False
    return sample['memory_free_percent'] <= 100


def training_resource_decision(start: dict, samples: list[dict], *, limits=None) -> str | None:
    """Return the first stop reason, or None for complete in-limit samples.

    The monitor must call this continuously and terminate its owned process;
    this calculation alone is not a resource monitor or authorization.
    Three strictly increasing swap observations define a sustained trend.
    """
    if limits is None:
        limits = _DEFAULT_LIMITS
    if not valid_resource_limits(limits) or not _valid(start) or not isinstance(samples, list) or not samples:
        return 'monitor_evidence_invalid'
    previous_time = start['monotonic_seconds']
    for index, state in enumerate(samples):
        if not _valid(state):
            return 'monitor_evidence_invalid'
        current_time = state['monotonic_seconds']
        if (current_time < previous_time or current_time - previous_time > limits['sample_interval_seconds'] or
                (index > 0 and current_time == previous_time)):
            return 'monitor_evidence_invalid'
        previous_time = current_time
    swaps = []
    for state in [start, *samples]:
        if state['disk_free_bytes'] <= limits['minimum_disk_gib'] * _GIB:
            return 'insufficient_disk'
        if state['memory_free_percent'] <= limits['minimum_free_memory_percent']:
            return 'memory_pressure_limit'
        if state['swap_used_mib'] - start['swap_used_mib'] >= limits['maximum_swap_growth_mib']:
            return 'swap_increase_limit'
        if state['process_tree_rss_bytes'] >= limits['maximum_process_rss_gib'] * _GIB:
            return 'process_rss_limit'
        swaps.append(state['swap_used_mib'])
        if len(swaps) >= 3 and swaps[-3] < swaps[-2] < swaps[-1]:
            return 'swap_growth_trend'
    return None
