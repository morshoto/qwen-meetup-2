"""Evaluate recorded training samples; this does not start or stop a process."""
import math

_GIB = 1024 ** 3
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


def training_resource_decision(start: dict, samples: list[dict]) -> str | None:
    """Return the first stop reason, or None for complete in-limit samples.

    The monitor must call this continuously and terminate its owned process;
    this calculation alone is not a resource monitor or authorization.
    Three strictly increasing swap observations define a sustained trend.
    """
    if not _valid(start) or not isinstance(samples, list) or not samples:
        return 'monitor_evidence_invalid'
    previous_time = start['monotonic_seconds']
    for index, state in enumerate(samples):
        if not _valid(state):
            return 'monitor_evidence_invalid'
        current_time = state['monotonic_seconds']
        if (current_time < previous_time or current_time - previous_time > 5 or
                (index > 0 and current_time == previous_time)):
            return 'monitor_evidence_invalid'
        previous_time = current_time
    swaps = []
    for state in [start, *samples]:
        if state['disk_free_bytes'] <= 20 * _GIB:
            return 'insufficient_disk'
        if state['memory_free_percent'] <= 20:
            return 'memory_pressure_limit'
        if state['swap_used_mib'] - start['swap_used_mib'] >= 512:
            return 'swap_increase_limit'
        if state['process_tree_rss_bytes'] >= 32 * _GIB:
            return 'process_rss_limit'
        swaps.append(state['swap_used_mib'])
        if len(swaps) >= 3 and swaps[-3] < swaps[-2] < swaps[-1]:
            return 'swap_growth_trend'
    return None
