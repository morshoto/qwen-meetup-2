"""Propose deterministic task partitions from a separately audited exposure set.

This function cannot establish the completeness of the exposure set or authorize a run.
"""
from collections.abc import Iterable


def make_split(task_ids: Iterable[str], exposed_ids: Iterable[str], *,
               development_size: int) -> dict[str, list[str]]:
    ids = list(task_ids)
    exposed = list(exposed_ids)
    if any(not isinstance(item, str) or not item for item in ids + exposed):
        raise ValueError('task IDs must be nonempty strings')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate task IDs')
    if type(development_size) is not int or development_size <= 0:
        raise ValueError('development size must be a positive integer')
    eligible = sorted(set(ids) - set(exposed))
    if len(eligible) < development_size:
        raise ValueError('insufficient eligible development tasks')
    if development_size >= len(ids):
        raise ValueError('generation partition must be nonempty')
    development = eligible[-development_size:]
    return {'generation': sorted(set(ids) - set(development)),
            'development': development}
