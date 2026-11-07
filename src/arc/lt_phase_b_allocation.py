"""Check Phase B task IDs against explicit exclusion inventories.

This checks identity disjointness, not semantic equivalence or historical exposure.
"""


def audit_allocation(*, training, universe, phase_a, development, protected,
                     exposed, official, scheduled, expected_count=100):
    groups = {'phase_a': phase_a, 'development': development,
              'protected': protected, 'exposed': exposed, 'official': official,
              'scheduled': scheduled}
    if (type(expected_count) is not int or expected_count <= 0 or
            not isinstance(training, list) or len(training) != expected_count or
            any(not isinstance(x, str) or not x for x in training) or
            len(set(training)) != expected_count or not isinstance(universe, set) or
            any(not isinstance(x, str) or not x for x in universe) or
            any(not isinstance(group, set) or any(not isinstance(x, str) or not x for x in group)
                for group in groups.values())):
        raise ValueError('invalid allocation inputs')
    selected = set(training)
    if not selected <= universe:
        raise ValueError('training task outside source')
    intersections = {name: len(selected & group) for name, group in groups.items()}
    if any(intersections.values()):
        raise ValueError('training task intersects exclusion')
    return {'training_count': len(selected), 'intersection_counts': intersections,
            'historical_exposure': 'unknown', 'pretraining_exposure': 'unknown'}
