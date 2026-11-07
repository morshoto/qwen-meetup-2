"""Propose a distinct exploratory 50-task comparison, never authorize a run.

Caller-supplied inventories must be separately bound to their source hashes and
reviewed. This selection does not prove historical or model pretraining exposure.
"""
import hashlib

_PREFIX = b'lt-phase-b-comparison-20261112\0'


def propose_comparison(universe, *, training, phase_a, development, protected, scheduled, exposed, official):
    groups = {'training': training, 'phase_a': phase_a, 'development': development,
              'protected': protected, 'scheduled': scheduled, 'exposed': exposed, 'official': official}
    if (not isinstance(universe, set) or len(universe) != 1000 or
            any(not isinstance(x, str) or not x for x in universe) or
            any(not isinstance(group, set) or not group or
                any(not isinstance(x, str) or not x for x in group) for group in groups.values()) or
            any(len(groups[name]) != count for name, count in
                (('training', 100), ('phase_a', 200), ('development', 3), ('protected', 250),
                 ('scheduled', 20), ('official', 120))) or
            any(not group <= universe for name, group in groups.items() if name != 'official')):
        raise ValueError('invalid or incomplete allocation inventories')
    eligible = universe.difference(*groups.values())
    if len(eligible) < 50:
        raise ValueError('insufficient comparison tasks')
    ranked = sorted(eligible, key=lambda task: (hashlib.sha256(_PREFIX + task.encode('utf-8')).digest(), task))
    return {'status': 'proposal_not_frozen', 'task_ids': sorted(ranked[:50]),
            'selection_rule': 'first 50 by sha256(lt-phase-b-comparison-20261112 NUL task_id)',
            'eligible_count': len(eligible), 'historical_exposure': 'unknown',
            'pretraining_exposure': 'unknown'}
