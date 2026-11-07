"""Replay the unfrozen comparison allocation from bound local inventories.

Only the official challenge IDs are used; no official evaluation solutions or
Phase A outcomes are read. A consistent proposal does not authorize inference.
"""
import hashlib
import json
from pathlib import Path

from src.arc.data import Task
from src.arc.lt_allocation import build_comparison_proposal
from src.arc.lt_attestation import _object
from src.arc.lt_phase_b_comparison_selection import propose_comparison
from src.arc.lt_phase_b_dataset import _check_task_sets
from src.arc.rsi_task_generation import generation_slots

_PATHS = {'phase_a': 'configs/lt.phase-a.frozen.json', 'training': 'configs/lt.phase-b.training-pool.json',
          'registry': 'research/rsi/bounded-exposure-proposal.json',
          'strict_split': 'research/rsi/exploratory-split-proposal.json',
          'development': 'configs/lt.phase-a.development.json',
          'selection_code': 'src/arc/lt_phase_b_comparison_selection.py',
          'content_check_code': 'src/arc/lt_phase_b_dataset.py'}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build_proposal(root, challenge_bytes, official_challenge_bytes):
    if not isinstance(challenge_bytes, bytes) or not isinstance(official_challenge_bytes, bytes):
        raise ValueError('invalid input bytes')
    root = Path(root)
    raw = {key: (root / path).read_bytes() for key, path in _PATHS.items()}
    a, training, registry, split, dev = (json.loads(raw[x]) for x in
                                        ('phase_a', 'training', 'registry', 'strict_split', 'development'))
    challenges, official = _object(challenge_bytes), set(_object(official_challenge_bytes))
    if _sha(challenge_bytes) != a['challenges_sha256']:
        raise ValueError('challenge digest differs')
    existing, _ = build_comparison_proposal(challenge_bytes, raw['registry'], raw['strict_split'], dev, training,
                                             seed=20261108)
    if (existing != json.loads((root / 'configs/lt.phase-a.comparison-proposal.json').read_bytes()) or
            existing['task_ids'] != a['task_ids']):
        raise ValueError('Phase A proposal differs')
    generation = registry['partitions']['generation']
    scheduled = {slot['task_id'] for index in (0, 1) for slot in generation_slots(generation, index)}
    protected = set(split['confirmatory_task_ids']) | set(split['exploratory_task_ids'])
    groups = {'training': set(training), 'phase_a': set(a['task_ids']), 'development': set(dev['task_ids']),
              'protected': protected, 'scheduled': scheduled, 'exposed': set(registry['known_exposed_task_ids']),
              'official': official}
    for name in ('phase_a', 'development', 'protected', 'exposed', 'official'):
        if groups['training'] & groups[name]:
            raise ValueError('training pool intersects exclusion')
    proposal = propose_comparison(set(challenges), **groups)

    def select(ids):
        result = {}
        for key in ids:
            item = challenges[key]
            if any(set(pair) != {'input'} for pair in item['test']):
                raise ValueError('unexpected test labels')
            result[key] = Task(key, tuple((v['input'], v['output']) for v in item['train']),
                               tuple(v['input'] for v in item['test']))
        return result

    for name in ('training', 'phase_a', 'development', 'protected', 'scheduled'):
        _check_task_sets(select(proposal['task_ids']), select(groups[name]))
    return {**proposal, 'source_hashes': {**{k: _sha(v) for k, v in raw.items()},
            'training_challenges': _sha(challenge_bytes), 'official_evaluation_challenges': _sha(official_challenge_bytes)},
            'inventory_counts': {k: len(v) for k, v in groups.items()},
            'scheduled_training_overlap': sorted(groups['scheduled'] & groups['training']),
            'content_check': 'exact visible content disjoint from training, phase A, development, strict evaluation and scheduled tasks',
            'limitations': ['proposal only; allocation decision unresolved',
                            'independent exposure and semantic ancestry unverified',
                            'official evaluation IDs only; never scored']}
