"""Recompute a Phase B allocation receipt from local challenge inventories.

This checks exact IDs and visible content only. It does not prove semantic
independence, complete historical exposure, official data provenance, or
independent review, and never authorizes inference or training.
"""
import hashlib
import json
from pathlib import Path

from src.arc.data import Task
from src.arc.lt_attestation import _object
from src.arc.lt_phase_b_allocation import audit_allocation
from src.arc.lt_phase_b_dataset import _check_task_sets
from src.arc.lt_phase_b_proposal import build_proposal
from src.arc.rsi_task_generation import generation_slots


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def derive_allocation(root, challenge_bytes, official_challenge_bytes, training_task_ids):
    root = Path(root)
    proposal = build_proposal(root, challenge_bytes, official_challenge_bytes)
    saved = json.loads((root / 'configs/lt.phase-b.comparison-proposal.json').read_bytes())
    if proposal != saved:
        raise ValueError('comparison proposal differs')
    challenges = _object(challenge_bytes)
    registry = _object((root / 'research/rsi/bounded-exposure-proposal.json').read_bytes())
    split = _object((root / 'research/rsi/exploratory-split-proposal.json').read_bytes())
    dev = _object((root / 'configs/lt.phase-a.development.json').read_bytes())
    phase_a = _object((root / 'configs/lt.phase-a.frozen.json').read_bytes())
    official = set(_object(official_challenge_bytes))
    protected = set(split['confirmatory_task_ids']) | set(split['exploratory_task_ids'])
    scheduled = {slot['task_id'] for i in (0, 1) for slot in generation_slots(registry['partitions']['generation'], i)}
    exclusions = {'phase_a': set(phase_a['task_ids']), 'development': set(dev['task_ids']),
                  'protected': protected, 'scheduled': scheduled,
                  'exposed': set(registry['known_exposed_task_ids']), 'official': official}
    checked = audit_allocation(training=training_task_ids, universe=set(challenges), **exclusions)
    comparison = proposal['task_ids']
    if set(training_task_ids) & set(comparison):
        raise ValueError('training and comparison IDs overlap')

    def select(ids):
        result = {}
        for key in ids:
            item = challenges[key]
            if any(set(pair) != {'input'} for pair in item['test']):
                raise ValueError('unexpected test labels')
            result[key] = Task(key, tuple((v['input'], v['output']) for v in item['train']),
                               tuple(v['input'] for v in item['test']))
        return result

    training = select(training_task_ids)
    for group in (comparison, exclusions['phase_a'], exclusions['development'], exclusions['protected'],
                  exclusions['scheduled'], exclusions['exposed']):
        _check_task_sets(training, select(group))
    allocation_code = (root / 'src/arc/lt_phase_b_allocation.py').read_bytes()
    receipt_code = (root / 'src/arc/lt_phase_b_allocation_receipt.py').read_bytes()
    return {'status': 'allocation_consistent_unverified', 'execution_ready': False,
            'training_task_sha256': {key: _digest(challenges[key]) for key in sorted(training_task_ids)},
            'comparison_task_sha256': {key: _digest(challenges[key]) for key in comparison},
            'source_hashes': {**proposal['source_hashes'],
                'allocation_code': hashlib.sha256(allocation_code).hexdigest(),
                'allocation_receipt_code': hashlib.sha256(receipt_code).hexdigest()},
            'intersection_counts': checked['intersection_counts'],
            'historical_exposure': 'unknown', 'pretraining_exposure': 'unknown'}
