"""Check Phase A exclusions against the separate prior RSI proposal.

Caller-supplied training IDs and local evidence are not independently
verified. This function never selects tasks or authorizes execution.
"""
import hashlib
import json
from pathlib import Path

from src.arc.lt_attestation import _object
from src.arc.lt_comparison import make_manifest
from src.arc.lt_development import verify_development_manifest
from src.arc.rsi_exploratory_split import propose_exploratory_split


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def check_protected_allocation(proposal: dict, registry_bytes: bytes, split_bytes: bytes,
                               development: dict, training_task_ids: list[str]) -> dict:
    if (not isinstance(proposal, dict) or not isinstance(development, dict) or
            not isinstance(registry_bytes, bytes) or not isinstance(split_bytes, bytes) or
            not isinstance(training_task_ids, list) or
            any(not isinstance(x, str) or not x for x in training_task_ids) or
            len(training_task_ids) != len(set(training_task_ids))):
        raise ValueError('invalid allocation inputs')
    registry, split = _object(registry_bytes), _object(split_bytes)
    try:
        expected = propose_exploratory_split(registry_bytes, expected_sha256=_sha(registry_bytes))
        code_sha = _sha(Path(__file__).with_name('rsi_exploratory_split.py').read_bytes())
        if split != {**expected, 'selection_code_sha256': code_sha}:
            raise ValueError('prior RSI split differs')
        universe = registry['partitions']['generation'] + registry['partitions']['development']
        known = set(registry['known_exposed_task_ids'])
        protected = set(split['confirmatory_task_ids']) | set(split['exploratory_task_ids'])
        verify_development_manifest(development, universe, known_exposed=known,
            reserved=set(registry['partitions']['development']), dataset_sha256=registry['task_source_sha256'],
            exposure_sha256=_sha(registry_bytes), seed=development['seed'])
        dev = set(development['task_ids'])
        training = set(training_task_ids)
        if (not training <= set(universe) or training & dev or
                not dev <= set(proposal['development_exclusions']) or
                not training <= set(proposal['training_exclusions']) or
                not known <= set(proposal['known_exposure_exclusions']) or
                not protected <= set(proposal['protected_exclusions']) or
                set(proposal['task_ids']) & (protected | dev | training | known)):
            raise ValueError('required protected or excluded task IDs missing')
        expected_proposal = make_manifest(universe, seed=proposal['seed'],
            development=set(proposal['development_exclusions']), training=set(proposal['training_exclusions']),
            exposures=set(proposal['known_exposure_exclusions']), protected=set(proposal['protected_exclusions']),
            dataset_sha256=registry['task_source_sha256'])
        if proposal != expected_proposal:
            raise ValueError('comparison selection differs')
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError('invalid prior allocation evidence') from exc
    return {'status': 'allocation_consistent_unverified', 'execution_ready': False,
            'training_inventory_independently_verified': False, 'protected_count': len(protected),
            'registry_sha256': _sha(registry_bytes), 'prior_split_sha256': _sha(split_bytes),
            'development_manifest_sha256': _sha(json.dumps(development, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()),
            'training_ids_sha256': _sha(json.dumps(sorted(training), separators=(',', ':')).encode())}


def build_comparison_proposal(challenge_bytes: bytes, registry_bytes: bytes, split_bytes: bytes,
                              development: dict, training_task_ids: list[str], *, seed: int) -> tuple[dict, dict]:
    """Derive exclusions from supplied evidence; the result is not execution-ready."""
    if (not isinstance(challenge_bytes, bytes) or not isinstance(registry_bytes, bytes) or
            not isinstance(split_bytes, bytes) or not isinstance(development, dict) or
            not isinstance(training_task_ids, list) or
            any(not isinstance(x, str) or not x for x in training_task_ids) or
            len(training_task_ids) != len(set(training_task_ids))):
        raise ValueError('invalid proposal inputs')
    tasks, registry, split = _object(challenge_bytes), _object(registry_bytes), _object(split_bytes)
    try:
        universe = registry['partitions']['generation'] + registry['partitions']['development']
        if _sha(challenge_bytes) != registry['task_source_sha256'] or set(tasks) != set(universe):
            raise ValueError('training source differs from prior registry')
        proposal = make_manifest(list(tasks), seed=seed, development=set(development['task_ids']),
            training=set(training_task_ids), exposures=set(registry['known_exposed_task_ids']),
            protected=set(split['confirmatory_task_ids']) | set(split['exploratory_task_ids']),
            dataset_sha256=_sha(challenge_bytes))
        audit = check_protected_allocation(proposal, registry_bytes, split_bytes, development, training_task_ids)
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid proposal evidence') from exc
    return proposal, audit
