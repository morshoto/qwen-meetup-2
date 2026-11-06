"""Propose a development-only subset from known prior exposures.

No challenge contents, solutions, model or official evaluation set are read.
The manifest does not establish provenance or authorize inference.
"""
import random
import re


def build_development_manifest(ids: list[str], *, known_exposed: set[str], reserved: set[str],
                               dataset_sha256: str, exposure_sha256: str, seed: int) -> dict:
    valid_hash = lambda value: isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value))
    if (not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids) or
            len(ids) != len(set(ids)) or type(seed) is not int or not valid_hash(dataset_sha256) or
            not valid_hash(exposure_sha256) or not isinstance(known_exposed, set) or
            not isinstance(reserved, set) or not known_exposed <= set(ids) or not reserved <= set(ids)):
        raise ValueError('invalid development selection inputs')
    eligible = sorted(known_exposed - reserved)
    if len(eligible) < 3:
        raise ValueError('fewer than three eligible development tasks')
    random.Random(seed).shuffle(eligible)
    return {'protocol': 'lt-phase-a-development', 'status': 'proposed_dev_only',
            'seed': seed, 'task_ids': sorted(eligible[:3]), 'dataset_sha256': dataset_sha256,
            'exposure_sha256': exposure_sha256, 'known_exposed': True,
            'pretraining_exposure_unknown': True, 'execution_ready': False}


def verify_development_manifest(manifest: dict, ids: list[str], **kwargs) -> None:
    if manifest != build_development_manifest(ids, **kwargs):
        raise ValueError('development selection differs from input identities')
