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


def main(argv: list[str] | None = None) -> int:
    import argparse
    import hashlib
    import json
    import sys
    from pathlib import Path
    from src.arc.lt_attestation import _object

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--challenges', required=True, type=Path)
    parser.add_argument('--exposure', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args(argv)
    try:
        source, exposure = args.challenges.read_bytes(), args.exposure.read_bytes()
        tasks, registry = _object(source), _object(exposure)
        digest = hashlib.sha256(source).hexdigest()
        known = registry['known_exposed_task_ids']
        reserved = registry['partitions']['development']
        if (registry.get('status') != 'bounded_exposure_proposed_unverified' or
                registry.get('task_source_sha256') != digest or not isinstance(known, list) or
                not isinstance(reserved, list) or len(set(known)) != len(known) or
                len(set(reserved)) != len(reserved)):
            raise ValueError('exposure registry differs from task source')
        result = build_development_manifest(list(tasks), known_exposed=set(known), reserved=set(reserved),
                    dataset_sha256=digest, exposure_sha256=hashlib.sha256(exposure).hexdigest(), seed=args.seed)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'development selection failed: {type(exc).__name__}', file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
