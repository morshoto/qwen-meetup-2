"""Prepare a development-only public task export without reading solutions.

The export is not a verified operating-system boundary or permission to run.
"""
import hashlib
import json

from src.arc.lt_attestation import _object
from src.arc.lt_development import verify_development_manifest
from src.arc.lt_public import build_public_export, load_public_export


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def build_development_export(challenge_bytes: bytes, exposure_bytes: bytes, manifest: dict) -> tuple[dict, dict]:
    if not isinstance(challenge_bytes, bytes) or not isinstance(exposure_bytes, bytes) or not isinstance(manifest, dict):
        raise ValueError('invalid development export inputs')
    tasks, registry = _object(challenge_bytes), _object(exposure_bytes)
    try:
        known, reserved = registry['known_exposed_task_ids'], registry['partitions']['development']
        if (registry.get('status') != 'bounded_exposure_proposed_unverified' or
                registry.get('task_source_sha256') != _sha(challenge_bytes) or
                not isinstance(known, list) or not isinstance(reserved, list) or
                len(set(known)) != len(known) or len(set(reserved)) != len(reserved)):
            raise ValueError('exposure registry differs')
        verify_development_manifest(manifest, list(tasks), known_exposed=set(known), reserved=set(reserved),
            dataset_sha256=_sha(challenge_bytes), exposure_sha256=_sha(exposure_bytes), seed=manifest['seed'])
        identity = _sha(_canonical(manifest))
        hashes = {task_id: _sha(_canonical(tasks[task_id])) for task_id in manifest['task_ids']}
        public = build_public_export(tasks, {'comparison_sha256': identity, 'selected_task_sha256': hashes})
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid development manifest or registry') from exc
    export = {'development_manifest_sha256': identity, 'tasks': public['tasks']}
    receipt = {'development_manifest_sha256': identity, 'selected_task_sha256': hashes,
               'purpose': 'development_only', 'execution_ready': False}
    return export, receipt


def load_development_export(export: dict, receipt: dict) -> dict:
    if (not isinstance(export, dict) or set(export) != {'development_manifest_sha256', 'tasks'} or
            not isinstance(receipt, dict) or set(receipt) != {'development_manifest_sha256',
                'selected_task_sha256', 'purpose', 'execution_ready'} or
            receipt['purpose'] != 'development_only' or receipt['execution_ready'] is not False or
            export['development_manifest_sha256'] != receipt['development_manifest_sha256']):
        raise ValueError('development export identity differs')
    return load_public_export({'comparison_sha256': export['development_manifest_sha256'], 'tasks': export['tasks']},
        {'comparison_sha256': receipt['development_manifest_sha256'], 'selected_task_sha256': receipt['selected_task_sha256']})


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('challenges', 'exposure', 'manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--part', choices=('export', 'receipt'), required=True)
    args = parser.parse_args(argv)
    try:
        export, receipt = build_development_export(args.challenges.read_bytes(), args.exposure.read_bytes(),
                                                   _object(args.manifest.read_bytes()))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'development export failed: {type(exc).__name__}', file=sys.stderr)
        return 1
    print(json.dumps(export if args.part == 'export' else receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
