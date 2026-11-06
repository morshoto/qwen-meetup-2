"""Bind an exploratory comparison proposal to local input bytes.

These checks establish consistency, not dataset provenance, isolation, or
permission to run a model. Preserve the receipt before inspecting results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from src.arc.lt_comparison import make_manifest, verify_manifest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _object(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    parsed = json.loads(data, object_pairs_hook=unique)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError('expected a nonempty task object')
    return parsed


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(sources: Mapping[str, Path]) -> str:
    if not sources or any(not isinstance(name, str) or not name for name in sources):
        raise ValueError('source inventory is empty or invalid')
    return _sha(_canonical({name: _file_sha(path) for name, path in sources.items()}))


def attest_inputs(frozen: dict, challenges_path: Path, solutions_path: Path,
                  sources: Mapping[str, Path], prompt_path: Path, *, source_root: Path | None = None,
                  source_inventory: dict | None = None) -> dict:
    verify_manifest(frozen)
    if (source_root is None) != (source_inventory is None):
        raise ValueError('source root and inventory must be provided together')
    if source_root is not None:
        from src.arc.lt_source import verify_source_tree
        root = Path(source_root).resolve()
        verify_source_tree(root, source_inventory)
        if (set(sources) != set(source_inventory['files']) or
                any(Path(path).resolve() != root / name for name, path in sources.items())):
            raise ValueError('source path map differs from source inventory')
    challenges_bytes, solutions_bytes = Path(challenges_path).read_bytes(), Path(solutions_path).read_bytes()
    challenges, solutions = _object(challenges_bytes), _object(solutions_bytes)
    manifest, config = frozen['manifest'], frozen['config']
    if challenges.keys() != solutions.keys() or _sha(challenges_bytes) != manifest['dataset_sha256']:
        raise ValueError('dataset identities differ')
    proposal = make_manifest(list(challenges), seed=manifest['seed'],
        development=set(manifest['development_exclusions']), training=set(manifest['training_exclusions']),
        exposures=set(manifest['known_exposure_exclusions']), protected=set(manifest['protected_exclusions']),
        dataset_sha256=manifest['dataset_sha256'])
    if proposal != manifest:
        raise ValueError('task selection is not the registered selection')
    if source_digest(sources) != config['code_sha256'] or _sha(Path(prompt_path).read_bytes()) != config['prompt_sha256']:
        raise ValueError('source or prompt identity differs')
    if source_root is not None:
        verify_source_tree(root, source_inventory)
    scope = ({'source_scope': 'conservative_source_tree',
              'source_inventory_sha256': _sha(_canonical(source_inventory))}
             if source_root is not None else {'source_scope': 'caller_supplied_files'})
    return {**scope, 'comparison_sha256': frozen['sha256'], 'challenges_sha256': _sha(challenges_bytes),
            'solutions_sha256': _sha(solutions_bytes), 'code_sha256': config['code_sha256'],
            'prompt_sha256': config['prompt_sha256'],
            'selected_task_sha256': {task_id: _sha(_canonical(challenges[task_id]))
                                     for task_id in manifest['task_ids']}}


def verify_inputs(receipt: dict, frozen: dict, challenges_path: Path, solutions_path: Path,
                  sources: Mapping[str, Path], prompt_path: Path, *, source_root: Path | None = None,
                  source_inventory: dict | None = None) -> None:
    if receipt != attest_inputs(frozen, challenges_path, solutions_path, sources, prompt_path,
                                source_root=source_root, source_inventory=source_inventory):
        raise ValueError('input attestation differs')


def attest_model_runtime(frozen: dict, model: str, model_files: Mapping[str, Path],
                         runtime_files: Mapping[str, Path]) -> dict:
    """Check supplied inventories; does not prove inventories are complete or loaded."""
    verify_manifest(frozen)
    if model not in ('14b', '32b'):
        raise ValueError('invalid model')
    model_sha, runtime_sha = source_digest(model_files), source_digest(runtime_files)
    config = frozen['config']
    if model_sha != config['model_files_sha256'][model] or runtime_sha != config['runtime_sha256']:
        raise ValueError('model or runtime identity differs')
    return {'comparison_sha256': frozen['sha256'], 'model': model, 'revision': config['models'][model],
            'model_files_sha256': model_sha, 'runtime_sha256': runtime_sha,
            'inventory_completeness_verified': False, 'loaded_identity_verified': False}


def verify_model_runtime(receipt: dict, frozen: dict, model: str, model_files: Mapping[str, Path],
                         runtime_files: Mapping[str, Path]) -> None:
    if receipt != attest_model_runtime(frozen, model, model_files, runtime_files):
        raise ValueError('model runtime receipt differs')
