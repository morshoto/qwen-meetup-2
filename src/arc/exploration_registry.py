"""Build and validate conservative manifests of previously used task IDs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = 'p3c_exploration_task_exclusions_v1'


def _add(registry: dict[str, set[str]], task_id: Any, source: str) -> None:
    if isinstance(task_id, str) and task_id:
        registry.setdefault(task_id, set()).add(source)


def _walk_known_ids(value: Any, registry: dict[str, set[str]], source: str) -> None:
    if isinstance(value, dict):
        for key in ('task_id', 'task_ids', 'comparison_task_ids', 'fixed_batch_task_ids',
                    'diagnostic_task_ids_excluded', 'verified_task_ids', 'verified_unique_task_ids',
                    'attempted_task_ids', 'used_task_ids'):
            item = value.get(key)
            if key == 'task_id':
                _add(registry, item, source)
            elif isinstance(item, list):
                for task_id in item:
                    _add(registry, task_id, source)
        for child in value.values():
            if isinstance(child, (dict, list)):
                _walk_known_ids(child, registry, source)
    elif isinstance(value, list):
        for child in value:
            _walk_known_ids(child, registry, source)


def build_registry(repo_root: str | Path, *, additional_manifests: list[str | Path] | None = None) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    registry: dict[str, set[str]] = {}
    artifacts = root / 'artifacts'
    if artifacts.is_dir():
        relevant_names = {'results.json', 'holdout_manifest.json', 'collection.json',
                          'source_manifest.json', 'selection_manifest.json',
                          'comparison_identity.json', 'collection_manifest.json',
                          'collection_summary.json', 'comparison.json', 'task_manifest.json',
                          'identity.json'}
        json_paths = sorted(path for path in artifacts.rglob('*.json')
                            if path.name in relevant_names)
        jsonl_names = {'task_diagnostics.jsonl', 'candidate_traces.jsonl',
                       'attempted_tasks.jsonl', 'verified_solutions.jsonl',
                       'accepted_solutions.jsonl'}
        jsonl_paths = sorted(path for path in artifacts.rglob('*.jsonl')
                             if path.name in jsonl_names)
        for path in [*json_paths, *jsonl_paths]:
            if path.name not in relevant_names and path.name not in jsonl_names:
                continue
            run_parts = path.relative_to(artifacts).parts
            run_root = artifacts / run_parts[0] if run_parts else artifacts
            fake_run = False
            for marker in ('selection_manifest.json', 'collection_manifest.json', 'collection_summary.json'):
                marker_path = run_root / marker
                if marker_path.is_file():
                    try:
                        marker_value = json.loads(marker_path.read_text(encoding='utf-8'))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        marker_value = {}
                    if isinstance(marker_value, dict) and marker_value.get('inference_mode') == 'fake_cpu':
                        fake_run = True
                        break
            if fake_run:
                continue
            try:
                raw = path.read_text(encoding='utf-8')
                source = path.relative_to(root).as_posix()
                if path.suffix == '.jsonl':
                    for line in raw.splitlines():
                        if line.strip():
                            _walk_known_ids(json.loads(line), registry, source)
                else:
                    _walk_known_ids(json.loads(raw), registry, source)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
    configs = root / 'configs'
    if configs.is_dir():
        try:
            import yaml
            for path in sorted(configs.glob('*.yml')):
                try:
                    value = yaml.safe_load(path.read_text(encoding='utf-8'))
                except (OSError, UnicodeDecodeError, yaml.YAMLError):
                    continue
                p3c = value.get('p3c', {}) if isinstance(value, dict) else {}
                if isinstance(p3c, dict):
                    for task_id in p3c.get('task_ids', []):
                        _add(registry, task_id, path.relative_to(root).as_posix())
                promotion = value.get('p3b', {}).get('promotion', {}) if isinstance(value, dict) else {}
                manifest = promotion.get('diagnostic_task_manifest') if isinstance(promotion, dict) else None
                if isinstance(manifest, str) and manifest:
                    manifest_path = (root / manifest).resolve()
                    if manifest_path.is_file():
                        try:
                            data = json.loads(manifest_path.read_text(encoding='utf-8'))
                            for task_id in data.get('task_ids', []):
                                _add(registry, task_id, manifest_path.relative_to(root).as_posix())
                        except (OSError, ValueError, TypeError):
                            continue
        except ImportError:
            pass
    for manifest in additional_manifests or []:
        path = Path(manifest).expanduser().resolve()
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'exclusion manifest is unreadable: {path}') from exc
        if (not isinstance(value, dict) or value.get('protocol') not in
                {PROTOCOL, 'p3b_diagnostic_task_exclusion_v1',
                 'p3c_exploration_diagnosis_exclusion_v1'}):
            raise ValueError(f'unsupported exclusion manifest: {path}')
        _walk_known_ids(value, registry, str(path))
    return {'protocol': PROTOCOL, 'version': 1,
            'exclusions': {task_id: sorted(sources) for task_id, sources in sorted(registry.items())},
            'task_ids': sorted(registry),
            'sources': sorted({source for sources in registry.values() for source in sources})}


def validate_registry(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict) or value.get('protocol') != PROTOCOL or
            value.get('version') != 1 or not isinstance(value.get('exclusions'), dict)):
        raise ValueError('unsupported Exploration exclusion manifest')
    exclusions = value['exclusions']
    if any(not isinstance(task_id, str) or not isinstance(sources, list) or
           not sources or any(not isinstance(source, str) for source in sources)
           for task_id, sources in exclusions.items()):
        raise ValueError('invalid Exploration exclusion entries')
    ids = sorted(exclusions)
    if value.get('task_ids') != ids:
        raise ValueError('exclusion manifest task ID list does not match its provenance map')
    return value


def registry_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
