"""Validated, versioned task-ID manifests used by diagnostic exclusion gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_diagnostic_task_ids(path: str | Path) -> tuple[list[str], str]:
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('diagnostic task manifest is not valid UTF-8 JSON') from exc
    if not isinstance(manifest, dict):
        raise ValueError('diagnostic task manifest must be a JSON object')
    task_ids = manifest.get('task_ids')
    if (manifest.get('protocol') != 'p3b_diagnostic_task_exclusion_v1' or
            manifest.get('version') != 1 or not isinstance(task_ids, list) or not task_ids or
            any(type(task_id) is not str or not task_id for task_id in task_ids) or
            len(set(task_ids)) != len(task_ids) or not manifest.get('source_artifact') or
            not manifest.get('source_protocol')):
        raise ValueError('diagnostic task manifest has an unsupported protocol or invalid task IDs')
    return task_ids, hashlib.sha256(raw).hexdigest()
