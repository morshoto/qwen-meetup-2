"""Registration for a single exploratory model on the Phase A task set."""
import copy
import re
from pathlib import Path

from src.arc.lt_benchmark_manifest import build as build_phase_a, canonical, sha

SOURCE_FILES = ('src/arc/lt_single_manifest.py', 'src/arc/lt_single_score.py',
                'src/cli/lt_single.py', 'src/cli/lt_single_worker.py')


def build(root: Path, model: dict) -> dict:
    if not isinstance(model, dict) or set(model) != {
        'model_id', 'revision', 'files_sha256', 'chat_template_sha256'
    }:
        raise ValueError('invalid model configuration')
    files = model['files_sha256']
    if (not isinstance(model['model_id'], str) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', model['model_id']) or
            not isinstance(model['revision'], str) or
            not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', model['revision']) or
            not isinstance(model['chat_template_sha256'], str) or
            not re.fullmatch(r'[0-9a-f]{64}', model['chat_template_sha256']) or
            not isinstance(files, dict) or not files or
            any(not isinstance(name, str) or name in ('', '.', '..') or
                '/' in name or '\\' in name or not isinstance(digest, str) or
                not re.fullmatch(r'[0-9a-f]{64}', digest) for name, digest in files.items())):
        raise ValueError('invalid model identity or inventory')
    phase_a = build_phase_a(root)
    value = {key: copy.deepcopy(val) for key, val in phase_a.items()
             if key not in ('models', 'sha256')}
    value.update(protocol='lt-single-model-exploratory', phase_a_sha256=phase_a['sha256'],
                 model=copy.deepcopy(model),
                 single_source_sha256={name: sha((root / name).read_bytes()) for name in SOURCE_FILES})
    return {**value, 'sha256': sha(canonical(value))}


def verify(root: Path, manifest: dict) -> None:
    try:
        if manifest != build(root, manifest['model']):
            raise ValueError('single-model registration differs')
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid single-model registration') from exc
