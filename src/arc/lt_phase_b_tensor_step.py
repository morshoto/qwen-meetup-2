"""Persist before/after tensor observations around a supplied canary callback.

The caller must enforce the one-step trainer configuration, supervision, and
model provenance. These observations do not establish a live training result.
"""
import os
import stat
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_tensor_evidence import capture_tensor_state, compare_tensor_states


def observe_one_step(model, *, output_dir, run_step, flatten, evaluate):
    path = Path(output_dir)
    if path.is_symlink() or not callable(run_step):
        raise ValueError('invalid canary output or callback')
    path = path.resolve(strict=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('canary output directory must be private')
    before = capture_tensor_state(model, flatten=flatten, evaluate=evaluate)
    _create(path / 'tensor-before.json', before)
    run_step()
    after = capture_tensor_state(model, flatten=flatten, evaluate=evaluate)
    _create(path / 'tensor-after.json', after)
    result = compare_tensor_states(before, after)
    _create(path / 'tensor-evidence.json', result)
    return result
