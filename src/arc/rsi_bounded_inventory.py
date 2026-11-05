"""Build a reproducible local proposal for the v3 exposure inventory."""
import hashlib
import json
from pathlib import Path

from src.arc.exploration_registry import validate_registry
from src.arc.rsi_candidate_history import audit_candidate_journals
from src.arc.rsi_exposure_review import audit_extra_artifacts
from src.arc.rsi_history import audit_history
from src.arc.rsi_split import make_split

_CODE = ('src/arc/rsi_bounded_inventory.py', 'src/arc/rsi_history.py',
         'src/arc/exploration_registry.py', 'src/arc/rsi_exposure_review.py',
         'src/arc/rsi_candidate_history.py', 'src/arc/rsi_split.py')


def _code_hashes(root):
    try:
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _CODE}
    except OSError as exc:
        raise ValueError('review code unavailable') from exc


def propose_bounded_inventory(root, *, task_path, registry_path, required_paths, development_size=200):
    root = Path(root).resolve()
    if not isinstance(required_paths, list) or any(not isinstance(p, str) for p in required_paths):
        raise ValueError('invalid required history paths')
    def read(relative):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('invalid input path')
        path = root
        for part in Path(relative).parts:
            path = path / part
            if path.is_symlink(): raise ValueError('symlinked inventory input')
        try:
            raw = path.read_bytes()
            return raw, json.loads(raw)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError('inventory input unavailable') from exc
    tasks_raw, tasks = read(task_path)
    registry_raw, registry = read(registry_path)
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError('invalid dataset')
    ids = set(tasks)
    registry = validate_registry(registry)
    code = _code_hashes(root)
    def review():
        history = audit_history(root, required_paths=[*required_paths, task_path, registry_path])
        extra = audit_extra_artifacts(root, ids)
        candidates = audit_candidate_journals(root, ids)
        return history, extra, candidates
    history, extra, candidates = review()
    exposed = (set(registry['task_ids']) | set(history['task_ids']) |
               set(extra['task_ids']) | set(candidates['task_ids'])) & ids
    partitions = make_split(ids, exposed, development_size=development_size)
    if (read(task_path)[0] != tasks_raw or read(registry_path)[0] != registry_raw or
            _code_hashes(root) != code or review() != (history, extra, candidates)):
        raise ValueError('inventory inputs changed during proposal')
    return {'status':'bounded_exposure_proposed_unverified', 'execution_ready':False,
            'independent_review_completed':False, 'historical_independence_established':False,
            'unknown_project_exposure':True, 'unknown_pretraining_exposure':True,
            'task_source':task_path, 'task_source_sha256':hashlib.sha256(tasks_raw).hexdigest(),
            'prior_registry':registry_path, 'prior_registry_sha256':hashlib.sha256(registry_raw).hexdigest(),
            'required_paths':sorted(set(required_paths)), 'audit_code_sha256':code,
            'known_exposed_task_ids':sorted(exposed), 'partitions':partitions,
            'history_files':history['files'], 'extra_review':extra, 'candidate_review':candidates}
