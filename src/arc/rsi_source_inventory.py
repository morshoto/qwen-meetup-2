"""Conservative local source inventory, not an execution-ready registration.

Inventory all files under the listed code/config roots to avoid relying solely
on static import discovery. Runtime packages, model files, data, container image,
and out-of-tree dependencies require separate frozen evidence.
"""
import hashlib
from pathlib import Path

_ROOTS = ('src', 'sandbox', 'configs', 'scripts')
_FILES = ('pyproject.toml', 'uv.lock')


def _scan(root: Path) -> dict[str, str]:
    paths = []
    for relative in _ROOTS:
        directory = root / relative
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('required source directory is missing or symlinked: ' + relative)
        for path in directory.rglob('*'):
            if '__pycache__' in path.relative_to(root).parts or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                raise ValueError('symlink in source inventory: ' + path.relative_to(root).as_posix())
            if path.is_file():
                paths.append(path)
    for relative in _FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('required source file is missing or symlinked: ' + relative)
        paths.append(path)
    result = {}
    try:
        for path in sorted(paths):
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError('source inventory could not be read') from exc
    return result


def build_source_inventory(root: str | Path) -> dict:
    root = Path(root).resolve()
    first = _scan(root)
    if _scan(root) != first:
        raise ValueError('source inventory changed during capture')
    return {'status': 'local_source_inventory_only', 'execution_ready': False, 'files': first}


def verify_source_inventory(root: str | Path, saved: dict) -> None:
    if build_source_inventory(root) != saved:
        raise ValueError('local source inventory differs from registration')
