"""Hash an entire local virtual environment without executing model inference."""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _scan(root):
    entries, directories = {}, {}
    def onerror(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(directory)
        directories[parent.relative_to(root).as_posix()] = _signature(parent.stat())
        for name in dirs:
            path = parent / name
            if path.is_symlink():
                raise ValueError('symlinked environment directory')
        for name in files:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                raise ValueError('unsupported environment file')
            entries[relative] = (_signature(info), os.readlink(path) if stat.S_ISLNK(info.st_mode) else None)
            if len(entries) > 20000:
                raise ValueError('environment file limit exceeded')
    return entries, directories


def _hash(path):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('environment target is not a file')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if _signature(before) != _signature(after) or _signature(after) != _signature(os.stat(path, follow_symlinks=False)):
        raise ValueError('environment file changed while hashing')
    return digest.hexdigest(), after.st_size


def inventory(environment_root, interpreter):
    root = Path(environment_root).resolve(strict=True)
    executable = Path(interpreter).resolve(strict=True)
    if not root.is_dir() or not executable.is_file():
        raise ValueError('environment or interpreter unavailable')
    entries, directories = _scan(root)
    if not entries:
        raise ValueError('environment is empty')
    rows = [{'path': name, 'mode': stat.S_IMODE(signature[2]), 'kind': 'directory'}
            for name, signature in sorted(directories.items())]
    total = 0
    for name, (signature, link) in sorted(entries.items()):
        path = root / name
        target = path.resolve(strict=True)
        if not target.is_relative_to(root) and target != executable:
            raise ValueError('environment link escapes interpreter boundary')
        digest, size = _hash(target)
        total += size
        if total > 4 * 1024 ** 3:
            raise ValueError('environment size limit exceeded')
        rows.append({'path': name, 'mode': stat.S_IMODE(signature[2]), 'kind': 'file', 'link': link,
                     'target_sha256': digest, 'size': size})
    if _scan(root) != (entries, directories):
        raise ValueError('environment changed while hashing')
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return {'environment_root': str(root), 'interpreter_realpath': str(executable),
            'interpreter_sha256': _hash(executable)[0], 'inventory_sha256': hashlib.sha256(encoded).hexdigest(),
            'file_count': len(entries), 'directory_count': len(directories),
            'total_bytes': total, 'scope': 'venv_files_only'}


if __name__ == '__main__':
    print(json.dumps(inventory(sys.prefix, sys.executable), sort_keys=True))
