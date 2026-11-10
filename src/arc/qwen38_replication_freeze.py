"""Identity checks for the previously completed exploratory outcome."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest

ADAPTER_SHA256 = 'b363fd2bd15114577d96141b2e03500e7d3d241a4b4a64b6176f25dbb46ac9f6'
REPORT_SHA256 = '023ac3d4afa8982d43a24e2b328f4471e127a7c2f9a24109d582d9e89d193b70'


def verify_frozen_outcome(adapter_bytes: bytes, report_bytes: bytes) -> dict:
    """Check the immutable adapter and content-digested report, without changing either."""
    try:
        report = json.loads(report_bytes)
        valid = (isinstance(report, dict) and
                 report.get('sha256') == REPORT_SHA256 and
                 digest({k: v for k, v in report.items() if k != 'sha256'}) == REPORT_SHA256 and
                 report.get('status') == 'complete' and report.get('schedule_status') == 'complete' and
                 report.get('adapter_sha256') == ADAPTER_SHA256 and
                 hashlib.sha256(adapter_bytes).hexdigest() == ADAPTER_SHA256)
    except (ValueError, TypeError, UnicodeDecodeError):
        valid = False
    if not valid:
        raise ValueError('frozen outcome identity mismatch')
    return {'status': 'frozen_outcome_verified', 'adapter_sha256': ADAPTER_SHA256,
            'report_sha256': REPORT_SHA256}


def inventory_tree(root):
    """Hash all regular files in an evidence tree; reject links and special files."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('invalid evidence root')
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs:
            if (base / name).is_symlink():
                raise ValueError('evidence contains a symlink')
        for name in names:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('evidence contains a non-regular file')
            value = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
            files[path.relative_to(root).as_posix()] = value.hexdigest()
    return files


def verify_inventory(root, expected):
    actual = inventory_tree(root)
    if not expected or actual != expected:
        raise ValueError('evidence inventory changed')
    return actual


def freeze_snapshot(root, output, pinned):
    """Write an exclusive private inventory only after the caller's pins match."""
    root = Path(root).resolve()
    output = Path(output)
    if output.resolve().is_relative_to(root) or not pinned:
        raise ValueError('invalid snapshot destination or pins')
    files = inventory_tree(root)
    if any(files.get(name) != value for name, value in pinned.items()):
        raise ValueError('pinned source evidence changed')
    snapshot = {'status': 'frozen_inventory', 'source_root': str(root), 'files': files}
    snapshot['sha256'] = digest(snapshot)
    raw = (json.dumps(snapshot, sort_keys=True, separators=(',', ':')) + '\n').encode()
    try:
        fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if output.is_symlink() or not output.is_file() or output.read_bytes() != raw:
            raise ValueError('snapshot changed')
    else:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return snapshot
