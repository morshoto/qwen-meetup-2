"""Sample macOS host pressure and a process tree; samples are not peak proofs."""
import subprocess
import time
from pathlib import Path

from src.arc.e8_resource import snapshot


def process_tree_rss_bytes(listing: str, root_pid: int) -> int:
    if not isinstance(listing, str) or type(root_pid) is not int or root_pid <= 0:
        raise ValueError('invalid process listing or root')
    processes = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('malformed process listing')
        try:
            pid, parent, rss = map(int, parts)
        except ValueError as exc:
            raise ValueError('malformed process listing') from exc
        if pid <= 0 or parent < 0 or rss < 0 or pid == parent or pid in processes:
            raise ValueError('invalid process listing identity')
        processes[pid] = (parent, rss)
    if root_pid not in processes:
        raise ValueError('owned process is absent from listing')
    included = {root_pid}
    while True:
        added = {pid for pid, (parent, _) in processes.items() if parent in included} - included
        if not added:
            break
        included.update(added)
    return sum(processes[pid][1] for pid in included) * 1024


def sample_process(root: str | Path, pid: int) -> dict:
    host = snapshot(Path(root))
    listing = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True, timeout=3)
    rss = process_tree_rss_bytes(listing, pid)
    return {'monotonic_seconds': time.monotonic(), 'disk_free_bytes': host['disk_free_bytes'],
            'memory_free_percent': host['memory_free_percent'], 'swap_used_mib': host['swap_used_mib'],
            'process_tree_rss_bytes': rss}


def sample_baseline(root: str | Path) -> dict:
    host = snapshot(Path(root))
    return {'monotonic_seconds': time.monotonic(), 'disk_free_bytes': host['disk_free_bytes'],
            'memory_free_percent': host['memory_free_percent'], 'swap_used_mib': host['swap_used_mib'],
            'process_tree_rss_bytes': 0}
