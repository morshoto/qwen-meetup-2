"""Mac resource checks and bounded monitoring for E8-owned servers."""
from __future__ import annotations
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def snapshot(path: Path) -> dict[str, Any]:
    swap = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True, timeout=10)
    match = re.search(r'used = ([\d.]+)([MGT])', swap)
    if not match:
        raise RuntimeError('cannot parse macOS swap usage')
    factor = {'M': 1, 'G': 1024, 'T': 1024 * 1024}[match.group(2)]
    pressure = subprocess.check_output(['memory_pressure'], text=True, timeout=15)
    m = re.search(r'System-wide memory free percentage:\s*(\d+)%', pressure)
    if not m:
        raise RuntimeError('cannot parse macOS memory pressure')
    return {'swap_used_mib': float(match.group(1)) * factor, 'memory_free_percent': int(m.group(1)),
            'disk_free_bytes': shutil.disk_usage(path).free, 'unix_time': time.time()}


def check_resources(start: dict, current: dict, *, min_free_percent: int,
                    max_swap_increase_mib: int, min_disk_bytes: int) -> str | None:
    if current['disk_free_bytes'] < min_disk_bytes:
        return 'insufficient_disk'
    if current['memory_free_percent'] < min_free_percent:
        return 'memory_pressure_limit'
    if current['swap_used_mib'] - start['swap_used_mib'] > max_swap_increase_mib:
        return 'swap_increase_limit'
    return None


class ResourceMonitor:
    """Sample RSS and stop only guarded servers whose argv contains this run path."""
    def __init__(self, root: Path, start: dict, *, min_free_percent: int,
                 max_swap_increase_mib: int, min_disk_bytes: int):
        self.root = root.resolve()
        self.start = start
        self.limits = dict(min_free_percent=min_free_percent,
                           max_swap_increase_mib=max_swap_increase_mib, min_disk_bytes=min_disk_bytes)
        self.samples: list[dict] = []
        self.stop_reason: str | None = None
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.event.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            raise RuntimeError('resource monitor did not stop')

    def _monitor(self):
        owned: list[tuple[int, int]] = []
        while not self.event.is_set():
            try:
                state = snapshot(self.root)
                listing = subprocess.check_output(['ps', '-axo', 'pid=,rss=,command='], text=True, timeout=10)
                owned = []
                for line in listing.splitlines():
                    if 'src.cli.guarded_mlx_server' in line and str(self.root) in line:
                        pid, rss, _ = line.strip().split(None, 2)
                        owned.append((int(pid), int(rss)))
                state['owned_server_rss_kib'] = [rss for _, rss in owned]
                self.samples.append(state)
                reason = check_resources(self.start, state, **self.limits)
                if reason:
                    self.stop_reason = reason
                    for pid, _ in owned:
                        try: os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError: pass
                    return
            except Exception as exc:
                self.stop_reason = 'monitor_failed_' + type(exc).__name__
                for pid, _ in owned:
                    try: os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                return
            self.event.wait(2)
