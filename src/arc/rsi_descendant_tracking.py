"""Track observed identities; snapshots alone cannot prove process cleanup."""
import re
import os
import subprocess


def parse_ps_listing(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('empty process listing')
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 8:
            raise ValueError('malformed process listing')
        try:
            pid, parent = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise ValueError('malformed process listing') from exc
        rows.append({'pid': pid, 'ppid': parent, 'start': ' '.join(fields[2:7]), 'state': fields[7]})
    return rows


def begin_owned_observation(process):
    """Observe one owned live session; never signal or certify it."""
    from src.arc.rsi_process_monitor import _exited_unreaped
    if not isinstance(process, subprocess.Popen) or process.returncode is not None:
        raise ValueError('owned live child required')
    try:
        if _exited_unreaped(process) is not None:
            raise ValueError('child already exited')
        pgid, session = os.getpgid(process.pid), os.getsid(process.pid)
        if pgid != process.pid or session != process.pid:
            raise ValueError('owned session leader required')
        listing = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,lstart=,state='],
                                          text=True, timeout=3)
        rows = parse_ps_listing(listing)
        matches = [row for row in rows if row['pid'] == process.pid]
        if len(matches) != 1 or matches[0]['ppid'] != os.getpid() or _exited_unreaped(process) is not None:
            raise ValueError('owned child identity changed during observation')
        tracker = DescendantTracker(root_pid=process.pid, root_start=matches[0]['start'])
        result = tracker.observe(rows)
    except (OSError, ChildProcessError, subprocess.SubprocessError) as exc:
        raise ValueError('process observation unavailable') from exc
    result.update(root_pgid=pgid, root_session=session, root_start=matches[0]['start'])
    return tracker, result


def sample_descendant_observation(tracker):
    """Sample previously observed identities without signalling or certifying them."""
    if not isinstance(tracker, DescendantTracker):
        raise ValueError('descendant tracker required')
    return tracker.observe(sample_process_rows())


def sample_process_rows():
    """Read process rows without updating any tracker."""
    try:
        listing = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,lstart=,state='],
                                          text=True, timeout=3)
        return parse_ps_listing(listing)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('process observation unavailable') from exc


class DescendantTracker:
    def __init__(self, *, root_pid, root_start):
        if type(root_pid) is not int or root_pid <= 0 or not isinstance(root_start, str) or not root_start:
            raise ValueError('invalid root identity')
        self._root = root_pid
        self._root_start = root_start
        self._tracked = {}

    def observe(self, rows):
        if not isinstance(rows, list) or not rows:
            raise ValueError('empty or invalid snapshot')
        current = {}
        for item in rows:
            if (not isinstance(item, dict) or set(item) != {'pid', 'ppid', 'start', 'state'} or
                    type(item['pid']) is not int or item['pid'] <= 0 or
                    type(item['ppid']) is not int or item['ppid'] < 0 or item['pid'] == item['ppid'] or
                    not isinstance(item['start'], str) or not item['start'] or
                    not isinstance(item['state'], str) or not re.fullmatch(r'[A-Z][A-Za-z+<>]*', item['state']) or
                    item['pid'] in current):
                raise ValueError('malformed process snapshot')
            current[item['pid']] = item
        if not self._tracked:
            root = current.get(self._root)
            if root is None or root['start'] != self._root_start:
                raise ValueError('root identity missing or changed')
            self._tracked[self._root] = self._root_start
        # A reused PID is never accepted as the previously observed process.
        matching = {pid for pid, start in self._tracked.items()
                    if pid in current and current[pid]['start'] == start}
        reused = sorted(pid for pid, start in self._tracked.items()
                        if pid in current and current[pid]['start'] != start)
        while True:
            added = {pid for pid, item in current.items()
                     if item['ppid'] in matching and pid not in self._tracked}
            if not added:
                break
            for pid in added:
                self._tracked[pid] = current[pid]['start']
            matching.update(added)
        return {'live_observed_pids': sorted(pid for pid in matching if not current[pid]['state'].startswith('Z')),
                'zombie_observed_pids': sorted(pid for pid in matching if current[pid]['state'].startswith('Z')),
                'missing_observed_pids': sorted(pid for pid in self._tracked if pid not in current),
                'reused_pids': reused, 'cleanup_verified': False, 'escaped_processes_excluded': False}
