"""Parse a process-group listing without asserting containment or cleanup."""
import re


def parse_group_listing(listing, group_id):
    if not isinstance(listing, str) or type(group_id) is not int or group_id <= 0:
        raise ValueError('invalid group observation')
    seen, members = set(), []
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('invalid process listing row')
        try:
            pid, group = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError('invalid process listing identity') from exc
        state = parts[2]
        if pid <= 0 or group <= 0 or pid in seen or not re.fullmatch(r'[A-Z][A-Za-z+<>]*', state):
            raise ValueError('invalid process listing row')
        seen.add(pid)
        if group == group_id:
            members.append({'pid': pid, 'state': state})
    if not any(row['pid'] == group_id for row in members):
        raise ValueError('unreaped group leader not observed')
    members.sort(key=lambda row: row['pid'])
    return {'status':'process_group_listing_unverified', 'members':members,
            'live_member_pids':[row['pid'] for row in members if not row['state'].startswith('Z')],
            'group_cleanup_verified':False, 'escaped_processes_excluded':False}


def observe_exited_group(process):
    """Observe a retained child group's listing; this never proves cleanup."""
    import subprocess
    from src.arc.rsi_process_monitor import _exited_unreaped
    if not isinstance(process, subprocess.Popen) or process.returncode is not None:
        raise ValueError('owned unreaped process required')
    try:
        if _exited_unreaped(process) is None:
            raise ValueError('group leader has not exited')
        listing = subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,state='], text=True, timeout=3)
        result = parse_group_listing(listing, process.pid)
        if _exited_unreaped(process) is None:
            raise ValueError('unreaped leader observation changed')
    except (OSError, ChildProcessError, subprocess.SubprocessError) as exc:
        raise ValueError('process group listing unavailable') from exc
    return result
