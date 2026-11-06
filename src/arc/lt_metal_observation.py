"""Compare Metal compiler process-list observations, not cleanup evidence."""

_SERVICE = ('/System/Library/Frameworks/Metal.framework/Versions/A/XPCServices/'
            'MTLCompilerService.xpc/Contents/MacOS/MTLCompilerService')


def parse_metal_processes(listing: str) -> dict[int, dict[str, int]]:
    if not isinstance(listing, str):
        raise ValueError('process listing must be text')
    result = {}
    for line in listing.splitlines():
        if _SERVICE not in line:
            continue
        parts = line.strip().split(None, 3)
        if (len(parts) != 4 or any(not item.isdigit() or int(item) <= 0 for item in parts[:3]) or
                parts[3].split(None, 1)[0] != _SERVICE):
            raise ValueError('ambiguous compiler service observation')
        pid, ppid, pgid = map(int, parts[:3])
        if pid in result:
            raise ValueError('duplicate compiler service PID')
        result[pid] = {'ppid': ppid, 'pgid': pgid}
    return result


def compare_observations(before: dict, during: dict, after: dict) -> dict:
    for sample in (before, during, after):
        if (not isinstance(sample, dict) or any(type(pid) is not int or pid <= 0 or
                not isinstance(value, dict) or set(value) != {'ppid', 'pgid'} or
                any(type(value[k]) is not int or value[k] <= 0 for k in ('ppid', 'pgid'))
                for pid, value in sample.items())):
            raise ValueError('invalid process observation')
    new = set(during) - set(before)
    return {'newly_observed': sorted(new), 'newly_observed_still_listed': sorted(new & set(after)),
            'ownership_verified': False, 'cleanup_verified': False, 'execution_ready': False,
            'evidence_type': 'process_list_snapshots_only'}


def classify_responsibility(pid: int, responsible_pid: int, error: int) -> dict:
    if (type(pid) is not int or pid <= 0 or type(responsible_pid) is not int or
            type(error) is not int or error < 0):
        raise ValueError('invalid responsibility observation')
    observed = responsible_pid > 0 and error == 0
    return {'pid': pid, 'status': 'observed_unverified' if observed else 'unavailable',
            'responsible_pid': responsible_pid if observed else None, 'errno': error,
            'ownership_verified': False}


def observe_responsibility(pid: int) -> dict:
    """Query an undocumented macOS API; results are advisory and unauthenticated."""
    import ctypes
    import sys
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid PID')
    if sys.platform != 'darwin':
        return classify_responsibility(pid, -1, 0)
    try:
        library = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        lookup = library.responsibility_get_pid_responsible_for_pid
        lookup.argtypes = [ctypes.c_int]
        lookup.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = lookup(pid)
        error = ctypes.get_errno()
    except (OSError, AttributeError):
        return classify_responsibility(pid, -1, 0)
    return classify_responsibility(pid, result, error)
