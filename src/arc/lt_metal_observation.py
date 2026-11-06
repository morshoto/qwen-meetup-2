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


def observe_process_identity(pid: int) -> dict:
    """Sample macOS BSD process metadata; this does not prove ownership or cleanup."""
    import ctypes
    import sys
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid PID')
    unavailable = {'pid': pid, 'status': 'unavailable', 'ownership_verified': False}
    if sys.platform != 'darwin':
        return unavailable

    class BSDInfo(ctypes.Structure):
        _fields_ = ([(name, ctypes.c_uint32) for name in ('flags', 'status', 'xstatus', 'pid', 'ppid',
                    'uid', 'gid', 'ruid', 'rgid', 'svuid', 'svgid', 'reserved')] +
                   [('comm', ctypes.c_char * 16), ('name', ctypes.c_char * 32)] +
                   [(name, ctypes.c_uint32) for name in ('nfiles', 'pgid', 'pjobc', 'tdev', 'tpgid')] +
                   [('nice', ctypes.c_int32), ('start_seconds', ctypes.c_uint64),
                    ('start_microseconds', ctypes.c_uint64)])
    try:
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        call = lib.proc_pidinfo
        call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        call.restype = ctypes.c_int
        info = BSDInfo()
        ctypes.set_errno(0)
        size = call(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        error = ctypes.get_errno()
    except (OSError, AttributeError):
        return unavailable
    if (size != ctypes.sizeof(info) or info.pid != pid or info.pgid <= 0 or
            info.start_seconds <= 0 or info.start_microseconds >= 1000000):
        return {**unavailable, 'errno': error}
    return {'pid': pid, 'status': 'observed_unverified', 'ppid': info.ppid, 'pgid': info.pgid,
            'uid': info.uid, 'start_seconds': info.start_seconds, 'start_microseconds': info.start_microseconds,
            'ownership_verified': False}


def merge_process_identity(previous: dict | None, observation: dict) -> dict:
    """Retain start-time conflicts and missing samples without asserting ownership."""
    if (not isinstance(observation, dict) or type(observation.get('pid')) is not int or
            observation['pid'] <= 0 or observation.get('ownership_verified') is not False or
            observation.get('status') not in ('observed_unverified', 'unavailable')):
        raise ValueError('invalid process identity observation')
    pid = observation['pid']
    if previous is None:
        previous = {'pid': pid, 'start_times': [], 'unavailable_seen': False,
                    'conflicting_identities': False, 'ownership_verified': False}
    if (not isinstance(previous, dict) or previous.get('pid') != pid or
            previous.get('ownership_verified') is not False or not isinstance(previous.get('start_times'), list)):
        raise ValueError('invalid process identity history')
    starts = [list(value) for value in previous['start_times']]
    if observation['status'] == 'observed_unverified':
        seconds, micros = observation.get('start_seconds'), observation.get('start_microseconds')
        if type(seconds) is not int or seconds <= 0 or type(micros) is not int or not 0 <= micros < 1000000:
            raise ValueError('invalid process start time')
        stamp = [seconds, micros]
        if stamp not in starts:
            starts.append(stamp)
    return {'pid': pid, 'start_times': starts,
            'unavailable_seen': previous.get('unavailable_seen') is True or observation['status'] == 'unavailable',
            'conflicting_identities': len(starts) > 1, 'ownership_verified': False}
