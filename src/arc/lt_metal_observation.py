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
