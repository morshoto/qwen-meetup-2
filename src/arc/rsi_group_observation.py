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
