"""Construct a synthetic macOS profile for boundary investigation only.

This profile permits broad Mach and IOKit access and does not defend against
hard links in allowed directories or inherited descriptors. Never use it to
launch a model or infer that evaluation data is isolated.
"""
import json
import os


def build_probe_profile(read_dirs: list[str], write_dirs: list[str]) -> dict:
    def checked(paths: list[str]) -> list[str]:
        if (not isinstance(paths, list) or not paths or
                any(not isinstance(path, str) or not path or '\x00' in path or
                    not os.path.isabs(path) or os.path.realpath(path) == '/' for path in paths)):
            raise ValueError('invalid synthetic profile directories')
        return sorted(set(os.path.realpath(path) for path in paths))
    reads = checked(read_dirs)
    writes = checked(write_dirs)
    roots = sorted(set(reads + ['/usr', '/System', '/bin', '/dev', '/opt/homebrew']))
    read_rules = ' '.join('(subpath ' + json.dumps(path) + ')' for path in roots)
    write_rules = ' '.join('(subpath ' + json.dumps(path) + ')' for path in writes)
    profile = ('(version 1) (deny default) (allow process-exec) (deny process-fork) '
               '(allow sysctl*) (allow mach*) (allow iokit-open) (allow file-read-metadata) '
               '(allow file-read* (literal "/") ' + read_rules + ') '
               '(allow file-write* (literal "/dev/null") ' + write_rules + ')')
    return {'profile': profile, 'execution_ready': False, 'isolation_verified': False,
            'evidence_type': 'synthetic_profile_proposal'}
