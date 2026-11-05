"""Build a proposed Darwin fork-restricted command; never launch it.

The profile is only one containment layer. It cannot prove that services or
other processes cannot act for the child, nor that a model workload will run.
"""
import os

_PROFILE = '(version 1) (allow default) (deny process-fork)'


def fork_restricted_command(argv):
    if (not isinstance(argv, list) or not argv or
            any(not isinstance(item, str) or not item or '\x00' in item for item in argv) or
            not os.path.isabs(argv[0])):
        raise ValueError('invalid proposed child command')
    return {'argv': ['/usr/bin/sandbox-exec', '-p', _PROFILE, *argv],
            'execution_ready': False, 'training_allowed': False, 'containment_verified': False,
            'evidence_type': 'proposed_fork_restriction_only'}
