"""Durable slot markers for cooperating local processes, not run authorization.

A marker alone cannot establish authenticated provenance or protect against a
writer with access to the run directory. Independent audit remains required.
"""
import json
import os
import re
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path

_HASH = re.compile(r'^[0-9a-f]{64}$')


def _identity(registration_sha256, generation, slot):
    if (not isinstance(registration_sha256, str) or not _HASH.fullmatch(registration_sha256) or
            type(generation) is not int or generation not in (0, 1) or
            type(slot) is not int or not 0 <= slot < 10):
        raise ValueError('invalid registered slot identity')
    return {'registration_sha256': registration_sha256, 'generation': generation, 'slot': slot}


@contextmanager
def _directory(root):
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY'):
        raise ValueError('safe directory operations are unavailable')
    try:
        fd = os.open(Path(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError('run directory is unavailable or symlinked') from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError('run directory must be owner-only')
        yield fd
    finally:
        os.close(fd)


def _name(identity, phase):
    return f"g{identity['generation']}-slot-{identity['slot']:02d}.{phase}.json"


def _exists(directory, name):
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _read(directory, name):
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError('slot marker must be a private regular file')
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError('slot marker exceeds size limit')
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError('duplicate slot marker key')
                value[key] = item
            return value
        value = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(value, dict):
            raise ValueError('slot marker must be an object')
        return value
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError('slot marker is unreadable or invalid') from exc


def _create(directory, name, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(directory)


def _start(directory, identity):
    value = _read(directory, _name(identity, 'started'))
    if (set(value) != {*identity, 'claim_token'} or any(value[key] != item for key, item in identity.items()) or
            type(value['generation']) is not int or type(value['slot']) is not int or
            not isinstance(value.get('claim_token'), str) or not _HASH.fullmatch(value['claim_token'])):
        raise ValueError('slot start marker differs from registration')
    return value


def claim_slot(root, *, registration_sha256: str, generation: int, slot: int) -> dict:
    identity = _identity(registration_sha256, generation, slot)
    with _directory(root) as directory:
        started, completed = _name(identity, 'started'), _name(identity, 'completed')
        if _exists(directory, completed):
            if not _exists(directory, started):
                raise ValueError('completion marker exists without a start marker')
            raise FileExistsError('slot has already been consumed')
        value = {**identity, 'claim_token': secrets.token_hex(32)}
        _create(directory, started, value)
        return dict(value)


def complete_slot(root, *, registration_sha256: str, generation: int, slot: int,
                  claim_token: str, evidence_sha256: str) -> None:
    identity = _identity(registration_sha256, generation, slot)
    if (not isinstance(claim_token, str) or not _HASH.fullmatch(claim_token) or
            not isinstance(evidence_sha256, str) or not _HASH.fullmatch(evidence_sha256)):
        raise ValueError('invalid claim token or evidence digest')
    with _directory(root) as directory:
        started = _start(directory, identity)
        if not secrets.compare_digest(started['claim_token'], claim_token):
            raise ValueError('slot claim token differs')
        _create(directory, _name(identity, 'completed'),
                {**identity, 'claim_token': claim_token, 'evidence_sha256': evidence_sha256})


def observe_slot(root, *, registration_sha256: str, generation: int, slot: int) -> dict:
    identity = _identity(registration_sha256, generation, slot)
    with _directory(root) as directory:
        started_name, completed_name = _name(identity, 'started'), _name(identity, 'completed')
        if not _exists(directory, started_name):
            if _exists(directory, completed_name):
                raise ValueError('completion marker exists without a start marker')
            return {'status': 'unstarted', 'eligible_for_training': False}
        started = _start(directory, identity)
        if not _exists(directory, completed_name):
            return {'status': 'started_outcome_uncertain', 'eligible_for_training': False}
        completed = _read(directory, completed_name)
        if (set(completed) != {*identity, 'claim_token', 'evidence_sha256'} or
                any(completed[key] != item for key, item in identity.items()) or
                type(completed['generation']) is not int or type(completed['slot']) is not int or
                completed.get('claim_token') != started['claim_token'] or
                not isinstance(completed.get('evidence_sha256'), str) or not _HASH.fullmatch(completed['evidence_sha256'])):
            raise ValueError('completion marker differs from slot identity')
        return {'status': 'completion_recorded_unverified', 'eligible_for_training': False,
                'evidence_sha256': completed['evidence_sha256']}
