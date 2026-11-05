"""Conservative shared collection-stage reservations for cooperating processes.

Local mutable state is not an authenticated ledger, a final registration, or
permission to dispatch. Calls must reserve here before any transport or sandbox
start, including canaries; a reservation is never refunded after interruption.
"""
import fcntl
import hashlib
import re
import math
import os
import secrets
import stat
import time
from contextlib import contextmanager

from src.arc.rsi_slot_lifecycle import _create, _directory, _identity, _read


class CollectionBudget:
    max_model_calls = 32
    max_reserved_tokens = 200000
    max_sandbox_runs = 300
    max_wall_seconds = 3600
    model_kinds = frozenset(('generation', 'canary'))

    def __init__(self, root, identity):
        self.root = root
        self.identity = identity
        self.name = f"g{identity['generation']}.budget.json"
        self.lock_name = f"g{identity['generation']}.budget.lock"

    @classmethod
    def create(cls, root, *, registration_sha256, generation, reference_setup_bytes=None,
               expected_reference_setup_sha256=None, model_key=None):
        reference = None
        supplied = (reference_setup_bytes, expected_reference_setup_sha256, model_key)
        if any(item is not None for item in supplied):
            from src.arc.rsi_transport_audit import _json_object
            if (generation != 0 or not isinstance(reference_setup_bytes, bytes) or
                    not 1 <= len(reference_setup_bytes) <= 65536 or
                    not isinstance(expected_reference_setup_sha256, str) or
                    not re.fullmatch(r'[0-9a-f]{64}', expected_reference_setup_sha256) or
                    hashlib.sha256(reference_setup_bytes).hexdigest() != expected_reference_setup_sha256 or
                    model_key not in ('14b', '32b')):
                raise ValueError('invalid reference setup budget identity')
            reference = _json_object(reference_setup_bytes)
            if (reference.get('status') != 'legacy_behavior_reference_local_unverified' or
                    reference.get('model_key') != model_key or type(reference.get('sandbox_starts')) is not int or
                    reference['sandbox_starts'] != 10 or type(reference.get('model_calls')) is not int or
                    reference['model_calls'] != 0 or type(reference.get('training_steps')) is not int or
                    reference['training_steps'] != 0 or any(reference.get(k) is not False for k in
                    ('private_outputs_exposed', 'eligible_for_training', 'independently_verified', 'generation_authorized')) or
                    type(reference.get('elapsed_seconds')) not in (int, float) or
                    not math.isfinite(reference['elapsed_seconds']) or not 0 <= reference['elapsed_seconds'] <= 60 or
                    not isinstance(reference.get('pool_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', reference['pool_sha256'])):
                raise ValueError('reference setup accounting differs')
        identity = _identity(registration_sha256, generation, 0)
        identity.pop('slot')
        budget = cls(root, identity)
        with budget._locked() as directory:
            wall, monotonic = time.time(), time.monotonic()
            state = {**identity, 'started_wall': wall, 'last_wall': wall,
                     'started_monotonic': monotonic, 'last_monotonic': monotonic,
                     'model_calls': 0, 'reserved_tokens': 0, 'sandbox_runs': 0}
            if reference is not None:
                state.update(sandbox_runs=reference['sandbox_starts'], reference_setup_sha256=expected_reference_setup_sha256,
                             reference_model_key=model_key, reference_pool_sha256=reference['pool_sha256'],
                             reference_elapsed_seconds=reference['elapsed_seconds'])
            _create(directory, budget.name, state)
        return budget

    @classmethod
    def open(cls, root, *, registration_sha256, generation):
        identity = _identity(registration_sha256, generation, 0)
        identity.pop('slot')
        budget = cls(root, identity)
        budget.snapshot()
        return budget

    @contextmanager
    def _locked(self):
        with _directory(self.root) as directory:
            fd = os.open(self.lock_name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                    raise ValueError('budget lock must be owner-only')
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield directory
            finally:
                os.close(fd)

    def _state(self, directory):
        state = _read(directory, self.name)
        keys = {*self.identity, 'started_wall', 'last_wall', 'started_monotonic',
                'last_monotonic', 'model_calls', 'reserved_tokens', 'sandbox_runs'}
        reference_keys = {'reference_setup_sha256', 'reference_model_key', 'reference_pool_sha256', 'reference_elapsed_seconds'}
        if set(state) not in (keys, keys | reference_keys) or any(state[k] != v or type(state[k]) is not type(v) for k, v in self.identity.items()):
            raise ValueError('budget identity differs from registration')
        if reference_keys <= set(state) and (self.identity['generation'] != 0 or state['reference_model_key'] not in ('14b','32b') or
                any(not isinstance(state[k], str) or not re.fullmatch(r'[0-9a-f]{64}', state[k])
                    for k in ('reference_setup_sha256','reference_pool_sha256')) or
                type(state['reference_elapsed_seconds']) not in (int, float) or
                not math.isfinite(state['reference_elapsed_seconds']) or not 0 <= state['reference_elapsed_seconds'] <= 60 or
                type(state['sandbox_runs']) is not int or state['sandbox_runs'] < 10):
            raise ValueError('invalid reference setup budget state')
        for key, limit in (('model_calls', self.max_model_calls), ('reserved_tokens', self.max_reserved_tokens),
                           ('sandbox_runs', self.max_sandbox_runs)):
            if type(state[key]) is not int or not 0 <= state[key] <= limit:
                raise ValueError('invalid collection budget counter')
        for key in ('started_wall', 'last_wall', 'started_monotonic', 'last_monotonic'):
            if type(state[key]) not in (int, float) or not math.isfinite(state[key]) or state[key] < 0:
                raise ValueError('invalid collection budget clock')
        if state['last_wall'] < state['started_wall'] or state['last_monotonic'] < state['started_monotonic']:
            raise ValueError('invalid collection budget clock order')
        return state

    def snapshot(self):
        with self._locked() as directory:
            return self._state(directory)

    def _remaining(self, state, wall, monotonic):
        if (not math.isfinite(wall) or not math.isfinite(monotonic) or
                wall < state['last_wall'] or monotonic < state['last_monotonic']):
            raise TimeoutError('collection budget expired or clock moved backwards')
        remaining = self.max_wall_seconds - state.get('reference_elapsed_seconds', 0) - max(
            wall - state['started_wall'], monotonic - state['started_monotonic'])
        if remaining <= 0:
            raise TimeoutError('collection budget expired or clock moved backwards')
        return remaining

    def remaining(self):
        with self._locked() as directory:
            return self._remaining(self._state(directory), time.time(), time.monotonic())

    def _reserve(self, *, tokens=0, sandbox=False):
        with self._locked() as directory:
            state = self._state(directory)
            wall, monotonic = time.time(), time.monotonic()
            self._remaining(state, wall, monotonic)
            if sandbox:
                if state['sandbox_runs'] >= self.max_sandbox_runs:
                    raise TimeoutError('collection sandbox budget exhausted')
                state['sandbox_runs'] += 1
            else:
                if state['model_calls'] >= self.max_model_calls or state['reserved_tokens'] + tokens > self.max_reserved_tokens:
                    raise TimeoutError('collection call or token budget exhausted')
                state['model_calls'] += 1
                state['reserved_tokens'] += tokens
            state['last_wall'], state['last_monotonic'] = wall, monotonic
            temporary = f'.{self.name}.{secrets.token_hex(16)}.tmp'
            _create(directory, temporary, state)
            try:
                os.replace(temporary, self.name, src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass

    def reserve_model(self, tokens, *, kind):
        if type(tokens) is not int or tokens <= 0 or kind not in self.model_kinds:
            raise ValueError('invalid model reservation')
        self._reserve(tokens=tokens)

    def reserve_sandbox(self):
        self._reserve(sandbox=True)
