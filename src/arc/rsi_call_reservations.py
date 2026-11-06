"""Durable scheduled-attempt reservations; not an execution authorization.

The fixed schedule limits generation requests to 30 per generation. It does not
account for calibration, tokens, sandbox starts, or elapsed time, and cannot
protect against uncooperative writers or authenticate an actual dispatch.
"""
import secrets

from src.arc.rsi_slot_lifecycle import _create, _directory, _exists, _identity, _name, _start


def reserve_call(root, *, registration_sha256: str, generation: int, slot: int,
                 attempt: int, claim_token: str) -> dict:
    identity = _identity(registration_sha256, generation, slot)
    if type(attempt) is not int or not 0 <= attempt <= 2 or not isinstance(claim_token, str):
        raise ValueError('invalid attempt or claim token')
    with _directory(root) as directory:
        started = _start(directory, identity)
        if not secrets.compare_digest(started['claim_token'], claim_token):
            raise ValueError('slot claim token differs')
        if _exists(directory, _name(identity, 'completed')):
            raise ValueError('completed slot cannot reserve a call')
        if attempt and not _exists(directory, _name(identity, f'call-{attempt - 1}')):
            raise ValueError('previous attempt was not reserved')
        marker = {**identity, 'attempt': attempt, 'claim_token': claim_token}
        _create(directory, _name(identity, f'call-{attempt}'), marker)
    return {'status': 'call_reserved_unverified', 'attempt': attempt,
            'eligible_for_training': False}
