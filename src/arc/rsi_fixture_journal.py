"""Journal fixture-only slot output; this is not an authenticated run audit."""
import hashlib
import json
import os
import stat

from src.arc.rsi_call_reservations import reserve_call
from src.arc.rsi_fixture import run_fixture_slot
from src.arc.rsi_slot_lifecycle import _create, _directory, claim_slot, complete_slot, observe_slot


def _filename(slot: int) -> str:
    return f'g0-slot-{slot:02d}.fixture.json'


def run_fixture_journaled(root, *, registration_sha256: str, slot: int, task, replies: list[str],
                          public_probes: list, private_probes: list) -> dict:
    """Claim before fixture execution; any interruption consumes the slot."""
    claim = claim_slot(root, registration_sha256=registration_sha256, generation=0, slot=slot)
    def before_attempt(attempt):
        reserve_call(root, registration_sha256=registration_sha256, generation=0, slot=slot,
                     attempt=attempt, claim_token=claim['claim_token'])

    result = run_fixture_slot(task, replies, public_probes=public_probes, private_probes=private_probes,
                              before_attempt=before_attempt)
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n').encode()
    if len(raw) > 1024 * 1024:
        raise ValueError('fixture evidence exceeds size limit')
    evidence_hash = hashlib.sha256(raw).hexdigest()
    with _directory(root) as directory:
        _create(directory, _filename(slot), result)
    complete_slot(root, registration_sha256=registration_sha256, generation=0, slot=slot,
                  claim_token=claim['claim_token'], evidence_sha256=evidence_hash)
    return {'status': 'fixture_completion_recorded', 'evidence_sha256': evidence_hash,
            'eligible_for_training': False}


def verify_fixture_journal(root, *, registration_sha256: str, slot: int) -> dict:
    """Check byte binding only; local writable markers are not an independent seal."""
    observed = observe_slot(root, registration_sha256=registration_sha256, generation=0, slot=slot)
    if observed['status'] != 'completion_recorded_unverified':
        raise ValueError('fixture completion evidence is unavailable')
    with _directory(root) as directory:
        try:
            fd = os.open(_filename(slot), os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
                        stat.S_IMODE(info.st_mode) != 0o600):
                    raise ValueError('fixture evidence is not owner-only')
                raw = stream.read(1024 * 1024 + 1)
        except OSError as exc:
            raise ValueError('fixture evidence is unavailable') from exc
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != observed['evidence_sha256']:
        raise ValueError('fixture evidence hash differs from completion')
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid fixture evidence') from exc
    if not isinstance(value, dict) or not isinstance(value.get('status'), str):
        raise ValueError('invalid fixture evidence')
    if value['status'] == 'fixture_draft':
        draft = value.get('draft')
        if (not isinstance(draft, dict) or draft.get('eligible_for_training') is not False or
                not isinstance(draft.get('provenance'), dict) or
                draft['provenance'].get('execution_mode') != 'fake_fixture'):
            raise ValueError('fixture evidence has invalid draft provenance')
    return {'status': 'fixture_integrity_only', 'evidence_sha256': observed['evidence_sha256'],
            'eligible_for_training': False}
