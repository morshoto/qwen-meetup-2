"""Local once-only diagnostic bookkeeping; never authorizes execution."""
import hashlib
import secrets

from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_comparison_freeze import _hash
from src.arc.rsi_slot_lifecycle import _create, _directory, _exists

_PROPOSAL = 'final.diagnostic.proposal.json'
_START = 'final.diagnostic.started.json'
_DONE = 'final.diagnostic.completed.json'


def _proposal(directory, expected):
    if not _hash(expected):
        raise ValueError('invalid proposal digest')
    raw, value = _read(directory, _PROPOSAL)
    if (len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected or
            value.get('status') != 'final_diagnostic_proposed_unverified' or
            value.get('execution_ready') is not False or not _hash(value.get('selected_adapter')) or
            value.get('competitive_target_percent') != 42.4 or
            value.get('target_split') != 'arc_agi_2_semi_private' or
            value.get('diagnostic_split') != 'arc_agi_2_public_evaluation'):
        raise ValueError('diagnostic proposal differs')
    return value


def _started(directory, expected):
    _, marker = _read(directory, _START)
    if (set(marker) != {'proposal_sha256', 'claim_token'} or marker['proposal_sha256'] != expected or
            not _hash(marker['claim_token'])):
        raise ValueError('diagnostic claim differs')
    return marker


def claim_diagnostic(root, *, expected_proposal_sha256):
    with _directory(root) as directory:
        _proposal(directory, expected_proposal_sha256)
        if _exists(directory, _DONE) and not _exists(directory, _START):
            raise ValueError('completion exists without diagnostic claim')
        marker = {'proposal_sha256': expected_proposal_sha256, 'claim_token': secrets.token_hex(32)}
        _create(directory, _START, marker)
    return {**marker, 'status': 'diagnostic_claimed_unverified', 'execution_ready': False}


def complete_diagnostic(root, *, expected_proposal_sha256, claim_token, evidence_sha256):
    if not _hash(claim_token) or not _hash(evidence_sha256):
        raise ValueError('invalid diagnostic completion identity')
    with _directory(root) as directory:
        _proposal(directory, expected_proposal_sha256)
        marker = _started(directory, expected_proposal_sha256)
        if not secrets.compare_digest(marker['claim_token'], claim_token):
            raise ValueError('diagnostic claim token differs')
        _create(directory, _DONE, {**marker, 'evidence_sha256': evidence_sha256})


def observe_diagnostic(root, *, expected_proposal_sha256):
    with _directory(root) as directory:
        _proposal(directory, expected_proposal_sha256)
        if not _exists(directory, _START):
            if _exists(directory, _DONE):
                raise ValueError('completion exists without diagnostic claim')
            status, evidence = 'unstarted', None
        else:
            marker = _started(directory, expected_proposal_sha256)
            if not _exists(directory, _DONE):
                status, evidence = 'started_outcome_uncertain', None
            else:
                _, completed = _read(directory, _DONE)
                if (set(completed) != {'proposal_sha256', 'claim_token', 'evidence_sha256'} or
                        completed['proposal_sha256'] != marker['proposal_sha256'] or
                        completed['claim_token'] != marker['claim_token'] or not _hash(completed['evidence_sha256'])):
                    raise ValueError('diagnostic completion differs')
                status, evidence = 'completion_recorded_unverified', completed['evidence_sha256']
    return {'status': status, 'evidence_sha256': evidence, 'execution_ready': False, 'independently_verified': False}
