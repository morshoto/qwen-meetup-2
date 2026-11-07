"""Offline score only fully replayed matched-comparison journal pairs.

The caller must bind allocation, source, public inputs, prior training/reload,
completed batch receipts and the selected answer source before calling this.
The result is local consistency evidence, not independent attestation.
"""
from src.arc.data import Verifier
from src.arc.lt_phase_b_comparison_journal import _read, _replay, validate_comparison_journals
from src.arc.lt_phase_b_paired import score_paired


def score_completed_comparison(*, off_journal, on_journal, registration, resource_limits, verifier):
    if not isinstance(verifier, Verifier):
        raise ValueError('offline verifier required')
    validate_comparison_journals(off_journal=off_journal, on_journal=on_journal,
                                 registration=registration, resource_limits=resource_limits)
    for state, journal in (('off', off_journal), ('on', on_journal)):
        if _replay(journal, state, registration, resource_limits) is not None or journal.pending(1):
            raise ValueError('comparison is incomplete or failed')
    rows = [_read(journal.root / (task + '.result.json'))
            for journal in (off_journal, on_journal) for task in journal.ids]
    report = score_paired(off_journal.ids, rows, verifier, registration=registration)
    return {'status': 'paired_score_consistent_unverified', 'evaluation_verified': False, **report}
