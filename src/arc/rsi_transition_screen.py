"""Replay and provisionally screen up to two unverified RSI transitions."""
import hashlib

from src.arc.rsi_comparison_audit import _read, audit_comparison_captures, replay_comparison
from src.arc.rsi_comparison_freeze import _hash, _validate
from src.arc.rsi_slot_lifecycle import _directory


def _screen(entry, *, generation, parent_adapter, tasks, sandbox, verifier, require_captures):
    if not isinstance(entry, dict) or set(entry) != {'root', 'sha256'} or not _hash(entry['sha256']):
        raise ValueError('invalid comparison reference')
    root, digest = entry['root'], entry['sha256']
    with _directory(root) as directory:
        frozen_bytes, frozen = _read(directory, 'comparison.frozen.json')
        started_bytes, started = _read(directory, 'comparison.started.json')
    if hashlib.sha256(frozen_bytes).hexdigest() != digest:
        raise ValueError('comparison digest differs')
    _validate(frozen)
    if (type(started.get('generation')) is not int or started['generation'] != generation or
            started.get('registration_sha256') != frozen['registration_sha256'] or
            frozen['parent_adapter'] != parent_adapter):
        raise ValueError('comparison generation or parent differs')
    auditor = audit_comparison_captures if require_captures else replay_comparison
    auditor(root, expected_sha256=digest, tasks=tasks, sandbox=sandbox, verifier=verifier)
    with _directory(root) as directory:
        frozen_after, _ = _read(directory, 'comparison.frozen.json')
        started_after, _ = _read(directory, 'comparison.started.json')
    if frozen_after != frozen_bytes or started_after != started_bytes:
        raise ValueError('comparison changed during screening')
    ids = frozen['task_ids']
    parent_solved = sum(frozen['parent'][key]['solved'] for key in ids)
    candidate_solved = sum(frozen['candidate'][key]['solved'] for key in ids)
    return {'candidate_adapter': frozen['candidate_adapter'], 'parent_solved': parent_solved,
            'candidate_solved': candidate_solved, 'positive': candidate_solved > parent_solved,
            'task_ids': ids, 'task_hashes': frozen['task_hashes'], 'comparison_sha256': digest}


def screen_rsi_chain(*, first, second=None, tasks, sandbox, verifier):
    return _screen_chain(first=first, second=second, tasks=tasks, sandbox=sandbox, verifier=verifier,
                         require_captures=False)


def screen_captured_rsi_chain(*, first, second=None, tasks, sandbox, verifier):
    return _screen_chain(first=first, second=second, tasks=tasks, sandbox=sandbox, verifier=verifier,
                         require_captures=True)


def _screen_chain(*, first, second, tasks, sandbox, verifier, require_captures):
    initial = _screen(first, generation=0, parent_adapter='none', tasks=tasks, sandbox=sandbox, verifier=verifier,
                      require_captures=require_captures)
    screened = [initial]
    if not initial['positive']:
        if second is not None:
            raise ValueError('second generation cannot follow a failed first screen')
        status, selected = 'first_screen_nonpositive_unverified', 'none'
    elif second is None:
        status, selected = 'first_screen_positive_unverified', initial['candidate_adapter']
    else:
        later = _screen(second, generation=1, parent_adapter=initial['candidate_adapter'],
                        tasks=tasks, sandbox=sandbox, verifier=verifier, require_captures=require_captures)
        if later['task_ids'] != initial['task_ids'] or later['task_hashes'] != initial['task_hashes']:
            raise ValueError('development tasks differ between transitions')
        screened.append(later)
        status = 'second_screen_positive_unverified' if later['positive'] else 'second_screen_nonpositive_unverified'
        selected = later['candidate_adapter'] if later['positive'] else initial['candidate_adapter']
    return {'status': status, 'provisional_selected_adapter': selected, 'transitions': screened,
            'execution_ready': False, 'independently_verified': False, 'improvement_claim_supported': False,
            'evidence_type': ('caller_supplied_captures_sandbox_and_verifier' if require_captures else
                              'caller_supplied_comparisons_sandbox_and_verifier')}
