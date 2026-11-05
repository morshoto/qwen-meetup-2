"""Descriptive base comparison from replayed, captured development outcomes."""
import hashlib

from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_comparison_freeze import _validate
from src.arc.rsi_paired import paired_summary
from src.arc.rsi_slot_lifecycle import _directory
from src.arc.rsi_transition_screen import screen_captured_rsi_chain


def _paired_frozen_summary(first, selected):
    _validate(first)
    _validate(selected)
    if (first['parent_adapter'] != 'none' or first['task_ids'] != selected['task_ids'] or
            first['task_hashes'] != selected['task_hashes'] or first['seeds'] != selected['seeds'] or
            first['settings'] != selected['settings']):
        raise ValueError('base and selected comparison settings or tasks differ')
    ids = first['task_ids']
    return paired_summary({key: first['parent'][key]['solved'] for key in ids},
                          {key: selected['candidate'][key]['solved'] for key in ids})


def _frozen(entry):
    with _directory(entry['root']) as directory:
        raw, value = _read(directory, 'comparison.frozen.json')
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('frozen comparison digest differs')
    return raw, value


def summarize_captured_base_comparison(*, first, second=None, tasks, sandbox, verifier):
    screen = screen_captured_rsi_chain(first=first, second=second, tasks=tasks, sandbox=sandbox, verifier=verifier)
    selected = screen['provisional_selected_adapter']
    if selected == 'none':
        raise ValueError('no candidate passed development screening')
    selected_entry = second if len(screen['transitions']) == 2 and screen['transitions'][1]['positive'] else first
    base_bytes, base = _frozen(first)
    selected_bytes, candidate = _frozen(selected_entry)
    if candidate['candidate_adapter'] != selected:
        raise ValueError('selected adapter differs from frozen comparison')
    result = _paired_frozen_summary(base, candidate)
    criterion = result.pop('improvement_criterion_met')
    if _frozen(first)[0] != base_bytes or _frozen(selected_entry)[0] != selected_bytes:
        raise ValueError('frozen comparison changed during summary')
    return {**result, 'status': 'matched_base_summary_unverified', 'selected_adapter': selected,
            'base_comparison_sha256': first['sha256'], 'selected_comparison_sha256': selected_entry['sha256'],
            'descriptive_bootstrap_criterion_met': criterion, 'independently_verified': False,
            'improvement_claim_supported': False, 'evidence_type': 'caller_supplied_captures_sandbox_and_verifier'}
