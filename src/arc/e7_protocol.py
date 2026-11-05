"""Pre-registered E7 schedule and conservative feasibility decisions."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.arc.e7_config import E7Settings

ARMS = ('G0_baseline', 'G1_novelty_feedback')


def schedule(settings: E7Settings) -> list[dict[str, Any]]:
    result = []
    for slot in range(settings.slots):
        arms = ARMS if slot % 2 == 0 else tuple(reversed(ARMS))
        for arm in arms:
            result.append({'slot': slot, 'arm': arm,
                           'attempt_seeds': [settings.seed + slot * 3 + attempt
                                             for attempt in range(settings.max_repairs + 1)]})
    return result


def selection(rows: list[dict[str, Any]], settings: E7Settings, *, protocol_valid: bool = True) -> dict[str, Any]:
    expected = {(entry['slot'], entry['arm']) for entry in schedule(settings)}
    keys = [(row['slot'], row['arm']) for row in rows]
    if not protocol_valid or len(set(keys)) != len(keys) or not set(keys) <= expected:
        decision = 'protocol_invalid'
    elif set(keys) != expected or any(row.get('completion') != 'completed' for row in rows):
        decision = 'inconclusive_budget_or_missing_data'
    else:
        counts = {arm: sum(bool(row.get('verified_novel')) for row in rows if row['arm'] == arm)
                  for arm in ARMS}
        decision = 'novelty_gain_observed' if counts[ARMS[1]] > counts[ARMS[0]] else 'no_gain_observed'
    completed = {arm: sum(row.get('completion') == 'completed' for row in rows if row['arm'] == arm)
                 for arm in ARMS}
    novel = {arm: sum(bool(row.get('verified_novel')) for row in rows
                      if row['arm'] == arm and row.get('completion') == 'completed') for arm in ARMS}
    by_slot = {(row['slot'], row['arm']): row for row in rows}
    paired = sum(all(by_slot.get((slot, arm), {}).get('completion') == 'completed' for arm in ARMS)
                 for slot in range(settings.slots))
    return {'decision': decision, 'planned_slots_per_arm': settings.slots,
            'completed_slots': completed, 'novel_oracles': novel, 'paired_completed_slots': paired,
            'settings': asdict(settings)}


def feasibility(*, strict_count: int, new_strict_count: int, selection_decision: str,
                resource_limited: bool = False) -> str:
    if strict_count < 0 or new_strict_count < 0 or new_strict_count > strict_count:
        raise ValueError('invalid strict Oracle counts')
    if selection_decision == 'protocol_invalid':
        return 'inconclusive_protocol_or_evidence'
    if strict_count >= 5:
        return 'ready_for_gen0_to_gen1'
    if new_strict_count:
        return 'novel_oracles_but_below_gate'
    if selection_decision in {'novelty_gain_observed', 'no_gain_observed'}:
        return 'no_novel_oracles_in_valid_comparison'
    return 'inconclusive_resource_limited' if resource_limited else 'inconclusive_protocol_or_evidence'


def arm_metrics(rows: list[dict[str, Any]], arm: str, planned: int) -> dict[str, Any]:
    selected = [r for r in rows if r['arm'] == arm]
    done = [r for r in selected if r.get('completion') == 'completed']
    attempts = [a for r in selected for a in r.get('attempts', [])]
    verified = [r for r in done if r.get('oracle_valid')]
    repaired = [r for r in done if any(a.get('attempt', 0) > 0 for a in r.get('attempts', []))]
    observed = [r.get('observed_usage', {}) for r in selected]
    budgets = [r.get('budget', {}) for r in selected]
    known_calls = sum(x.get('calls_with_usage', 0) for x in observed)
    calls = sum(x.get('model_calls', 0) for x in budgets)
    rate = lambda x, n: x / n if n else None
    return {'planned_slots': planned, 'started_slots': len(selected), 'completed_slots': len(done),
        'incomplete_slots': len(selected) - len(done), 'attempts': len(attempts),
        'verified_novel_oracles': sum(bool(r.get('verified_novel')) for r in done),
        'valid_oracles': len(verified), 'ast_duplicate_candidates': sum(bool(r.get('ast_duplicate')) for r in done),
        'ast_duplicate_rate_per_completed_candidate': rate(sum(bool(r.get('ast_duplicate')) for r in done), len(done)),
        'repair_attempted_candidates': len(repaired), 'repair_successes': sum(bool(r.get('repair_pass')) for r in repaired),
        'repair_success_rate': rate(sum(bool(r.get('repair_pass')) for r in repaired), len(repaired)),
        'sandbox_valid_candidates': len(verified),
        'syntax_valid_attempts': sum(a.get('category') not in {'syntax_error', 'invalid_model_response', 'model_request_failed', 'budget_exhausted'} and bool(a.get('code_sha256')) for a in attempts),
        'syntax_attempt_denominator': len(attempts),
        'syntax_valid_rate_per_attempt': rate(sum(a.get('category') not in {'syntax_error', 'invalid_model_response', 'model_request_failed', 'budget_exhausted'} and bool(a.get('code_sha256')) for a in attempts), len(attempts)),
        'sandbox_execution_successes': sum(r.get('sandbox_successful_executions', 0) for r in selected),
        'sandbox_execution_denominator': sum(r.get('sandbox_finished_executions', 0) for r in selected),
        'sandbox_execution_success_rate': rate(sum(r.get('sandbox_successful_executions', 0) for r in selected), sum(r.get('sandbox_finished_executions', 0) for r in selected)),
        'sampled_behavior_suspects': sum(bool(r.get('behavior_match')) for r in done),
        'behavior_fingerprint_denominator': sum('behavior_sha256' in r for r in done),
        'model_calls_per_verified_candidate': rate(sum(r.get('budget', {}).get('model_calls', 0) for r in verified), len(verified)),
        'reserved_tokens_per_verified_candidate': rate(sum(r.get('budget', {}).get('reserved_tokens', 0) for r in verified), len(verified)),
        'sandbox_starts_per_verified_candidate': rate(sum(r.get('budget', {}).get('sandbox_runs', 0) for r in verified), len(verified)),
        'seconds_per_verified_candidate': rate(sum(r.get('elapsed_seconds', 0) for r in verified), len(verified)),
        'model_calls_per_completed_slot': rate(calls, len(done)),
        'sandbox_starts_per_completed_slot': rate(sum(x.get('sandbox_runs', 0) for x in budgets), len(done)),
        'seconds_per_completed_slot': rate(sum(r.get('elapsed_seconds', 0) for r in selected), len(done)),
        'model_calls': calls,
        'reserved_tokens': sum(x.get('reserved_tokens', 0) for x in budgets),
        'observed_tokens': sum(x.get('observed_total_tokens') or 0 for x in observed) if calls and known_calls == calls else None,
        'observed_prompt_tokens': sum(x.get('observed_prompt_tokens') or 0 for x in observed) if calls and known_calls == calls else None,
        'sandbox_starts': sum(x.get('sandbox_runs', 0) for x in budgets),
        'slot_elapsed_seconds': sum(r.get('elapsed_seconds', 0) for r in selected),
        'model_inference_seconds': sum(r.get('model_inference_seconds', 0) for r in selected),
        'candidate_completion_rate': rate(len(done), planned),
        'verification_cost_denominator': len(verified)}
