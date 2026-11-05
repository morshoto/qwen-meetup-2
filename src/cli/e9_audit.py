"""Independent offline audit for a sealed E9 model run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.arc.e5_selfplay import _generator_prompt
from src.arc.e6f_repair import repair_prompt
from src.arc.e6g_novelty import classify_novelty
from src.arc.e9_config import MODELS, E9Settings
from src.arc.http_backend import final_answer
from src.pipelines.e5 import _verify_mutable_file
from src.pipelines.e6 import _observed_token_usage
from src.pipelines.e6f import _sha
from src.pipelines.e6g import _seal_g, _verify_g
from src.pipelines.e8 import _model_dir, _model_manifest
from src.pipelines.e9 import IMPLEMENTATION_FILES, _audit_sft, _live_provenance

PROJECT = Path(__file__).resolve().parents[2]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def _audit_slot(root: Path, index: int, protocol: dict, pool: list[dict], fake: bool) -> tuple[dict, int, int]:
    slot_root = root / 'slots' / str(index)
    _verify_g(slot_root)
    row = json.loads((slot_root / 'result.json').read_text(encoding='utf-8'))
    if row.get('slot') != index or row.get('arm') != 'G0_baseline':
        raise ValueError('E9 slot identity mismatch')
    budget = row.get('budget', {})
    if (budget.get('model_calls', 0) > 3 or budget.get('reserved_tokens', 0) > 15000 or
            budget.get('sandbox_runs', 0) > protocol['settings']['candidate_sandbox']):
        raise ValueError('E9 slot budget exceeded')
    prompt_events = row.get('prompts', [])
    if fake:
        if any(event.get('feedback_applied') for event in prompt_events):
            raise ValueError('E9 Fake run unexpectedly applied feedback')
        for event in prompt_events:
            expected_index = index * 3 + event['attempt']
            if event.get('seed_requested') != protocol['settings']['seed'] + expected_index:
                raise ValueError('E9 Fake requested seed differs from protocol')
        fake_events = _jsonl(slot_root / 'events.jsonl')
        return row, len(prompt_events), sum(event.get('event') == 'sandbox_dispatch' for event in fake_events)
    events = _jsonl(slot_root / 'events.jsonl')
    responses = _jsonl(slot_root / 'logs/e6/llm_responses.jsonl')
    requests = _jsonl(slot_root / 'logs/e6/llm_requests.jsonl')
    response_by_candidate: dict[int, dict] = {}
    for response in responses:
        candidate = response.get('candidate')
        if type(candidate) is not int or candidate in response_by_candidate:
            raise ValueError('E9 response candidate is missing or duplicated')
        response_by_candidate[candidate] = response
    attempts: dict[int, str] = {}
    failures: dict[int, tuple[str, str | None]] = {}
    request_ids: set[str] = set()
    for event in events:
        if event.get('event') == 'attempt_finished':
            failures[event['attempt']] = (event['category'], event.get('exception_type'))
    model_id = protocol['model_identity'][0]
    revision = protocol['model_identity'][1]
    for event in prompt_events:
        attempt = event['attempt']
        prompt = (_generator_prompt() if attempt == 0 else
                  repair_prompt(attempts[attempt - 1], *failures[attempt - 1]))
        expected_index = index * 3 + attempt
        if (event.get('feedback_applied') or event.get('seed_requested') != protocol['settings']['seed'] + expected_index or
                hashlib.sha256(prompt.encode()).hexdigest() != event.get('prompt_sha256')):
            raise ValueError('E9 prompt, seed or feedback audit failed')
        response = response_by_candidate.get(expected_index)
        if not response:
            raise ValueError('E9 response log lacks a requested candidate')
        code = final_answer(response['generated_text']).strip()
        _live_provenance(slot_root, prompt, expected_index, code)
        matches = [item for item in requests if item.get('candidate') == expected_index]
        if len(matches) != 1:
            raise ValueError('E9 request log missing or duplicated')
        request = matches[0]
        if (request.get('model') != model_id or request.get('revision') != revision or
                request.get('thinking_mode_effective') != 'off'):
            raise ValueError('E9 request model, revision or thinking mode mismatch')
        if request.get('request_id') in request_ids:
            raise ValueError('E9 request ID duplicated')
        request_ids.add(request['request_id'])
        attempts[attempt] = code
    if len(request_ids) != len(requests) or len(requests) != len(prompt_events):
        raise ValueError('E9 model request log has unaccounted calls')
    if row.get('oracle_valid'):
        if (any(row.get('phase_starts', {}).get(phase, 0) < 10
                for phase in ('public_validation', 'private_acceptance')) or
                row.get('sandbox_successful_executions', 0) < 20):
            raise ValueError('E9 verified candidate lacks full public/private execution count')
        final_attempt = max(attempts)
        if row.get('verified_novel') != (
                classify_novelty(attempts[final_attempt], row.get('behavior_sha256'), pool) == 'novel'):
            raise ValueError('E9 strict novelty classification differs from the sealed pool')
        if row.get('verified_novel'):
            item = row.get('accepted_item')
            if not isinstance(item, dict) or not isinstance(item.get('record'), dict):
                raise ValueError('E9 strict candidate lacks its accepted Oracle record')
            record = item['record']
            if (record.get('messages', [{}])[-1].get('content') != attempts[final_attempt] or
                    record.get('qwen_provenance', {}).get('request_id') not in request_ids):
                raise ValueError('E9 accepted Oracle does not bind to its live generation response')
            pool.append({'record': record, 'behavior_sha256': item['behavior_sha256'],
                         'origin': 'E9_same_model', 'historical_duplicate_observed': False})
    elif row.get('verified_novel'):
        raise ValueError('E9 unverified candidate was marked strict')
    dispatches = sum(event.get('event') == 'sandbox_dispatch' for event in events)
    if dispatches > budget.get('sandbox_runs', 0):
        raise ValueError('E9 sandbox trace exceeds the candidate budget')
    return row, len(requests), dispatches


def audit(run_root: Path, output: Path) -> dict[str, Any]:
    root = run_root.expanduser().resolve()
    _verify_g(root)
    protocol_path = root / 'protocol.lock.yml'
    _verify_mutable_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if (root / 'protocol.sha256').read_text().strip() != _sha(protocol_path):
        raise ValueError('E9 protocol hash mismatch')
    if protocol.get('protocol') != 'e9_dual_model_autonomous_rsi_v1':
        raise ValueError('unknown E9 protocol')
    for name, digest in protocol['e9_source_hashes'].items():
        if _sha(PROJECT / name) != digest:
            raise ValueError('E9 implementation dependency differs: ' + name)
    settings = E9Settings(**protocol['settings'])
    model_key = protocol['model_key']
    if model_key not in settings.models or protocol.get('model_identity') != list(MODELS[model_key]):
        raise ValueError('E9 model identity does not match the pinned registry')
    fake = settings.profile == 'fake'
    model_manifest = protocol['model_manifest']
    if not fake:
        actual = _model_manifest(_model_dir(MODELS[model_key][0], MODELS[model_key][1]), MODELS[model_key][1])
        if actual != model_manifest:
            raise ValueError('E9 model files or chat template changed')
    _verify_mutable_file(root / 'initial_pool.json')
    _verify_mutable_file(root / 'novelty_pool.json')
    fixed_pool = json.loads((root / 'initial_pool.json').read_text(encoding='utf-8'))
    fixed_sha = hashlib.sha256(json.dumps(fixed_pool, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=False).encode()).hexdigest()
    if fixed_sha != protocol['initial_pool_sha256']:
        raise ValueError('E9 fixed initial pool differs from protocol')
    initial_pool = json.loads((root / 'novelty_pool.json').read_text(encoding='utf-8'))
    pool_sha = hashlib.sha256(json.dumps(initial_pool, sort_keys=True, separators=(',', ':'),
                                           ensure_ascii=False).encode()).hexdigest()
    if pool_sha != protocol['starting_novelty_pool_sha256']:
        raise ValueError('E9 starting novelty pool differs from protocol')
    result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    if (result.get('model_key') != model_key or result.get('revision') != MODELS[model_key][1] or
            result.get('campaign_protocol_sha256') != _sha(protocol_path)):
        raise ValueError('E9 result model or protocol identity mismatch')
    rows, requests, dispatches = [], 0, 0
    for index in range(settings.slot_offset, settings.slot_offset + settings.slots):
        slot_root = root / 'slots' / str(index)
        if not slot_root.is_dir():
            continue
        row, slot_requests, slot_dispatches = _audit_slot(root, index, protocol, initial_pool, fake)
        rows.append(row)
        requests += slot_requests
        dispatches += slot_dispatches
    strict_new = 0 if fake else sum(bool(row.get('verified_novel')) for row in rows)
    if (result.get('verified_new_oracles') != strict_new or
            result.get('completed_slots') != sum(row.get('completion') == 'completed' for row in rows)):
        raise ValueError('E9 result counts differ from candidate slots')
    total_budget = json.loads((root / 'total-budget.json').read_text(encoding='utf-8'))
    model_budget = json.loads((root / 'model-budget.json').read_text(encoding='utf-8'))
    for path in (root / 'total-budget.json', root / 'model-budget.json'):
        _verify_mutable_file(path)
    if total_budget['model_calls'] > settings.calls or total_budget['reserved_tokens'] > settings.reserved_tokens or total_budget['sandbox_runs'] > settings.sandbox:
        raise ValueError('E9 aggregate budget exceeded')
    if any(model_budget.get(key) != total_budget.get(key)
           for key in ('model_calls', 'reserved_tokens', 'sandbox_runs')):
        raise ValueError('E9 total and per-model budget ledgers disagree')
    request_files = sorted(root.rglob('llm_requests.jsonl'))
    all_requests = [request for path in request_files for request in _jsonl(path)]
    observed_total = 0
    observed_calls = 0
    usage_complete = bool(all_requests) and not fake
    for path in request_files:
        usage = _observed_token_usage(path)
        observed_calls += usage['calls_with_usage']
        if usage['observed_total_tokens'] is not None:
            observed_total += usage['observed_total_tokens']
        if usage['calls_with_usage'] != len(_jsonl(path)):
            usage_complete = False
    if fake:
        usage_complete = False
    logged_call_count = len(all_requests) if not fake else 2 + requests
    if logged_call_count != total_budget['model_calls']:
        raise ValueError('E9 aggregate model request count differs from budget ledger')
    if dispatches != total_budget['sandbox_runs']:
        raise ValueError('E9 aggregate sandbox dispatch count differs from budget ledger')
    accepted = _jsonl(root / 'strict_oracles.jsonl')
    sft_rows, sft_audit = _audit_sft(accepted, model_key, protocol['sandbox_image_id'], model_manifest)
    saved_sft = _jsonl(root / 'sft.jsonl')
    if saved_sft != sft_rows or json.loads((root / 'sft-audit.json').read_text()) != sft_audit:
        raise ValueError('E9 SFT records or mask audit differ from strict Oracle provenance')
    training = json.loads((root / 'training-gate.json').read_text())
    if training != result.get('training_gate') or training['strict_total'] != len(sft_rows):
        raise ValueError('E9 strict training gate differs from audited SFT records')
    if any((result.get('training_invoked'), result.get('weights_updated'), result.get('promoted'),
            result.get('monitoring_run'), result.get('public_evaluation_run'), result.get('operator_search_used'))):
        raise ValueError('E9 collection unexpectedly trained, promoted, or used forbidden evaluation')
    summary = {
        'audit_status': ('passed' if result.get('observed_tokens') ==
                         (observed_total if usage_complete else None) else 'passed_with_reconciliation'),
        'run_dir': str(root), 'model_key': model_key,
        'model_id': MODELS[model_key][0], 'revision': MODELS[model_key][1],
        'protocol_sha256': _sha(protocol_path), 'auditor_sha256': _sha(Path(__file__)),
        'planned_slots': settings.slots, 'slots_audited': len(rows),
        'strict_new_oracles': strict_new, 'strict_total': training['strict_total'],
        'strict_existing': training['strict_eligible_existing'],
        'strict_parent_e9': training['strict_parent_e9'],
        'generation_requests_audited': requests,
        'aggregate_request_count': len(all_requests), 'sandbox_dispatches_audited': dispatches,
        'reserved_tokens': total_budget['reserved_tokens'],
        'observed_tokens_reconciled': observed_total if usage_complete else None,
        'observed_usage_calls': observed_calls, 'observed_usage_complete': usage_complete,
        'result_summary_observed_tokens_matches': result.get('observed_tokens') == (observed_total if usage_complete else None),
        'training': {'trained': training['trained'], 'weights_updated': training['weights_updated'],
                     'promotion': training['promotion'], 'reason': training['reason']},
        'checks': {'protocol_and_tree_seals': True, 'source_hashes': True,
                   'model_revision_and_template': True, 'request_provenance': not fake,
                   'budgets': True, 'strict_gate_and_sft_mask': True,
                   'operator_search_unreferenced_by_pipeline': True,
                   'monitoring_and_public_eval_unused': True},
        'note': 'Observed tokens are independently summed from per-slot request logs; they may reconcile a null pipeline summary field.',
    }
    out = output.expanduser().resolve()
    if out.exists():
        raise FileExistsError('E9 audit output already exists')
    out.mkdir(parents=True, mode=0o700)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (out / 'report.md').write_text(
        '# E9 offline audit\n\n'
        f"- Model: `{model_key}` (`{MODELS[model_key][0]}`)\n"
        f"- Protocol SHA-256: `{summary['protocol_sha256']}`\n"
        f"- Slots audited: {len(rows)} / {settings.slots}\n"
        f"- Strict new / total: {strict_new} / {training['strict_total']}\n"
        f"- Calls / reserved tokens / observed tokens: {len(all_requests)} / {total_budget['reserved_tokens']} / "
        f"{summary['observed_tokens_reconciled']}\n\n"
        'This audit checks sealed request provenance, model identity, slot and aggregate budgets, verifier counts, '
        'strict Oracle/SFT provenance, and the explicit no-training boundary.\n', encoding='utf-8')
    for path in out.iterdir():
        path.chmod(0o600)
    _seal_g(out)
    _verify_g(out)
    return summary


def write_source_mismatch_exclusion(run_root: Path, output: Path) -> dict[str, Any]:
    """Preserve a sealed run while explicitly excluding it after code drift."""
    root = run_root.expanduser().resolve()
    _verify_g(root)
    protocol_path = root / 'protocol.lock.yml'
    _verify_mutable_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    mismatches = []
    for name, expected in protocol.get('e9_source_hashes', {}).items():
        actual = _sha(PROJECT / name)
        if actual != expected:
            mismatches.append({'path': name, 'registered_sha256': expected, 'current_sha256': actual})
    if not mismatches:
        raise ValueError('run source hashes match; source-mismatch exclusion is not applicable')
    result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    summary = {
        'audit_status': 'excluded_from_strict_gate', 'run_dir': str(root),
        'model_key': result.get('model_key'), 'model_id': result.get('model_id'),
        'revision': result.get('revision'), 'protocol_sha256': _sha(protocol_path),
        'root_seal_sha256': _sha(root / 'seal.json'),
        'registered_strict_new_oracles': result.get('verified_new_oracles'),
        'strict_new_oracles_eligible': 0,
        'reason': 'source files changed after this run was sealed; strict provenance cannot be re-audited',
        'source_hash_mismatches': mismatches,
        'original_artifact_modified': False, 'training_or_promotion_permitted': False,
    }
    out = output.expanduser().resolve()
    if out.exists():
        raise FileExistsError('E9 exclusion note output already exists')
    out.mkdir(parents=True, mode=0o700)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (out / 'report.md').write_text(
        '# E9 source-mismatch exclusion\n\n'
        f"- Preserved run: `{root}`\n"
        f"- Registered strict new count: {result.get('verified_new_oracles')}\n"
        '- Strict eligible count: 0\n'
        '- The original sealed run was not changed.\n', encoding='utf-8')
    for path in out.iterdir():
        path.chmod(0o600)
    _seal_g(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-source-mismatch', action='store_true')
    args = parser.parse_args()
    action = write_source_mismatch_exclusion if args.exclude_source_mismatch else audit
    print(json.dumps(action(args.run_dir, args.output), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
