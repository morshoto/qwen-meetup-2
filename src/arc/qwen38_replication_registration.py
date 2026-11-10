"""Construct a provisional answer-blind registration for a frozen replication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION
from src.arc.qwen38_replication_allocation import audit_visible
from src.arc.qwen38_replication_freeze import ADAPTER_SHA256
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_eval_registration import EVALUATION_SOURCES

EXECUTION_SHA256 = '08e4c2cabaca7f331c43a5bb12ddf5e1924ac68e573e3819e1e685044cf1038b'
PROPOSAL_SHA256 = '581d0250bf5aa1b9874094e12b16b151fdb9ea5dad8f9ef406279ab2b3f669de'
EVALUATION_SHA256 = '70fec8e84c0bf03e4bcc524399cd7551405b0708eb4ee9fffa2d2b8a746f3277'
SNAPSHOT_SHA256 = 'ea309257ef1e10e081a68ec6878b22b90ef408806c09f0279989d6095483d16e'
ADAPTER_CONFIG_SHA256 = '8c58de57198438da131ccb7ad55db5a4a2dac90c74e32cc30fc763dfdd503ce0'
AUDIT_SHA256 = '572ce6a11cf545b8245de098c235ef3a81b38d5ef0b3c97484d0759597246b3a'
REPLICATION_SOURCES = tuple(dict.fromkeys(EVALUATION_SOURCES + (
    'src/arc/qwen38_replication_allocation.py', 'src/arc/qwen38_replication_audit.py',
    'src/arc/qwen38_replication_freeze.py',
    'src/arc/qwen38_replication_fingerprints.py',
    'src/arc/qwen38_replication_registration.py', 'src/arc/qwen38_replication_schedule.py',
    'src/arc/qwen38_replication_statistics.py', 'src/arc/qwen38_replication_report.py',
    'src/cli/qwen38_weight_worker.py', 'src/cli/qwen38_replication_worker.py',
)))


def _valid(value, expected):
    return (isinstance(value, dict) and value.get('sha256') == expected and
            digest({k: v for k, v in value.items() if k != 'sha256'}) == expected)


def build_replication_registration(execution, proposal, evaluation, snapshot, challenges, repo,
                                   *, additional_exclusions, official):
    """Describe fixed inputs; caller must replay and verify the sources independently.

    The returned pending descriptor never by itself authorizes a model call.
    """
    if (not _valid(execution, EXECUTION_SHA256) or not _valid(proposal, PROPOSAL_SHA256) or
        not _valid(evaluation, EVALUATION_SHA256) or not _valid(snapshot, SNAPSHOT_SHA256) or
        snapshot.get('status') != 'frozen_inventory' or
        hashlib.sha256(challenges).hexdigest() != execution.get('input_hashes', {}).get('challenges') or
        digest(official) != execution.get('input_hashes', {}).get('official_challenges')):
        raise ValueError('replication provenance mismatch')
    model = execution.get('model', {})
    if (model.get('id') != MODEL_ID or model.get('revision') != MODEL_REVISION or
        model.get('quantization') != '4-bit MLX' or not model.get('files_sha256') or
        evaluation.get('adapter_sha256') != ADAPTER_SHA256 or evaluation.get('adapter_config_sha256') != ADAPTER_CONFIG_SHA256 or
        evaluation.get('candidate_count') != 2 or
        evaluation.get('temperature') != .6 or evaluation.get('max_tokens') != 4096 or
        evaluation.get('max_context') != 18432 or evaluation.get('format_recovery') is not True):
        raise ValueError('replication model or evaluation settings mismatch')
    selected = proposal['allocation']['validation']
    exclusions = {**proposal['exclusions']}
    if set(exclusions) & set(additional_exclusions):
        raise ValueError('additional exclusion name collides')
    exclusions.update(additional_exclusions)
    visible = audit_visible(json.loads(challenges), selected, exclusions, proposal['known_lineage'], official)
    repo = Path(repo)
    sources = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in REPLICATION_SOURCES}
    result = {'protocol': 'qwen38-frozen-replication-v1', 'status': 'answer_audit_pending',
              'inference_authorized': False, 'designation': 'exploratory', 'task_ids': selected,
              'task_fingerprints': visible['challenge_fingerprints'],
              'visible_audit_sha256': digest(visible), 'challenge_sha256': execution['input_hashes']['challenges'],
              'solution_sha256': execution['input_hashes']['solutions'],
              'exclusions_sha256': digest(exclusions), 'official_challenges_sha256': digest(official),
              'official_file_sha256': execution['input_hashes']['official_challenges_file'],
              'execution_sha256': EXECUTION_SHA256, 'proposal_sha256': PROPOSAL_SHA256,
              'original_evaluation_sha256': EVALUATION_SHA256, 'source_inventory_sha256': SNAPSHOT_SHA256,
              'model': model, 'adapter_sha256': ADAPTER_SHA256, 'adapter_config_sha256': ADAPTER_CONFIG_SHA256,
              'prompt_sha256': evaluation['prompt_sha256'],
              'candidate_count': 2, 'temperature': .6, 'max_tokens': 4096, 'max_context': 18432,
              'format_recovery': True, 'seed_rule': '20261111 + candidate_index',
              'reserved_completion_tokens': len(selected) * 4 * 4096, 'wall_seconds_limit': 86400,
              'per_attempt_seconds': 600, 'source_files_sha256': sources,
              'decision_rule': 'paired gains and losses with exact two-sided sign test; no tuning',
              'limitations': ['Unknown semantic and pretraining exposure.', 'Previously examined 50 are exploratory.',
                              'Two recorded candidates are not assumed independent.']}
    result['sha256'] = digest(result)
    return result


def build_ready_registration(pending, audit):
    """Bind the pinned audit; caller must first replay pending and audit sources."""
    if (not isinstance(pending, dict) or not _valid(pending, pending.get('sha256')) or
        pending.get('status') != 'answer_audit_pending' or pending.get('inference_authorized') is not False or
        not _valid(audit, AUDIT_SHA256) or audit.get('status') != 'bounded_disjoint' or
        audit.get('designation') != 'exploratory' or audit.get('unresolved_exposure') is not True or
        audit.get('task_ids') != pending.get('task_ids') or
        audit.get('proposal_sha256') != pending.get('proposal_sha256') or
        audit.get('snapshot_sha256') != pending.get('source_inventory_sha256') or
        audit.get('challenge_sha256') != pending.get('challenge_sha256') or
        audit.get('official_sha256') != pending.get('official_file_sha256') or
        audit.get('solutions_sha256') != pending.get('solution_sha256') or
        audit.get('exclusions_sha256') != pending.get('exclusions_sha256')):
        raise ValueError('replication audit does not bind registration')
    ready = {k: v for k, v in pending.items() if k != 'sha256'}
    ready.update({'status': 'evaluation_ready', 'inference_authorized': True, 'audit_sha256': AUDIT_SHA256})
    ready['sha256'] = digest(ready)
    return ready
