"""Fail-closed local preflight for a separately frozen weight collection."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

from src.arc.qwen38_canary_receipt import verify_canary_bundle
from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_review import PHASE_B_HEAD, HISTORY_MANIFEST_SHA256, HISTORY_AGGREGATE_SHA256

PROPOSAL_SHA256 = '581d0250bf5aa1b9874094e12b16b151fdb9ea5dad8f9ef406279ab2b3f669de'
PINS_PATH = 'research/rsi/qwen38-weight-canary-pins.json'
REVIEW_SOURCE_PATH = 'src/arc/qwen38_weight_review.py'
REVIEW_CLI_PATH = 'src/cli/qwen38_weight_review.py'
PACKAGES = {'mlx': '0.32.4', 'mlx-vlm': '0.6.8', 'mlx-lm': '0.32.2', 'transformers': '5.22.2', 'datasets': '5.1.0'}
SUPERVISION = {'per_attempt_seconds': 600, 'minimum_free_percent': 10,
               'maximum_swap_bytes': 16 * 1024**3, 'max_output_bytes': 2 * 1024**2}
EXECUTION_SOURCES = (
    'src/arc/__init__.py', 'src/arc/agent.py', 'src/arc/guarded_contract.py',
    'src/arc/http_backend.py', 'src/arc/prompting.py', 'src/arc/response_diagnostics.py',
    'src/arc/sandbox.py', 'src/arc/training.py', 'src/arc/data.py', 'src/arc/scoring.py', 'src/arc/lt_direct_format.py', 'src/arc/lt_live_backend.py',
    'src/arc/qwen38_local_runner.py', 'src/arc/qwen38_rsi_comparison.py',
    'src/arc/qwen38_weight_generation.py', 'src/arc/qwen38_weight_schedule.py',
    'src/arc/qwen38_weight_journal.py', 'src/arc/qwen38_weight_supervisor.py',
    'src/arc/qwen38_weight_fresh_verify.py', 'src/arc/qwen38_weight_verification_receipt.py',
    'src/arc/qwen38_weight_canary.py', 'src/arc/qwen38_canary_receipt.py',
    'src/arc/qwen38_weight_preflight.py', 'src/arc/qwen38_weight_review.py',
    'src/arc/qwen38_weight_allocation.py', 'src/arc/qwen38_weight_register.py',
    'src/arc/rsi_task_generation.py', 'src/arc/e5_selfplay.py',
    'src/cli/qwen38_weight_review.py', 'src/cli/qwen38_weight_register.py', 'src/cli/qwen38_canary.py',
    'src/cli/qwen38_weight_worker.py', 'src/cli/qwen38_weight_collect.py', 'src/cli/qwen38_weight_execution.py',
)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _valid_digest(value):
    return isinstance(value, dict) and value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'})


def verify_preflight(manifest, proposal, review, repo, model_path, challenges, solutions, *, canary_dir):
    """Recheck identities and evidence; this function does not dispatch a model."""
    repo, model_path = Path(repo).resolve(), Path(model_path).resolve()
    if (not _valid_digest(manifest) or manifest.get('protocol') != 'qwen38-weight-execution-v1' or
        manifest.get('status') != 'execution_ready' or not _valid_digest(proposal) or
        proposal.get('sha256') != PROPOSAL_SHA256 or proposal.get('status') != 'proposal_not_authorization' or
        manifest.get('proposal_sha256') != PROPOSAL_SHA256):
        raise ValueError('execution or proposal identity mismatch')
    for key in ('allocation', 'generation', 'model', 'input_hashes', 'comparison_sha256'):
        if manifest.get(key) != proposal.get(key):
            raise ValueError(f'execution differs from proposal: {key}')
    if (not _valid_digest(review) or review.get('status') != 'reviewed_bounded' or
        review.get('proposal_sha256') != PROPOSAL_SHA256 or
        review.get('known_lineage_sha256') != digest(proposal['known_lineage']) or
        review.get('unresolved_exposure') is not True or manifest.get('lineage_review_sha256') != review['sha256']):
        raise ValueError('lineage review missing or mismatched')
    selection = review.get('selection', {}); history = review.get('history_inspection', {})
    if (review.get('input_hashes') != proposal['input_hashes'] or review.get('phase_b_head') != PHASE_B_HEAD or
        review.get('review_source_sha256') != _sha(repo / REVIEW_SOURCE_PATH) or
        review.get('review_cli_sha256') != _sha(repo / REVIEW_CLI_PATH) or
        review.get('history_manifest_sha256') != HISTORY_MANIFEST_SHA256 or
        review.get('history_aggregate_sha256') != HISTORY_AGGREGATE_SHA256 or
        history.get('manifest_sha256') != HISTORY_MANIFEST_SHA256 or history.get('aggregate_sha256') != HISTORY_AGGREGATE_SHA256 or
        history.get('parents') != 200 or history.get('candidates') != 400 or
        not isinstance(history.get('checkpoint_inventory_sha256'), str) or len(history['checkpoint_inventory_sha256']) != 64 or
        selection.get('fresh_parents') != len(proposal['allocation']['fresh']) or
        selection.get('validation_parents') != len(proposal['allocation']['validation']) or
        selection.get('official_challenges') != 120 or selection.get('selected_parent_overlap') != 0 or
        selection.get('selected_input_overlap') != 0 or selection.get('unresolved_semantic_exposure') is not True):
        raise ValueError('bounded review evidence incomplete')
    if (hashlib.sha256(challenges).hexdigest() != proposal['input_hashes']['challenges'] or
        hashlib.sha256(solutions).hexdigest() != proposal['input_hashes']['solutions']):
        raise ValueError('training dataset hash mismatch')
    if manifest.get('packages') != PACKAGES or {name: metadata.version(name) for name in PACKAGES} != PACKAGES:
        raise ValueError('runtime package mismatch')
    if manifest.get('supervision') != SUPERVISION:
        raise ValueError('supervision limits mismatch')
    if set(manifest.get('source_files_sha256', {})) != set(EXECUTION_SOURCES):
        raise ValueError('incomplete execution source inventory')
    if any(_sha(repo / name) != manifest['source_files_sha256'][name] for name in EXECUTION_SOURCES):
        raise ValueError('execution source changed')
    identity = proposal['model']
    if (model_path.name != MODEL_REVISION or identity.get('id') != MODEL_ID or
        identity.get('revision') != MODEL_REVISION or identity.get('quantization') != '4-bit MLX'):
        raise ValueError('model identity mismatch')
    files = {path.name: _sha(path) for path in sorted(model_path.iterdir()) if path.is_file()}
    if files != identity.get('files_sha256'):
        raise ValueError('model snapshot file inventory changed')
    pins_path = repo / PINS_PATH
    if _sha(pins_path) != manifest.get('canary_pins_sha256'):
        raise ValueError('canary pins changed')
    pins = json.loads(pins_path.read_bytes())
    if (pins.get('model_id') != MODEL_ID or pins.get('model_revision') != MODEL_REVISION or
        pins.get('comparison_sha256') != proposal['comparison_sha256'] or
        pins.get('base_files_sha256') != {k: v for k, v in files.items() if k.endswith('.safetensors')}):
        raise ValueError('canary pins do not match the registered model')
    canary = verify_canary_bundle(canary_dir, model_path, pins['artifact_sha256'], pins['base_files_sha256'])
    if canary.get('status') != 'passed':
        raise ValueError('canary replay did not pass')
    return {'status': 'passed', 'manifest_sha256': manifest['sha256'],
            'proposal_sha256': PROPOSAL_SHA256, 'lineage_review_sha256': review['sha256']}
