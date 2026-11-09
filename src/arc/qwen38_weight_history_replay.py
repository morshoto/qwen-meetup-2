"""Source-bound offline replay of archived FP8 model-generated responses."""
from __future__ import annotations

import hashlib
import json

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_history import audit_history, historical_blocked_ids
from src.arc.qwen38_weight_review import review_selection

PROPOSAL_SHA256 = '581d0250bf5aa1b9874094e12b16b151fdb9ea5dad8f9ef406279ab2b3f669de'
HISTORY_MANIFEST_SHA256 = '911763ed410c9bafc12e4259bd8c29a26c5d00a2e12a3d67c5cce045c0f9cc7b'
HISTORY_AGGREGATE_SHA256 = 'f07534b2a20704cda4eb19879b43d047bfb34d865e94e24028be552ace58f77e'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _valid_digest(value):
    return isinstance(value, dict) and value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'})


def replay_history(*, manifest, proposal, review, challenges, solutions, official_bytes,
                   history_manifest, aggregate_bytes, checkpoint_bytes):
    """Return private verified records; training still requires source preflight and gate."""
    if (not _valid_digest(manifest) or manifest.get('status') != 'execution_ready' or
        not _valid_digest(proposal) or proposal.get('sha256') != PROPOSAL_SHA256 or
        manifest.get('proposal_sha256') != PROPOSAL_SHA256 or not _valid_digest(review) or
        review.get('status') != 'reviewed_bounded' or review.get('proposal_sha256') != PROPOSAL_SHA256 or
        manifest.get('lineage_review_sha256') != review['sha256'] or
        manifest.get('allocation') != proposal['allocation'] or manifest.get('input_hashes') != proposal['input_hashes']):
        raise ValueError('historical replay registration mismatch')
    hashes = proposal['input_hashes']
    official = json.loads(official_bytes)
    if (_sha(challenges) != hashes['challenges'] or _sha(solutions) != hashes['solutions'] or
        _sha(official_bytes) != hashes['official_challenges_file'] or digest(official) != hashes['official_challenges']):
        raise ValueError('historical replay dataset identity mismatch')
    tasks, answers = json.loads(challenges), json.loads(solutions)
    if review_selection(proposal, tasks, answers, official) != review.get('selection'):
        raise ValueError('bounded selection review no longer matches')
    actual_checkpoints = {k: _sha(v) for k, v in checkpoint_bytes.items()}
    if digest(actual_checkpoints) != review['history_inspection']['checkpoint_inventory_sha256']:
        raise ValueError('historical checkpoint inventory changed')
    checkpoints = {k: json.loads(v) for k, v in checkpoint_bytes.items()}
    exclusions = set().union(*(set(v) for k, v in proposal['exclusions'].items() if k != 'prior_phase_a'))
    exclusions.update(proposal['allocation']['fresh']); exclusions.update(proposal['allocation']['validation'])
    blocked = historical_blocked_ids(tasks, official, exclusions, proposal['known_lineage'])
    forbidden = {digest(answers[k]) for k in blocked}
    result = audit_history(tasks=tasks, solutions=answers, manifest=history_manifest,
                           aggregate_bytes=aggregate_bytes, checkpoints=checkpoints, blocked_ids=blocked,
                           expected_aggregate_sha256=HISTORY_AGGREGATE_SHA256,
                           expected_manifest_sha256=HISTORY_MANIFEST_SHA256,
                           expected_challenge_sha256=hashes['challenges'],
                           expected_prompt_sha256=proposal['generation']['prompt_sha256'],
                           forbidden_answer_fingerprints=forbidden)
    for record in result['eligible']:
        record['checkpoint_content_sha256'] = record['checkpoint_sha256']
        record['checkpoint_sha256'] = actual_checkpoints[record['task_id']]
        record['aggregate_sha256'] = HISTORY_AGGREGATE_SHA256
        record['model_id'] = history_manifest['model_id']
        record['model_revision'] = history_manifest['model_revision']
        record['quantization'] = history_manifest['precision']
        record['prompt_sha256'] = history_manifest['prompt_sha256']
    return {'source': 'historical_fp8', 'counts': result['counts'], 'records': result['eligible'],
            'blocked_training_parents': len(blocked), 'limitations': result['limitations']}
