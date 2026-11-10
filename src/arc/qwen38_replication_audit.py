"""Replay the bounded replication allocation audit from frozen source evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.arc.qwen38_replication_allocation import audit_reserved, audit_visible
from src.arc.qwen38_replication_fingerprints import permitted_fingerprints
from src.arc.qwen38_replication_freeze import verify_inventory
from src.arc.qwen38_rsi_comparison import digest

PROPOSAL_SHA256 = '581d0250bf5aa1b9874094e12b16b151fdb9ea5dad8f9ef406279ab2b3f669de'
HISTORY_SHA256 = '911763ed410c9bafc12e4259bd8c29a26c5d00a2e12a3d67c5cce045c0f9cc7b'
SFT_SHA256 = '52d6f734d61dc1aeb90dfc2160246ea62e6d775b58b9756e83e1cb65cb3379a0'
SNAPSHOT_SHA256 = 'ea309257ef1e10e081a68ec6878b22b90ef408806c09f0279989d6095483d16e'
FRESH_START_COUNT = 294


def _valid(value, expected):
    return (isinstance(value, dict) and value.get('sha256') == expected and
            digest({k: v for k, v in value.items() if k != 'sha256'}) == expected)


def build_audit_receipt(proposal, snapshot, history, challenges, official_bytes, solutions_path):
    """Recheck recorded parents and permitted fingerprints without decoding targets."""
    if (not _valid(proposal, PROPOSAL_SHA256) or not _valid(snapshot, SNAPSHOT_SHA256) or
        snapshot.get('status') != 'frozen_inventory' or not _valid(history, HISTORY_SHA256) or
        hashlib.sha256(challenges).hexdigest() != proposal['input_hashes']['challenges'] or
        hashlib.sha256(official_bytes).hexdigest() != proposal['input_hashes']['official_challenges_file']):
        raise ValueError('audit source identity mismatch')
    root = Path(snapshot['source_root'])
    verify_inventory(root, snapshot['files'])
    sft = json.loads((root / 'training/sft-v1/manifest.json').read_bytes())
    if not _valid(sft, SFT_SHA256):
        raise ValueError('SFT source changed')
    starts = sorted((root / 'journal').glob('*.start.json'))
    if len(starts) != FRESH_START_COUNT:
        raise ValueError('fresh generation inventory incomplete')
    fresh = set()
    for i, path in enumerate(starts):
        start = json.loads(path.read_bytes())
        if (path.name != f'{i:06}.start.json' or start.get('sequence') != i or
            start.get('sha256') != digest({k: v for k, v in start.items() if k != 'sha256'})):
            raise ValueError('fresh generation start changed')
        fresh.add(start['request']['task_id'])
    exclusions = dict(proposal['exclusions'])
    extra = {'actual_fresh_generation': sorted(fresh),
             'historical_fp8': history['phase_a_task_ids'] + history['dev_task_ids'],
             'sft_actual': [parent['task_id'] for parent in sft['parents']]}
    if set(exclusions) & set(extra): raise ValueError('exclusion source name collision')
    exclusions.update(extra)
    tasks, official = json.loads(challenges), json.loads(official_bytes)
    selected = proposal['allocation']['validation']
    visible = audit_visible(tasks, selected, exclusions, proposal['known_lineage'], official)
    if visible['challenge_fingerprints'] != {k: proposal['allocation']['challenge_fingerprints'][k] for k in selected}:
        raise ValueError('reserved challenge identity changed')
    blocked = permitted_fingerprints(solutions_path, set(visible['blocked_ids']), set(selected),
                                     proposal['input_hashes']['solutions'])
    reserved = {k: proposal['solution_fingerprints'][k] for k in selected}
    audit = audit_reserved(tasks, selected, exclusions, proposal['known_lineage'], blocked | reserved, official)
    receipt = {**audit, 'task_ids': selected, 'proposal_sha256': PROPOSAL_SHA256,
               'snapshot_sha256': SNAPSHOT_SHA256, 'history_sha256': HISTORY_SHA256, 'sft_sha256': SFT_SHA256,
               'challenge_sha256': proposal['input_hashes']['challenges'],
               'solutions_sha256': proposal['input_hashes']['solutions'],
               'official_sha256': proposal['input_hashes']['official_challenges_file'],
               'exclusions_sha256': digest(exclusions), 'blocked_answer_fingerprints_sha256': digest(blocked),
               'reserved_answer_fingerprints_sha256': digest(reserved), 'fresh_start_count': len(starts),
               'limitations': ['Exact hashes and known lineage only; unrecorded and pretraining exposure remain unknown.',
                               'Reserved answer values were not decoded by this audit.']}
    receipt['sha256'] = digest(receipt)
    return receipt
