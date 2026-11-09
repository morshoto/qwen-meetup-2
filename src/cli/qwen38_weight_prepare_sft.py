"""Replay fresh and, only when needed, historical records for provisional SFT."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_history import historical_blocked_ids
from src.arc.qwen38_weight_history_replay import HISTORY_AGGREGATE_SHA256, replay_history
from src.arc.qwen38_weight_preflight import EXECUTION_SOURCES
from src.arc.qwen38_weight_sft import select_training_examples
from src.arc.qwen38_weight_token_count import make_trainer_counter
from src.cli.qwen38_weight_fresh_replay import replay_fresh

SFT_SOURCES = tuple(dict.fromkeys(EXECUTION_SOURCES + (
    'src/arc/qwen38_weight_data_audit.py', 'src/arc/qwen38_weight_history.py',
    'src/arc/qwen38_weight_history_replay.py', 'src/arc/qwen38_weight_sft.py',
    'src/arc/qwen38_weight_sft_bundle.py', 'src/arc/qwen38_weight_token_count.py',
    'src/cli/qwen38_weight_fresh_replay.py', 'src/cli/qwen38_weight_prepare_sft.py',
    'src/arc/qwen38_weight_train.py', 'src/cli/qwen38_weight_train_worker.py',
    'src/cli/qwen38_canary_worker.py', 'src/cli/qwen38_canary_verify.py',
    'src/cli/qwen38_local_train.py', 'src/cli/qwen38_training_data.py',
    'src/cli/qwen38_expand_training.py', 'src/cli/qwen38_weight_train.py',
)))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_history(root):
    root = Path(root)
    historical = json.loads((root / 'manifest.json').read_bytes())
    ids = historical['phase_a_task_ids']
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not re.fullmatch('[A-Za-z0-9_-]+', i) for i in ids):
        raise ValueError('invalid historical task inventory')
    return {'history_manifest': historical, 'aggregate_bytes': (root / 'phase_a_results.json').read_bytes(),
            'checkpoint_bytes': {i: (root / 'phase_a' / f'{i}.json').read_bytes() for i in ids}}


def prepare_selection(manifest, proposal, review, repo, model_path, challenges, solutions,
                      official_bytes, canary_dir, run_dir, history_dir, *, token_counter=None):
    """Recompute source evidence before choosing examples; does not train."""
    fresh = replay_fresh(manifest, proposal, review, repo, model_path, challenges, solutions, canary_dir, run_dir)
    hashes = proposal['input_hashes']
    official = json.loads(official_bytes)
    if (_sha(challenges) != hashes['challenges'] or _sha(solutions) != hashes['solutions'] or
        _sha(official_bytes) != hashes['official_challenges_file'] or digest(official) != hashes['official_challenges'] or
        manifest['input_hashes'] != hashes):
        raise ValueError('SFT source dataset identity mismatch')
    tasks, answers = json.loads(challenges), json.loads(solutions)
    excluded = set().union(*(set(ids) for key, ids in proposal['exclusions'].items() if key != 'prior_phase_a'))
    excluded.update(proposal['allocation']['validation'])
    blocked = historical_blocked_ids(tasks, official, excluded, proposal['known_lineage'])
    forbidden = {digest(answers[task_id]) for task_id in blocked}
    if set(proposal['allocation']['fresh']) & blocked:
        raise ValueError('fresh pool overlaps blocked lineage')
    counter = token_counter if token_counter is not None else make_trainer_counter(model_path)
    options = {'blocked_ids': blocked, 'forbidden_answer_fingerprints': forbidden, 'token_counter': counter}
    selection = select_training_examples(tasks, answers, fresh['records'], [], **options)
    historical = None
    if selection['counts']['unique_parents'] < 5:
        historical = replay_history(manifest=manifest, proposal=proposal, review=review, challenges=challenges,
                                    solutions=solutions, official_bytes=official_bytes, **load_history(history_dir))
        selection = select_training_examples(tasks, answers, fresh['records'], historical['records'], **options)
    files = {name: _sha((Path(repo) / name).read_bytes()) for name in SFT_SOURCES}
    registration = {'execution_sha256': manifest['sha256'], 'fresh_journal_sha256': fresh['journal_sha256'],
                    'fresh_verification_sha256': fresh['verification_sha256'], 'challenge_sha256': hashes['challenges'],
                    'solution_sha256': hashes['solutions'], 'source_files_sha256': files, 'training_code_sha256': digest(files),
                    'historical_aggregate_sha256': HISTORY_AGGREGATE_SHA256 if historical is not None else None,
                    'historical_checkpoint_inventory_sha256': review['history_inspection']['checkpoint_inventory_sha256'] if historical is not None else None,
                    'packages': manifest['packages'], 'model_files_sha256': manifest['model']['files_sha256'],
                    'training_settings': manifest['training_settings']}
    return selection, registration


def _bundle_path(manifest, run_dir, *, create_parent):
    root = Path(run_dir).resolve()
    if manifest.get('run_dir') != str(root) or not root.is_dir() or root.stat().st_mode & 0o077:
        raise ValueError('SFT destination differs from the registered private run')
    parent = root / 'training'
    if create_parent:
        parent.mkdir(mode=0o700, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink() or parent.stat().st_mode & 0o077:
        raise ValueError('training directory is not private')
    return parent / 'sft-v1'


def prepare_bundle(**kwargs):
    from src.arc.qwen38_weight_sft_bundle import write_bundle
    selection, registration = prepare_selection(**kwargs)
    path = _bundle_path(kwargs['manifest'], kwargs['run_dir'], create_parent=True)
    return write_bundle(path, selection, registration)


def replay_bundle(**kwargs):
    from src.arc.qwen38_weight_sft_bundle import verify_bundle
    selection, registration = prepare_selection(**kwargs)
    path = _bundle_path(kwargs['manifest'], kwargs['run_dir'], create_parent=False)
    result = verify_bundle(path, selection, registration)
    return selection, registration, result
