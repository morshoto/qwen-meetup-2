"""Replay the bounded allocation and historical-source review without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from src.arc.qwen38_local_runner import parse_prediction
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import freeze_registration
from src.arc import qwen38_weight_review as review
from src.cli.qwen38_weight_register import load_sources, propose


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inspect_history(root, challenge_sha256, prompt_sha256, allowed_tasks):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    raw = (root / 'phase_a_results.json').read_bytes()
    if (manifest.get('sha256') != review.HISTORY_MANIFEST_SHA256 or
        digest({k: v for k, v in manifest.items() if k != 'sha256'}) != review.HISTORY_MANIFEST_SHA256 or
        _sha(raw) != review.HISTORY_AGGREGATE_SHA256 or
        manifest.get('protocol') != 'qwen38-quick-exploratory-v1' or
        manifest.get('model_id') != 'Qwen/Qwen3.8-27B-FP8' or
        manifest.get('model_revision') != 'c3a462cc3d38cf351f3b26cbd763469eeb74ef81' or
        manifest.get('precision') != 'fp8' or manifest.get('candidate_count') != 2 or
        manifest.get('challenge_sha256') != challenge_sha256 or manifest.get('prompt_sha256') != prompt_sha256):
        raise ValueError('historical source identity mismatch')
    aggregate = json.loads(raw); ids = manifest['phase_a_task_ids']; rows = aggregate['task_results']
    if (aggregate.get('manifest') != manifest or aggregate.get('complete') is not True or
        len(ids) != len(set(ids)) or set(rows) != set(ids) or not set(ids) <= allowed_tasks.keys()):
        raise ValueError('historical source is incomplete')
    checkpoint_hashes = {}; candidate_count = 0
    for task_id in ids:
        raw_checkpoint = (root / 'phase_a' / f'{task_id}.json').read_bytes()
        row = json.loads(raw_checkpoint)
        if (row != rows[task_id] or row.get('task_id') != task_id or row.get('status') != 'complete' or
            row.get('manifest_sha256') != manifest['sha256'] or row.get('test_count') != len(allowed_tasks[task_id]['test']) or
            len(row.get('candidates', [])) != 2):
            raise ValueError('historical checkpoint mismatch')
        for index, candidate in enumerate(row['candidates']):
            if candidate.get('candidate_index') != index or not isinstance(candidate.get('raw'), str):
                raise ValueError('missing historical response')
            parsed, mode = parse_prediction(candidate['raw'], row['test_count'])
            if parsed != candidate.get('outputs') or (parsed is not None) != candidate.get('valid') or mode != candidate.get('parse_mode'):
                raise ValueError('historical response differs from saved parse')
            candidate_count += 1
        checkpoint_hashes[task_id] = _sha(raw_checkpoint)
    return {'manifest_sha256': manifest['sha256'], 'aggregate_sha256': _sha(raw),
            'parents': len(ids), 'candidates': candidate_count, 'checkpoint_inventory_sha256': digest(checkpoint_hashes)}


def run_review(*, repo, data_repo, baseline, lineage, phase_b_comparison, phase_b_repo, history_dir, proposal_path, output):
    repo = Path(repo).resolve()
    proposal = json.loads(Path(proposal_path).read_bytes())
    source = load_sources(repo, data_repo, baseline, lineage, phase_b_comparison)
    # Rebuild the whole proposal from the source files; never overwrite the committed proposal.
    with tempfile.TemporaryDirectory(prefix='qwen38-review-') as temp:
        replayed = propose(repo, data_repo, baseline, lineage, phase_b_comparison, Path(temp) / 'proposal.json')
    if replayed != proposal:
        raise ValueError('proposal no longer matches its source inventory')
    recorded_phase_b = subprocess.run(['git', '-C', str(phase_b_repo), 'show',
                                       review.PHASE_B_HEAD + ':configs/lt.phase-b.comparison-proposal.json'],
                                      capture_output=True, check=True, timeout=10).stdout
    if _sha(recorded_phase_b) != source['source_hashes']['phase_b_comparison']:
        raise ValueError('Phase B proposal differs from the pinned PR head')
    tasks = json.loads(source['challenges_bytes']); solutions = json.loads(source['solutions_bytes'])
    observed = review.review_selection(proposal, tasks, solutions, source['official'])
    history = inspect_history(history_dir, proposal['input_hashes']['challenges'],
                              proposal['generation']['prompt_sha256'],
                              {k: tasks[k] for k in proposal['exclusions']['prior_phase_a']})
    result = review.bind_review(proposal, observed, proposal['input_hashes'],
                                _sha((repo / 'src/arc/qwen38_weight_review.py').read_bytes()),
                                history['manifest_sha256'], history['aggregate_sha256'], review.PHASE_B_HEAD)
    result.pop('sha256')
    result['history_inspection'] = history
    result['review_cli_sha256'] = _sha((repo / 'src/cli/qwen38_weight_review.py').read_bytes())
    return freeze_registration(output, result)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for flag in ('repo', 'data-repo', 'baseline', 'lineage', 'phase-b-comparison', 'phase-b-repo', 'history-dir', 'proposal', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv)
    result = run_review(repo=args.repo, data_repo=args.data_repo, baseline=args.baseline, lineage=args.lineage,
                        phase_b_comparison=args.phase_b_comparison, phase_b_repo=args.phase_b_repo,
                        history_dir=args.history_dir, proposal_path=args.proposal, output=args.output)
    print(json.dumps({'status': result['status'], 'sha256': result['sha256'],
                      'selection': result['selection'], 'history_inspection': result['history_inspection']}, sort_keys=True))


if __name__ == '__main__':
    main()
