"""Prepare a non-authorizing, immutable weight-experiment allocation proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_rsi_comparison import digest, validate_allocation
from src.arc.rsi_task_generation import generation_slots
from src.arc.qwen38_weight_allocation import freeze_registration
from src.arc.qwen38_weight_register import audited_registration

COMPARISON_SHA256 = 'cff354ce06e7e1604725f87f1567b34f0d1408e57e2b73c1d3889e2d9fe2f626'


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def load_sources(repo, data_repo, baseline_path, lineage_path, phase_b_comparison_path):
    repo, data_repo = Path(repo), Path(data_repo)
    sources = {
        'phase_a_proposal': repo / 'configs/lt.phase-a.comparison-proposal.json',
        'phase_b_reservations': repo / 'configs/lt.phase-b.training-pool.json',
        'prior_rsi_split': repo / 'research/rsi/exploratory-split-proposal.json',
        'prior_provisional_split': repo / 'research/rsi/provisional-split.json',
        'exposure_review': repo / 'research/rsi/extended-exposure-review.json',
        'bounded_exposure': repo / 'research/rsi/bounded-exposure-proposal.json',
        'phase_b_comparison': Path(phase_b_comparison_path),
        'lineage_register': Path(lineage_path), 'comparison_manifest': Path(baseline_path),
    }
    raw = {name: path.read_bytes() for name, path in sources.items()}
    decoded = {name: json.loads(content) for name, content in raw.items()}
    challenges_bytes = (data_repo / 'data/raw/arc-agi_training_challenges.json').read_bytes()
    solutions_bytes = (data_repo / 'data/raw/arc-agi_training_solutions.json').read_bytes()
    with ZipFile(data_repo / 'arc-prize-2025.zip') as archive:
        official_bytes = archive.read('arc-agi_evaluation_challenges.json')
        if (archive.read('arc-agi_training_challenges.json') != challenges_bytes or
            archive.read('arc-agi_training_solutions.json') != solutions_bytes):
            raise ValueError('training files differ from competition archive')
    official = json.loads(official_bytes)
    hashes = {name: _hash(content) for name, content in raw.items()}
    hashes['official_challenges'] = digest(official)
    hashes['official_challenges_file'] = _hash(official_bytes)
    registry = decoded['bounded_exposure']
    scheduled = sorted(slot['task_id'] for index in (0, 1)
                       for slot in generation_slots(registry['partitions']['generation'], index))
    return {'scheduled': scheduled, 'phase_b_comparison': decoded['phase_b_comparison']['task_ids'],
            'bounded_registry': registry, 'challenges_bytes': challenges_bytes,
            'solutions_bytes': solutions_bytes,
            'official': official, 'baseline': decoded['comparison_manifest'],
            'proposal': decoded['phase_a_proposal'], 'phase_b': decoded['phase_b_reservations'],
            'prior': decoded['prior_rsi_split'], 'exposure': decoded['exposure_review'],
            'lineage': decoded['lineage_register'], 'source_hashes': hashes,
            'provisional': decoded['prior_provisional_split']}


def propose(repo, data_repo, baseline_path, lineage_path, phase_b_comparison_path, output):
    source = load_sources(repo, data_repo, baseline_path, lineage_path, phase_b_comparison_path)
    baseline, proposal, prior, exposure = (source[k] for k in ('baseline', 'proposal', 'prior', 'exposure'))
    provisional = source.pop('provisional')
    registry = source.pop('bounded_registry')
    if registry['task_source_sha256'] != _hash(source['challenges_bytes']) or set(registry['known_exposed_task_ids']) != set(proposal['known_exposure_exclusions']):
        raise ValueError('bounded exposure inventory differs')
    for key, value in [('proposal_sha256', proposal), ('prior_split_sha256', prior),
                       ('extended_exposure_sha256', exposure), ('provisional_split_sha256', provisional)]:
        if baseline.get(key) != digest(value):
            raise ValueError(f'comparison source mismatch: {key}')
    tasks = json.loads(source['challenges_bytes'])
    validate_allocation(baseline['allocation'], proposal, tasks)
    source['expected_comparison_sha256'] = COMPARISON_SHA256
    source['prompt'] = PROMPT
    result = audited_registration(**source)
    result['status'] = 'proposal_not_authorization'
    result['limitations'] = ['Lineage register and historical exposure require independent review.',
                             'No collection or evaluation is authorized by this proposal.']
    return freeze_registration(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--data-repo', required=True)
    parser.add_argument('--comparison-manifest', required=True)
    parser.add_argument('--lineage-register', required=True)
    parser.add_argument('--phase-b-comparison', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = propose(args.repo, args.data_repo, args.comparison_manifest, args.lineage_register, args.phase_b_comparison, args.output)
    print(json.dumps({'status': result['status'], 'sha256': result['sha256'],
                      'fresh': len(result['allocation']['fresh']),
                      'validation': len(result['allocation']['validation']),
                      'shortfall': result['allocation']['shortfall']}, indent=2))


if __name__ == '__main__':
    main()
