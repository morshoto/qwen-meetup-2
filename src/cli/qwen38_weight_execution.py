"""Freeze a source-bound collection descriptor only after full preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.arc import qwen38_weight_preflight as preflight
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import freeze_registration


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_descriptor(repo, proposal, review):
    if (proposal.get('sha256') != preflight.PROPOSAL_SHA256 or
        digest({k: v for k, v in proposal.items() if k != 'sha256'}) != preflight.PROPOSAL_SHA256 or
        proposal.get('status') != 'proposal_not_authorization' or review.get('status') != 'reviewed_bounded' or
        review.get('proposal_sha256') != preflight.PROPOSAL_SHA256 or
        review.get('sha256') != digest({k: v for k, v in review.items() if k != 'sha256'})):
        raise ValueError('proposal or bounded review is invalid')
    repo = Path(repo)
    result = {'protocol': 'qwen38-weight-execution-v1', 'status': 'execution_ready',
              'proposal_sha256': proposal['sha256'], 'lineage_review_sha256': review['sha256'],
              **{k: proposal[k] for k in ('allocation', 'generation', 'model', 'input_hashes', 'comparison_sha256')},
              'training_gate': proposal.get('training_gate'), 'training_settings': proposal.get('training_settings'),
              'packages': preflight.PACKAGES, 'supervision': preflight.SUPERVISION,
              'source_files_sha256': {name: _sha(repo / name) for name in preflight.EXECUTION_SOURCES},
              'canary_pins_sha256': _sha(repo / preflight.PINS_PATH)}
    result['sha256'] = digest(result)
    return result


def freeze_execution(repo, proposal, review, model_path, challenges, solutions, canary_dir, output):
    descriptor = make_descriptor(repo, proposal, review)
    result = preflight.verify_preflight(descriptor, proposal, review, repo, model_path, challenges, solutions,
                                        canary_dir=canary_dir)
    if result.get('status') != 'passed':
        raise ValueError('execution preflight did not pass')
    return freeze_registration(output, {k: v for k, v in descriptor.items() if k != 'sha256'})


def main(argv=None):
    parser = argparse.ArgumentParser()
    for flag in ('repo', 'proposal', 'review', 'model-path', 'challenges', 'solutions', 'canary-dir', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv)
    result = freeze_execution(Path(args.repo).resolve(), json.loads(Path(args.proposal).read_bytes()),
                              json.loads(Path(args.review).read_bytes()), Path(args.model_path).resolve(),
                              Path(args.challenges).read_bytes(), Path(args.solutions).read_bytes(),
                              Path(args.canary_dir).resolve(), Path(args.output))
    print(json.dumps({'status': result['status'], 'sha256': result['sha256']}, sort_keys=True))


if __name__ == '__main__':
    main()
