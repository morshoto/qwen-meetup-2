import json

import pytest

from src.arc.data import Verifier
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_replication_report import score_replication


def setup():
    descriptor = {'status': 'evaluation_ready', 'task_ids': ['a', 'b', 'c'],
                  'prompt_sha256': 'b' * 64, 'adapter_sha256': 'c' * 64}
    descriptor['sha256'] = digest(descriptor)
    return descriptor, Verifier({k: [[[1]]] for k in descriptor['task_ids']})


def row(descriptor, task, arm, values):
    label = 'evaluation/baseline' if arm == 'base' else 'evaluation/weights'
    variant = None if arm == 'base' else descriptor['adapter_sha256']
    candidates = []
    for index, val in enumerate(values):
        raw = json.dumps([val], separators=(',', ':'))
        candidates.append({'candidate_index': index, 'raw': raw, 'outputs': [val], 'valid': True, 'parse_mode': 'strict'})
    return {'task_id': task, 'status': 'complete', 'candidates': candidates, 'runtime_seconds': 1.0,
            'manifest_sha256': descriptor['sha256'], 'prompt_sha256': descriptor['prompt_sha256'],
            'variant_sha256': variant, 'identity': digest({'manifest': descriptor['sha256'], 'label': label,
                 'task_id': task, 'prompt_sha256': descriptor['prompt_sha256'], 'variant_sha256': variant})}


def test_report_counts_paired_outcomes_and_incomplete_without_raw_answers():
    d, verifier = setup()
    base = [row(d, 'a', 'base', [[[0]], [[0]]]), row(d, 'b', 'base', [[[1]], [[0]]])]
    adapter = [row(d, 'a', 'adapter', [[[1]], [[0]]]), row(d, 'b', 'adapter', [[[0]], [[0]]])]
    report = score_replication(d, base, adapter, verifier)
    assert report['status'] == 'incomplete'
    assert report['paired']['paired'] == {'complete': 2, 'gained': 1, 'lost': 1, 'both_correct': 0, 'both_incorrect': 0}
    assert report['paired']['baseline']['incomplete_tasks'] == 1
    assert report['uncertainty']['discordant'] == 2 and report['uncertainty']['two_sided_p'] == 1.0
    assert report['paired']['comparison'] == 'new exploratory replication tasks'
    assert '[[[1]]]' not in repr(report)


def test_report_rejects_pending_or_altered_registration():
    d, verifier = setup()
    d['status'] = 'answer_audit_pending'
    with pytest.raises(ValueError): score_replication(d, [], [], verifier)
