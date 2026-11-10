import hashlib
import json

import pytest

from src.arc.data import Verifier
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_replication_report import score_replication


def setup():
    descriptor = {'status': 'evaluation_ready', 'task_ids': ['a', 'b', 'c'],
                  'prompt_sha256': 'b' * 64, 'adapter_sha256': 'c' * 64,
                  'candidate_count': 2, 'max_tokens': 4096,
                  'reserved_completion_tokens': 12 * 4096, 'wall_seconds_limit': 1000}
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


def test_report_requires_terminal_receipt_for_complete_status():
    d, verifier = setup()
    base = [row(d, task, 'base', [[[1]], [[0]]]) for task in d['task_ids']]
    adapter = [row(d, task, 'adapter', [[[1]], [[0]]]) for task in d['task_ids']]
    report = score_replication(d, base, adapter, verifier)
    assert report['paired']['paired']['complete'] == 3
    assert report['status'] == 'incomplete'


def test_report_verifies_terminal_journal_and_registration():
    d, verifier = setup()
    base = [row(d, task, 'base', [[[1]], [[0]]]) for task in d['task_ids']]
    adapter = [row(d, task, 'adapter', [[[1]], [[0]]]) for task in d['task_ids']]
    rows = {'base': base, 'adapter': adapter}
    journal = [{'arm': arm, 'task_id': task, 'candidate_index': i, 'status': 'complete',
                'elapsed_seconds': 1.0, 'seconds': .5,
                'raw_sha256': hashlib.sha256(rows[arm][n]['candidates'][i]['raw'].encode()).hexdigest()}
               for arm in ('base', 'adapter') for n, task in enumerate(d['task_ids']) for i in (0, 1)]
    termination = {'status': 'complete', 'registration_sha256': d['sha256'], 'journal_sha256': digest(journal)}
    termination['sha256'] = digest(termination)
    assert score_replication(d, base, adapter, verifier, journal=journal, termination=termination)['status'] == 'complete'
    termination['journal_sha256'] = '0' * 64
    with pytest.raises(ValueError): score_replication(d, base, adapter, verifier, journal=journal, termination=termination)


def test_report_rejects_rows_not_bound_to_terminal_journal():
    d, verifier = setup()
    base = [row(d, task, 'base', [[[1]], [[0]]]) for task in d['task_ids']]
    adapter = [row(d, task, 'adapter', [[[1]], [[0]]]) for task in d['task_ids']]
    journal = [{'arm': arm, 'task_id': task, 'candidate_index': i, 'status': 'complete',
                'elapsed_seconds': 1.0, 'raw_sha256': '0' * 64}
               for arm in ('base', 'adapter') for task in d['task_ids'] for i in (0, 1)]
    termination = {'status': 'complete', 'registration_sha256': d['sha256'], 'journal_sha256': digest(journal)}
    termination['sha256'] = digest(termination)
    with pytest.raises(ValueError):
        score_replication(d, base, adapter, verifier, journal=journal, termination=termination)


def test_report_preserves_wall_budget_stop_as_incomplete():
    d, verifier = setup()
    journal = []
    termination = {'status': 'time_budget_exhausted', 'registration_sha256': d['sha256'],
                   'journal_sha256': digest(journal), 'wall_seconds': d['wall_seconds_limit']}
    termination['sha256'] = digest(termination)
    report = score_replication(d, [], [], verifier, journal=journal, termination=termination)
    assert report['status'] == 'incomplete'
    assert report['paired']['baseline']['incomplete_tasks'] == 3
    assert report['paired']['baseline']['first_correct'] is None
