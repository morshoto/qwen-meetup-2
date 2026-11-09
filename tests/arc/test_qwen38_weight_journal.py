import json

import pytest

from src.arc.qwen38_weight_journal import AttemptJournal


def start(task='a', index=0):
    return {'task_id': task, 'candidate_index': index, 'seed': 7, 'manifest_sha256': 'a' * 64}


def test_journal_survives_restart_and_never_replaces_evidence(tmp_path):
    with AttemptJournal(tmp_path / 'journal') as journal:
        journal.start(start())
        assert journal.read()[0]['status'] == 'started'
        journal.finish(0, {'status': 'complete', 'elapsed_seconds': 1.5, 'raw': 'original'})
        assert journal.read()[0]['raw'] == 'original'
        with pytest.raises(ValueError): journal.finish(0, {'status': 'error', 'elapsed_seconds': 1})
    with AttemptJournal(tmp_path / 'journal') as journal:
        assert journal.read()[0]['raw'] == 'original'
        journal.start(start('b', 0))
        assert journal.read()[-1]['status'] == 'started'
        with pytest.raises(ValueError): journal.start(start('c', 0))


def test_journal_rejects_tampering_and_unknown_files(tmp_path):
    root = tmp_path / 'journal'
    with AttemptJournal(root) as journal:
        journal.start(start()); journal.finish(0, {'status': 'error', 'elapsed_seconds': 1, 'error': 'timeout'})
    result_path = root / '000000.result.json'
    payload = json.loads(result_path.read_text()); payload['start_sha256'] = 'f' * 64
    result_path.write_text(json.dumps(payload))
    with AttemptJournal(root) as journal:
        with pytest.raises(ValueError): journal.read()
    result_path.unlink(); (root / 'unexpected').write_text('x')
    with AttemptJournal(root) as journal:
        with pytest.raises(ValueError): journal.read()


def test_journal_lock_excludes_second_writer(tmp_path):
    root = tmp_path / 'journal'
    with AttemptJournal(root):
        with pytest.raises(RuntimeError):
            with AttemptJournal(root): pass


def test_journal_rejects_reserved_identity_fields_and_invalid_elapsed_replay(tmp_path):
    from src.arc.qwen38_rsi_comparison import digest
    root = tmp_path / 'journal'
    with AttemptJournal(root) as journal:
        with pytest.raises(ValueError): journal.start({**start(), 'status': 'complete'})
        with pytest.raises(ValueError): journal.start({**start(), 'candidate_index': -1})
        journal.start(start()); journal.finish(0, {'status': 'error', 'elapsed_seconds': 1, 'error': 'timeout'})
    path = root / '000000.result.json'
    value = json.loads(path.read_text()); value['elapsed_seconds'] = -1
    value['sha256'] = digest({k: v for k, v in value.items() if k != 'sha256'})
    path.write_text(json.dumps(value))
    with AttemptJournal(root) as journal:
        with pytest.raises(ValueError): journal.read()
