import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_allocation import fingerprint
from src.arc.qwen38_replication_audit import build_audit_receipt


def signed(value):
    return dict(value, sha256=digest(value))


def fixture(tmp_path, monkeypatch):
    import src.arc.qwen38_replication_audit as audit
    tasks = {str(i): {'train': [{'input': [[i]], 'output': [[i + 1]]}],
                      'test': [{'input': [[i + 100]]}]} for i in range(3)}
    answers = {str(i): [[[i + 20]]] for i in range(3)}
    challenge = json.dumps(tasks).encode(); solutions = json.dumps(answers).encode(); official = b'{}'
    answer_path = tmp_path / 'answers.json'; answer_path.write_bytes(solutions)
    proposal = signed({'allocation': {'validation': ['0'], 'challenge_fingerprints': {k: fingerprint(v) for k,v in tasks.items()}},
                       'exclusions': {'prior': ['1']}, 'known_lineage': {},
                       'solution_fingerprints': {'0': digest(answers['0'])},
                       'input_hashes': {'challenges': hashlib.sha256(challenge).hexdigest(),
                                        'solutions': hashlib.sha256(solutions).hexdigest(),
                                        'official_challenges_file': hashlib.sha256(official).hexdigest()}})
    history = signed({'phase_a_task_ids': ['2'], 'dev_task_ids': []})
    source = tmp_path / 'old'; (source / 'training/sft-v1').mkdir(parents=True); (source / 'journal').mkdir()
    sft = signed({'parents': [{'task_id': '2'}]}); (source / 'training/sft-v1/manifest.json').write_text(json.dumps(sft))
    start = signed({'sequence': 0, 'request': {'task_id': '2'}}); (source / 'journal/000000.start.json').write_text(json.dumps(start))
    from src.arc.qwen38_replication_freeze import inventory_tree
    snapshot = signed({'status': 'frozen_inventory', 'source_root': str(source), 'files': inventory_tree(source)})
    for name, value in [('PROPOSAL_SHA256',proposal),('HISTORY_SHA256',history),('SFT_SHA256',sft),('SNAPSHOT_SHA256',snapshot)]:
        monkeypatch.setattr(audit, name, value['sha256'])
    monkeypatch.setattr(audit, 'FRESH_START_COUNT', 1)
    return proposal, snapshot, history, challenge, official, answer_path


def test_receipt_replays_sources_without_returning_answers(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    result = build_audit_receipt(*args)
    assert result['status'] == 'bounded_disjoint' and result['designation'] == 'exploratory'
    assert result['selected_count'] == 1 and result['blocked_count'] == 2
    assert '[[[20]]]' not in repr(result)
    assert result['sha256'] == digest({k:v for k,v in result.items() if k != 'sha256'})


@pytest.mark.parametrize('change', ['proposal', 'snapshot', 'history', 'challenges', 'official', 'solution', 'journal'])
def test_receipt_rejects_altered_evidence(tmp_path, monkeypatch, change):
    args = list(fixture(tmp_path, monkeypatch))
    if change == 'proposal': args[0]['allocation']['validation'] = ['1']
    elif change == 'snapshot': args[1]['files']['x'] = 'fake'
    elif change == 'history': args[2]['phase_a_task_ids'] = []
    elif change == 'challenges': args[3] += b' '
    elif change == 'official': args[4] += b' '
    elif change == 'solution': args[5].write_text('{}')
    else: (tmp_path / 'old/journal/000000.start.json').write_text('{}')
    with pytest.raises(ValueError): build_audit_receipt(*args)
