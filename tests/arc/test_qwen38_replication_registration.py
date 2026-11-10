import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_replication_registration import build_replication_registration


def signed(d):
    d['sha256'] = digest(d)
    return d


def fixture(tmp_path, monkeypatch):
    import src.arc.qwen38_replication_registration as r
    tasks = {str(i): {'train': [{'input': [[i]], 'output': [[i + 1]]}],
                      'test': [{'input': [[i + 100]]}]} for i in range(3)}
    raw = json.dumps(tasks, sort_keys=True).encode()
    selected = ['0', '1']
    execution = signed({'input_hashes': {'challenges': hashlib.sha256(raw).hexdigest(), 'solutions': 'e' * 64,
                                         'official_challenges_file': 'd' * 64, 'official_challenges': digest({})},
                        'model': {'id': r.MODEL_ID, 'revision': r.MODEL_REVISION, 'quantization': '4-bit MLX',
                                  'files_sha256': {'weight': 'b' * 64}}})
    proposal = signed({'allocation': {'validation': selected}, 'known_lineage': {}, 'exclusions': {'prior': ['2']}})
    evaluation = signed({'adapter_sha256': r.ADAPTER_SHA256, 'adapter_config_sha256': r.ADAPTER_CONFIG_SHA256, 'prompt_sha256': 'a' * 64,
                         'candidate_count': 2, 'temperature': .6, 'max_tokens': 4096,
                         'max_context': 18432, 'format_recovery': True})
    snapshot = signed({'status': 'frozen_inventory', 'files': {'result': 'c' * 64}})
    for name, obj in [('EXECUTION_SHA256', execution), ('PROPOSAL_SHA256', proposal),
                      ('EVALUATION_SHA256', evaluation), ('SNAPSHOT_SHA256', snapshot)]:
        monkeypatch.setattr(r, name, obj['sha256'])
    monkeypatch.setattr(r, 'REPLICATION_SOURCES', ('logic.py',))
    (tmp_path / 'logic.py').write_text('code')
    return execution, proposal, evaluation, snapshot, raw, tmp_path


def test_registration_is_fixed_and_never_authorizes_inference(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    result = build_replication_registration(*args, additional_exclusions={}, official={})
    assert result['task_ids'] == ['0', '1']
    assert result['status'] == 'answer_audit_pending' and result['inference_authorized'] is False
    assert result['adapter_sha256'] == args[2]['adapter_sha256']
    assert result['candidate_count'] == 2 and result['seed_rule'] == '20261111 + candidate_index'
    assert result['reserved_completion_tokens'] == 2 * 2 * 2 * 4096
    assert '[[100]]' not in json.dumps(result)


@pytest.mark.parametrize('change', ['execution', 'proposal', 'evaluation', 'snapshot', 'challenge', 'overlap'])
def test_registration_rejects_evidence_drift_or_visible_overlap(tmp_path, monkeypatch, change):
    args = list(fixture(tmp_path, monkeypatch)); exclusions = {}
    if change == 'execution': args[0]['model']['revision'] = 'other'
    elif change == 'proposal': args[1]['allocation']['validation'].reverse()
    elif change == 'evaluation': args[2]['temperature'] = .7
    elif change == 'snapshot': args[3]['files']['result'] = 'x'
    elif change == 'challenge': args[4] += b' '
    else: exclusions = {'new_exposure': ['0']}
    with pytest.raises(ValueError):
        build_replication_registration(*args, additional_exclusions=exclusions, official={})


def test_registration_binds_exclusion_and_official_evidence(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    first = build_replication_registration(*args, additional_exclusions={}, official={})
    second = build_replication_registration(*args, additional_exclusions={'repeat': ['2']}, official={})
    assert first['exclusions_sha256'] != second['exclusions_sha256']
    assert first['official_challenges_sha256'] == digest({})
    assert first['exclusions_sha256'] == digest({'prior': ['2']})


def test_source_inventory_includes_offline_report():
    import src.arc.qwen38_replication_registration as registration
    assert 'src/arc/qwen38_replication_report.py' in registration.REPLICATION_SOURCES


def test_registration_contains_pinned_visible_task_fingerprints(tmp_path, monkeypatch):
    from src.arc.qwen38_weight_allocation import fingerprint
    args = fixture(tmp_path, monkeypatch)
    result = build_replication_registration(*args, additional_exclusions={}, official={})
    tasks = json.loads(args[4])
    assert result['task_fingerprints'] == {key: fingerprint(tasks[key]) for key in ('0', '1')}


def test_registration_pins_adapter_configuration_and_worker(tmp_path, monkeypatch):
    import src.arc.qwen38_replication_registration as registration
    args = fixture(tmp_path, monkeypatch)
    args[2]['adapter_config_sha256'] = 'd' * 64
    args[2]['sha256'] = digest({k: v for k, v in args[2].items() if k != 'sha256'})
    monkeypatch.setattr(registration, 'EVALUATION_SHA256', args[2]['sha256'])
    monkeypatch.setattr(registration, 'ADAPTER_CONFIG_SHA256', 'd' * 64)
    result = build_replication_registration(*args, additional_exclusions={}, official={})
    assert result['adapter_config_sha256'] == 'd' * 64
    monkeypatch.undo()
    assert 'src/cli/qwen38_replication_worker.py' in registration.REPLICATION_SOURCES


def test_registration_pins_solution_source_and_fingerprint_auditor(tmp_path, monkeypatch):
    import src.arc.qwen38_replication_registration as registration
    args = fixture(tmp_path, monkeypatch)
    args[0]['input_hashes']['solutions'] = 'e' * 64
    args[0]['sha256'] = digest({k: v for k, v in args[0].items() if k != 'sha256'})
    monkeypatch.setattr(registration, 'EXECUTION_SHA256', args[0]['sha256'])
    result = build_replication_registration(*args, additional_exclusions={}, official={})
    assert result['solution_sha256'] == 'e' * 64
    monkeypatch.undo()
    assert 'src/arc/qwen38_replication_fingerprints.py' in registration.REPLICATION_SOURCES


def test_source_inventory_includes_replayable_exposure_audit():
    import src.arc.qwen38_replication_registration as registration
    assert 'src/arc/qwen38_replication_audit.py' in registration.REPLICATION_SOURCES


def test_ready_registration_requires_exact_bound_exposure_receipt(tmp_path, monkeypatch):
    import src.arc.qwen38_replication_registration as r
    args = fixture(tmp_path, monkeypatch)
    pending = build_replication_registration(*args, additional_exclusions={}, official={})
    receipt = signed({'status': 'bounded_disjoint', 'designation': 'exploratory', 'unresolved_exposure': True,
                      'task_ids': pending['task_ids'], 'proposal_sha256': pending['proposal_sha256'],
                      'snapshot_sha256': pending['source_inventory_sha256'], 'challenge_sha256': pending['challenge_sha256'],
                      'solutions_sha256': pending['solution_sha256'], 'exclusions_sha256': pending['exclusions_sha256'],
                      'official_sha256': pending['official_file_sha256']})
    monkeypatch.setattr(r, 'AUDIT_SHA256', receipt['sha256'])
    ready = r.build_ready_registration(pending, receipt)
    assert ready['status'] == 'evaluation_ready' and ready['inference_authorized'] is True
    assert ready['audit_sha256'] == receipt['sha256']
    assert ready['sha256'] == digest({k: v for k, v in ready.items() if k != 'sha256'})
    receipt['task_ids'] = ['other']
    with pytest.raises(ValueError): r.build_ready_registration(pending, receipt)


def test_ready_registration_rejects_mismatched_official_source(tmp_path, monkeypatch):
    import src.arc.qwen38_replication_registration as r
    args = fixture(tmp_path, monkeypatch)
    pending = build_replication_registration(*args, additional_exclusions={}, official={})
    receipt = signed({'status': 'bounded_disjoint', 'designation': 'exploratory', 'unresolved_exposure': True,
                      'task_ids': pending['task_ids'], 'proposal_sha256': pending['proposal_sha256'],
                      'snapshot_sha256': pending['source_inventory_sha256'], 'challenge_sha256': pending['challenge_sha256'],
                      'solutions_sha256': pending['solution_sha256'], 'exclusions_sha256': pending['exclusions_sha256'],
                      'official_sha256': 'f' * 64})
    monkeypatch.setattr(r, 'AUDIT_SHA256', receipt['sha256'])
    with pytest.raises(ValueError): r.build_ready_registration(pending, receipt)
