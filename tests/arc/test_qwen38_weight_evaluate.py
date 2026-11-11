import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
import src.cli.qwen38_weight_evaluate as evaluate


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'run'; root.mkdir(mode=0o700); (root / 'training').mkdir(mode=0o700)
    baseline = tmp_path / 'old'; baseline.mkdir(); (baseline / 'evaluation').mkdir(); (baseline / 'evaluation/baseline').mkdir()
    comparison = {'allocation': {'evaluation': ['a']}}; (baseline / 'manifest.json').write_text(json.dumps(comparison))
    (baseline / 'evaluation/baseline/a.json').write_text('{}')
    result = {'status': 'trained', 'sha256': 'a' * 64}; (root / 'training/training-result-v1.json').write_text(json.dumps(result))
    selection = {'status': 'eligible_pending_source_replay', 'mode': 'fresh_only', 'counts': {'unique_parents': 5}}
    registration = {'execution_sha256': 'b' * 64}; receipt = {'status': 'matches_supplied_replay', 'manifest_sha256': 'c' * 64}
    monkeypatch.setattr(evaluate, 'replay_bundle', lambda **kwargs: (selection, registration, receipt))
    monkeypatch.setattr(evaluate, 'verify_training_evidence', lambda *args: {'status': 'source_evidence_matches', 'adapter_sha256': 'd' * 64})
    descriptor = {'status': 'evaluation_ready', 'run_dir': str((root / 'evaluation/adapter-v1').resolve()), 'task_ids': ['a']}
    descriptor['sha256'] = digest(descriptor)
    monkeypatch.setattr(evaluate, 'build_evaluation_descriptor', lambda *args: descriptor)
    kwargs = {'manifest': {'run_dir': str(root)}, 'proposal': {}, 'review': {}, 'repo': tmp_path,
              'model_path': tmp_path / 'model', 'challenges': b'{}', 'solutions': b'{}', 'official_bytes': b'{}',
              'canary_dir': tmp_path, 'run_dir': root, 'history_dir': tmp_path, 'comparison_dir': baseline}
    return kwargs, descriptor


def test_evaluation_registration_requires_replayed_training_and_fixed_path(tmp_path, monkeypatch):
    kwargs, descriptor = fixture(tmp_path, monkeypatch)
    result = evaluate.freeze_evaluation(**kwargs)
    path = kwargs['run_dir'] / 'evaluation/evaluation-v1.json'
    assert result == descriptor and json.loads(path.read_text()) == descriptor
    assert evaluate.verify_evaluation_descriptor(**kwargs) == descriptor


def test_evaluation_refuses_changed_training_or_extra_baseline_rows(tmp_path, monkeypatch):
    kwargs, descriptor = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, 'verify_training_evidence', lambda *args: (_ for _ in ()).throw(ValueError('changed')))
    with pytest.raises(ValueError): evaluate.freeze_evaluation(**kwargs)
    assert not (kwargs['run_dir'] / 'evaluation/evaluation-v1.json').exists()
    monkeypatch.setattr(evaluate, 'verify_training_evidence', lambda *args: {'status': 'source_evidence_matches'})
    (kwargs['comparison_dir'] / 'evaluation/baseline/extra.json').write_text('{}')
    with pytest.raises(ValueError): evaluate.freeze_evaluation(**kwargs)


def test_evaluation_steps_never_receive_solutions_and_replay_pairs(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from src.arc.lt_live_backend import PROMPT
    from src.arc.qwen38_weight_generation import capture_response
    kwargs, descriptor = fixture(tmp_path, monkeypatch)
    task = {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
    kwargs['challenges'] = json.dumps({'a': task}).encode()
    kwargs['manifest']['model'] = {'id': 'model'}
    descriptor.update({'candidate_count': 2, 'max_tokens': 4096, 'reserved_completion_tokens': 8192,
                       'wall_seconds_limit': 100, 'per_attempt_seconds': 10,
                       'adapter_sha256': 'd' * 64, 'adapter_config_sha256': 'e' * 64,
                       'comparison_sha256': 'f' * 64, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                       'supervision': {'minimum_free_percent': 10, 'maximum_swap_bytes': 100, 'max_output_bytes': 10000}})
    descriptor['sha256'] = digest({k: v for k, v in descriptor.items() if k != 'sha256'})
    monkeypatch.setattr(evaluate, 'verify_evaluation_descriptor', lambda **kwargs: descriptor)
    (kwargs['run_dir'] / 'evaluation').mkdir(mode=0o700)
    calls = []
    def worker(command, payload, directory, **options):
        calls.append(payload); assert set(payload) == {'request', 'max_tokens'} and 'solutions' not in json.dumps(payload)
        directory = Path(directory); directory.mkdir()
        for name in ('stdout.log', 'stderr.log'): (directory / name).write_text('')
        (directory / 'request.json').write_text(json.dumps(payload))
        execution = {'status': 'complete', 'reason': None, 'exit_code': 0, 'process_group_gone': True,
                     'elapsed_seconds': 1, 'command': command, 'pid': 123,
                     'samples': [{'free_percent': 80, 'swap_bytes': 0}]}
        (directory / 'execution.json').write_text(json.dumps(execution))
        response = {'raw': '[[[4]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                    'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}
        (directory / 'response.json').write_text(json.dumps(capture_response(payload['request'], response, payload['max_tokens'])))
        return execution
    monkeypatch.setattr(evaluate, 'run_supervised', worker)
    assert evaluate.evaluate_step(**kwargs)['status'] == 'attempt_recorded'
    assert evaluate.evaluate_step(**kwargs)['status'] == 'attempt_recorded'
    assert evaluate.evaluate_step(**kwargs)['status'] == 'complete'
    assert [call['request']['seed'] for call in calls] == [20261111, 20261112]


def test_paired_report_uses_offline_answers_and_keeps_raw_outputs_private(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from src.arc.lt_live_backend import PROMPT
    from src.arc.qwen38_weight_generation import capture_response
    kwargs, descriptor = fixture(tmp_path, monkeypatch)
    task = {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
    kwargs['challenges'] = json.dumps({'a': task}).encode(); kwargs['solutions'] = json.dumps({'a': [[[4]]]}).encode()
    kwargs['manifest']['model'] = {'id': 'model'}
    descriptor.update({'candidate_count': 2, 'max_tokens': 4096, 'reserved_completion_tokens': 8192,
                       'wall_seconds_limit': 100, 'per_attempt_seconds': 10,
                       'adapter_sha256': 'd' * 64, 'adapter_config_sha256': 'e' * 64,
                       'comparison_sha256': 'f' * 64, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                       'solution_sha256': hashlib.sha256(kwargs['solutions']).hexdigest(),
                       'supervision': {'minimum_free_percent': 10, 'maximum_swap_bytes': 100, 'max_output_bytes': 10000}})
    descriptor['sha256'] = digest({k: v for k, v in descriptor.items() if k != 'sha256'})
    monkeypatch.setattr(evaluate, 'verify_evaluation_descriptor', lambda **kwargs: descriptor)
    (kwargs['run_dir'] / 'evaluation').mkdir(mode=0o700)
    def worker(command, payload, directory, **options):
        directory = Path(directory); directory.mkdir()
        for name in ('stdout.log', 'stderr.log'): (directory / name).write_text('')
        (directory / 'request.json').write_text(json.dumps(payload))
        execution = {'status': 'complete', 'reason': None, 'exit_code': 0, 'process_group_gone': True,
                     'elapsed_seconds': 1, 'command': command, 'pid': 123,
                     'samples': [{'free_percent': 80, 'swap_bytes': 0}]}
        (directory / 'execution.json').write_text(json.dumps(execution))
        response = {'raw': '[[[4]]]', 'prompt_tokens': 5, 'completion_tokens': 1,
                    'finish_reason': 'stop', 'seconds': 1, 'peak_memory_gb': 1}
        (directory / 'response.json').write_text(json.dumps(capture_response(payload['request'], response, payload['max_tokens'])))
        return execution
    monkeypatch.setattr(evaluate, 'run_supervised', worker)
    assert evaluate.evaluate_step(**kwargs)['status'] == 'attempt_recorded'
    assert evaluate.evaluate_step(**kwargs)['status'] == 'attempt_recorded'
    assert evaluate.evaluate_step(**kwargs)['status'] == 'complete'
    baseline = {'task_id': 'a', 'status': 'complete', 'manifest_sha256': descriptor['comparison_sha256'],
                'prompt_sha256': descriptor['prompt_sha256'], 'variant_sha256': None,
                'runtime_seconds': 2, 'candidates': [{'candidate_index': i, 'raw': '[[[4]]]',
                  'outputs': [[[4]]], 'valid': True, 'parse_mode': 'strict'} for i in range(2)]}
    baseline['identity'] = digest({'manifest': descriptor['comparison_sha256'], 'label': 'evaluation/baseline',
                                   'task_id': 'a', 'prompt_sha256': descriptor['prompt_sha256'], 'variant_sha256': None})
    selection = {'mode': 'fresh_only', 'counts': {'unique_parents': 5}}
    training = {'sha256': 'a' * 64, 'adapter_sha256': descriptor['adapter_sha256']}
    monkeypatch.setattr(evaluate, '_context', lambda **kwargs: (descriptor, {}, training, selection, {'a': json.dumps(baseline).encode()}))
    report = evaluate.report_evaluation(**kwargs)
    assert report['status'] == 'complete' and report['paired']['paired']['both_correct'] == 1
    assert '[[[4]]]' not in json.dumps(report)
    assert evaluate.report_evaluation(**kwargs) == report
    termination_path = Path(descriptor['run_dir']) / 'termination.json'
    termination = json.loads(termination_path.read_text()); termination['status'] = 'resource_stop'
    termination['sha256'] = digest({k: v for k, v in termination.items() if k != 'sha256'})
    termination_path.write_text(json.dumps(termination))
    with pytest.raises(ValueError): evaluate.report_evaluation(**kwargs)


def test_evaluation_termination_must_match_journal(tmp_path, monkeypatch):
    kwargs, descriptor = fixture(tmp_path, monkeypatch)
    root = kwargs['run_dir'] / 'evaluation/adapter-v1'; root.mkdir(parents=True, mode=0o700)
    rows = []
    original = evaluate._finalize_evaluation(root, descriptor, rows, 'complete', 3)
    altered = dict(original, status='resource_stop')
    altered['sha256'] = digest({k: v for k, v in altered.items() if k != 'sha256'})
    (root / 'termination.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError): evaluate._finalize_evaluation(root, descriptor, rows, 'complete', 3)


def test_evaluation_command_reads_only_official_challenges(tmp_path, monkeypatch, capsys):
    from zipfile import ZipFile
    kwargs, descriptor = fixture(tmp_path, monkeypatch)
    for name in ('manifest', 'proposal', 'review', 'challenges', 'solutions'):
        (tmp_path / (name + '.json')).write_text('{}')
    with ZipFile(tmp_path / 'archive.zip', 'w') as archive:
        archive.writestr('arc-agi_evaluation_challenges.json', '{}')
    calls = []
    monkeypatch.setattr(evaluate, 'freeze_evaluation', lambda **values: calls.append(values) or {'status': 'evaluation_ready', 'sha256': 'a' * 64})
    args = ['prepare', '--manifest', str(tmp_path / 'manifest.json'), '--proposal', str(tmp_path / 'proposal.json'),
            '--review', str(tmp_path / 'review.json'), '--repo', str(tmp_path), '--model-path', str(tmp_path),
            '--challenges', str(tmp_path / 'challenges.json'), '--solutions', str(tmp_path / 'solutions.json'),
            '--archive', str(tmp_path / 'archive.zip'), '--canary-dir', str(tmp_path), '--run-dir', str(kwargs['run_dir']),
            '--history-dir', str(tmp_path / 'history'), '--comparison-dir', str(kwargs['comparison_dir'])]
    assert evaluate.main(args) == 0 and calls[0]['official_bytes'] == b'{}'
    assert json.loads(capsys.readouterr().out)['status'] == 'evaluation_ready'
